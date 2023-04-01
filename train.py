import os
import math
import time
import torch
import inspect
import numpy as np
from torch import nn
from torch.nn import functional as F
from contextlib import nullcontext

GLOBAL_CONTEXT = nullcontext()


def configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device_type="cuda"):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
    # will appear in the no_decay and decay sets respectively after the above.
    # In addition, because named_parameters() doesn't return duplicates, it
    # will only return the first occurence, key'd by 'transformer.wte.weight', below.
    # so let's manually remove 'lm_head.weight' from decay set. This will include
    # this tensor into optimization via transformer.wte.weight only, and not decayed.
    decay.remove("lm_head.weight")

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    use_fused = (device_type == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
    print(f"using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    return optimizer


# helps estimate an arbitrarily accurate loss over either split using many batches
def estimate_loss(model, dataset, eval_iters=200):
    with torch.no_grad():
        model.eval()
        # losses = torch.zeros(eval_iters)
        losses = 0
        for iter in range(eval_iters):
            xx, yy = dataset.get_random_batch()
            with GLOBAL_CONTEXT:
                logits, loss = model(xx, yy)
            losses += loss.item()
        model.train()
    return losses / eval_iters


# learning rate decay scheduler (cosine with warmup)
def cosine_with_warmup_lr(it, learning_rate=6e-4, warmup_iters=2000, lr_decay_iters=600000, min_lr=6e-5):
    if it < warmup_iters:  # linear warmup for warmup_iters steps
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:  # it > lr_decay_iters, return min learning rate
        return min_lr
    # in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def train(
    model,
    optimizer,
    train_data,
    val_data,
    max_iters=600000,
    eval_interval=2000,
    gradient_accumulation_steps=5,
    log_interval=1,
    out_dir="checkpoints",
    device="cuda:0",
):
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda"))
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=torch.float16)

    iter_num = 0
    best_val_loss = 1e9
    train_x, train_y = train_data.get_random_batch()
    while iter_num < max_iters:
        t0 = time.time()
        lr = cosine_with_warmup_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num > 0 and iter_num % eval_interval == 0:
            train_loss = estimate_loss(model, train_data)
            val_loss = estimate_loss(model, val_data)
            print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                pre_best_ckpt = os.path.join(out_dir, "ckpt_val_loss_{:.4f}.pt".format(best_val_loss))
                if os.path.exists(pre_best_ckpt):
                    os.remove(pre_best_ckpt)

                best_val_loss = val_loss
                checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt_val_loss_{:.4f}.pt".format(val_loss)))
            torch.save(checkpoint, os.path.join(out_dir, "ckpt_latest.pt"))

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for _ in range(gradient_accumulation_steps):
            with GLOBAL_CONTEXT:
                logits = model(train_x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), train_y.view(-1), ignore_index=-1)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            train_x, train_y = train_data.get_random_batch()
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip the gradient
        scaler.step(optimizer)  # step the optimizer and scaler if training in fp16
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  # flush the gradients as soon as we can, no need for this memory anymore

        if iter_num % log_interval == 0:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            dt = time.time() - t0
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1


if __name__ == "__main__":
    from gpt2 import GPT2
    from datasets import TinyShakespeare
    from train import configure_optimizers, train

    device = "cuda:0"
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    model = GPT2(n_layer=6, vocab_size=50304)  # defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)
    model.to(device)
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device_type=device_type)
    # model = torch.compile(model) # requires PyTorch 2.0

    # poor man's data loader
    save_path, block_size, batch_size = "data", 1024, 12
    train_data = TinyShakespeare(save_path=save_path, split="train", block_size=block_size, batch_size=batch_size, device=device)
    val_data = TinyShakespeare(save_path=save_path, split="val", block_size=block_size, batch_size=batch_size, device=device)

    train(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        max_iters=60000,
        eval_interval=2000,
        gradient_accumulation_steps=5,
        log_interval=1,
        out_dir="checkpoints",
        device=device,
    )
