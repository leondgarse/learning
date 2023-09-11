# Learning
***
- [2006.11239](https://arxiv.org/abs/2006.11239)
- [Github hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)
- [Github labmlai/annotated_deep_learning_paper_implementations/diffusion](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/ddpm)
```py
import torch
import torchvision
import unet
import denoise_diffusion


def mnist(image_size=32):
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size), torchvision.transforms.ToTensor()])
    return torchvision.datasets.MNIST("datasets", train=True, download=True, transform=transform)


def cifar10(image_size=32):
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size), torchvision.transforms.ToTensor()])
    return torchvision.datasets.CIFAR10("datasets", train=True, download=True, transform=transform)

image_size = 64
batch_size = 96
learning_rate = 1e-4
device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) > 0 else torch.device("cpu")

data_loader = torch.utils.data.DataLoader(cifar10(image_size), batch_size, shuffle=True, pin_memory=True)
xx, _ = next(iter(data_loader))
image_channels, image_size = xx.shape[1], xx.shape[2]
print("image_channels = {}, image_size = {}".format(image_channels, image_size))
print("xx.min() = {}, xx.max() = {}".format(xx.min(), xx.max()))

eps_model = unet.UNet(image_channels=image_channels, n_blocks=2, n_channels=64, is_attn=[False, False, True, True])
_ = eps_model.to(device)
optimizer = torch.optim.Adam(eps_model.parameters(), lr=learning_rate)
ddpm = denoise_diffusion.DenoiseDiffusion(model=eps_model)


""" Training """


import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def cosine_with_warmup_lr(it, learning_rate=6e-4, warmup_iters=3, lr_decay_iters=30, min_lr=1e-6):
    if it < warmup_iters:  # linear warmup for warmup_iters steps
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:  # it > lr_decay_iters, return min learning rate
        return min_lr
    # in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

epochs = 100
lr_decay_func = lambda step: cosine_with_warmup_lr(step, learning_rate=learning_rate, warmup_iters=10, lr_decay_iters=epochs)

save_path = "checkpoints"
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
eval_x0 = torch.randn([16, image_channels, image_size, image_size]).to(device)

hists = {"lr": [], "loss": []}
bar_format = "{n_fmt}/{total_fmt} [{bar:30}] - ETA: {elapsed}<{remaining} {rate_fmt}{postfix}{desc}"
for epoch_id in range(epochs):
    lr = lr_decay_func(epoch_id + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch {}/{}, lr: {}".format(epoch_id + 1, epochs. lr))

    process_bar = tqdm(enumerate(data_loader), total=len(data_loader), bar_format=bar_format, ascii=".>>=")
    avg_loss = 0.0
    for batch, (xx, _) in process_bar:
        optimizer.zero_grad()
        loss = ddpm.loss(xx.to(device))
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        process_bar.desc = " - loss: {:.4f}".format(avg_loss / (batch + 1))
        process_bar.refresh()
        # if batch == 10:
        #     break

    hists["lr"].append(lr)
    hists["loss"].append(avg_loss / (batch + 1))
    with open(os.path.join(save_path, "test_cifar10_hist.json"), "w") as ff:
        json.dump(hists, ff)

    torch.save(eps_model.state_dict(), os.path.join(save_path, "test_cifar10.pt"))
    eval_xt = ddpm.generate(x0=eval_x0, return_inner=False).permute([0, 2, 3, 1]).cpu().numpy()
    eval_xt = eval_xt[:, :, :, 0] if eval_xt.shape[-1] == 1 else eval_xt
    eval_xt = np.vstack([np.hstack(eval_xt[row * 4: row * 4 + 4]) for row in range(4)])
    eval_xt = np.clip(eval_xt, 0, 1)
    plt.imsave(os.path.join(save_path, "epoch_{}.jpg".format(epoch_id + 1)), eval_xt)
    print("")
print(hists)
```
