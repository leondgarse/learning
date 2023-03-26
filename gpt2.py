import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, block_size=1024, n_embd=768, n_head=12, bias=True, dropout=0.0):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.attn_out = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        causal_mask = (1 - np.tri(block_size).astype("float32")[None, None]) * -1e10
        self.register_buffer("causal_mask", torch.from_numpy(causal_mask), persistent=False)
        self.n_head, self.block_size, self.n_embd = n_head, block_size, n_embd

    def forward(self, inputs):
        batch, blocks, channels = inputs.size()
        key_dim = channels // self.n_head
        qq_scale = 1.0 / (float(key_dim) ** 0.5)

        # efficient attention using Flash Attention CUDA kernels
        # torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        qq, kk, vv = self.qkv(inputs).split(self.n_embd, dim=-1)
        qq = qq.view(batch, blocks, self.n_head, key_dim).transpose(1, 2)
        kk = kk.view(batch, blocks, self.n_head, key_dim).permute([0, 2, 3, 1])
        vv = vv.view(batch, blocks, self.n_head, key_dim).transpose(1, 2)

        # print(f"{qq.shape = }, {kk.shape = }, {vv.shape = }, {self.causal_mask.shape = }")
        attn = (qq @ kk) * qq_scale + self.causal_mask[:, :, :blocks, :blocks]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ vv
        out = out.transpose(1, 2).contiguous().view(batch, blocks, channels)
        out = self.attn_out(out)
        out = self.output_dropout(out)
        return out


class AttnMlpBlock(nn.Module):
    def __init__(self, block_size=1024, n_embd=768, n_head=12, bias=True, dropout=0.0):
        super().__init__()
        self.attn_ln = nn.LayerNorm(n_embd)  # bias=bias
        self.attn = CausalSelfAttention(block_size, n_embd, n_head, bias, dropout)
        self.mlp_ln = nn.LayerNorm(n_embd)  # bias=bias
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, inputs):
        attn_out = inputs + self.attn(self.attn_ln(inputs))
        mlp_out = attn_out + self.mlp(self.mlp_ln(attn_out))
        return mlp_out


class GPT2(nn.Module):
    def __init__(self, n_layer=12, vocab_size=50304, block_size=1024, n_embd=768, n_head=12, bias=True, dropout=0.0):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)  # Encoder
        self.wpe = nn.Embedding(block_size, n_embd)  # Encoder

        self.drop = nn.Dropout(dropout)
        blocks = [AttnMlpBlock(block_size, n_embd, n_head, bias, dropout) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*blocks)
        self.ln_f = nn.LayerNorm(n_embd)  # bias=bias

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # may not necessary, https://paperswithcode.com/method/weight-tying
        self.block_size = block_size

    def forward(self, idx, targets=None):
        batch, blocks = idx.size()
        pos_idx = torch.arange(0, blocks, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos_idx)
        out = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            out = block(out)
        out = self.ln_f(out)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(out)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(out[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=40, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    """Load wights from transformers"""

    import gpt2

    config_args = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768, vocab_size=50257),  # 124M params
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257),  # 350M params
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280, vocab_size=50257),  # 774M params
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600, vocab_size=50257),  # 1558M params
    }

    def weight_name_map(weight_name):
        weight_name = weight_name.replace("transformer.h.", "blocks.")
        weight_name = weight_name.replace("transformer.", "")
        weight_name = weight_name.replace(".ln_1.", ".attn_ln.")
        weight_name = weight_name.replace(".attn.c_attn.", ".attn.qkv.")
        weight_name = weight_name.replace(".attn.c_proj.", ".attn.attn_out.")
        weight_name = weight_name.replace(".ln_2.", ".mlp_ln.")
        weight_name = weight_name.replace(".mlp.c_fc.", ".mlp.0.")
        weight_name = weight_name.replace(".mlp.c_proj.", ".mlp.2.")
        return weight_name

    def convert_gpt2_state_dict(state_dict):
        need_transpose_sufixes = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        need_transpose = lambda weight_name: any([weight_name.endswith(ii) for ii in need_transpose_sufixes])
        exclude_sufixes = [".attn.masked_bias", ".attn.bias"]
        exclude = lambda weight_name: any([weight_name.endswith(ii) for ii in exclude_sufixes])
        result = {weight_name_map(kk): vv.T if need_transpose(kk) else vv for kk, vv in state_dict.items() if not exclude(kk)}
        return result

    model_type = "gpt2"
    model = GPT2(**config_args[model_type])
    # print({kk: vv.shape for kk, vv in model.state_dict().items()})

    import torch
    from transformers import GPT2LMHeadModel

    source_state_dict = GPT2LMHeadModel.from_pretrained(model_type).state_dict()
    target_state_dict = convert_gpt2_state_dict(model.state_dict())
    model.load_state_dict(target_state_dict)
    torch.save(model.state_dict(), "gpt2.pt")

    """ Run evaluation """

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    start = "hello world"
    start_ids = enc.encode(start)
    inputs = torch.tensor(start_ids, dtype=torch.long)[None]

    # run generation
    num_samples = 10  # number of samples to draw
    max_new_tokens = 500  # number of tokens generated in each sample
    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability

    for k in range(num_samples):
        out = model.generate(inputs, max_new_tokens, temperature=temperature, top_k=top_k)
        print(enc.decode(out[0].tolist()))
        print("---------------")
