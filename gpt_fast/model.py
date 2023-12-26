import torch
from torch import nn, Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else (n + k - (n % k))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.float16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # print(f"{input_pos.shape[0] = }, {k_val.shape[2] = }")
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        # print(">>>> self.k_cache.shape:", self.k_cache.shape, "self.v_cache.shape:", self.v_cache.shape)
        # print(">>>> k_val.shape:", k_val.shape, "v_val.shape:", v_val.shape)

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, n_layer=32, vocab_size=32000, block_size=16384, rope_base=10000, dim=4096, n_head=32, n_local_heads=-1, norm_eps=1e-5) -> None:
        super().__init__()
        self.vocab_size, self.dim, self.n_head, self.block_size, self.rope_base = vocab_size, dim, n_head, block_size, rope_base
        self.n_local_heads = n_local_heads if n_local_heads > 0 else n_head
        self.intermediate_size = find_multiple(int(2 * 4 * self.dim / 3), 256)

        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(TransformerBlock(dim, n_head, self.n_local_heads, self.intermediate_size, norm_eps) for _ in range(n_layer))
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.freqs_cis = None
        self.mask_cache = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        # self.run_prediction = RunPrediction(self, tokenizer="SentencePieceTokenizer")
        self.max_block_size = 2048
        self.output_shape = (None, vocab_size)

    def setup_caches(self, max_batch_size, max_seq_length, dtype=torch.float16):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.dim // self.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.blocks:
            b.self_attn.kv_cache = KVCache(max_batch_size, max_seq_length, self.n_local_heads, head_dim, dtype=dtype)

        self.freqs_cis = precompute_freqs_cis(self.block_size, self.dim // self.n_head, self.rope_base)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos=None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.embed_tokens(idx)

        for i, layer in enumerate(self.blocks):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_head, n_local_heads, intermediate_size, norm_eps=1e-5) -> None:
        super().__init__()
        self.self_attn = Attention(dim, n_head, n_local_heads)
        self.mlp = FeedForward(dim, intermediate_size)
        self.post_attention_layernorm = RMSNorm(dim, norm_eps)
        self.input_layernorm = RMSNorm(dim, norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x), freqs_cis, mask, input_pos)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Attention(nn.Module):
    def __init__(self, dim, n_head, n_local_heads):
        super().__init__()
        assert dim % n_head == 0
        head_dim = dim // n_head

        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(dim, n_head * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_local_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_local_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None
        self.dim, self.n_head, self.n_local_heads, self.head_dim = dim, n_head, n_local_heads, head_dim

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos=None) -> Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_head, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        repeat = self.n_head // self.n_local_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        # print(">>>> q.shape:", q.shape, "k.shape:", k.shape, "v.shape:", v.shape, "mask.shape:", mask.shape)
        # y = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=mask.contiguous(), dropout_p=0.0)
        y = F.softmax((q @ k.transpose(2, 3)) / (float(self.head_dim) ** 0.5) * mask, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.o_proj(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, dim, intermediate_size) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.up_proj(x)) * self.gate_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.gamma


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.float16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], 2, -1).permute([0, 1, 2, 4, 3])
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    left = xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1]
    right = xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1]
    return torch.stack([left, right], -2).flatten(3).type_as(x)


def LLaMA2_1B(vocab_size=32003, **kwargs):
    model = Transformer(n_layer=22, vocab_size=vocab_size, block_size=8192, rope_base=10000, dim=2048, n_head=32, n_local_heads=4, **kwargs)
    model.name = 'llama2_1b'
    return model


def LLaMA2_7B(vocab_size=32000, **kwargs):
    model = Transformer(n_layer=32, vocab_size=vocab_size, block_size=16384, rope_base=10000, dim=4096, n_head=32, n_local_heads=-1, **kwargs)
    model.name = 'llama2_7b'
    return model
