import os
import torch

GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else "cpu"
GLOBAL_PRECISSION = torch.float16 if GLOBAL_DEVICE == "cuda" else torch.float32


def find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else (n + k - (n % k))


def _check_linear_int4_k(k, groupsize=1, inner_k_tiles=1):
    return k % groupsize == 0 and k % (inner_k_tiles * 16) == 0


def get_group_qparams(w, n_bit=4, groupsize=128):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(GLOBAL_PRECISSION).reshape(w.shape[0], -1), zeros.to(GLOBAL_PRECISSION).reshape(w.shape[0], -1)


def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int32 = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int).to(torch.int32).reshape_as(w)
    return w_int32


def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == GLOBAL_PRECISSION
    assert zeros.dtype == GLOBAL_PRECISSION
    rr = torch.cat([scales.reshape(scales.size(0), scales.size(1), 1), zeros.reshape(zeros.size(0), zeros.size(1), 1)], 2)
    return rr.transpose(0, 1).contiguous()


def prepare_int4_weight_and_scales_and_zeros(weight_bf16, groupsize, inner_k_tiles):
    n_bit = 4
    scales, zeros = get_group_qparams(weight_bf16, n_bit, groupsize)
    weight_int32 = group_quantize_tensor_from_qparams(weight_bf16, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)

    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_int32, inner_k_tiles)
    return weight_int4pack, scales_and_zeros


def create_quantized_state_dict(model, groupsize=128, inner_k_tiles=8, padding=True):
    assert groupsize in [32, 64, 128, 256]
    assert inner_k_tiles in [2, 4, 8]

    cur_state_dict = model.state_dict()
    with torch.no_grad():
        for fqn, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert not mod.bias
                out_features = mod.out_features
                in_features = mod.in_features
                assert out_features % 8 == 0, "require out_features % 8 == 0"

                weight = mod.weight.data
                print(f"linear: {fqn}, in={in_features}, out={out_features}, weight={weight.shape}, {weight.dtype}, {weight.device}")
                if not _check_linear_int4_k(in_features, groupsize, inner_k_tiles):
                    if padding:
                        from model import find_multiple

                        print(f"warning: {fqn} is padded to satisfy in_features % 1024 == 0")
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = torch.nn.functional.pad(weight, pad=(0, padded_in_features - in_features))
                    else:
                        print(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                weight = weight.to(GLOBAL_PRECISSION).to(GLOBAL_DEVICE)
                weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(weight, groupsize, inner_k_tiles)
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to("cpu")
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to("cpu")

    return cur_state_dict


def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype=None,
        groupsize: int = 128,
        inner_k_tiles: int = 8,
        padding: bool = True,
    ) -> None:
        super().__init__()
        self.padding = padding
        if padding:
            from model import find_multiple

            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, "require in_features % (innerKTiles * 16) == 0"
        self.register_buffer("weight", torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32))
        self.register_buffer("scales_and_zeros", torch.empty((in_features // groupsize, out_features, 2), dtype=GLOBAL_PRECISSION))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(GLOBAL_PRECISSION)
        if self.padding:
            input = torch.nn.functional.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(input, self.weight, self.scales_and_zeros, self.out_features, self.groupsize)


def replace_linear_int4(module, groupsize=128, inner_k_tiles=8, padding=True):
    for name, child in module.named_children():
        if not isinstance(child, nn.Linear):
            replace_linear_int4(child, groupsize, inner_k_tiles, padding)
            continue

        if _check_linear_int4_k(child.in_features, groupsize, inner_k_tiles):
            linear = WeightOnlyInt4Linear(child.in_features, child.out_features, bias=False, groupsize=groupsize, inner_k_tiles=inner_k_tiles, padding=False)
            setattr(module, name, linear)
        elif padding:
            linear = WeightOnlyInt4Linear(child.in_features, child.out_features, bias=False, groupsize=groupsize, inner_k_tiles=inner_k_tiles, padding=True)
            setattr(module, name, linear)
