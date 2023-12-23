import os
import time
import torch
import numpy as np
import torch._dynamo.config
import torch._inductor.config
from torch.distributed._functional_collectives import all_reduce
# import matplotlib.pyplot as plt

import model, int8_quant


if hasattr(torch._inductor.config, "coordinate_descent_tuning"):
    torch._inductor.config.coordinate_descent_tuning = True
if hasattr(torch._inductor.config, "triton") and hasattr(torch._inductor.config.triton, "unique_kernel_names"):
    torch._inductor.config.triton.unique_kernel_names = True
if hasattr(torch._inductor.config, "fx_graph_cache"):
    torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else "cpu"
GLOBAL_PRECISSION = torch.float16 if GLOBAL_DEVICE == "cuda" else torch.float32
GLOBAL_DIST_BACKEND = "nccl" if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else "gloo"
GLOBAL_CONTEXT = torch.autocast(device_type=GLOBAL_DEVICE, dtype=GLOBAL_PRECISSION)

def maybe_init_dist():
    try:
        rank = int(os.environ.get("LOCAL_RANK", "0"))  # provided by torchrun
        world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))  # provided by torchrun
        print(f"{rank = }, {world_size = }")

        if world_size < 2:
            return None  # too few gpus to parallelize, tp is no-op
    except KeyError:
        return None  # not run via torchrun, no-op

    backend = "nccl" if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else "gloo"
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank


def apply_tensor_parallel_kecam(model):
    for layer in model.layers:
        if isinstance(layer, layers.Reshape):  # and ("query" in layer.name or "key" in layer.name or "value" in layer.name):
            print("[reshape], layer.name:", layer.name, "layer.target_shape:", layer.target_shape)
            layer.target_shape[1] //= LOCAL_WORLD_SIZE  # Split num_heads
            continue

        if not isinstance(layer, layers.Dense) or layer.name in ["lm_head"]:
            continue

        if layer.name.endswith("mlp.down_proj") or layer.name.endswith("o_proj"):  # row shard, shard on inputs
            print("[row shard] layer.name:", layer.name, "LOCAL_RANK:", LOCAL_RANK)
            shard_weights = torch.tensor_split(layer.module.weight, LOCAL_WORLD_SIZE, dim=1)[LOCAL_RANK]
            layer.module.weight = torch.nn.Parameter(shard_weights, requires_grad=False)
            layer.module.in_features //= LOCAL_WORLD_SIZE

            if LOCAL_WORLD_SIZE >= 2:
                layer.module.register_forward_hook(lambda _module, _input, output: all_reduce(output, "sum", list(range(LOCAL_WORLD_SIZE))))
        else:  # col shard, shard on outputs
            print("[col shard] layer.name:", layer.name, "LOCAL_RANK:", LOCAL_RANK)
            shard_weights = torch.tensor_split(layer.module.weight, LOCAL_WORLD_SIZE, dim=0)[LOCAL_RANK]
            layer.module.weight = torch.nn.Parameter(shard_weights, requires_grad=False)
            layer.module.out_features //= LOCAL_WORLD_SIZE

def apply_tp_linear_inplace(linear_layer, is_split_out=False):
    dim = 1 if is_split_out else 0
    shard_weights = torch.tensor_split(linear_layer.weight, LOCAL_WORLD_SIZE, dim=dim)[LOCAL_RANK]
    linear_layer.weight = torch.nn.Parameter(shard_weights, requires_grad=False)

    if hasattr(linear_layer, "scales") and not is_split_out:
        linear_layer.scales = torch.tensor_split(linear_layer.scales, LOCAL_WORLD_SIZE, dim=dim)[LOCAL_RANK]

    if is_split_out:
        linear_layer.out_features //= LOCAL_WORLD_SIZE
    else:
        linear_layer.in_features //= LOCAL_WORLD_SIZE

def apply_tensor_parallel_torch(model):
    model.n_head //= LOCAL_WORLD_SIZE
    model.n_local_heads //= LOCAL_WORLD_SIZE
    model.dim //= LOCAL_WORLD_SIZE

    for transformer_block in model.blocks:
        apply_tp_linear_inplace(transformer_block.mlp.gate_proj)
        apply_tp_linear_inplace(transformer_block.mlp.up_proj)
        apply_tp_linear_inplace(transformer_block.mlp.down_proj, is_split_out=True)
        if LOCAL_WORLD_SIZE >= 2:
            transformer_block.mlp.register_forward_hook(lambda _module, _input, output: all_reduce(output, "sum", list(range(LOCAL_WORLD_SIZE))))

        apply_tp_linear_inplace(transformer_block.self_attn.q_proj)
        apply_tp_linear_inplace(transformer_block.self_attn.k_proj)
        apply_tp_linear_inplace(transformer_block.self_attn.v_proj)
        apply_tp_linear_inplace(transformer_block.self_attn.o_proj, is_split_out=True)
        transformer_block.self_attn.n_head //= LOCAL_WORLD_SIZE
        transformer_block.self_attn.n_local_heads //= LOCAL_WORLD_SIZE
        transformer_block.self_attn.dim //= LOCAL_WORLD_SIZE
        if LOCAL_WORLD_SIZE >= 2:
            transformer_block.self_attn.register_forward_hook(lambda _module, _input, output: all_reduce(output, "sum", list(range(LOCAL_WORLD_SIZE))))


def decode_one_token(model, inputs, input_pos):
    out = model(inputs, input_pos)
    if GLOBAL_DEVICE == "cuda":
        torch.cuda.synchronize()
    return out

if __name__ == "__main__":
    """ Quant -> Tensor Parallel -> To device and precission -> Compile -> Speculative """
    maybe_init_dist()

    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    tt = model.LLaMA2_1B()
    quant_save_path = (tt.name if hasattr(tt, 'name') else tt.__class__.__name__) + '_int8.pth'
    if os.path.exists("llama2_1b.pt") and not os.path.exists(quant_save_path):
        ss = torch.load('llama2_1b.pt')
        tt.load_state_dict({ii: ss['state_dict'][('_'.join(ii.split('.')[:-1]) + '.' + ii.split('.')[-1])] for ii in tt.state_dict().keys()})
    # tt = tt.to(device=GLOBAL_DEVICE, dtype=GLOBAL_PRECISSION).eval()

    print(">>>> Quant") if LOCAL_RANK == 0 else None
    if not os.path.exists(quant_save_path):
        quantized_state_dict = int8_quant.create_quantized_state_dict(tt)
        torch.save(quantized_state_dict, quant_save_path)
    else:
        quantized_state_dict = torch.load(quant_save_path)
    int8_quant.replace_linear_weight_only_int8_per_channel(tt)
    tt.load_state_dict(quantized_state_dict, assign=True)

    if LOCAL_WORLD_SIZE >= 2:
        print(">>>> Tensor Parallel") if LOCAL_RANK == 0 else None
        print("     [Before] Total parameters:", sum([int(np.prod(ii.shape)) for ii in tt.parameters()])) if LOCAL_RANK == 0 else None
        # apply_tensor_parallel_kecam(tt)
        apply_tensor_parallel_torch(tt)
        print("     [After] Total parameters:", sum([int(np.prod(ii.shape)) for ii in tt.parameters()])) if LOCAL_RANK == 0 else None

    print(">>>> To device and precission") if LOCAL_RANK == 0 else None
    tt = tt.to(device=GLOBAL_DEVICE, dtype=GLOBAL_PRECISSION).eval()

    print(">>>> compile") if LOCAL_RANK == 0 else None
    # global decode_one_token
    decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

    repeat = 100
    with torch.no_grad(), torch.device(GLOBAL_DEVICE):
        tt.setup_caches(max_batch_size=1, max_seq_length=2048)
        print(">>>> Warmup") if LOCAL_RANK == 0 else None
        for id in range(5):
            inputs = torch.randint(low=0, high=32000, size=[1, 1])
            input_pos = torch.ones([1], dtype=torch.int64) + id
            print(decode_one_token(tt, inputs, input_pos).shape) if LOCAL_RANK == 0 else None

        print(">>>> Repeat test") if LOCAL_RANK == 0 else None
        times = []
        for id in range(repeat):
            inputs = torch.randint(low=0, high=32000, size=[1, 1])
            input_pos = torch.ones([1], dtype=torch.int64) + id
            ss = time.time()
            out = decode_one_token(tt, inputs, input_pos)
            times.append((time.time() - ss) * 1000)
    print("Mean of time(ms) token for the inner 80%:", np.mean(sorted(times)[len(times) // 10: -len(times) // 10])) if LOCAL_RANK == 0 else None
    # plt.plot(times)
