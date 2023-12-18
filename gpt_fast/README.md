- [Github pytorch-labs/gpt-fast](https://github.com/pytorch-labs/gpt-fast)
- [Colab gpt_fast_test.ipynb](https://colab.research.google.com/drive/1C3w-jI3BinXuqDqaS3e2lvU8TUkOf0nH?usp=sharing)
- Install torch nightly from [Start Locally Torch](https://pytorch.org/get-started/locally/)
  ```sh
  pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
  pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
  ```
- **int8 / int4 quant -> tensor parallel -> compile**
- **Run parallel**
  ```py
  import os
  import torch
  from typing import List, Optional

  def maybe_init_dist() -> Optional[int]:
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

  if __name__ == "__main__":
      print(f"{maybe_init_dist() = }")
  ```
  ```sh
  torchrun --standalone --nproc_per_node=2 torch_parallel.py
  ```
- **Model**
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'
  import torch
  from keras_cv_attention_models import llama2
  model = llama2.LLaMA2_42M()

  GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else "cpu"
  GLOBAL_PRECISSION = torch.bfloat16 if GLOBAL_DEVICE == "cuda" else torch.float32
  GLOBAL_DIST_BACKEND = "nccl" if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else "gloo"
  ```
- **compile**
  ```py
  if compile:
      if is_speculative and use_tp:
          torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case

      if is_speculative:
          global model_forward, logits_to_prob
          model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

      global decode_one_token, prefill
      decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)


  import torch

  def model_forward(model, x, input_pos):
      return model(x, input_pos)

  model = model.to(device=GLOBAL_DEVICE, dtype=GLOBAL_PRECISSION)
  model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)
  ```
- **int8 quant**
  ```py
  import torch

  def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
      # assumes symmetric quantization, assumes axis == 0, assumes dense memory format
      # default setup for affine quantization of activations
      eps = torch.finfo(torch.float32).eps
      # get min and max
      min_val, max_val = torch.aminmax(x, dim=1)
      # calculate scales and zero_points based on min and max, reference: https://fburl.com/code/srbiybme
      min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
      max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
      device = min_val_neg.device

      # reference: https://fburl.com/code/4wll53rk
      max_val_pos = torch.max(-min_val_neg, max_val_pos)
      scales = max_val_pos / (float(quant_max - quant_min) / 2)
      # ensure scales is the same dtype as the original tensor
      scales = torch.clamp(scales, min=eps).to(x.dtype)
      zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

      # quantize based on qmin/qmax/scales/zp
      # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
      x_div = x / scales.unsqueeze(-1)
      x_round = torch.round(x_div)
      x_zp = x_round + zero_points.unsqueeze(-1)
      quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

      return quant, scales, zero_points

  class WeightOnlyInt8Linear(torch.nn.Module):
      __constants__ = ['in_features', 'out_features']
      in_features: int
      out_features: int
      weight: torch.Tensor

      def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
          factory_kwargs = {'device': device, 'dtype': dtype}
          super().__init__()
          self.in_features = in_features
          self.out_features = out_features
          self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
          self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

      def forward(self, input: torch.Tensor) -> torch.Tensor:
          return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales

  def create_quantized_state_dict(model):
      cur_state_dict = model.state_dict()
      for fqn, mod in model.named_modules():
          if isinstance(mod, torch.nn.Linear):
              int8_weight, scales, _ = dynamically_quantize_per_channel(mod.weight.float(), -128, 127, torch.int8)
              cur_state_dict[f"{fqn}.weight"] = int8_weight
              cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype)

      return cur_state_dict

  def replace_linear_weight_only_int8_per_channel(module):
      for name, child in module.named_children():
          if isinstance(child, torch.nn.Linear):
              setattr(module, name, WeightOnlyInt8Linear(child.in_features, child.out_features))
          else:
              replace_linear_weight_only_int8_per_channel(child)

  quantized_state_dict = create_quantized_state_dict(model)
  save_path = (model.name if hasattr(model, 'name') else model.__class__.__name__) + '_int8.pth'
  torch.save(quantized_state_dict, save_path)

  replace_linear_weight_only_int8_per_channel(model)
  model.load_state_dict(quantized_state_dict, assign=True)
  model = model.to(device=GLOBAL_DEVICE, dtype=GLOBAL_PRECISSION).eval()
  ```
- **Int4 quant [CUDA only as torch.ops.aten._convert_weight_to_int4pack not supporting CPU]**
  ```py
  def find_multiple(n: int, k: int) -> int:
      return n if n % k == 0 else (n + k - (n % k))

  def _check_linear_int4_k(k, groupsize = 1, inner_k_tiles = 1):
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
      return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(torch.bfloat16).reshape(w.shape[0], -1)

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
      w_int32 = (to_quant.sub(min_val) .div(scales) .round() .clamp_(min_int, max_int) .to(torch.int32) .reshape_as(w))
      return w_int32

  def pack_scales_and_zeros(scales, zeros):
      assert scales.shape == zeros.shape
      assert scales.dtype == torch.bfloat16
      assert zeros.dtype == torch.bfloat16
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
                  print(f"linear: {fqn}, in={in_features}, out={out_features}")

                  weight = mod.weight.data
                  if not _check_linear_int4_k(in_features, groupsize, inner_k_tiles):
                      if padding:
                          from model import find_multiple
                          print(f"warning: {fqn} is padded to satisfy in_features % 1024 == 0")
                          padded_in_features = find_multiple(in_features, 1024)
                          weight = torch.nn.functional.pad(weight, pad=(0, padded_in_features - in_features))
                      else:
                          print(f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, " +
                              "and that groupsize and inner_k_tiles*16 evenly divide into it")
                          continue
                  weight = weight.to(GLOBAL_PRECISSION).to(GLOBAL_DEVICE)
                  weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(weight, groupsize, inner_k_tiles)
                  cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to('cpu')
                  cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to('cpu')

      return cur_state_dict

  def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
      origin_x_size = x.size()
      x = x.reshape(-1, origin_x_size[-1])
      c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
      new_shape = origin_x_size[:-1] + (out_features,)
      c = c.reshape(new_shape)
      return c

  class WeightOnlyInt4Linear(torch.nn.Module):
      __constants__ = ['in_features', 'out_features']
      in_features: int
      out_features: int
      weight: torch.Tensor

      def __init__(
              self, in_features: int, out_features: int,
              bias=True, device=None, dtype=None, groupsize: int = 128, inner_k_tiles: int = 8, padding: bool = True,
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
          self.register_buffer(
              "weight", torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32)
          )
          self.register_buffer("scales_and_zeros", torch.empty((in_features // groupsize, out_features, 2), dtype=torch.bfloat16))

      def forward(self, input: torch.Tensor) -> torch.Tensor:
          input = input.to(torch.bfloat16)
          if self.padding:
              input = torch.nn.functional.pad(input, pad=(0, self.in_features - self.origin_in_features))
          return linear_forward_int4(input, self.weight, self.scales_and_zeros, self.out_features, self.groupsize)

  def replace_linear_int4(module, groupsize=128, inner_k_tiles=8, padding=True):
      for name, child in module.named_children():
          if isinstance(child, nn.Linear):
              if _check_linear_int4_k(child.in_features, groupsize, inner_k_tiles):
                  setattr(module, name, WeightOnlyInt4Linear(
                      child.in_features, child.out_features, bias=False, groupsize=groupsize, inner_k_tiles=inner_k_tiles, padding=False,
                  ))
              elif padding:
                  setattr(module, name, WeightOnlyInt4Linear(
                      child.in_features, child.out_features, bias=False, groupsize=groupsize, inner_k_tiles=inner_k_tiles, padding=True,
                  ))
          else:
              replace_linear_int4(child, groupsize, inner_k_tiles, padding)

  groupsize = 128
  quantized_state_dict = create_quantized_state_dict(model, groupsize=groupsize)
  save_path = (model.name if hasattr(model, 'name') else model.__class__.__name__) + '_int4.g{}.pth'.format(groupsize)
  torch.save(quantized_state_dict, save_path)

  replace_linear_int4(model, groupsize=groupsize)
  model.load_state_dict(quantized_state_dict, assign=True)
  model = model.to(device=GLOBAL_DEVICE, dtype=GLOBAL_PRECISSION).eval()
  ```
- **tensor parallel**
  ```py
  import os
  import time
  import torch
  from torch import nn
  from typing import List, Optional
  from torch.distributed._functional_collectives import all_reduce

  import torch._dynamo.config
  import torch._inductor.config

  torch._inductor.config.coordinate_descent_tuning = True
  torch._inductor.config.triton.unique_kernel_names = True
  torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

  GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else "cpu"
  GLOBAL_PRECISSION = torch.float16 if GLOBAL_DEVICE == "cuda" else torch.float32
  GLOBAL_DIST_BACKEND = "nccl" if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else "gloo"
  GLOBAL_CONTEXT = torch.autocast(device_type=GLOBAL_DEVICE, dtype=GLOBAL_PRECISSION)

  os.environ["KECAM_BACKEND"] = "torch"
  from keras_cv_attention_models import llama2
  from keras_cv_attention_models.pytorch_backend import models, layers


  def maybe_init_dist() -> Optional[int]:
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


  def apply_tensor_parallel(model):
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


  if __name__ == "__main__":
      maybe_init_dist()

      LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
      LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

      with torch.device(GLOBAL_DEVICE), GLOBAL_CONTEXT:
          model = llama2.LLaMA2_1B()
      with torch.no_grad(), torch.device(GLOBAL_DEVICE):
          inputs = torch.randint(low=0, high=32000, size=[1, 1])
          print(model(inputs).shape)

          repeat = 10
          start = time.time()
          _ = [model(inputs) for ii in range(repeat)]
          end = time.time()
          print("[Before tensor parallel] {} ms per loop, loop: {}".format((end - start) * 1000, repeat))

      apply_tensor_parallel(model)

      # print(model)
      # model.set_debug(True)
      # model.run_prediction('As evening fell, a maiden stood at the edge of a wood. In her hands,', top_k=1)
      with torch.no_grad(), torch.device(GLOBAL_DEVICE):
          inputs = torch.randint(low=0, high=32000, size=[1, 1])
          print(model(inputs).shape)

          repeat = 10
          start = time.time()
          _ = [model(inputs) for ii in range(repeat)]
          end = time.time()
          print("[After tensor parallel] {} ms per loop, loop: {}".format((end - start) * 1000, repeat))
  ```
- **Speculative**
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'
  import torch
  from keras_cv_attention_models import llama2
  draft = llama2.LLaMA2_1B()
  target = llama2.LLaMA2_7B(pretrained="llama2_chat/llama2_7b_chat_hf.h5")
  # draft = llama2.LLaMA2_42M()
  # target = llama2.LLaMA2_110M()
  draft.run_prediction.build()
  min_vocab_size = min(draft.output_shape[-1], target.output_shape[-1])

  speculate_k = 8
  temperature = 0.8
  inputs = "As evening fell, a maiden stood at the edge of a wood. In her hands,"
  start_ids = np.array(draft.run_prediction.tokenizer.encode(inputs, add_sot=True))
  draft_tokens, draft_probs = draft.run_prediction(inputs, max_new_tokens=speculate_k, top_k=1, temperature=temperature, return_token_and_probs=True)

  with torch.no_grad():
      target_logits = target(np.concatenate([start_ids, draft_tokens])[None])[0].cpu().numpy()
  target_probs = target.run_prediction.softmax_numpy(target_logits / max(temperature, 1e-5), axis=-1)

  # target_sub_prob >= draft_sub_prob: always accept draft token
  # target_sub_prob < draft_sub_prob: draft_sub_prob/target_sub_prob prob to accept draft token
  target_start_pos = target_probs.shape[0] - speculate_k - 1
  draft_sub_prob = draft_probs[np.arange(0, speculate_k), draft_tokens]
  target_sub_prob = target_probs[np.arange(target_start_pos, target_start_pos + speculate_k), draft_tokens]
  accept_draft_prob = np.minimum(target_sub_prob / draft_sub_prob, 1)
  rejected_locations = np.nonzero(np.random.uniform(size=accept_draft_prob.shape) > accept_draft_prob)[0]


  if rejected_locations.shape[0] == 0:  # All draft tokens have been accepted
      accept_length = speculate_k + 1
      last_token = target_probs[-1].argmax(-1)
      accept_tokens = np.concatenate([draft_tokens, last_token[None]])
  else:
      accept_length = rejected_locations[0]  # The first rejected
      draft_prob = draft_probs[accept_length, :min_vocab_size]
      target_prob = target_probs[target_start_pos + accept_length, :min_vocab_size]
      next_token = (target_prob - draft_prob).argmax()
      accept_tokens = np.concatenate([draft_tokens[:accept_length], next_token[None]])
  ```
