- [Github pytorch-labs/gpt-fast](https://github.com/pytorch-labs/gpt-fast)
- Install torch nightly from [Start Locally Torch](https://pytorch.org/get-started/locally/)
  ```sh
  pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
  pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
  ```
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
- **int8 / int4 quant -> tensor parallel -> compile**
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
  import torch
  from torch import nn
  from typing import List, Optional
  from torch.distributed._functional_collectives import all_reduce


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
          # if isinstance(layer, layers.Dropout) and LOCAL_WORLD_SIZE >= 2:
          #     layer.module.register_forward_hook(lambda _module, _input, output: all_reduce(output, "sum", list(range(LOCAL_WORLD_SIZE))))

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
              # if LOCAL_WORLD_SIZE >= 2:
              #     layer.module.register_forward_hook(lambda _module, _input, output: all_reduce(output, "sum", list(range(LOCAL_WORLD_SIZE))))


  if __name__ == "__main__":
      maybe_init_dist()

      LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
      LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

      model = llama2.LLaMA2_42M()
      apply_tensor_parallel(model)
      # print(model)
      # model.set_debug(True)
      # model.run_prediction('As evening fell, a maiden stood at the edge of a wood. In her hands,')
      model.run_prediction("hello")
  ```
- **Speculative**
  ```py
  import torch

  def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
      q = torch.empty_like(probs_sort).exponential_(1)
      return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

  def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
      logits = logits / max(temperature, 1e-5)

      if top_k is not None:
          v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
          pivot = v.select(-1, -1).unsqueeze(-1)
          logits = torch.where(logits < pivot, -float("Inf"), logits)
      probs = torch.nn.functional.softmax(logits, dim=-1)
      return probs

  def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
      probs = logits_to_probs(logits[0, -1], temperature, top_k)
      idx_next = multinomial_sample_one_no_sync(probs)
      return idx_next, probs

  def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
      # input_pos: [B, S]
      logits = model(x, input_pos)
      return sample(logits, **sampling_kwargs)[0]

  def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
      # input_pos: [B, 1]
      assert input_pos.shape[-1] == 1
      logits = model(x, input_pos)
      return sample(logits, **sampling_kwargs)

  def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
      new_tokens, new_probs = [], []
      for i in range(num_new_tokens):
          with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
              next_token, next_prob = decode_one_token(model, cur_token, input_pos, **sampling_kwargs)
          input_pos += 1
          new_tokens.append(next_token.clone())
          callback(new_tokens[-1])
          new_probs.append(next_prob.clone())
          cur_token = next_token.view(1, -1)
      return new_tokens, new_probs

  def speculative_decode(model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs):
      # draft model inference sequentially
      device = cur_token.device
      orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
      draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

      draft_tokens = torch.cat(draft_tokens)
      # parallel inference on target model using draft tokens
      input_pos = torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
      target_logits = model(torch.cat([cur_token.view(1), draft_tokens]).view(1, -1), input_pos)
      target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
      draft_probs = torch.stack(draft_probs)
      # q: target prob, p: draft prob
      # q >= p: always accept draft token
      # q < p: q/p prob to accept draft token
      p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
      q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
      accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
      rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

      if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
          accept_length = speculate_k + 1
          last_token = multinomial_sample_one_no_sync(target_probs[-1])
          # fill last token into draft model
          draft_model(draft_tokens[-1].view(1, -1), orig_input_pos + speculate_k)
          return torch.cat([draft_tokens, last_token])
      else:
          accept_length = rejected_locations[0].item()
          p = draft_probs[accept_length]
          q = target_probs[accept_length]
          new = q - p
          new = torch.where(new > 0, new, 0.0)
          new = new / new.sum()
          next_token = multinomial_sample_one_no_sync(new)
          return torch.cat([draft_tokens[:accept_length], next_token])

  @torch.no_grad()
  def generate(model, prompt, max_new_tokens, *, interactive, draft_model, speculate_k=8, callback=lambda x: x, **sampling_kwargs):
      """
      Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
      """

      is_speculative = draft_model is not None
      # create an empty tensor of the expected final shape and fill in the current tokens
      T = prompt.size(0)
      T_new = T + max_new_tokens
      if interactive:
          max_seq_length = 350
      else:
          max_seq_length = min(T_new, model.config.block_size)

      device, dtype = prompt.device, prompt.dtype
      max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
      with torch.device(device):
          model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
          if is_speculative and draft_model is not model:
              draft_model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

      # create an empty tensor of the expected final shape and fill in the current tokens
      empty = torch.empty(T_new, dtype=dtype, device=device)
      empty[:T] = prompt
      seq = empty
      input_pos = torch.arange(0, T, device=device)

      next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
      if is_speculative:
          prefill(draft_model, prompt.view(1, -1), input_pos, **sampling_kwargs)
      seq[T] = next_token

      input_pos = torch.tensor([T], device=device, dtype=torch.int)
      accept_counts = [0] * (speculate_k + 1)

      if is_speculative:
          input_pos = input_pos.item()  # for speculative decoding easier to keep on host
          while input_pos < T_new - 1:
              cur_token = next_token.view(())

              next_tokens = speculative_decode(model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs)

              accept_counts[len(next_tokens) - 1] += 1
              num_added = min(T_new - input_pos - 1, len(next_tokens))
              seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
              for i in next_tokens[: num_added,]:
                  callback(i)
              input_pos = input_pos + num_added
              next_token = next_tokens[-1]
      else:
          generated_tokens, _ = decode_n_tokens(model, next_token.view(1, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
          seq[T + 1:] = torch.cat(generated_tokens)
      return seq, {'accept_counts': accept_counts}
  ```
