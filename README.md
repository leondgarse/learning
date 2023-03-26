# learning_gpt2
***
- [Github karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [Github jaymody/picoGPT](https://github.com/jaymody/picoGPT)
```sh
pip instal tiktoken
```
## Model load statedict from huggingface
  ```py
  config_args = {
      'gpt2': dict(n_layer=12, n_head=12, n_embd=768, vocab_size=50257),  # 124M params
      'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257), # 350M params
      'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280, vocab_size=50257), # 774M params
      'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600, vocab_size=50257), # 1558M params
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

  from transformers import GPT2LMHeadModel

  source_state_dict = GPT2LMHeadModel.from_pretrained(model_type).state_dict()
  target_state_dict = convert_gpt2_state_dict(model.state_dict())
  model.load_state_dict(target_state_dict)
  ```
## Model evaluation
  ```py
  import tiktoken
  enc = tiktoken.get_encoding('gpt2')

  start = "hello world"
  start_ids = enc.encode(start)
  inputs = (torch.tensor(start_ids, dtype=torch.long)[None])

  # run generation
  num_samples = 10 # number of samples to draw
  max_new_tokens = 500 # number of tokens generated in each sample
  temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
  top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability

  for k in range(num_samples):
      out = model.generate(inputs, max_new_tokens, temperature=temperature, top_k=top_k)
      print(enc.decode(out[0].tolist()))
      print('---------------')
  ```
***

```py
with open('gpt2_60000.log') as ff:
    aa = ff.readlines()
losses = [float(ii.strip().split("loss ")[-1].split(', ')[0]) for ii in aa[1:] if ii.startswith('iter ')]
val_losses = [float(ii.strip().split("loss ")[-1].split(', ')[0]) for ii in aa[1:] if ii.startswith('step ')]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].plot(np.log(losses), label="train losses")
axes[1].plot(val_losses, label="val losses")

axes[0].grid(True)
axes[0].legend()
axes[1].grid(True)
axes[1].legend()
plt.tight_layout()
```
