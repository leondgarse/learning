# learning_gpt2
***
- [Github karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [Github jaymody/picoGPT](https://github.com/jaymody/picoGPT)
```sh
pip instal tiktoken kecam
```
## Model load statedict from huggingface and run prediction
  ```py
  import gpt2, kecam, torch

  mm = gpt2.GPT2_Base()
  gpt2.run_prediction(mm, "hello world")
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
