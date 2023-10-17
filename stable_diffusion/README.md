## Prepare
- [Paper 2112.10752 High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Github CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [Github stability-ai/stablediffusion](https://github.com/stability-ai/stablediffusion)
- [Github labmlai/annotated_deep_learning_paper_implementations/stable_diffusion](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion)
- [Huggingface CompVis/stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
```py
!pip install pytorch-lightning transformers

import torch
ss = torch.load('sd-v1-4.ckpt')
pp = {kk.replace('model.diffusion_model.', ''): vv.half() for kk, vv in ss['state_dict'].items() if kk.startswith("model.diffusion_model.")}
torch.save(pp, 'diffusion_model.pt')

pp = {kk.replace('cond_stage_model.transformer.', ''): vv.half() for kk, vv in ss['state_dict'].items() if kk.startswith("cond_stage_model.transformer.")}
torch.save(pp, 'clip_model.pt')

pp = {kk.replace('first_stage_model.encoder.', ''): vv.half() for kk, vv in ss['state_dict'].items() if kk.startswith("first_stage_model.encoder.")}
torch.save(pp, 'encoder_model.pt')

pp = {kk.replace('first_stage_model.', ''): vv.half() for kk, vv in ss['state_dict'].items() if kk.startswith("first_stage_model.quant_conv.")}
torch.save(pp, 'post_encoder_model.pt')

pp = {kk.replace('first_stage_model.', ''): vv.half() for kk, vv in ss['state_dict'].items() if kk.startswith("first_stage_model.post_quant_conv.")}
torch.save(pp, 'pre_decoder_model.pt')

pp = {kk.replace('first_stage_model.decoder.', ''): vv.half() for kk, vv in ss['state_dict'].items() if kk.startswith("first_stage_model.decoder.")}
torch.save(pp, 'decoder_model.pt')

!GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/openai/clip-vit-large-patch14
```
## Diffusers
```py
!pip install -q diffusers transformers accelerate

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cpu")

prompt = "a photo of an astonaut riding a horse on mars"
image = pipe(prompt).images[0]
image
```
## Deconstruct
```py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from contextlib import nullcontext

device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else torch.device("cpu")
if device.type == "cpu":
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    global_context = nullcontext()
else:
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    global_context = torch.amp.autocast(device_type=device.type, dtype=torch.float16)

import unet, autoencoder, ddim_sampler, unet_attention
from transformers import CLIPTokenizer, CLIPTextModel

unet_attention.CrossAttention.use_flash_attention = False  # opt.flash

# Initialize the autoencoder
pre_decoder = autoencoder.PreDecoder()
pre_decoder.load_state_dict(torch.load("pre_decoder_model.pt"))
pre_decoder.eval().to(device)

decoder = autoencoder.Decoder()
decoder.load_state_dict(torch.load("decoder_model.pt"))
decoder.eval().to(device)

clip_model = "clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_model)
ss = torch.load('clip_model.pt')
clip_text_embedder = CLIPTextModel.from_pretrained(clip_model, state_dict=ss)
clip_text_embedder.load_state_dict(ss, strict=False)
clip_text_embedder.eval().to(device)

unet_model = unet.UNetModel()
unet_model.load_state_dict(torch.load("diffusion_model.pt"))
unet_model.eval().to(device)
sampler = ddim_sampler.DDIMSampler(unet_model, n_steps=50, ddim_eta=0)  # sampler = DDPMSampler(model)

# Not required for text-to-image
encoder = autoencoder.Encoder()
encoder.load_state_dict(torch.load("encoder_model.pt"))
encoder.eval().to(device)

post_encoder = autoencoder.PostEncoder()
post_encoder.load_state_dict(torch.load("post_encoder_model.pt"))
post_encoder.eval().to(device)


def prompt_embedding(prompts, clip_text_embedder, tokenizer, max_length=77):
    batch_encoding = tokenizer(
        prompts, truncation=True, max_length=max_length, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt"
    ) # Get token ids
    tokens = batch_encoding["input_ids"].to(device)
    # Get CLIP embeddings
    return clip_text_embedder(input_ids=tokens).last_hidden_state
```
```py
batch_size = 4
c = 4  # Number of channels in the image
f = 8  # Image to latent space resolution reduction
height = 512
width = 512
uncond_scale = 7.5  # unconditional guidance scale: "eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
latent_scaling_factor = 0.18215
n_steps = 50
ddim_eta = 0
prompt = "a photo of an astonaut riding a horse on mars"
prompts = batch_size * [prompt]  # Make a batch of prompts

with torch.no_grad(), global_context:
    uncond_cond = prompt_embedding(batch_size * [""], clip_text_embedder, tokenizer)  # get the embeddings for empty prompts (no conditioning).
    cond = prompt_embedding(prompts, clip_text_embedder, tokenizer)  # Get the prompt embeddings
    # Sample in the latent space. `x` will be of shape `[batch_size, c, h / f, w / f]`
    x = sampler.sample(cond=cond, shape=[batch_size, c, height // f, width // f], uncond_scale=uncond_scale, uncond_cond=uncond_cond)
    # Decode the image
    latent = pre_decoder(x / latent_scaling_factor)
    images = decoder(latent)

# Save images
images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).cpu().permute(0, 2, 3, 1).numpy().astype("float32")
plt.imshow(np.hstack(images))
```
