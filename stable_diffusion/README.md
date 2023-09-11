- [Paper 2112.10752 High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Github CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [Github stability-ai/stablediffusion](https://github.com/stability-ai/stablediffusion)
- [Github labmlai/annotated_deep_learning_paper_implementations/stable_diffusion](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion)
- [Huggingface CompVis/stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
```py
# e8c7b5341445394de4fee061c34137b4  clip_model.pt
# dd8f085ecbd26d4cb0c2a5301369b07f  diffusion_model.pt
# 7ebbef498e4a24597125703d34be9cdc  encoder_decoder_model.pt
!pip install pytorch-lightning

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
```py
import torch
import unet
import autoencoder
import ddim_sampler
from transformers import CLIPTokenizer, CLIPTextModel

from unet_attention import CrossAttention
CrossAttention.use_flash_attention = False  # opt.flash

unet_model = unet.UNetModel()
unet_model.load_state_dict(torch.load("diffusion_model.pt"))
unet_model.eval()
sampler = ddim_sampler.DDIMSampler(unet_model)  # sampler = DDPMSampler(model)

# Initialize the autoencoder
encoder = autoencoder.Encoder()
encoder.load_state_dict(torch.load("encoder_model.pt"))
post_encoder = autoencoder.PostEncoder()
post_encoder.load_state_dict(torch.load("post_encoder_model.pt"))
encoder.eval(), post_encoder.eval()

pre_decoder = autoencoder.PreDecoder()
pre_decoder.load_state_dict(torch.load("pre_decoder_model.pt"))
decoder = autoencoder.Decoder()
decoder.load_state_dict(torch.load("decoder_model.pt"))
pre_decoder.eval(), decoder.eval()

# autoencoder = autoencoder.Autoencoder(encoder=encoder, decoder=decoder, emb_channels=4, z_channels=4)
# autoencoder.load_state_dict(torch.load("encoder_decoder_model.pt"))
# autoencoder.eval()

clip_model = "clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_model)
ss = torch.load('clip_model.pt')
clip_text_embedder = CLIPTextModel.from_pretrained(clip_model, state_dict=ss).eval()

def prompt_embedding(prompt, clip_text_embedder, tokenizer, max_length=77):
    batch_encoding = tokenizer(
        prompts, truncation=True, max_length=max_length, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt"
    ) # Get token ids
    tokens = batch_encoding["input_ids"]
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
prompt = "a painting of a virus monster playing guitar"
prompts = batch_size * [prompt]  # Make a batch of prompts

with torch.cuda.amp.autocast():
    un_cond = prompt_embedding(batch_size * [""], clip_text_embedder, tokenizer)  # get the embeddings for empty prompts (no conditioning).
    cond = prompt_embedding(prompts, clip_text_embedder, tokenizer)  # Get the prompt embeddings
    # [Sample in the latent space](../sampler/index.html).
    # `x` will be of shape `[batch_size, c, h / f, w / f]`
    x = sampler.sample(cond=cond, shape=[batch_size, c, height // f, width // f], uncond_scale=uncond_scale, uncond_cond=un_cond)
    # Decode the image from the [autoencoder](../model/autoencoder.html)
    latent = pre_decoder(x / latent_scaling_factor)
    images = decoder(latent)

# Save images
images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).cpu().permute(0, 2, 3, 1).numpy()
for id, image in enumerate(images):
    plt.imsave("test_{}.jpg".format(id), image)
```
