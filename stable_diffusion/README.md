- [Huggingface CompVis/stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
```py
from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion
from labml_nn.diffusion.stable_diffusion.model.autoencoder import Encoder, Decoder, Autoencoder
from labml_nn.diffusion.stable_diffusion.model.clip_embedder import CLIPTextEmbedder
from labml_nn.diffusion.stable_diffusion.model.unet import UNetModel

# Initialize the autoencoder
encoder = Encoder(z_channels=4, in_channels=3, channels=128, channel_multipliers=[1, 2, 4, 4], n_resnet_blocks=2)
decoder = Decoder(out_channels=3, z_channels=4, channels=128, channel_multipliers=[1, 2, 4, 4], n_resnet_blocks=2)
autoencoder = Autoencoder(emb_channels=4, encoder=encoder, decoder=decoder, z_channels=4)

# Initialize the CLIP text embedder
clip_text_embedder = CLIPTextEmbedder()

# Initialize the U-Net
unet_model = UNetModel(
    in_channels=4, out_channels=4, channels=320, attention_levels=[0, 1, 2], n_res_blocks=2, channel_multipliers=[1, 2, 4, 4], n_heads=8, tf_layers=1, d_cond=768
)

# Initialize the Latent Diffusion model
model = LatentDiffusion(
    linear_start=0.00085, linear_end=0.0120, n_steps=1000, latent_scaling_factor=0.18215, autoencoder=autoencoder, clip_embedder=clip_text_embedder, unet_model=unet_model
)

# Load the checkpoint
checkpoint = torch.load(path, map_location="cpu")
missing_keys, extra_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
model.eval()
```
LatentDiffusion model
```py
self.model = DiffusionWrapper(unet_model)
# Auto-encoder and scaling factor
self.first_stage_model = autoencoder
self.latent_scaling_factor = latent_scaling_factor
# [CLIP embeddings generator](model/clip_embedder.html)
self.cond_stage_model = clip_embedder

# Number of steps $T$
self.n_steps = n_steps

# $\beta$ schedule
beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64) ** 2
self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
# $\alpha_t = 1 - \beta_t$
alpha = 1. - beta
# $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
alpha_bar = torch.cumprod(alpha, dim=0)
self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
```
DiffusionSampler get_eps
```py
def get_eps(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, *,
            uncond_scale: float, uncond_cond: Optional[torch.Tensor]):
    """
    ## Get $\epsilon(x_t, c)$

    :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
    :param t: is $t$ of shape `[batch_size]`
    :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
    :param uncond_scale: is the unconditional guidance scale $s$. This is used for
        $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
    :param uncond_cond: is the conditional embedding for empty prompt $c_u$
    """
    # When the scale $s = 1$
    # $$\epsilon_\theta(x_t, c) = \epsilon_\text{cond}(x_t, c)$$
    if uncond_cond is None or uncond_scale == 1.:
        return self.model(x, t, c)

    # Duplicate $x_t$ and $t$
    x_in = torch.cat([x] * 2)
    t_in = torch.cat([t] * 2)
    # Concatenated $c$ and $c_u$
    c_in = torch.cat([uncond_cond, c])
    # Get $\epsilon_\text{cond}(x_t, c)$ and $\epsilon_\text{cond}(x_t, c_u)$
    e_t_uncond, e_t_cond = self.model(x_in, t_in, c_in).chunk(2)
    # Calculate
    # $$\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$$
    e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)

    #
    return e_t
```
```py
from labml_nn.diffusion.stable_diffusion.sampler.ddim import DDIMSampler

sampler = DDIMSampler(model, n_steps=n_steps, ddim_eta=ddim_eta)
# sampler = DDPMSampler(model)

c = 4  # Number of channels in the image
f = 8  # Image to latent space resolution reduction
height = 512
width = 512
uncond_scale = 7.5  # unconditional guidance scale: "eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
prompts = batch_size * [prompt]  # Make a batch of prompts

with torch.cuda.amp.autocast():
    un_cond = clip_text_embedder(batch_size * [""])  # get the embeddings for empty prompts (no conditioning).
    cond = clip_text_embedder(prompts)  # Get the prompt embeddings
    # [Sample in the latent space](../sampler/index.html).
    # `x` will be of shape `[batch_size, c, h / f, w / f]`
    x = sampler.sample(cond=cond, shape=[batch_size, c, height // f, width // f], uncond_scale=uncond_scale, uncond_cond=un_cond)
    # Decode the image from the [autoencoder](../model/autoencoder.html)
    images = model.autoencoder_decode(x)

# Save images
images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).cpu().permute(0, 2, 3, 1).numpy()
for id, image in enumerate(images):
    plt.imsave("test_{}.jpg".format(id), image)
```
```py
from labml_nn.diffusion.stable_diffusion.model.unet_attention import CrossAttention
CrossAttention.use_flash_attention = opt.flash

#
txt2img = Txt2Img(checkpoint_path=lab.get_data_path() / 'stable-diffusion' / 'sd-v1-4.ckpt',
                  sampler_name=opt.sampler_name,
                  n_steps=opt.steps)

with monit.section('Generate'):
    txt2img(dest_path='outputs',
            batch_size=opt.batch_size,
            prompt=opt.prompt,
            uncond_scale=opt.scale)
```
