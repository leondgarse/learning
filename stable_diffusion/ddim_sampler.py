"""
This implements DDIM sampling from the paper [Denoising Diffusion Implicit Models](https://papers.labml.ai/paper/2010.02502)
"""

from typing import Optional, List

import numpy as np
import torch

class DDIMSampler:
    def __init__(self, unet_model, n_steps=50, n_steps_training=1000, ddim_discretize="uniform", linear_start=0.00085, linear_end=0.0120, ddim_eta=0.):
        """
        :param unet_model: is the unet_model to predict noise $\epsilon_\text{cond}(x_t, c)$
        :param n_steps: is the number of DDIM sampling steps, $S$
        :param n_steps_training: is the number of training sampling steps, $S$
        :param ddim_discretize: specifies how to extract $\tau$ from $[1,2,\dots,T]$. It can be either `uniform` or `quad`.
        :param ddim_eta: is $\eta$ used to calculate $\sigma_{\tau_i}$. $\eta = 0$ makes the sampling process deterministic.
        """
        if ddim_discretize == 'uniform':
            c = n_steps_training // n_steps
            self.time_steps = np.asarray(list(range(0, n_steps_training, c))) + 1
        elif ddim_discretize == 'quad':
            self.time_steps = ((np.linspace(0, np.sqrt(n_steps_training * .8), n_steps)) ** 2).astype(int) + 1
        else:
            raise NotImplementedError(ddim_discretize)

        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps_training, dtype=torch.float64) ** 2
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0).float()

        self.ddim_alpha = alpha_bar[self.time_steps].clone().float()
        self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
        self.ddim_alpha_prev = torch.cat([alpha_bar[:1], alpha_bar[self.time_steps[:-1]]])
        ddim_sigma = ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) * (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5
        self.ddim_sigma = ddim_eta * ddim_sigma
        self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5

        self.unet_model = unet_model
        self.device = next(unet_model.parameters()).device

    def get_eps(self, xt, timestep, cond, uncond_scale=1, uncond_cond=None):
        """ xt: [batch_size, channels, height, width], timestep: [batch_size], cond: [batch_size, emb_size]"""
        if uncond_cond is None or uncond_scale == 1.:
            return self.unet_model(xt, timestep, cond_embeddings)

        x_in = torch.cat([xt] * 2)
        t_in = torch.cat([timestep] * 2)
        c_in = torch.cat([uncond_cond, cond])
        e_t_uncond, e_t_cond = self.unet_model(x_in, t_in, c_in).chunk(2)
        # $$\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$$
        return e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)

    def get_x_prev_and_pred_x0(self, e_t, index, xt, temperature=1, repeat_noise=False):
        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)  # Current prediction for x_0
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t  # Direction pointing to x_t

        noise = 0. if sigma == 0 else (torch.randn((1, *x.shape[1:])) if repeat_noise else torch.randn(x.shape))
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * temperature * noise
        return x_prev, pred_x0

    @torch.no_grad()
    def sample(self, shape, cond, repeat_noise=False, temperature=1., x_last=None, uncond_scale=1., uncond_cond=None, skip_steps=0):
        """
        :param shape: is the shape of the generated images in the form `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_{\tau_S}$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$.
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip $i'$. We start sampling from $S - i'$. And `x_last` is then $x_{\tau_{S - i'}}$.
        """
        bs = shape[0]
        x = x_last if x_last is not None else torch.randn(shape)
        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in enumerate(time_steps):
            index = len(time_steps) - i - 1  # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            ts = x.new_full((bs,), step, dtype=torch.long)  # Time step $\tau_i$

            e_t = self.get_eps(x, ts, cond, uncond_scale=uncond_scale, uncond_cond=uncond_cond)
            x, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x, temperature=temperature, repeat_noise=repeat_noise)
        return x

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        ### Sample from $q_{\sigma,\tau}(x_{\tau_i}|x_0)$

        $$q_{\sigma,\tau}(x_t|x_0) =
         \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$

        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $\tau_i$ index $i$
        :param noise: is the noise, $\epsilon$
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample from
        #  $$q_{\sigma,\tau}(x_t|x_0) =
        #          \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$
        return self.ddim_alpha_sqrt[index] * x0 + self.ddim_sqrt_one_minus_alpha[index] * noise

    @torch.no_grad()
    def paint(self, x: torch.Tensor, cond: torch.Tensor, t_start: int, *,
              orig: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None, orig_noise: Optional[torch.Tensor] = None,
              uncond_scale: float = 1.,
              uncond_cond: Optional[torch.Tensor] = None,
              ):
        """
        ### Painting Loop

        :param x: is $x_{S'}$ of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param t_start: is the sampling step to start from, $S'$
        :param orig: is the original image in latent page which we are in paining.
            If this is not provided, it'll be an image to image transformation.
        :param mask: is the mask to keep the original image.
        :param orig_noise: is fixed noise to be added to the original image.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        # Get  batch size
        bs = x.shape[0]

        # Time steps to sample at $\tau_{S`}, \tau_{S' - 1}, \dots, \tau_1$
        time_steps = np.flip(self.time_steps[:t_start])

        for i, step in monit.enum('Paint', time_steps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            index = len(time_steps) - i - 1
            # Time step $\tau_i$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$
            x, _, _ = self.p_sample(x, cond, ts, step, index=index,
                                    uncond_scale=uncond_scale,
                                    uncond_cond=uncond_cond)

            # Replace the masked area with original image
            if orig is not None:
                # Get the $q_{\sigma,\tau}(x_{\tau_i}|x_0)$ for original image in latent space
                orig_t = self.q_sample(orig, index, noise=orig_noise)
                # Replace the masked area
                x = orig_t * mask + x * (1 - mask)

        #
        return x
