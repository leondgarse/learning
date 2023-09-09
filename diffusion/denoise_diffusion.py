import torch


class DenoiseDiffusion:
    def __init__(self, model, n_steps=100):
        self.model, self.n_steps = model, n_steps
        self.device = next(model.parameters()).device

        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(self.device)
        self.beta = self.beta[:, None, None, None]  # expand to calculation on batch dimension

        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

    def denoise_sample(self, x0, timestep, noise=None):
        noise = torch.randn_like(x0).to(self.device) if noise is None else noise
        cur_alpha = self.alpha_bar[timestep]
        # Sample from $q(x_t|x_0)$
        return cur_alpha ** 0.5 * x0 + (1 - cur_alpha) ** 0.5 * noise

    def loss(self, x0):
        timestep = torch.randint(0, self.n_steps, (x0.shape[0],)).to(self.device)
        noise = torch.randn_like(x0).to(self.device)
        xt = self.denoise_sample(x0, timestep, noise)
        xt_noise = self.model(xt, timestep)
        return torch.functional.F.mse_loss(noise, xt_noise)

    def diffusion_sample(self, xt, timestep):
        xt_noise = self.model(xt, timestep)

        cur_alpha_bar = self.alpha_bar[timestep]
        cur_alpha = self.alpha[timestep]
        eps_coef = (1 - cur_alpha) / (1 - cur_alpha_bar) ** .5

        eps = torch.randn(xt.shape).to(self.device)
        return 1 / (cur_alpha ** 0.5) * (xt - eps_coef * xt_noise) + ((1 - cur_alpha) ** 0.5) * eps

    def generate(self, x0=None, image_size=160, n_samples=1, n_channels=3, return_inner=False):
        xt = x0 = torch.randn([n_samples, n_channels, image_size, image_size]).to(self.device) if x0 is None else x0
        timestep = torch.full([x0.shape[0]], self.n_steps).to(self.device)

        rr = []
        with torch.no_grad():
            for iter in range(self.n_steps):
                timestep -= 1
                xt = self.diffusion_sample(xt, timestep)
                if return_inner:
                    rr.append(xt)
        return rr if return_inner else xt
