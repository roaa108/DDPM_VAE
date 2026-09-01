import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, time):
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000) * torch.arange(half, device=time.device) / max(half - 1, 1)
        )
        embedding = time.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat([embedding.sin(), embedding.cos()], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_dim=256):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.enc2 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(base_channels * 2, base_channels, time_dim)
        self.output = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, time):
        t = self.time_embedding(time)
        x1 = self.enc1(self.input(x), t)
        x2 = self.enc2(self.down1(x1), t)
        h = self.mid(self.down2(x2), t)
        h = self.dec1(torch.cat([self.up1(h), x2], dim=1), t)
        h = self.dec2(torch.cat([self.up2(h), x1], dim=1), t)
        return self.output(h)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


class DDPM(nn.Module):
    def __init__(self, network, timesteps=1_000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.network = network
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        for name, value in {
            "betas": betas, "alphas": alphas, "alpha_bars": alpha_bars,
            "sqrt_alpha_bars": alpha_bars.sqrt(),
            "sqrt_one_minus_alpha_bars": (1 - alpha_bars).sqrt(),
            "posterior_variance": betas * (1 - alpha_bars_previous) / (1 - alpha_bars),
        }.items():
            self.register_buffer(name, value)

    @staticmethod
    def _extract(values, time, x):
        return values.gather(0, time).view(-1, 1, 1, 1).to(x.dtype)

    def q_sample(self, clean_images, time, noise=None):
        noise = torch.randn_like(clean_images) if noise is None else noise
        return (self._extract(self.sqrt_alpha_bars, time, clean_images) * clean_images +
                self._extract(self.sqrt_one_minus_alpha_bars, time, clean_images) * noise)

    def compute_loss(self, clean_images, generator=None):
        time = torch.randint(0, self.timesteps, (clean_images.size(0),), device=clean_images.device,generator=generator)
        noise = torch.randn(clean_images.shape, device=clean_images.device,
                        dtype=clean_images.dtype, generator=generator)
        noisy_images = self.q_sample(clean_images, time, noise)
        predicted_noise = self.network(noisy_images, time)
        loss = F.mse_loss(predicted_noise, noise)
        return loss, {"total_loss": loss.detach().item()}

    @torch.no_grad()
    def p_sample(self, x, time, generator=None):
        beta_t = self._extract(self.betas, time, x)
        alpha_t = self._extract(self.alphas, time, x)
        alpha_bar_t = self._extract(self.alpha_bars, time, x)
        mean = (x - beta_t * self.network(x, time) / (1 - alpha_bar_t).sqrt()) / alpha_t.sqrt()
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        nonzero = (time != 0).float().view(-1, 1, 1, 1)
        variance = self._extract(self.posterior_variance, time, x).clamp_min(1e-20).sqrt()
        return mean + nonzero * variance * noise

    @torch.no_grad()
    def sample(self, num_samples, device=None, noise=None, generator=None, return_intermediates=False, every=100,progress=False):
        device = device or next(self.parameters()).device
        x = noise if noise is not None else torch.randn(
            num_samples, 3, 32, 32, device=device, generator=generator
        )
        intermediates = [x.detach().cpu()] if return_intermediates else None

        steps = reversed(range(self.timesteps))
        if progress:
            steps = tqdm(steps, total=self.timesteps, desc="Sampling", leave=False)
        for step in steps:
            time = torch.full((x.size(0),), step, device=device, dtype=torch.long)
            x = self.p_sample(x, time, generator)
            if return_intermediates and (step % every == 0 or step == 0):
                intermediates.append(x.detach().cpu())
        return (x, intermediates) if return_intermediates else x


class ModelEMA:
    #EMA of DDPM weights, used for sampling-evaluation.
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_parameter, parameter in zip(self.ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(self.decay).add_(parameter, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)
