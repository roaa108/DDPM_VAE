import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3= nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
           
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x).clamp(-30,20) # to avoid Nan

        return mu, logvar

class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE_Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = z.view(-1, 128, 4, 4)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = torch.tanh(self.deconv3(z)) 
        return z

class VAE(nn.Module):
    def __init__(self, latent_dim=128,beta=1.0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta= beta
        self.encoder = VAE_Encoder(latent_dim)
        self.decoder = VAE_Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples, device=None, noise=None,progress=False): # noise = z
        if noise is None:
            if device is None:
                device = next(self.parameters()).device
            noise = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(noise)
    
    @torch.no_grad()
    def reconstruct(self, x):
        mu, logvar= self.encoder(x)
        z= self.reparameterize(mu, logvar)
        reconstruction= self.decoder(z)
        return reconstruction

    @torch.no_grad()
    def interpolate(self, x1, x2, steps=8):
        mu1, _ = self.encoder(x1[:1])
        mu2, _ = self.encoder(x2[:1])
        alpha = torch.linspace(0, 1, steps, device=x1.device).view(-1, 1)
        return self.decoder((1 - alpha) * mu1 + alpha * mu2)
    
    def compute_loss(self,x, generator=None):
        reconstruction, mu, logvar= self.forward(x)
        rec = F.mse_loss(reconstruction, x, reduction="none").flatten(1).sum(1).mean()
        kld = ( -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()
        loss= rec + self.beta * kld
        metrics = {
            'reconstruction_loss': rec.item(),
            'kl_divergence': kld.item(),
            'total_loss': loss.item()
        }   
        return loss,metrics

    @torch.no_grad()
    def kl_per_dimension(self, dataloader, device):
        """Mean KL contribution for every latent variable on held-out data."""
        total = torch.zeros(self.latent_dim, device=device)
        count = 0
        self.eval()
        for images, _ in dataloader:
            mu, logvar = self.encoder(images.to(device))
            total += (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(0)
            count += images.size(0)
        return (total / count).cpu()

