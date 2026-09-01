import os
import torch
from torchvision.utils import save_image 
from dataset import cifar_dataloader
from models.VAE import VAE
from trainer import Trainer
from trainer import Trainer, set_seed

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, validation_loader,test_loader = cifar_dataloader(
    data_root='./data', 
    batch_size=128,
    validation_size=5000
)

vae = VAE(
    latent_dim=128,
    beta=1.0
).to(device)


optimizer = torch.optim.Adam(
    vae.parameters(),
    lr=1e-3
)

trainer = Trainer(
    model = vae,
    optimizer = optimizer,
    device = device,
    use_amp = True
)


progress_samples_dir = ("results/vae/progress/samples")
progress_reconstructions_dir = ("results/vae/progress/reconstructions")
os.makedirs(progress_samples_dir, exist_ok=True)
os.makedirs(progress_reconstructions_dir,exist_ok=True)

# fix randomly picked z & data samples to visualize quality progress during training
fixed_z = torch.randn(64,vae.latent_dim,device=device)

fixed_images, _ = next(iter(validation_loader))
fixed_images = fixed_images[:8].to(device)

@torch.no_grad()
def save_progress(epoch, model):

    if epoch % 5 != 0:
        return

    model.eval()

    # generated images from fixed latent vectors
    generated_images = model.decoder(fixed_z)
    generated_images = (generated_images.clamp(-1, 1) + 1) / 2
    save_image(generated_images,f"{progress_samples_dir}/samples_epoch_{epoch:03d}.png", nrow=8)
   
    # reconstructed fixed validation images
    mu, _ = model.encoder(fixed_images)
    reconstructed_images = model.decoder(mu)
    reconstructed_images = (reconstructed_images.clamp(-1, 1) + 1) / 2

    # originals and reconstructed together.
    comparison = torch.cat(
        [
            (fixed_images.clamp(-1, 1) + 1) / 2,
            reconstructed_images,
        ],
        dim=0,
    )

    save_image(comparison, f"{progress_reconstructions_dir}/reconstructions_epoch_{epoch:03d}.png", nrow=8)

history = trainer.train(
    train_loader=train_loader,
    validation_loader=validation_loader,
    checkpoint_dir="checkpoints/vae",
    sample_callback=save_progress,
    epochs=100,
    resume_from="checkpoints/vae/latest.pt",
    snapshot_interval=25,
)