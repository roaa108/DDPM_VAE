import os, random, torch
import numpy as np
from torchvision.utils import save_image
from dataset import cifar_dataloader
from models.DDPM import DDPM, UNet, ModelEMA
from trainer import Trainer, set_seed

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, validation_loader,test_loader = cifar_dataloader(
    data_root='./data', 
    batch_size=128,
    validation_size=5000,
    num_workers=2
)
network = UNet(
    in_channels=3, 
    base_channels=128, 
    time_dim=256
    )
ddpm = DDPM(
    network, 
    timesteps=1000
    ).to(device)

optimizer = torch.optim.Adam(
    ddpm.parameters(),
      lr=2e-4
      ) 

ema = ModelEMA(ddpm, decay=0.9999)

trainer = Trainer(
    model=ddpm,
    optimizer=optimizer, 
    device=device,
    ema=ema, 
    use_amp=True
    )

progress_dir = "results/ddpm/progress/samples"
os.makedirs(progress_dir, exist_ok=True)

g = torch.Generator(device=device).manual_seed(0)
fixed_noise = torch.randn(16, 3, 32, 32, device=device, generator=g)

@torch.no_grad()
def save_progress(epoch, model):
    if epoch % 20 != 0:      
        return
    imgs = ema.ema_model.sample(16, device=device, noise=fixed_noise)
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    save_image(imgs, f"{progress_dir}/samples_epoch_{epoch:03d}.png", nrow=4)

history = trainer.train(
    train_loader=train_loader,
    validation_loader=validation_loader,
    epochs=400,
    checkpoint_dir="checkpoints/ddpm",
    sample_callback=save_progress,
    resume_from="checkpoints/ddpm/latest.pt",  
    snapshot_interval=25,
    save_best=False,                          
)
