import argparse
import json
import os
import time
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from torchmetrics.image.fid import (FrechetInceptionDistance)
from torchmetrics.image.inception import (InceptionScore)
from torchmetrics.image.kid import (KernelInceptionDistance)

from dataset import cifar_dataloader
from models.VAE import VAE
from models.DDPM import DDPM, UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dirs(model_name):
    dirs = {
        "generation": f"results/{model_name}/final/generation",
        "reconstruction": f"results/{model_name}/final/reconstruction",
        "latent_space": f"results/{model_name}/final/latent_space",
        "metrics": f"results/{model_name}/metrics",
    }
    for folder in dirs.values():
        os.makedirs(folder, exist_ok=True)
    return dirs

def to_uint8(images):   
    images = (images.clamp(-1, 1) + 1) / 2
    return (images * 255).to(torch.uint8)


@torch.no_grad()
def evaluate_losses(model, test_loader, device):
    model.eval()
    totals, batches = {}, 0
    for images, _ in test_loader:
        images = images.to(device)
        _, metrics = model.compute_loss(images)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        batches += 1
    return {f"test_{k}": v / batches for k, v in totals.items()}

@torch.no_grad()
def calculate_generation_metrics(model, test_loader, device, num_samples=10_000):
    """
    Calculates FID, KID, and Inception Score.

    FID and KID compare real test images against
    newly generated images.

    Inception Score uses generated images only.
    """
    model.eval()

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(subset_size=100, normalize=False).to(device)
    inception_score = InceptionScore(normalize=False, splits=10,).to(device)

    generated_count = 0

    for real_images, _ in test_loader:
        if generated_count >= num_samples:
            break

        real_images = real_images.to(device)

        batch_size = min(real_images.size(0), num_samples - generated_count)

        real_images = real_images[:batch_size]

        fake_images = model.sample(batch_size, device=device, progress=True)

        real_uint8 = to_uint8(real_images)
        fake_uint8 = to_uint8(fake_images)

        fid.update(real_uint8, real=True)
        fid.update(fake_uint8, real=False)

        kid.update(real_uint8, real=True)
        kid.update(fake_uint8, real=False)

        # Inception Score uses generated images only.
        inception_score.update(fake_uint8)

        generated_count += batch_size

    fid_value = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    inception_mean, inception_std = (inception_score.compute())

    return {
        "fid": fid_value,
        "kid_mean": kid_mean.item(),
        "kid_std": kid_std.item(),
        "inception_score_mean": inception_mean.item(),
        "inception_score_std": inception_std.item(),
        "number_of_generated_images": generated_count,
    }

@torch.no_grad()
def measure_generation_time(model, device, num_samples=1000, batch_size=100):

    model.eval()

    # Warm up the GPU first
    model.sample(batch_size, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    generated_count = 0

    while generated_count < num_samples:

        current_batch_size = min(batch_size,num_samples - generated_count)
        model.sample(current_batch_size, device=device)
        generated_count += current_batch_size

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_seconds = time.perf_counter() - start_time

    return {
        "number_of_images": num_samples,
        "generation_time_seconds": elapsed_seconds,
        "images_per_second": (
            num_samples / elapsed_seconds
        ),
    }


@torch.no_grad()
def save_final_qualitative_results(model, test_loader, device,dirs,is_vae):
    
    model.eval()
    samples = model.sample(64, device=device)
    samples = (samples.clamp(-1, 1) + 1) / 2
    save_image(samples, os.path.join(f"{dirs['generation']}/final_samples.png"),nrow=8)

    # Original test images and their reconstructions.
    images, _ = next(iter(test_loader))
    images = images[:8].to(device)

    if not is_vae:
        return 
    # latent interpolation: first test image to second test image.
    interpolations = model.interpolate(images[0:1], images[1:2], steps=10)
    interpolations = ( interpolations.clamp(-1, 1) + 1) / 2
    save_image(interpolations,os.path.join(f"{dirs['latent_space']}/latent_interpolation.png"),nrow=10)

    mu, _ = model.encoder(images)
    reconstructions = model.decoder(mu)

    comparison = torch.cat([(images.clamp(-1, 1) + 1) / 2,
                            (reconstructions.clamp(-1, 1) + 1) / 2,],dim=0,)

    save_image(comparison,os.path.join(f"{dirs['reconstruction']}/final_reconstructions.png"), nrow=8)



def save_kl_per_dimension_plot(model, test_loader, device, results_dir):
    """
    Plots the average KL contribution of every
    VAE latent dimension across the full test set.
    """
    kl_dimensions = model.kl_per_dimension(test_loader, device)

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(kl_dimensions)),kl_dimensions.numpy())

    plt.xlabel("Latent dimension")
    plt.ylabel("Mean KL contribution")
    plt.title("VAE KL Divergence per Latent Dimension")

    plt.savefig(os.path.join(results_dir,"kl_per_dimension.png"),dpi=200,bbox_inches="tight")

    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vae", "ddpm"], required=True)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_vae = args.model == "vae"

    if is_vae:
        from models.VAE import VAE
        model = VAE(latent_dim=128, beta=1.0).to(device)
        default_ckpt, state_key = "checkpoints/vae/best.pt", "model_state_dict"
        timing_samples = 1000
    else:
        from models.DDPM import DDPM, UNet
        model = DDPM(UNet(base_channels=64), timesteps=1000).to(device)
        default_ckpt, state_key = "checkpoints/ddpm/latest.pt", "ema_state_dict"
        timing_samples = 100        # كل صورة = 1000 مرور على الشبكة

    checkpoint = torch.load(args.checkpoint or default_ckpt, map_location=device)
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    dirs = make_dirs(args.model)
    _, _, test_loader = cifar_dataloader(data_root="./data", batch_size=128,
                                         validation_size=5000)

    results = {
        "model": args.model,
        "epoch": checkpoint["epoch"],
        "test_losses": evaluate_losses(model, test_loader, device),
        "generation_metrics": calculate_generation_metrics(
            model, test_loader, device, num_samples=args.num_samples),
        "generation_time": measure_generation_time(
            model, device, num_samples=timing_samples, batch_size=50),
    }

    save_final_qualitative_results(model, test_loader, device, dirs, is_vae)
    if is_vae:
        save_kl_per_dimension_plot(model, test_loader, device, dirs["metrics"])

    with open(f"{dirs['metrics']}/final_results.json", "w") as file:
        json.dump(results, file, indent=2)
    print(json.dumps(results, indent=2))
    

if __name__ == "__main__":
    main()