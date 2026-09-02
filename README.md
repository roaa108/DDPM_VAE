# VAE vs DDPM on CIFAR-10

From-scratch implementations of a Variational Autoencoder and a Denoising
Diffusion Probabilistic Model, trained and benchmarked on CIFAR-10.

## Results

| Metric | VAE | DDPM |
|---|---|---|
| FID ↓ | 134.03 | **23.28** |
| KID ↓ | 0.1319 | **0.0162** |
| Inception Score ↑ | 3.35 | **7.29** |
| Generation speed | 132,900 img/s | 10.7 img/s |
| Time for 10,000 images | ~0.08 s | ~15.6 min |
| Parameters | base 64 ch | base 128 ch |
| Epochs trained | 100 | 400 |

FID / KID / IS computed on 10,000 generated images against the CIFAR-10 test set.

## Setup

Requirements: Python 3.11+, NVIDIA GPU with CUDA 12.x.

```bash
git clone https://github.com/roaa108/DDPM_VAE.git
cd DDPM_VAE
pip install -r requirements.txt
```

CIFAR-10 downloads automatically on first run.

## Reproducing the results

Run in this order:

```bash
python -m experiments.train_VAE      # ~10 min on A100
python -m experiments.train_ddpm     # ~2.5 h on A100
python plot_training.py --model vae
python plot_training.py --model ddpm
python evaluate.py --model vae
python evaluate.py --model ddpm
```

Training resumes automatically from `checkpoints/<model>/latest.pt` if it exists.
Re-run the same command after an interruption.

`evaluate.py` accepts `--num-samples` (default 10000) and `--checkpoint`.

## No local GPU?

`notebooks/colab_train.ipynb` runs the same pipeline on Google Colab.


## Structure

```
models/VAE.py            encoder, decoder, VAE, ELBO
models/DDPM.py           U-Net, noise schedule, forward/reverse process, EMA
dataset.py               CIFAR-10 loaders, normalised to [-1, 1]
trainer.py               shared training loop for both models
experiments/             training entry points
evaluate.py              FID / KID / IS, sampling speed, qualitative figures
plot_training.py         loss curves from history.json
```


Train split: 45,000 / validation: 5,000 / test: 10,000.

## Notes

- Results were produced on a single NVIDIA A100 (Google Colab).
- Metric values may vary slightly across GPU models due to non-deterministic
  CUDA kernels.
