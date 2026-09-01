import argparse, json, os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["vae", "ddpm"], required=True)
args = parser.parse_args()

results_dir = f"results/{args.model}"
os.makedirs(results_dir, exist_ok=True)

with open(f"checkpoints/{args.model}/history.json") as file:
    history = json.load(file)

keys = [k[len("train_"):] for k in history if k.startswith("train_")]
epochs = range(1, len(history["train_total_loss"]) + 1)

for key in keys:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history[f"train_{key}"], label="Training")
    plt.plot(epochs, history[f"validation_{key}"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(key.replace("_", " "))
    plt.title(f"{args.model.upper()} — {key.replace('_', ' ')}")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{results_dir}/{key}_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

print(f"Saved {len(keys)} plots to {results_dir}/")