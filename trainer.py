import json
import os
import torch
import random, numpy as np
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class Trainer:
    
    def __init__(self, model, optimizer, device, ema=None, use_amp=False):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.ema = ema
        self.use_amp = use_amp and device.type == "cuda"               
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) 
        
    def train_one_epoch(self, train_loader, epoch=None, epochs=None):
        self.model.train()

        total_metrics = {}
        number_of_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for images, _ in progress:
            images = images.to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast("cuda", enabled=self.use_amp):   
                loss, metrics = self.model.compute_loss(images)

            self.scaler.scale(loss).backward()   # REPLACES loss.backward()
            self.scaler.step(self.optimizer)     # REPLACES self.optimizer.step()
            self.scaler.update()                 # ADD

            if self.ema is not None:             # ADD
                self.ema.update(self.model)

            for name, value in metrics.items():
                if name not in total_metrics:
                    total_metrics[name] = 0.0

                total_metrics[name] += value

            number_of_batches += 1
            progress.set_postfix(loss=f"{metrics['total_loss']:.4f}")
        for name in total_metrics:
            total_metrics[name] /= number_of_batches

        return total_metrics

    def validate(self, validation_loader):
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(42) 
        total_metrics = {}
        number_of_batches = 0

        with torch.no_grad():
            for images, _ in tqdm(validation_loader, desc="Validation", leave=False):
                images = images.to(self.device)

                loss, metrics = self.model.compute_loss(images, generator=generator)

                for name, value in metrics.items():
                    if name not in total_metrics:
                        total_metrics[name] = 0.0

                    total_metrics[name] += value

                number_of_batches += 1

        for name in total_metrics:
            total_metrics[name] /= number_of_batches

        return total_metrics


    def resume(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
        return checkpoint["epoch"], checkpoint["history"]

    def train(self, train_loader, validation_loader, epochs, checkpoint_dir, 
              sample_callback=None,resume_from=None,snapshot_interval=25,
              save_best=True):

        os.makedirs(checkpoint_dir, exist_ok=True)
        start_epoch, history = 0, {}
        if resume_from is not None and os.path.exists(resume_from):
            start_epoch, history = self.resume(resume_from)

        best_validation_loss = float("inf")
        for epoch in range(start_epoch + 1, epochs + 1):
        
            train_metrics = self.train_one_epoch(train_loader, epoch=epoch, epochs=epochs)

            validation_metrics = self.validate(validation_loader)

            for name, value in train_metrics.items():
                history_name = "train_" + name

                if history_name not in history:
                    history[history_name] = []

                history[history_name].append(value)

            for name, value in validation_metrics.items():
                history_name = "validation_" + name

                if history_name not in history:
                    history[history_name] = []

                history[history_name].append(value)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train loss: {train_metrics['total_loss']:.4f} | "
                f"Validation loss: "
                f"{validation_metrics['total_loss']:.4f}"
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": (
                    self.optimizer.state_dict()
                ),
                "history": history,
            }
            if self.ema is not None:                                  
                checkpoint["ema_state_dict"] = self.ema.state_dict() 
            if self.use_amp:                                         
                checkpoint["scaler_state_dict"] = self.scaler.state_dict()
            
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, "latest.pt"),
            )

            if save_best and validation_metrics["total_loss"] < best_validation_loss:
                best_validation_loss = (validation_metrics["total_loss"])

                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_dir, "best.pt"),
                )

            with open(os.path.join(checkpoint_dir, "history.json"),"w",) as file:
                json.dump(history, file, indent=2)

            if snapshot_interval and epoch % snapshot_interval == 0:
                torch.save(checkpoint, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))

            if sample_callback is not None:
                sample_callback(epoch, self.model)

        return history