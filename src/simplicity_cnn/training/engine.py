"""Model training engine with mixed-precision, early stopping, and checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from simplicity_cnn.training.tracker import TrainingTracker

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates model training, validation, checkpointing, and metric logging."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module | None = None,
        scheduler: Any = None,
        device: torch.device | None = None,
        output_dir: Path | str = "outputs",
        num_epochs: int = 30,
        patience: int = 10,
        grad_clip: float = 1.0,
        use_amp: bool = True,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.patience = patience
        self.grad_clip = grad_clip

        # Enable AMP only if CUDA is available and requested
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)

        self.tracker = TrainingTracker(self.output_dir)
        self.best_val_acc: float = -1.0
        self.best_epoch: int = -1

    def train_one_epoch(self, epoch: int) -> tuple[float, float]:
        """Execute one training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch:02d}/{self.num_epochs:02d} [Train]", leave=False
        )
        for images, targets, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += batch_size

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{(correct / total) * 100:.2f}%"}
            )

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = (correct / max(total, 1)) * 100.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """Evaluate model performance on the validation split."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets, _ in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += batch_size

        val_loss = running_loss / max(total, 1)
        val_acc = (correct / max(total, 1)) * 100.0
        return val_loss, val_acc

    def fit(self) -> dict[str, Any]:
        """Run complete training cycle with validation, early stopping, and checkpointing."""
        logger.info(
            "Starting training for %d epochs on device: %s (AMP: %s)",
            self.num_epochs,
            self.device,
            self.use_amp,
        )

        epochs_no_improve = 0

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Learning rate step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            self.tracker.record(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr,
            )

            logger.info(
                "Epoch %02d/%02d | Train Loss: %.4f, Train Acc: %.2f%% | Val Loss: %.4f, Val Acc: %.2f%% | LR: %.6f",
                epoch,
                self.num_epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr,
            )

            # Checkpoint best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                epochs_no_improve = 0
                self._save_checkpoint("best_model.pt", epoch, val_loss, val_acc)
            else:
                epochs_no_improve += 1

            # Early stopping check
            if epochs_no_improve >= self.patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement. Best Val Acc: %.2f%% at epoch %d",
                    self.patience,
                    self.best_val_acc,
                    self.best_epoch,
                )
                break

        # Save final model state and export plots
        self._save_checkpoint("last_model.pt", epoch, val_loss, val_acc)
        self.tracker.save_json("history.json")
        plots = self.tracker.plot_curves()

        return {
            "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "final_epoch": epoch,
            "plots": plots,
            "history": self.tracker.history,
        }

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float, val_acc: float) -> Path:
        checkpoint_path = self.output_dir / filename
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": self.best_val_acc,
        }
        torch.save(checkpoint_data, checkpoint_path)
        logger.info("Saved model checkpoint: %s (Val Acc: %.2f%%)", checkpoint_path, val_acc)
        return checkpoint_path
