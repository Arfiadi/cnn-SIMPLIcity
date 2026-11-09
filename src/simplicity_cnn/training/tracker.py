"""Training history tracking and gradient descent curve plotting."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TrainingTracker:
    """Records epoch-wise metrics and renders training/validation gradient descent curves."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: dict[str, list[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "lr": [],
        }

    def record(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        val_macro_f1: float = 0.0,
    ) -> None:
        """Append one epoch's performance metrics."""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["val_macro_f1"].append(val_macro_f1)
        self.history["lr"].append(lr)

    def save_json(self, filename: str = "history.json") -> Path:
        """Export history dictionary to a JSON file."""
        file_path = self.output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        logger.debug("Saved training history to %s", file_path)
        return file_path

    def plot_curves(self, prefix: str = "") -> dict[str, Path]:
        """Render and save loss, accuracy, and combined gradient descent curves.

        Returns:
            dict[str, Path]: Map of curve types to saved image paths.
        """
        epochs = self.history["epoch"]
        if not epochs:
            logger.warning("No history to plot.")
            return {}

        saved_plots: dict[str, Path] = {}
        prefix_str = f"{prefix}_" if prefix else ""

        # 1. Loss curve
        fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
        ax_loss.plot(
            epochs, self.history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2
        )
        ax_loss.plot(
            epochs, self.history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2
        )
        ax_loss.set_title("Training & Validation Loss Curve", fontsize=13, fontweight="bold")
        ax_loss.set_xlabel("Epoch", fontsize=11)
        ax_loss.set_ylabel("Loss", fontsize=11)
        ax_loss.grid(True, linestyle="--", alpha=0.6)
        ax_loss.legend(fontsize=10)
        fig_loss.tight_layout()
        loss_path = self.output_dir / f"{prefix_str}loss_curve.png"
        fig_loss.savefig(loss_path, dpi=200)
        plt.close(fig_loss)
        saved_plots["loss"] = loss_path

        # 2. Accuracy curve
        fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
        ax_acc.plot(
            epochs,
            self.history["train_acc"],
            label="Train Accuracy",
            color="#2ca02c",
            linewidth=2,
        )
        ax_acc.plot(
            epochs,
            self.history["val_acc"],
            label="Validation Accuracy",
            color="#d62728",
            linewidth=2,
        )
        ax_acc.set_title("Training & Validation Accuracy Curve", fontsize=13, fontweight="bold")
        ax_acc.set_xlabel("Epoch", fontsize=11)
        ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
        ax_acc.grid(True, linestyle="--", alpha=0.6)
        ax_acc.legend(fontsize=10)
        fig_acc.tight_layout()
        acc_path = self.output_dir / f"{prefix_str}accuracy_curve.png"
        fig_acc.savefig(acc_path, dpi=200)
        plt.close(fig_acc)
        saved_plots["accuracy"] = acc_path

        # 3. Combined side-by-side plot
        fig_combo, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(
            epochs, self.history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2.2
        )
        ax1.plot(epochs, self.history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2.2)
        ax1.set_title("Loss Convergence", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        ax2.plot(
            epochs, self.history["train_acc"], label="Train Acc", color="#2ca02c", linewidth=2.2
        )
        ax2.plot(epochs, self.history["val_acc"], label="Val Acc", color="#d62728", linewidth=2.2)
        if any(v > 0 for v in self.history["val_macro_f1"]):
            f1_pct = [v * 100 for v in self.history["val_macro_f1"]]
            ax2.plot(
                epochs,
                f1_pct,
                label="Val Macro F1 (%)",
                color="#9467bd",
                linestyle="--",
                linewidth=2.0,
            )

        ax2.set_title("Accuracy & F1 Progression", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Metric (%)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        fig_combo.suptitle(
            "SIMPLIcity CNN Experiment: Training Summary", fontsize=14, fontweight="bold"
        )
        fig_combo.tight_layout()
        combo_path = self.output_dir / f"{prefix_str}gradient_descent_summary.png"
        fig_combo.savefig(combo_path, dpi=200)
        plt.close(fig_combo)
        saved_plots["combined"] = combo_path

        logger.debug("Saved curve plots to: %s", self.output_dir)
        return saved_plots
