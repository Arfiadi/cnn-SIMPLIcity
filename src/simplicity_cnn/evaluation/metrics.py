"""Evaluation metrics calculation and confusion matrix visualization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    output_dir: Path | str,
) -> dict[str, Any]:
    """Perform detailed evaluation on test dataset and save diagnostic artifacts.

    Args:
        model: Evaluated PyTorch model.
        test_loader: DataLoader for test split.
        device: Device to run evaluation on.
        class_names: List of class labels (ordered 0..C-1).
        output_dir: Path to store generated plots and metrics report.

    Returns:
        dict: Evaluation metrics and error analysis findings.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    all_preds: list[int] = []
    all_targets: list[int] = []
    all_confs: list[float] = []
    all_filenames: list[str] = []

    for images, targets, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confidences, preds = probs.max(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.numpy().tolist())
        all_confs.extend(confidences.cpu().numpy().tolist())
        all_filenames.extend(filenames)

    all_targets_np = np.array(all_targets)
    all_preds_np = np.array(all_preds)

    # 1. Classification report
    label_indices = list(range(len(class_names)))
    report_dict = classification_report(
        all_targets_np,
        all_preds_np,
        labels=label_indices,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        all_targets_np,
        all_preds_np,
        labels=label_indices,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    logger.debug("Classification Report:\n%s", report_text)

    # Save reports
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # 2. Confusion matrix
    cm = confusion_matrix(all_targets_np, all_preds_np, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title("SIMPLIcity Test Set Confusion Matrix", fontsize=13, fontweight="bold")

    # Add numeric annotations
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    logger.debug("Saved confusion matrix to: %s", cm_path)

    # 3. Misclassification error analysis
    misclassified: list[dict[str, Any]] = []
    for idx, (target, pred, conf, fname) in enumerate(
        zip(all_targets, all_preds, all_confs, all_filenames)
    ):
        if target != pred:
            misclassified.append(
                {
                    "filename": fname,
                    "true_label": class_names[target],
                    "predicted_label": class_names[pred],
                    "confidence": round(conf, 4),
                }
            )

    # Sort by highest confidence errors
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)
    with open(out_dir / "misclassified_samples.json", "w", encoding="utf-8") as f:
        json.dump(misclassified, f, indent=2)

    return {
        "accuracy": report_dict.get("accuracy", 0.0),
        "macro_f1": report_dict.get("macro avg", {}).get("f1-score", 0.0),
        "report_dict": report_dict,
        "confusion_matrix_path": str(cm_path),
        "total_misclassified": len(misclassified),
        "misclassified_samples": misclassified,
    }
