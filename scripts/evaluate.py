"""Standalone evaluation script for evaluating saved model checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from simplicity_cnn.data.dataset import CLASS_NAMES, SIMPLIcityDataset
from simplicity_cnn.data.transforms import build_transforms
from simplicity_cnn.evaluation.metrics import evaluate_model
from simplicity_cnn.models.factory import build_model
from simplicity_cnn.utils.env import get_default_data_dir, get_device
from simplicity_cnn.utils.logging_utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN checkpoint on SIMPLIcity test set."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .pt checkpoint file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="baseline_custom",
        help="Model architecture: baseline_custom, custom_scratch, resnet18, mobilenet_v3_small.",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root folder.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--img-size", type=int, default=128, help="Square image resolution.")
    parser.add_argument(
        "--device", type=str, default="auto", help="Compute device: auto, cuda, cpu."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval_standalone",
        help="Output directory for reports.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("eval_logger", log_file=out_dir / "eval.log")

    device = get_device(args.device)
    data_dir = Path(args.data_dir) if args.data_dir else get_default_data_dir()

    logger.info("Evaluating model: %s from checkpoint: %s", args.model, args.checkpoint)

    # 1. Dataset & loader
    eval_transform = build_transforms(img_size=(args.img_size, args.img_size), augment=False)
    test_ds = SIMPLIcityDataset(data_dir, split="test", transform=eval_transform)
    num_workers = 2 if sys.platform != "win32" else 0
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )

    # 2. Build model & load weights
    model = build_model(
        model_name=args.model,
        num_classes=len(CLASS_NAMES),
        input_size=(args.img_size, args.img_size),
    )
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    # 3. Evaluate
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=CLASS_NAMES,
        output_dir=out_dir,
    )

    print("\n" + "=" * 60)
    print("Standalone Evaluation Summary")
    print("=" * 60)
    print(f"Test Set Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:    {results['macro_f1']:.4f}")
    print(f"Misclassified:     {results['total_misclassified']} / {len(test_ds)}")
    print(f"Artifacts saved:   {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
