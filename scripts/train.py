"""Main training and evaluation CLI entrypoint using Comprehensive Configuration Schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simplicity_cnn.config.schema import load_config
from simplicity_cnn.data.dataset import CLASS_NAMES, SIMPLIcityDataset
from simplicity_cnn.data.transforms import build_transforms
from simplicity_cnn.evaluation.metrics import evaluate_model
from simplicity_cnn.models.factory import build_model
from simplicity_cnn.training.engine import Trainer
from simplicity_cnn.utils.env import get_default_data_dir, get_device
from simplicity_cnn.utils.logging_utils import setup_logger
from simplicity_cnn.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN model on SIMPLIcity dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset root folder.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture override: baseline_custom, custom_scratch, resnet18, mobilenet_v3_small.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Device: auto, cuda, cpu, mps.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run ultra-fast 2-epoch CPU verification on a micro subset of data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration schema
    overrides: dict[str, Any] = {}
    if args.data_dir:
        overrides["data.data_dir"] = args.data_dir
    if args.model:
        overrides["model.name"] = args.model
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    if args.batch_size:
        overrides["data.batch_size"] = args.batch_size
    if args.lr:
        overrides["training.lr"] = args.lr
    if args.device:
        overrides["environment.device"] = args.device
    if args.output_dir:
        overrides["control.output_dir"] = args.output_dir

    cfg = load_config(args.config, overrides=overrides)

    # Smoke test adjustments
    sample_size = None
    if args.smoke_test:
        print("\n>>> SMOKE TEST MODE ACTIVATED (Micro-batch verification on CPU) <<<")
        cfg.training.epochs = 2
        cfg.data.batch_size = 4
        cfg.data.num_workers = 0
        cfg.environment.device = "cpu"
        cfg.control.output_dir = "outputs/smoke_test"
        sample_size = 8

    output_dir = Path(cfg.control.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("simplicity_cnn", log_file=output_dir / "train.log")

    data_dir = Path(cfg.data.data_dir) if cfg.data.data_dir else get_default_data_dir()
    img_size = tuple(cfg.data.img_size)

    logger.info("=== Experiment: %s ===", cfg.experiment_name)
    logger.info("  Config Source: %s", args.config)
    logger.info("  Data Dir:      %s", data_dir)
    logger.info("  Model Name:    %s", cfg.model.name)
    logger.info("  Image Size:    %s", img_size)
    logger.info("  Batch Size:    %d", cfg.data.batch_size)
    logger.info("  Epochs:        %d", cfg.training.epochs)
    logger.info("  LR:            %.6f", cfg.training.lr)
    logger.info("  Monitor:       %s (%s)", cfg.control.monitor, cfg.control.mode)
    logger.info("  Label Smooth:  %.2f", cfg.training.label_smoothing)
    logger.info("  Output Dir:    %s", output_dir)

    # 1. Seed
    seed_everything(cfg.environment.seed)

    # 2. Device
    device = get_device(cfg.environment.device)
    logger.info("Active compute device: %s", device)

    # 3. Data transforms & datasets
    train_transform = build_transforms(
        img_size=img_size,
        augment=cfg.data.augmentation.enabled,
        mean=cfg.data.mean,
        std=cfg.data.std,
        flip_prob=cfg.data.augmentation.horizontal_flip_prob,
        rotation_deg=cfg.data.augmentation.rotation_degrees,
        color_jitter_params=cfg.data.augmentation.color_jitter,
        crop_scale=cfg.data.augmentation.crop_scale,
        random_erasing_prob=cfg.data.augmentation.random_erasing_prob,
    )
    eval_transform = build_transforms(
        img_size=img_size,
        augment=False,
        mean=cfg.data.mean,
        std=cfg.data.std,
    )

    train_ds = SIMPLIcityDataset(
        data_dir, split="train", transform=train_transform, sample_size=sample_size
    )
    val_ds = SIMPLIcityDataset(
        data_dir, split="validation", transform=eval_transform, sample_size=sample_size
    )
    test_ds = SIMPLIcityDataset(
        data_dir, split="test", transform=eval_transform, sample_size=sample_size
    )

    num_workers = cfg.data.num_workers if sys.platform != "win32" else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda" and cfg.data.pin_memory),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda" and cfg.data.pin_memory),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda" and cfg.data.pin_memory),
    )

    # 4. Build Model
    model = build_model(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        input_size=img_size,
        activation=cfg.model.activation,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
    )

    # 5. Optimizer & Criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    opt_name = cfg.training.optimizer.lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            trainable_params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # 6. Scheduler
    scheduler = None
    sched_name = cfg.training.scheduler.lower()
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.min_lr
        )
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(cfg.training.epochs // 3, 1), gamma=0.2
        )

    # 7. Trainer execution
    use_amp = (cfg.environment.precision == "amp_fp16") and (device.type == "cuda")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        num_epochs=cfg.training.epochs,
        patience=cfg.control.patience,
        grad_clip=cfg.training.gradient_clipping,
        use_amp=use_amp,
        monitor=cfg.control.monitor,
        mode=cfg.control.mode,
        min_delta=cfg.control.min_delta,
        label_smoothing=cfg.training.label_smoothing,
    )

    train_results = trainer.fit()

    # 8. Final evaluation on Test Set
    best_checkpoint = output_dir / "best_model.pt"
    if best_checkpoint.exists():
        logger.info("Loading best checkpoint for test set evaluation: %s", best_checkpoint)
        ckpt = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    logger.info("Running evaluation on test split (%d samples)...", len(test_ds))
    eval_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=CLASS_NAMES,
        output_dir=output_dir / "evaluation",
    )

    print("\n" + "=" * 65)
    print("SIMPLIcity CNN Experiment Complete!")
    print("=" * 65)
    print(f"Experiment Name:      {cfg.experiment_name}")
    print(f"Model Architecture:   {cfg.model.name}")
    print(
        f"Best {cfg.control.monitor}:   {train_results['best_score']:.4f} (Epoch {train_results['best_epoch']})"
    )
    print(f"Test Set Accuracy:    {eval_results['accuracy'] * 100:.2f}%")
    print(f"Test Set Macro F1:    {eval_results['macro_f1']:.4f}")
    print(f"Output Artifacts:     {output_dir.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
