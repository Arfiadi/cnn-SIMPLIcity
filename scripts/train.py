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
from simplicity_cnn.models.transfer import get_parameter_groups
from simplicity_cnn.training.engine import Trainer
from simplicity_cnn.utils.env import get_cuda_device_info, get_default_data_dir, get_device
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
        help="Model architecture override: baseline_custom, resnet18, resnet34, mobilenet_v3_large, efficientnet_b0, convnext_tiny.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Device: auto, cuda, cpu, mps.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable DataParallel across all available CUDA GPUs.",
    )
    parser.add_argument(
        "--backbone-lr-ratio",
        type=float,
        default=None,
        help="Backbone LR ratio (e.g. 0.1 for 10x lower LR on backbone). Enables differential learning rate.",
    )
    parser.add_argument(
        "--warmup-head-epochs",
        type=int,
        default=None,
        help="Number of head-only warm-up epochs before unfreezing backbone.",
    )
    parser.add_argument(
        "--preserve-aspect-ratio",
        action="store_true",
        help="Preserve aspect ratio using Letterbox padding instead of center crop.",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default=None,
        help="Head architecture type: standard or enhanced.",
    )
    parser.add_argument(
        "--use-tta",
        action="store_true",
        help="Enable Test-Time Augmentation (TTA) during final evaluation.",
    )
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
    if args.warmup_head_epochs is not None:
        overrides["training.warmup_head_epochs"] = args.warmup_head_epochs
    if args.preserve_aspect_ratio:
        overrides["data.preserve_aspect_ratio"] = True
    if args.head_type:
        overrides["model.head_type"] = args.head_type
    if args.use_tta:
        overrides["evaluation.use_tta"] = True

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
    img_size: tuple[int, int] = (int(cfg.data.img_size[0]), int(cfg.data.img_size[1]))

    # Log detailed diagnostics to file
    cuda_info = get_cuda_device_info()
    logger.debug(
        "Environment & Hardware: CUDA=%s, Count=%d, Devices=%s",
        cuda_info["available"],
        cuda_info["count"],
        cuda_info["devices"],
    )
    logger.debug(
        "Experiment: %s, Model: %s, Data: %s, Output: %s",
        cfg.experiment_name,
        cfg.model.name,
        data_dir,
        output_dir,
    )

    # 1. Seed
    seed_everything(cfg.environment.seed)

    # 2. Device
    device = get_device(cfg.environment.device)

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
        preserve_aspect_ratio=cfg.data.preserve_aspect_ratio,
    )
    eval_transform = build_transforms(
        img_size=img_size,
        augment=False,
        mean=cfg.data.mean,
        std=cfg.data.std,
        preserve_aspect_ratio=cfg.data.preserve_aspect_ratio,
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

    num_workers = min(cfg.data.num_workers, 4) if sys.platform != "win32" else 0
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
        head_type=cfg.model.head_type,
    )

    # 5. Optimizer & Criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    # Parameter groups (differential learning rate if requested)
    backbone_ratio = args.backbone_lr_ratio or 0.1
    if 0.0 < backbone_ratio < 1.0:
        param_groups = get_parameter_groups(
            model,
            base_lr=cfg.training.lr,
            backbone_lr_ratio=backbone_ratio,
            weight_decay=cfg.training.weight_decay,
        )
        logger.debug("Configured differential learning rate (backbone ratio: %.3f)", backbone_ratio)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": trainable_params}]

    opt_name = cfg.training.optimizer.lower()
    optimizer: torch.optim.Optimizer
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            param_groups, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # 6. Scheduler
    scheduler: Any = None
    sched_name = cfg.training.scheduler.lower()
    sched_epochs = (
        max(cfg.training.epochs - cfg.training.warmup_head_epochs, 1)
        if cfg.training.warmup_head_epochs > 0
        else cfg.training.epochs
    )
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sched_epochs, eta_min=cfg.training.min_lr
        )
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(sched_epochs // 3, 1), gamma=0.2
        )

    # 7. Trainer execution
    use_amp = (cfg.environment.precision == "amp_fp16") and (device.type == "cuda")
    trainer_multi_gpu = args.multi_gpu and (device.type == "cuda") and (cuda_info["count"] > 1)

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
        multi_gpu=args.multi_gpu,
        warmup_head_epochs=cfg.training.warmup_head_epochs,
        backbone_lr_ratio=backbone_ratio,
        base_lr=cfg.training.lr,
        min_lr=cfg.training.min_lr,
        weight_decay=cfg.training.weight_decay,
    )

    # Print concise execution summary card
    device_desc = (
        f"{device} ({', '.join(cuda_info['devices'])})"
        if cuda_info["available"]
        else str(device)
    )
    lr_desc = f"{cfg.training.lr:.2e}"
    if 0.0 < backbone_ratio < 1.0:
        lr_desc += f" (backbone: {cfg.training.lr * backbone_ratio:.2e})"

    banner_width = 93
    aspect_desc = "Letterbox" if cfg.data.preserve_aspect_ratio else "CenterCrop"
    print("=" * banner_width)
    print(f" SIMPLIcity Experiment : {cfg.experiment_name}")
    print(f" Architecture          : {cfg.model.name} (Head: {cfg.model.head_type}, Pretrained: {cfg.model.pretrained})")
    print(f" Compute Device        : {device_desc} | Multi-GPU: {trainer_multi_gpu} | AMP: {use_amp}")
    print(f" Data Configuration    : Batch: {cfg.data.batch_size} | Size: {img_size[0]}x{img_size[1]} ({aspect_desc}) | Workers: {num_workers}")
    print(f" Optimization          : {cfg.training.epochs} ep (Warmup: {cfg.training.warmup_head_epochs} ep) | Opt: {cfg.training.optimizer} (LR: {lr_desc})")
    print("=" * banner_width)

    train_results = trainer.fit()

    # 8. Final evaluation on Test Set
    eval_model = trainer.raw_model
    best_checkpoint = output_dir / "best_model.pt"
    if best_checkpoint.exists():
        logger.debug("Loading best checkpoint for test set evaluation: %s", best_checkpoint)
        ckpt = torch.load(best_checkpoint, map_location=device, weights_only=True)
        eval_model.load_state_dict(ckpt["model_state_dict"])

    use_tta = cfg.evaluation.use_tta or args.use_tta
    logger.debug("Running evaluation on test split (%d samples, TTA=%s)...", len(test_ds), use_tta)
    eval_results = evaluate_model(
        model=eval_model,
        test_loader=test_loader,
        device=device,
        class_names=CLASS_NAMES,
        output_dir=output_dir / "evaluation",
        use_tta=use_tta,
    )

    print("\n" + "=" * banner_width)
    print(" SIMPLIcity CNN Experiment Complete!")
    print("=" * banner_width)
    print(f" Experiment Name    : {cfg.experiment_name}")
    print(f" Model Architecture : {cfg.model.name} (Head: {cfg.model.head_type})")
    print(f" Best {cfg.control.monitor:<13} : {train_results['best_score']:.4f} (Epoch {train_results['best_epoch']})")
    print(f" Test Accuracy      : {eval_results['accuracy'] * 100:.2f}% (TTA: {use_tta})")
    print(f" Test Macro F1      : {eval_results['macro_f1']:.4f}")
    print(f" Artifacts Saved To : {output_dir.resolve()}")
    print("=" * banner_width)


if __name__ == "__main__":
    main()
