"""Main training and evaluation CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from simplicity_cnn.data.dataset import CLASS_NAMES, SIMPLIcityDataset
from simplicity_cnn.data.transforms import build_transforms
from simplicity_cnn.evaluation.metrics import evaluate_model
from simplicity_cnn.models.factory import build_model
from simplicity_cnn.training.engine import Trainer
from simplicity_cnn.utils.env import get_default_data_dir, get_device
from simplicity_cnn.utils.logging_utils import setup_logger
from simplicity_cnn.utils.seed import seed_everything


def load_yaml_config(config_path: Path | str | None) -> dict:
    """Load configuration dictionary from YAML file."""
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN model on SIMPLIcity dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset root folder.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture: baseline_custom, custom_scratch, resnet18, mobilenet_v3_small.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for training and evaluation."
    )
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use: auto, cuda, cpu, mps."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save checkpoints and plots."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run ultra-fast 2-epoch CPU verification on a micro subset of data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load base config
    cfg = load_yaml_config(args.config) if Path(args.config).exists() else {}
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})

    # Apply CLI overrides
    data_dir_arg = args.data_dir or data_cfg.get("data_dir")
    data_dir = Path(data_dir_arg) if data_dir_arg else get_default_data_dir()

    model_name = args.model or model_cfg.get("name", "baseline_custom")
    num_classes = model_cfg.get("num_classes", 10)
    img_size = tuple(data_cfg.get("img_size", [128, 128]))

    batch_size = args.batch_size or data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 2 if sys.platform != "win32" else 0)
    epochs = args.epochs or train_cfg.get("epochs", 30)
    lr = args.lr or train_cfg.get("lr", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    scheduler_name = train_cfg.get("scheduler", "cosine").lower()
    patience = train_cfg.get("patience", 10)
    seed = train_cfg.get("seed", 42)
    device_pref = args.device or train_cfg.get("device", "auto")
    output_dir = Path(args.output_dir or output_cfg.get("dir", f"outputs/{model_name}"))

    # Smoke test override
    sample_size = None
    if args.smoke_test:
        print("\n>>> SMOKE TEST MODE ACTIVATED (Micro-batch verification on CPU) <<<")
        epochs = 2
        batch_size = 4
        num_workers = 0
        sample_size = 8
        device_pref = "cpu"
        output_dir = Path("outputs/smoke_test")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("simplicity_cnn", log_file=output_dir / "train.log")

    logger.info("Configuration summary:")
    logger.info("  Data Dir:    %s", data_dir)
    logger.info("  Model Name:  %s", model_name)
    logger.info("  Image Size:  %s", img_size)
    logger.info("  Batch Size:  %d", batch_size)
    logger.info("  Epochs:      %d", epochs)
    logger.info("  LR:          %.6f", lr)
    logger.info("  Output Dir:  %s", output_dir)

    # 1. Seed
    seed_everything(seed)

    # 2. Device
    device = get_device(device_pref)
    logger.info("Active compute device: %s", device)

    # 3. Data transforms & datasets
    train_transform = build_transforms(img_size=img_size, augment=True)
    eval_transform = build_transforms(img_size=img_size, augment=False)

    train_ds = SIMPLIcityDataset(
        data_dir, split="train", transform=train_transform, sample_size=sample_size
    )
    val_ds = SIMPLIcityDataset(
        data_dir, split="validation", transform=eval_transform, sample_size=sample_size
    )
    test_ds = SIMPLIcityDataset(
        data_dir, split="test", transform=eval_transform, sample_size=sample_size
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Build Model
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        input_size=img_size,
        activation=model_cfg.get("activation", "relu"),
        pretrained=model_cfg.get("pretrained", True),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    )

    # 5. Optimizer & Criterion
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # 6. Scheduler
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(epochs // 3, 1), gamma=0.2
        )

    # 7. Trainer execution
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        num_epochs=epochs,
        patience=patience,
        use_amp=train_cfg.get("use_amp", True),
    )

    train_results = trainer.fit()

    # 8. Final evaluation on Test Set
    best_checkpoint = output_dir / "best_model.pt"
    if best_checkpoint.exists():
        logger.info("Loading best checkpoint for test set evaluation: %s", best_checkpoint)
        ckpt = (
            torch.save
            if False
            else torch.load(best_checkpoint, map_location=device, weights_only=False)
        )
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
    print(f"Model Architecture:   {model_name}")
    print(
        f"Best Validation Acc:  {train_results['best_val_acc']:.2f}% (Epoch {train_results['best_epoch']})"
    )
    print(f"Test Set Accuracy:    {eval_results['accuracy'] * 100:.2f}%")
    print(f"Test Set Macro F1:    {eval_results['macro_f1']:.4f}")
    print(f"Output Artifacts:     {output_dir.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
