"""Comprehensive configuration schema for Deep Learning experiments on SIMPLIcity."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    device: str = "auto"
    precision: str = "amp_fp16"
    seed: int = 42
    deterministic_cudnn: bool = True


@dataclass
class DataAugmentationConfig:
    enabled: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_degrees: int = 10
    color_jitter: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    random_erasing_prob: float = 0.0
    crop_scale: list[float] = field(default_factory=lambda: [0.8, 1.0])


@dataclass
class DataConfig:
    dataset_name: str = "SIMPLIcity"
    data_dir: str | None = None
    img_size: list[int] = field(default_factory=lambda: [128, 128])
    num_classes: int = 10
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    preserve_aspect_ratio: bool = False
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    augmentation: DataAugmentationConfig = field(default_factory=DataAugmentationConfig)


@dataclass
class ModelConfig:
    name: str = "baseline_custom"
    num_classes: int = 10
    activation: str = "relu"
    dropout_rate: float = 0.4
    use_batch_norm: bool = True
    pretrained: bool = True
    freeze_backbone: bool = False
    head_type: str = "standard"


@dataclass
class TrainingConfig:
    epochs: int = 30
    optimizer: str = "adamw"
    lr: float = 0.001
    min_lr: float = 0.00001
    weight_decay: float = 0.0001
    momentum: float = 0.9
    scheduler: str = "cosine"
    loss_name: str = "cross_entropy"
    label_smoothing: float = 0.0
    gradient_clipping: float = 1.0
    warmup_head_epochs: int = 0


@dataclass
class ControlConfig:
    early_stopping_enabled: bool = True
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 10
    min_delta: float = 0.0005
    output_dir: str = "outputs/run_default"


@dataclass
class EvaluationConfig:
    batch_size: int = 32
    metrics: list[str] = field(
        default_factory=lambda: ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    )
    generate_confusion_matrix: bool = True
    error_analysis_top_k: int = 10
    use_tta: bool = False


@dataclass
class ExperimentConfig:
    experiment_name: str = "simplicity_cnn"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        """Validate parameter ranges and logical consistency."""
        if self.training.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.training.epochs}")
        if self.training.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.training.lr}")
        if self.data.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.data.batch_size}")
        if not (0.0 <= self.training.label_smoothing < 1.0):
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {self.training.label_smoothing}"
            )
        if self.control.monitor not in {"val_loss", "val_acc", "val_macro_f1"}:
            logger.warning(
                "Monitor metric '%s' is non-standard. Expected 'val_loss', 'val_acc', or 'val_macro_f1'.",
                self.control.monitor,
            )


def load_config(
    yaml_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load, merge, and validate an ExperimentConfig from YAML and CLI overrides.

    Args:
        yaml_path: Optional path to a YAML configuration file.
        overrides: Optional flat or nested dictionary of overrides.

    Returns:
        ExperimentConfig: Instantiated and validated configuration object.
    """
    raw_dict: dict[str, Any] = {}

    if yaml_path:
        p = Path(yaml_path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                raw_dict = loaded

    # Apply overrides
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            if "." in k:
                sec, subk = k.split(".", 1)
                if sec not in raw_dict:
                    raw_dict[sec] = {}
                raw_dict[sec][subk] = v
            else:
                raw_dict[k] = v

    env_data = raw_dict.get("environment", {})
    data_data = raw_dict.get("data", {})
    aug_data = data_data.get("augmentation", {})
    model_data = raw_dict.get("model", {})
    train_data = raw_dict.get("training", {})
    ctrl_data = raw_dict.get("control", {})
    eval_data = raw_dict.get("evaluation", {})

    aug_cfg = DataAugmentationConfig(
        **{k: v for k, v in aug_data.items() if k in DataAugmentationConfig.__dataclass_fields__}
    )
    data_kwargs = {
        k: v
        for k, v in data_data.items()
        if k in DataConfig.__dataclass_fields__ and k != "augmentation"
    }
    data_cfg = DataConfig(augmentation=aug_cfg, **data_kwargs)

    env_cfg = EnvironmentConfig(
        **{k: v for k, v in env_data.items() if k in EnvironmentConfig.__dataclass_fields__}
    )
    model_cfg = ModelConfig(
        **{k: v for k, v in model_data.items() if k in ModelConfig.__dataclass_fields__}
    )
    train_cfg = TrainingConfig(
        **{k: v for k, v in train_data.items() if k in TrainingConfig.__dataclass_fields__}
    )
    ctrl_cfg = ControlConfig(
        **{k: v for k, v in ctrl_data.items() if k in ControlConfig.__dataclass_fields__}
    )
    eval_cfg = EvaluationConfig(
        **{k: v for k, v in eval_data.items() if k in EvaluationConfig.__dataclass_fields__}
    )

    exp_cfg = ExperimentConfig(
        experiment_name=raw_dict.get("experiment_name", "simplicity_cnn"),
        environment=env_cfg,
        data=data_cfg,
        model=model_cfg,
        training=train_cfg,
        control=ctrl_cfg,
        evaluation=eval_cfg,
    )
    exp_cfg.validate()
    return exp_cfg


def print_config_summary(yaml_path: str | Path) -> ExperimentConfig:
    """Load and display a sleek, condensed experiment configuration summary card.

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        ExperimentConfig: Loaded configuration object.
    """
    p = Path(yaml_path)
    cfg = load_config(p)

    width = 86
    print("=" * width)
    print(f" EXPERIMENT CONFIG CARD: {cfg.experiment_name} ({p.name})")
    print("-" * width)
    print(
        f"  Model       : {cfg.model.name} (Classes: {cfg.model.num_classes}, Input: {cfg.data.img_size}, Pretrained: {cfg.model.pretrained}, Head: {cfg.model.head_type})"
    )
    print(
        f"  Training    : {cfg.training.epochs} epochs | Batch: {cfg.data.batch_size} | Opt: {cfg.training.optimizer} (LR: {cfg.training.lr}, WD: {cfg.training.weight_decay}) | Warmup: {cfg.training.warmup_head_epochs} ep"
    )
    print(
        f"  Objective   : {cfg.training.loss_name} (label_smoothing: {cfg.training.label_smoothing}) | EarlyStop: {cfg.control.early_stopping_enabled} (patience: {cfg.control.patience}, monitor: {cfg.control.monitor} {cfg.control.mode})"
    )
    aug = cfg.data.augmentation
    aug_desc = (
        f"Flip: {aug.horizontal_flip_prob}, Rot: {aug.rotation_degrees}deg, Jitter: {aug.color_jitter}, Erase: {aug.random_erasing_prob}"
        if aug.enabled
        else "Disabled"
    )
    aspect_desc = "Letterbox" if cfg.data.preserve_aspect_ratio else "CenterCrop"
    print(f"  Geometry    : {aspect_desc} | Augment: {aug_desc}")
    print(f"  Artifacts   : {cfg.control.output_dir}")
    print("=" * width)
    return cfg
