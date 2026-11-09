"""Unit tests for configuration schema and YAML parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from simplicity_cnn.config.schema import ExperimentConfig, load_config


def test_default_config_loading():
    config = load_config("configs/default_config.yaml")
    assert isinstance(config, ExperimentConfig)
    assert config.model.name == "baseline_custom"
    assert config.data.num_classes == 10
    assert config.training.epochs == 30
    assert config.training.lr == 0.001


def test_all_dedicated_yaml_configs():
    yaml_files = [
        "configs/baseline_custom.yaml",
        "configs/custom_scratch_regularized.yaml",
        "configs/resnet18_transfer.yaml",
        "configs/smoke_test.yaml",
    ]
    for yf in yaml_files:
        assert Path(yf).exists(), f"Configuration file {yf} must exist"
        cfg = load_config(yf)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.data.num_classes == 10
        assert cfg.training.epochs > 0
        assert cfg.training.lr > 0


def test_eda_treatment_config_parameters():
    cfg = load_config("configs/custom_scratch_regularized.yaml")
    assert cfg.model.name == "custom_scratch"
    assert cfg.data.augmentation.color_jitter[0] == 0.25
    assert cfg.data.augmentation.random_erasing_prob == 0.20
    assert cfg.data.augmentation.crop_scale == [0.75, 1.0]
    assert cfg.training.label_smoothing == 0.10
    assert cfg.control.monitor == "val_macro_f1"
    assert cfg.control.mode == "max"


def test_config_overrides():
    overrides = {
        "training.lr": 0.005,
        "training.epochs": 50,
        "model.name": "mobilenet_v3_small",
    }
    cfg = load_config("configs/default_config.yaml", overrides=overrides)
    assert cfg.training.lr == 0.005
    assert cfg.training.epochs == 50
    assert cfg.model.name == "mobilenet_v3_small"


def test_config_validation_failures():
    with pytest.raises(ValueError, match="epochs must be > 0"):
        load_config("configs/default_config.yaml", overrides={"training.epochs": -5})

    with pytest.raises(ValueError, match="lr must be > 0"):
        load_config("configs/default_config.yaml", overrides={"training.lr": -0.001})

    with pytest.raises(ValueError, match="label_smoothing must be in"):
        load_config("configs/default_config.yaml", overrides={"training.label_smoothing": 1.5})
