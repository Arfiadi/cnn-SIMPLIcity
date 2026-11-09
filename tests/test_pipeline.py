"""Deterministic unit tests for SIMPLIcity CNN pipeline."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from simplicity_cnn.data.dataset import CLASS_NAMES, SIMPLIcityDataset
from simplicity_cnn.data.transforms import build_transforms
from simplicity_cnn.models.baseline_cnn import SimplicityBaselineCNN
from simplicity_cnn.models.custom_scratch import SimplicityCustomCNN
from simplicity_cnn.models.factory import build_model
from simplicity_cnn.models.transfer import (
    SUPPORTED_BACKBONES,
    freeze_backbone_batchnorm,
    get_parameter_groups,
)
from simplicity_cnn.utils.env import (
    detect_environment,
    get_cuda_device_info,
    get_default_data_dir,
    get_device,
)
from simplicity_cnn.utils.seed import seed_everything


def test_seed_everything():
    seed_everything(1234)
    t1 = torch.rand(5)
    seed_everything(1234)
    t2 = torch.rand(5)
    assert torch.equal(t1, t2), "Seeding should produce identical random tensors."


def test_environment_and_device():
    env = detect_environment()
    assert env in {"local", "colab", "kaggle"}

    dev = get_device("auto")
    assert isinstance(dev, torch.device)

    cpu_dev = get_device("cpu")
    assert cpu_dev.type == "cpu"


def test_cuda_device_info():
    info = get_cuda_device_info()
    assert "available" in info
    assert "count" in info
    assert "devices" in info
    assert "is_multi_gpu" in info
    assert isinstance(info["devices"], list)


def test_baseline_cnn_forward_shape():
    model = SimplicityBaselineCNN(num_classes=10, input_size=(128, 128), activation="relu")
    dummy_input = torch.randn(2, 3, 128, 128)
    output = model(dummy_input)
    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"


def test_custom_scratch_cnn_forward_shape():
    model = SimplicityCustomCNN(num_classes=10, in_channels=3)
    dummy_input = torch.randn(2, 3, 128, 128)
    output = model(dummy_input)
    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"


def test_gradient_flow_and_optimization():
    model = SimplicityBaselineCNN(num_classes=10, input_size=(128, 128))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(4, 3, 128, 128)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    assert not torch.isnan(loss), "Loss must not be NaN"
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} was not computed"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"

    optimizer.step()


def test_transforms_pipeline():
    train_tf = build_transforms(img_size=(224, 224), augment=True)
    val_tf = build_transforms(img_size=(224, 224), augment=False)

    from PIL import Image

    dummy_img = Image.new("RGB", (384, 256), color=(100, 150, 200))

    tensor_train = train_tf(dummy_img)
    tensor_val = val_tf(dummy_img)

    assert tensor_train.shape == (3, 224, 224)
    assert tensor_val.shape == (3, 224, 224)
    assert tensor_train.dtype == torch.float32
    assert tensor_val.dtype == torch.float32


def test_dataset_loading_if_exists():
    data_dir = get_default_data_dir()
    if not (data_dir / "train").exists():
        pytest.skip(f"Dataset train folder not present at {data_dir}")

    transform = build_transforms(img_size=(128, 128), augment=False)
    ds = SIMPLIcityDataset(data_dir, split="train", transform=transform, sample_size=4)

    assert len(ds) == 4
    image, label, fname = ds[0]
    assert image.shape == (3, 128, 128)
    assert 0 <= label < len(CLASS_NAMES)
    assert isinstance(fname, str)


def test_dataset_split_disjointness():
    """Verify strictly disjoint partitions: Train ∩ Val ∩ Test = ∅ to guarantee zero leakage."""
    data_dir = get_default_data_dir()
    if not (data_dir / "train").exists():
        pytest.skip(f"Dataset train folder not present at {data_dir}")

    transform = build_transforms(img_size=(128, 128), augment=False)
    train_ds = SIMPLIcityDataset(data_dir, split="train", transform=transform)
    val_ds = SIMPLIcityDataset(data_dir, split="validation", transform=transform)
    test_ds = SIMPLIcityDataset(data_dir, split="test", transform=transform)

    train_files = {p.name for p, _ in train_ds.samples}
    val_files = {p.name for p, _ in val_ds.samples}
    test_files = {p.name for p, _ in test_ds.samples}

    assert len(train_files) == 500, f"Expected 500 train images, got {len(train_files)}"
    assert len(val_files) == 300, f"Expected 300 val images, got {len(val_files)}"
    assert len(test_files) == 200, f"Expected 200 test images, got {len(test_files)}"

    assert train_files.isdisjoint(val_files), "Train and validation splits must be disjoint!"
    assert train_files.isdisjoint(test_files), "Train and test splits must be disjoint!"
    assert val_files.isdisjoint(test_files), "Validation and test splits must be disjoint!"


def test_transfer_models_forward_shape():
    """Verify all supported transfer learning backbones output correct shape on 224x224."""
    for model_name in SUPPORTED_BACKBONES:
        model = build_model(model_name=model_name, num_classes=10, pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2, 10), f"Expected shape (2, 10) for {model_name}, got {output.shape}"


def test_parameter_groups_differential_lr():
    """Verify get_parameter_groups splits backbone vs head correctly with differential LR."""
    model = build_model(model_name="resnet18", num_classes=10, pretrained=False)
    groups = get_parameter_groups(model, base_lr=3e-4, backbone_lr_ratio=0.1)

    assert len(groups) == 2, "Should produce exactly 2 parameter groups (backbone and head)"
    backbone_lr = groups[0]["lr"]
    head_lr = groups[1]["lr"]
    assert abs(backbone_lr - 3e-5) < 1e-8, f"Backbone LR should be 3e-5, got {backbone_lr}"
    assert abs(head_lr - 3e-4) < 1e-8, f"Head LR should be 3e-4, got {head_lr}"

    total_group_params = sum(p.numel() for g in groups for p in g["params"])
    total_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_group_params == total_model_params, "All trainable parameters must be accounted for"


def test_freeze_backbone_batchnorm():
    """Verify BatchNorm layers are frozen in eval mode with requires_grad=False."""
    model = build_model(model_name="resnet18", num_classes=10, pretrained=False)
    freeze_backbone_batchnorm(model)

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            assert not m.training, "BatchNorm should be set to eval mode"
            for p in m.parameters():
                assert not p.requires_grad, "BatchNorm parameters should not require grad"


def test_dataparallel_checkpoint_unwrap():
    """Verify checkpoint saved from DataParallel unwraps cleanly without 'module.' prefix."""
    model = build_model(model_name="resnet18", num_classes=10, pretrained=False)
    wrapped = torch.nn.DataParallel(model)

    raw = wrapped.module if isinstance(wrapped, torch.nn.DataParallel) else wrapped
    state = raw.state_dict()

    for key in state:
        assert not key.startswith("module."), f"Key '{key}' should not have 'module.' prefix"


def test_letterbox_pad_transforms():
    """Verify LetterboxPad preserves aspect ratio and fills padding with ImageNet mean."""
    from PIL import Image

    from simplicity_cnn.data.transforms import LetterboxPad

    padder = LetterboxPad(target_size=(240, 240))

    # 1. Landscape tensor [3, 256, 384]
    t_land = torch.randn(3, 256, 384)
    out_land = padder(t_land)
    assert out_land.shape == (3, 240, 240), f"Expected (3, 240, 240), got {out_land.shape}"
    # Verify pad area (top row) has ImageNet mean (~0.486, ~0.455, ~0.408)
    assert abs(out_land[0, 0, 0].item() - (124 / 255.0)) < 1e-4

    # 2. Portrait tensor [3, 384, 256]
    t_port = torch.randn(3, 384, 256)
    out_port = padder(t_port)
    assert out_port.shape == (3, 240, 240), f"Expected (3, 240, 240), got {out_port.shape}"
    # Verify pad area (left column) has ImageNet mean
    assert abs(out_port[0, 0, 0].item() - (124 / 255.0)) < 1e-4

    # 3. PIL image
    img = Image.new("RGB", (384, 256), color=(255, 0, 0))
    out_pil = padder(img)
    assert out_pil.size == (240, 240), f"Expected (240, 240), got {out_pil.size}"
    # Check top-left pixel in padding
    assert out_pil.getpixel((0, 0)) == (124, 116, 104)


def test_efficientnet_enhanced_head():
    """Verify EfficientNet-B0 with enhanced two-layer head outputs correct shape."""
    model = build_model(
        model_name="efficientnet_b0",
        num_classes=10,
        pretrained=False,
        head_type="enhanced",
    )
    dummy_input = torch.randn(2, 3, 240, 240)
    output = model(dummy_input)
    assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"

    groups = get_parameter_groups(model, base_lr=3e-4, backbone_lr_ratio=0.1)
    assert len(groups) == 2, "Expected 2 parameter groups (backbone and enhanced head)"
    assert groups[1]["lr"] == 3e-4, "Head LR should match base_lr"
    assert abs(groups[0]["lr"] - 3e-5) < 1e-8, "Backbone LR should be 10x lower"


def test_evaluate_model_with_tta(tmp_path):
    """Verify evaluate_model runs cleanly with use_tta=True."""
    from torch.utils.data import DataLoader, Dataset

    from simplicity_cnn.evaluation.metrics import evaluate_model

    model = build_model("efficientnet_b0", num_classes=10, pretrained=False)
    dummy_images = torch.randn(6, 3, 240, 240)
    dummy_targets = torch.tensor([0, 1, 2, 3, 4, 5])
    dummy_names = [f"test_{i}.jpg" for i in range(6)]

    class MockDataset(Dataset):
        def __len__(self):
            return 6

        def __getitem__(self, idx):
            return dummy_images[idx], dummy_targets[idx], dummy_names[idx]

    loader = DataLoader(MockDataset(), batch_size=2)
    dev = torch.device("cpu")
    class_names = [f"class_{i}" for i in range(10)]

    results = evaluate_model(
        model=model,
        test_loader=loader,
        device=dev,
        class_names=class_names,
        output_dir=tmp_path / "eval_tta",
        use_tta=True,
    )
    assert "accuracy" in results
    assert "macro_f1" in results
