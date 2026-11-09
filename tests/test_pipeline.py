"""Deterministic unit tests for SIMPLIcity CNN pipeline."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from simplicity_cnn.data.dataset import CLASS_NAMES, SIMPLIcityDataset
from simplicity_cnn.data.transforms import build_transforms
from simplicity_cnn.models.baseline_cnn import SimplicityBaselineCNN
from simplicity_cnn.models.custom_scratch import SimplicityCustomCNN
from simplicity_cnn.utils.env import detect_environment, get_default_data_dir, get_device
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
    train_tf = build_transforms(img_size=(128, 128), augment=True)
    val_tf = build_transforms(img_size=(128, 128), augment=False)

    from PIL import Image

    dummy_img = Image.new("RGB", (384, 256), color=(100, 150, 200))

    tensor_train = train_tf(dummy_img)
    tensor_val = val_tf(dummy_img)

    assert tensor_train.shape == (3, 128, 128)
    assert tensor_val.shape == (3, 128, 128)
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
