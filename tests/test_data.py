"""Dataset discovery, layout tolerance and normalisation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cyclegan.data import (
    UnpairedImageDataset,
    build_dataloaders,
    build_transform,
    list_images,
    resolve_split_dirs,
)


def test_resolves_canonical_layout(fake_dataset: Path) -> None:
    train_a, train_b = resolve_split_dirs(fake_dataset, "train")
    assert train_a.name == "trainA" and train_b.name == "trainB"


def test_resolves_legacy_summer_winter_layout(tmp_path: Path, fake_dataset: Path) -> None:
    """Some re-uploads name the folders summer/winter/test_summer/test_winter."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    renames = (("trainA", "summer"), ("trainB", "winter"), ("testA", "test_summer"), ("testB", "test_winter"))
    for old, new in renames:
        (fake_dataset / old).rename(legacy / new)

    train_a, _ = resolve_split_dirs(legacy, "train")
    test_a, test_b = resolve_split_dirs(legacy, "test")
    assert train_a.name == "summer"
    assert (test_a.name, test_b.name) == ("test_summer", "test_winter")


def test_finds_images_nested_one_level_deeper(tmp_path: Path, fake_dataset: Path) -> None:
    """An ImageFolder-style trainA/data/*.jpg nesting must still load."""
    nested = fake_dataset / "trainA" / "data"
    nested.mkdir()
    for image in list(fake_dataset.glob("trainA/*.jpg")):
        image.rename(nested / image.name)
    assert len(list_images(fake_dataset / "trainA")) == 6


def test_missing_directory_names_what_it_tried(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="trainA"):
        resolve_split_dirs(tmp_path, "train")


def test_rejects_unknown_split(fake_dataset: Path) -> None:
    with pytest.raises(ValueError):
        resolve_split_dirs(fake_dataset, "validation")


def test_length_follows_the_larger_domain(fake_dataset: Path) -> None:
    train_a, train_b = resolve_split_dirs(fake_dataset, "train")
    dataset = UnpairedImageDataset(train_a, train_b, load_size=72, crop_size=64)
    assert len(dataset) == 6  # 6 in A, 4 in B


def test_images_are_normalized_to_tanh_range(fake_dataset: Path) -> None:
    """The generator emits [-1, 1]; inputs must live on the same scale."""
    train_a, train_b = resolve_split_dirs(fake_dataset, "train")
    dataset = UnpairedImageDataset(train_a, train_b, load_size=72, crop_size=64)
    sample = dataset[0]
    assert sample["A"].shape == (3, 64, 64)
    assert sample["A"].min() >= -1.0 and sample["A"].max() <= 1.0
    assert sample["A"].min() < 0, "noise images should use the negative half of the range"


def test_eval_transform_is_deterministic(fake_dataset: Path) -> None:
    train_a, train_b = resolve_split_dirs(fake_dataset, "train")
    dataset = UnpairedImageDataset(train_a, train_b, 72, 64, train=False)
    torch.testing.assert_close(dataset[0]["A"], dataset[0]["A"])
    assert dataset[0]["path_B"] == dataset[0]["path_B"]


def test_train_transform_pipeline_includes_augmentation() -> None:
    train_ops = [type(t).__name__ for t in build_transform(72, 64, train=True).transforms]
    eval_ops = [type(t).__name__ for t in build_transform(72, 64, train=False).transforms]
    assert "RandomCrop" in train_ops and "RandomHorizontalFlip" in train_ops
    assert "RandomCrop" not in eval_ops and "RandomHorizontalFlip" not in eval_ops


def test_max_images_caps_both_domains(fake_dataset: Path) -> None:
    train_a, train_b = resolve_split_dirs(fake_dataset, "train")
    dataset = UnpairedImageDataset(train_a, train_b, 72, 64, max_images=2)
    assert len(dataset) == 2


def test_build_dataloaders_yields_batches(fake_dataset: Path) -> None:
    train_loader, test_loader = build_dataloaders(
        fake_dataset, load_size=72, crop_size=64, batch_size=2, num_workers=0
    )
    batch = next(iter(train_loader))
    assert batch["A"].shape == (2, 3, 64, 64)
    assert batch["B"].shape == (2, 3, 64, 64)
    assert next(iter(test_loader))["A"].shape == (1, 3, 64, 64)
