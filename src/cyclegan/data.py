"""Unpaired dataset and loaders for summer2winter_yosemite.

The Kaggle mirror of the dataset ships the canonical CycleGAN layout::

    data/summer2winter_yosemite/
    ├── trainA/   summer, training     (1231 images)
    ├── trainB/   winter, training     ( 962 images)
    ├── testA/    summer, held out     ( 309 images)
    └── testB/    winter, held out     ( 238 images)

``resolve_split_dirs`` also accepts the ``summer/ winter/ test_summer/
test_winter/`` naming used by some re-uploads, and looks for images
recursively, so a stray ``trainA/data/*.jpg`` nesting still loads.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Domain A is summer, domain B is winter. Each tuple lists the directory names
# that have been seen in the wild for that (split, domain), in priority order.
_SPLIT_ALIASES: dict[tuple[str, str], tuple[str, ...]] = {
    ("train", "A"): ("trainA", "summer", "train_summer"),
    ("train", "B"): ("trainB", "winter", "train_winter"),
    ("test", "A"): ("testA", "test_summer"),
    ("test", "B"): ("testB", "test_winter"),
}


def list_images(directory: Path) -> list[Path]:
    """Return every image under ``directory``, recursively, sorted by path."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No images found under {directory}")
    return files


def resolve_split_dirs(root: str | Path, split: str) -> tuple[Path, Path]:
    """Locate the (domain A, domain B) directories for ``split``.

    Raises:
        FileNotFoundError: If either domain directory is missing, with the
            names that were tried -- the most common setup mistake is pointing
            ``--data-root`` at the zip's parent rather than the dataset folder.
    """
    root = Path(root)
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    resolved: list[Path] = []
    for domain in ("A", "B"):
        names = _SPLIT_ALIASES[(split, domain)]
        match = next((root / n for n in names if (root / n).is_dir()), None)
        if match is None:
            raise FileNotFoundError(
                f"Could not find the {split} domain-{domain} directory under {root}. "
                f"Tried: {', '.join(names)}. Run scripts/prepare_data.py first."
            )
        resolved.append(match)
    return resolved[0], resolved[1]


def build_transform(load_size: int, crop_size: int, train: bool) -> transforms.Compose:
    """Preprocessing pipeline.

    Training jitters (upscale, random crop, random flip) for augmentation;
    evaluation resizes deterministically so results are comparable across runs.

    Both normalise to [-1, 1] to match the generator's ``tanh`` output. The
    earlier version of this project left images in [0, 1] while the generator
    emitted [-1, 1], so half the value range was unreachable.
    """
    steps: list[torch.nn.Module] = []
    if train:
        steps += [
            transforms.Resize(load_size, transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        steps += [transforms.Resize((crop_size, crop_size), transforms.InterpolationMode.BICUBIC)]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(steps)


class UnpairedImageDataset(Dataset):
    """Pairs images from two unaligned domains.

    Length is the size of the larger domain, so every image in it is seen once
    per epoch. The partner index is drawn at random during training (the pairing
    carries no meaning and re-randomising avoids a fixed spurious correlation)
    and taken modulo during evaluation for repeatability.
    """

    def __init__(
        self,
        dir_a: str | Path,
        dir_b: str | Path,
        load_size: int = 143,
        crop_size: int = 128,
        train: bool = True,
        max_images: int | None = None,
        seed: int | None = None,
    ):
        self.files_a = list_images(Path(dir_a))
        self.files_b = list_images(Path(dir_b))
        if max_images is not None:
            self.files_a = self.files_a[:max_images]
            self.files_b = self.files_b[:max_images]
        self.transform = build_transform(load_size, crop_size, train)
        self.train = train
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return max(len(self.files_a), len(self.files_b))

    def _load(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            # A handful of images in the dataset are greyscale or CMYK.
            return self.transform(img.convert("RGB"))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path_a = self.files_a[index % len(self.files_a)]
        if self.train:
            path_b = self.files_b[self._rng.randrange(len(self.files_b))]
        else:
            path_b = self.files_b[index % len(self.files_b)]
        return {
            "A": self._load(path_a),
            "B": self._load(path_b),
            "path_A": str(path_a),
            "path_B": str(path_b),
        }


def build_dataloaders(
    data_root: str | Path,
    load_size: int = 143,
    crop_size: int = 128,
    batch_size: int = 4,
    num_workers: int = 0,
    max_train_images: int | None = None,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create the train and test loaders for a dataset root.

    Returns:
        ``(train_loader, test_loader)``. The test loader uses batch size 1 and
        no shuffling so sample grids stay comparable between epochs.
    """
    train_a, train_b = resolve_split_dirs(data_root, "train")
    test_a, test_b = resolve_split_dirs(data_root, "test")

    train_set = UnpairedImageDataset(
        train_a, train_b, load_size, crop_size, train=True, max_images=max_train_images, seed=seed
    )
    test_set = UnpairedImageDataset(test_a, test_b, load_size, crop_size, train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
