"""Seeding, device selection, checkpoint I/O and image helpers."""

from __future__ import annotations

import csv
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch so a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a device spec into a concrete device, falling back to CPU."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[warn] {spec} requested but CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return device


def init_weights(module: nn.Module, gain: float = 0.02) -> None:
    """Normal(0, gain) initialisation, as used in the reference implementation.

    Apply with ``model.apply(init_weights)``. GANs are sensitive to this: the
    PyTorch default (Kaiming) makes the discriminator win early and starves the
    generator of gradient.
    """
    name = module.__class__.__name__
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, gain)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "Norm2d" in name and getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Map a [-1, 1] image tensor back to [0, 1] for display or saving."""
    return (tensor.clamp(-1, 1) + 1) / 2


def save_image_grid(images: Sequence[torch.Tensor], path: str | Path, nrow: int | None = None) -> None:
    """Save a row-per-tensor grid, e.g. ``[real_A, fake_B, rec_A]``.

    Args:
        images: Same-shaped batches. Each becomes one row of the grid.
        path: Destination PNG.
        nrow: Images per row. Defaults to the batch size, giving one row each.
    """
    if not images:
        raise ValueError("images must not be empty")
    nrow = nrow or images[0].size(0)
    stacked = torch.cat([denormalize(i.detach().cpu().float()) for i in images], dim=0)
    grid = make_grid(stacked, nrow=nrow, padding=2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(path))


def save_checkpoint(path: str | Path, **state: Any) -> None:
    """Write a checkpoint atomically, so an interrupt cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}")
    # weights_only=True refuses to unpickle arbitrary objects; our checkpoints
    # only hold tensors and plain Python values.
    return torch.load(path, map_location=map_location, weights_only=True)


class CSVLogger:
    """Append-only CSV of metrics, one row per logged step.

    Writes the header on first use and keeps the file flushed, so a run killed
    mid-training still leaves a usable loss history to plot.
    """

    def __init__(self, path: str | Path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        if write_header:
            self._writer.writeheader()
            self._handle.flush()

    def log(self, row: dict[str, Any]) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()
