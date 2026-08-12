"""Shared fixtures. A synthetic dataset stands in for the real 200MB download."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _write_noise_image(path: Path, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """A canonical-layout dataset of tiny noise images.

    Domain sizes are deliberately unequal (6 vs 4) so the unpaired dataset's
    length and index-wrapping logic get exercised.
    """
    root = tmp_path / "summer2winter_yosemite"
    for split, count in (("trainA", 6), ("trainB", 4), ("testA", 3), ("testB", 2)):
        for i in range(count):
            _write_noise_image(root / split / f"{i:03d}.jpg", size=80, seed=hash(split) % 1000 + i)
    return root
