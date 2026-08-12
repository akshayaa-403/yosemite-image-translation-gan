"""Evaluation metrics.

Unpaired translation has no ground-truth target, so there is nothing to compute
a plain PSNR against. Two things *can* be measured honestly:

* **Cycle reconstruction fidelity** (L1 / PSNR / SSIM between an image and its
  round trip). High fidelity alone does not prove good translation -- an
  identity mapping scores perfectly -- but a collapse in it is a reliable
  warning that the generators have stopped cooperating.
* **FID** between generated images and real images of the target domain, which
  is the standard proxy for translation quality. Requires ``torchmetrics``;
  ``fid_available()`` reports whether it can run.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .utils import denormalize


def l1_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean absolute error between two [-1, 1] batches, in [0, 1] units."""
    return (denormalize(a) - denormalize(b)).abs().mean().item()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB between two [-1, 1] batches."""
    x = denormalize(a).detach().cpu().numpy()
    y = denormalize(b).detach().cpu().numpy()
    return float(peak_signal_noise_ratio(y, x, data_range=1.0))


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean structural similarity over a batch of [-1, 1] images."""
    x = denormalize(a).detach().cpu().numpy()
    y = denormalize(b).detach().cpu().numpy()
    scores = [
        structural_similarity(xi, yi, channel_axis=0, data_range=1.0)
        for xi, yi in zip(x, y, strict=True)
    ]
    return float(np.mean(scores))


def cycle_metrics(real: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    """All three reconstruction metrics for one batch."""
    return {
        "l1": l1_distance(real, reconstructed),
        "psnr": psnr(real, reconstructed),
        "ssim": ssim(real, reconstructed),
    }


class FIDAccumulator:
    """Streaming FID between generated and real images of one domain.

    Feed batches as they come, then call :meth:`compute`. Inception expects
    uint8 in [0, 255], which is what :meth:`_to_uint8` produces from our
    [-1, 1] tensors.

    Note that FID is only meaningful with enough samples -- the standard is
    50k, and the Yosemite test splits hold a few hundred, so treat the number
    as a relative signal between checkpoints, not an absolute score.
    """

    def __init__(self, device: torch.device | str = "cpu", feature_dim: int = 2048):
        from torchmetrics.image.fid import FrechetInceptionDistance

        self.metric = FrechetInceptionDistance(feature=feature_dim, normalize=False).to(device)
        self.device = device
        self._n_real = 0
        self._n_fake = 0

    @staticmethod
    def _to_uint8(images: torch.Tensor) -> torch.Tensor:
        return (denormalize(images) * 255).round().to(torch.uint8)

    def add_real(self, images: torch.Tensor) -> None:
        self.metric.update(self._to_uint8(images).to(self.device), real=True)
        self._n_real += images.size(0)

    def add_fake(self, images: torch.Tensor) -> None:
        self.metric.update(self._to_uint8(images).to(self.device), real=False)
        self._n_fake += images.size(0)

    def compute(self) -> float:
        if self._n_real < 2 or self._n_fake < 2:
            raise RuntimeError("FID needs at least 2 real and 2 generated images")
        return float(self.metric.compute().item())
