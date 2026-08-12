"""Loss terms for CycleGAN.

The total generator objective is

    L_G = L_gan(G_AB) + L_gan(G_BA)
        + lambda_cycle * (||G_BA(G_AB(a)) - a||_1 + ||G_AB(G_BA(b)) - b||_1)
        + lambda_identity * lambda_cycle * (||G_BA(a) - a||_1 + ||G_AB(b) - b||_1)

and each discriminator minimises ``0.5 * (L_real + L_fake)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gan_loss_real(scores: torch.Tensor) -> torch.Tensor:
    """Least-squares GAN loss pushing patch scores towards 1 (real)."""
    return F.mse_loss(scores, torch.ones_like(scores))


def gan_loss_fake(scores: torch.Tensor) -> torch.Tensor:
    """Least-squares GAN loss pushing patch scores towards 0 (fake)."""
    return F.mse_loss(scores, torch.zeros_like(scores))


def cycle_loss(real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """L1 distance between an image and its round-trip reconstruction.

    This is the term that makes unpaired translation work at all: without it
    nothing ties the output back to the specific input image.
    """
    return F.l1_loss(reconstructed, real)


def identity_loss(real: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
    """L1 penalty for changing an image that is already in the target domain.

    Optional in the paper, but it noticeably preserves the colour composition
    of the Yosemite scenes -- without it the generators tend to tint the sky.
    """
    return F.l1_loss(same, real)


def discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Combined discriminator objective, halved as in the reference implementation.

    The 0.5 factor slows the discriminator relative to the generator, which is
    what keeps the pair from collapsing early in training.
    """
    return 0.5 * (gan_loss_real(real_scores) + gan_loss_fake(fake_scores))
