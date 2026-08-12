"""Loss terms and the replay buffer."""

from __future__ import annotations

import torch

from cyclegan.buffer import ImageBuffer
from cyclegan.losses import (
    cycle_loss,
    discriminator_loss,
    gan_loss_fake,
    gan_loss_real,
    identity_loss,
)


def test_gan_losses_are_zero_at_their_targets() -> None:
    assert gan_loss_real(torch.ones(2, 1, 4, 4)).item() == 0.0
    assert gan_loss_fake(torch.zeros(2, 1, 4, 4)).item() == 0.0


def test_gan_losses_are_symmetric_opposites() -> None:
    scores = torch.zeros(1, 1, 2, 2)
    assert gan_loss_real(scores).item() == 1.0
    assert gan_loss_fake(torch.ones(1, 1, 2, 2)).item() == 1.0


def test_cycle_and_identity_losses_vanish_for_perfect_reconstruction() -> None:
    x = torch.randn(2, 3, 8, 8)
    assert cycle_loss(x, x.clone()).item() == 0.0
    assert identity_loss(x, x.clone()).item() == 0.0


def test_cycle_loss_is_mean_absolute_error() -> None:
    a = torch.zeros(1, 3, 4, 4)
    b = torch.full((1, 3, 4, 4), 0.25)
    torch.testing.assert_close(cycle_loss(a, b), torch.tensor(0.25))


def test_discriminator_loss_is_halved_sum() -> None:
    """The 0.5 factor is what keeps D from outrunning G."""
    real = torch.zeros(1, 1, 2, 2)  # loss_real = 1
    fake = torch.ones(1, 1, 2, 2)   # loss_fake = 1
    torch.testing.assert_close(discriminator_loss(real, fake), torch.tensor(1.0))


def test_losses_are_differentiable() -> None:
    scores = torch.zeros(1, 1, 2, 2, requires_grad=True)
    gan_loss_real(scores).backward()
    assert scores.grad is not None and scores.grad.abs().sum() > 0


def test_buffer_returns_new_images_until_full() -> None:
    buffer = ImageBuffer(capacity=4, seed=0)
    batch = torch.randn(4, 3, 8, 8)
    torch.testing.assert_close(buffer.push_and_sample(batch), batch)
    assert len(buffer) == 4


def test_buffer_preserves_shape_when_full() -> None:
    buffer = ImageBuffer(capacity=2, seed=0)
    buffer.push_and_sample(torch.randn(2, 3, 8, 8))
    out = buffer.push_and_sample(torch.randn(2, 3, 8, 8))
    assert out.shape == (2, 3, 8, 8)
    assert len(buffer) == 2


def test_buffer_eventually_replays_old_images() -> None:
    buffer = ImageBuffer(capacity=2, seed=1)
    first = torch.zeros(2, 3, 4, 4)
    buffer.push_and_sample(first)
    replayed = any(
        buffer.push_and_sample(torch.ones(2, 3, 4, 4)).eq(0).any().item() for _ in range(20)
    )
    assert replayed, "a full buffer should sometimes hand back a stored image"


def test_zero_capacity_buffer_is_a_passthrough() -> None:
    buffer = ImageBuffer(capacity=0)
    batch = torch.randn(3, 3, 4, 4)
    torch.testing.assert_close(buffer.push_and_sample(batch), batch)


def test_buffer_detaches_samples_from_the_graph() -> None:
    """Buffered fakes feed D only; gradients must not flow back into G."""
    buffer = ImageBuffer(capacity=4, seed=0)
    source = torch.randn(2, 3, 4, 4, requires_grad=True)
    out = buffer.push_and_sample(source * 2)
    assert not out.requires_grad
