"""Shape and behaviour contracts for the two networks."""

from __future__ import annotations

import pytest
import torch

from cyclegan.discriminator import PatchDiscriminator
from cyclegan.generator import ResidualBlock, ResnetGenerator


@pytest.mark.parametrize("size", [64, 128, 256])
def test_generator_preserves_spatial_size(size: int) -> None:
    model = ResnetGenerator(base_channels=8, n_res_blocks=2)
    out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 3, size, size)


def test_generator_output_is_in_tanh_range() -> None:
    model = ResnetGenerator(base_channels=8, n_res_blocks=2)
    out = model(torch.randn(1, 3, 64, 64) * 5)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_generator_handles_non_square_input() -> None:
    """The architecture is fully convolutional, so rectangles must work too."""
    model = ResnetGenerator(base_channels=8, n_res_blocks=1)
    out = model(torch.randn(1, 3, 64, 96))
    assert out.shape == (1, 3, 64, 96)


def test_generator_rejects_zero_res_blocks() -> None:
    with pytest.raises(ValueError):
        ResnetGenerator(n_res_blocks=0)


def test_residual_block_is_identity_when_weights_are_zero() -> None:
    block = ResidualBlock(4)
    for param in block.parameters():
        torch.nn.init.zeros_(param)
    x = torch.randn(1, 4, 16, 16)
    torch.testing.assert_close(block(x), x)


@pytest.mark.parametrize("size,expected", [(128, 14), (256, 30), (70, 6)])
def test_discriminator_emits_a_patch_grid(size: int, expected: int) -> None:
    """A grid, not a scalar -- and it scales with the input instead of breaking."""
    model = PatchDiscriminator(base_channels=8)
    out = model(torch.randn(1, 3, size, size))
    assert out.shape == (1, 1, expected, expected)


def test_discriminator_first_block_has_no_normalization() -> None:
    """It must stay sensitive to absolute colour statistics."""
    model = PatchDiscriminator(base_channels=8)
    assert isinstance(model.model[0], torch.nn.Conv2d)
    assert isinstance(model.model[1], torch.nn.LeakyReLU)


def test_models_use_instance_norm_not_batch_norm() -> None:
    """Batch statistics are unusable at CycleGAN's batch sizes."""
    for model in (ResnetGenerator(base_channels=8, n_res_blocks=1), PatchDiscriminator(base_channels=8)):
        norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        assert norms == []
        assert any(isinstance(m, torch.nn.InstanceNorm2d) for m in model.modules())


def test_generator_gradients_reach_the_stem() -> None:
    model = ResnetGenerator(base_channels=8, n_res_blocks=2)
    model(torch.randn(1, 3, 64, 64)).mean().backward()
    first_conv = next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    assert first_conv.weight.grad is not None
    assert torch.isfinite(first_conv.weight.grad).all()
