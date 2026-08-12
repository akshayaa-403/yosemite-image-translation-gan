"""ResNet generator used by both translation directions.

Follows the architecture from the CycleGAN paper (Zhu et al., 2017): a c7s1-64
stem, two stride-2 downsampling blocks, ``n_res_blocks`` residual blocks at
256 channels, two fractionally strided upsampling blocks and a c7s1-3 head.

Two details matter for image quality and are easy to get wrong:

* **Instance normalisation, not batch norm.** CycleGAN trains with batch sizes
  of 1-4, where batch statistics are noisy and leak information between images.
* **Reflection padding on the 7x7 layers.** Zero padding at the border makes the
  generator paint a dark frame around its output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection, at constant resolution."""

    def __init__(self, channels: int, norm_layer: type[nn.Module] = nn.InstanceNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            norm_layer(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No activation after the addition; that is what the paper does.
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """Translate a 3-channel image into a 3-channel image of the same size.

    Args:
        in_channels: Channels of the input image.
        out_channels: Channels of the produced image.
        base_channels: Width of the stem. Doubled at each downsampling step.
        n_res_blocks: 6 is enough for 128px inputs, 9 is standard for 256px.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        n_res_blocks: int = 9,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
    ):
        super().__init__()
        if n_res_blocks < 1:
            raise ValueError("n_res_blocks must be >= 1")

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7, bias=False),
            norm_layer(base_channels),
            nn.ReLU(inplace=True),
        ]

        # Downsample twice: 64 -> 128 -> 256 channels, resolution / 4.
        channels = base_channels
        for _ in range(2):
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(channels * 2),
                nn.ReLU(inplace=True),
            ]
            channels *= 2

        layers += [ResidualBlock(channels, norm_layer) for _ in range(n_res_blocks)]

        # Upsample back to the input resolution.
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    channels,
                    channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                norm_layer(channels // 2),
                nn.ReLU(inplace=True),
            ]
            channels //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_channels, kernel_size=7),
            nn.Tanh(),  # outputs live in [-1, 1], matching the input normalisation
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
