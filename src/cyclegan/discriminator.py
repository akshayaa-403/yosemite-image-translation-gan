"""70x70 PatchGAN discriminator.

Unlike a classifier that emits one real/fake score per image, this network
emits a grid of scores, each covering a 70x70 receptive field of the input.
That keeps the parameter count low and the judgement local, which is what
pushes the generator towards realistic *texture* (snow, foliage) rather than
globally plausible but blurry images.

The output is a feature map, not a scalar, so it works at any input size --
the previous version of this file hard-coded a final 8x8 kernel and silently
broke for anything other than 128px inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """Args:
        in_channels: Channels of the image being judged.
        base_channels: Width of the first convolution.
        n_layers: Number of stride-2 blocks. 3 gives the 70x70 receptive field.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        norm_layer: type[nn.Module] = nn.InstanceNorm2d,
    ):
        super().__init__()

        # First block has no normalisation: it must be free to react to the
        # absolute colour statistics that distinguish the two domains.
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = base_channels
        for i in range(1, n_layers):
            out = min(base_channels * 2 ** i, base_channels * 8)
            layers += [
                nn.Conv2d(channels, out, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(out),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels = out

        # Stride-1 block widens the receptive field without shrinking the map.
        out = min(base_channels * 2 ** n_layers, base_channels * 8)
        layers += [
            nn.Conv2d(channels, out, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(out),
            nn.LeakyReLU(0.2, inplace=True),
            # One logit per patch. No sigmoid: the least-squares GAN loss in
            # losses.py operates on raw scores.
            nn.Conv2d(out, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
