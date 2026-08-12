"""Replay buffer of previously generated images.

Each discriminator is shown a mix of the generator's newest output and output
it produced some updates ago. This is the "history of generated images" trick
from Shrivastava et al. (2017) that the CycleGAN authors adopted: it stops the
discriminator from chasing the generator's most recent quirk and visibly
reduces oscillation in the loss curves.
"""

from __future__ import annotations

import random

import torch


class ImageBuffer:
    """Args:
        capacity: Number of images to retain. 0 disables buffering entirely.
        seed: Optional seed for the internal RNG, so runs stay reproducible.
    """

    def __init__(self, capacity: int = 50, seed: int | None = None):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = capacity
        self._images: list[torch.Tensor] = []
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._images)

    def push_and_sample(self, images: torch.Tensor) -> torch.Tensor:
        """Store the batch and return a batch of the same shape to train D on.

        For each image: while the buffer has room, keep it and return it. Once
        full, return a stored image half the time (swapping the new one in) and
        the fresh image otherwise.
        """
        if self.capacity == 0:
            return images

        out: list[torch.Tensor] = []
        for image in images:
            # detach(): the buffer feeds the discriminator only, so gradients
            # must not flow back into the generator through these samples.
            image = image.detach().unsqueeze(0)
            if len(self._images) < self.capacity:
                self._images.append(image)
                out.append(image)
            elif self._rng.random() > 0.5:
                idx = self._rng.randrange(self.capacity)
                out.append(self._images[idx].clone())
                self._images[idx] = image
            else:
                out.append(image)
        return torch.cat(out, dim=0)
