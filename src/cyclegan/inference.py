"""Loading trained generators and translating individual images."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .generator import ResnetGenerator
from .utils import denormalize, load_checkpoint, resolve_device

DIRECTIONS = {
    # name: (state_dict key, human-readable description)
    "summer2winter": ("g_ab", "summer -> winter"),
    "winter2summer": ("g_ba", "winter -> summer"),
}


def load_generator(
    checkpoint_path: str | Path,
    direction: str = "summer2winter",
    device: str | torch.device = "auto",
) -> tuple[ResnetGenerator, dict]:
    """Rebuild one generator from a training checkpoint.

    The architecture is taken from the config stored inside the checkpoint, so
    a checkpoint trained with different widths still loads correctly.

    Returns:
        ``(generator_in_eval_mode, stored_config)``.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}, got {direction!r}")

    device = resolve_device(device) if isinstance(device, str) else device
    state = load_checkpoint(checkpoint_path, map_location=device)
    config = state.get("config", {}) or {}

    model = ResnetGenerator(
        base_channels=int(config.get("g_base_channels", 64)),
        n_res_blocks=int(config.get("n_res_blocks", 6)),
    )
    model.load_state_dict(state[DIRECTIONS[direction][0]])
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, config


def preprocess(image: Image.Image, size: int) -> torch.Tensor:
    """PIL image -> normalised 1x3xSxS tensor in [-1, 1]."""
    pipeline = transforms.Compose(
        [
            transforms.Resize((size, size), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return pipeline(image.convert("RGB")).unsqueeze(0)


def postprocess(tensor: torch.Tensor, size: tuple[int, int] | None = None) -> Image.Image:
    """Model output -> PIL image, optionally resized back to ``size`` (w, h)."""
    array = denormalize(tensor.squeeze(0).cpu().float())
    image = transforms.functional.to_pil_image(array)
    if size is not None:
        image = image.resize(size, Image.BICUBIC)
    return image


@torch.no_grad()
def translate_image(
    model: ResnetGenerator,
    image: Image.Image,
    size: int = 128,
    keep_original_size: bool = True,
) -> Image.Image:
    """Translate a single PIL image.

    The network is fully convolutional, so it accepts any size divisible by 4;
    ``size`` still defaults to the training resolution because quality degrades
    when the input scale drifts far from what the model saw during training.
    """
    device = next(model.parameters()).device
    tensor = preprocess(image, size).to(device)
    output = model(tensor)
    return postprocess(output, image.size if keep_original_size else None)
