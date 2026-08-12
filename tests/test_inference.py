"""Loading a checkpoint back into a generator and translating an image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from tests.test_trainer import make_trainer

from cyclegan.inference import load_generator, postprocess, preprocess, translate_image


@pytest.fixture
def checkpoint(fake_dataset: Path, tmp_path: Path) -> Path:
    trainer = make_trainer(fake_dataset, tmp_path)
    path = Path(trainer.cfg.output_dir) / "checkpoints" / "test.pt"
    trainer.save(path, epoch=1)
    return path


def test_loads_both_directions_in_eval_mode(checkpoint: Path) -> None:
    for direction in ("summer2winter", "winter2summer"):
        model, config = load_generator(checkpoint, direction, device="cpu")
        assert not model.training
        assert config["n_res_blocks"] == 1
        assert all(not p.requires_grad for p in model.parameters())


def test_directions_load_different_weights(checkpoint: Path) -> None:
    g_ab, _ = load_generator(checkpoint, "summer2winter", device="cpu")
    g_ba, _ = load_generator(checkpoint, "winter2summer", device="cpu")
    a = g_ab.model[1].weight
    b = g_ba.model[1].weight
    assert not torch.allclose(a, b), "the two generators are trained separately"


def test_rejects_unknown_direction(checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="direction"):
        load_generator(checkpoint, "winter2spring", device="cpu")


def test_missing_checkpoint_raises_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_generator(tmp_path / "nope.pt", device="cpu")


def test_preprocess_produces_a_normalized_batch() -> None:
    image = Image.fromarray(np.zeros((30, 50, 3), dtype=np.uint8))
    tensor = preprocess(image, size=64)
    assert tensor.shape == (1, 3, 64, 64)
    torch.testing.assert_close(tensor.min(), torch.tensor(-1.0))


def test_postprocess_round_trips_to_pil() -> None:
    image = postprocess(torch.zeros(1, 3, 16, 16), size=(40, 20))
    assert isinstance(image, Image.Image)
    assert image.size == (40, 20)


def test_translate_keeps_the_original_resolution(checkpoint: Path) -> None:
    model, _ = load_generator(checkpoint, device="cpu")
    source = Image.fromarray(np.random.default_rng(0).integers(0, 256, (37, 53, 3), dtype=np.uint8))
    result = translate_image(model, source, size=64, keep_original_size=True)
    assert result.size == source.size
    assert result.mode == "RGB"


def test_translate_can_return_the_model_resolution(checkpoint: Path) -> None:
    model, _ = load_generator(checkpoint, device="cpu")
    source = Image.fromarray(np.zeros((37, 53, 3), dtype=np.uint8))
    assert translate_image(model, source, size=64, keep_original_size=False).size == (64, 64)


def test_translate_handles_grayscale_input(checkpoint: Path) -> None:
    """A few images in the real dataset are single-channel."""
    model, _ = load_generator(checkpoint, device="cpu")
    grayscale = Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L")
    assert translate_image(model, grayscale, size=64).mode == "RGB"
