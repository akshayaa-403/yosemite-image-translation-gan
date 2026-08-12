"""Config validation, YAML round-tripping and CLI overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from cyclegan.config import Config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_decay_start_defaults_to_half_the_run() -> None:
    assert Config(epochs=200).decay_start_epoch == 100


def test_rejects_crop_size_not_divisible_by_four() -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        Config(crop_size=130)


def test_rejects_load_size_smaller_than_crop() -> None:
    with pytest.raises(ValueError, match="load_size"):
        Config(load_size=64, crop_size=128)


def test_rejects_decay_start_beyond_the_run() -> None:
    with pytest.raises(ValueError, match="decay_start_epoch"):
        Config(epochs=10, decay_start_epoch=11)


def test_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="lerning_rate"):
        Config.from_dict({"lerning_rate": 0.1})


def test_yaml_round_trip(tmp_path: Path) -> None:
    original = Config(epochs=7, crop_size=64, load_size=72)
    path = tmp_path / "config.yaml"
    original.to_yaml(path)
    assert Config.from_yaml(path) == original


@pytest.mark.parametrize("name", ["yosemite_128.yaml", "yosemite_256.yaml", "smoke.yaml"])
def test_shipped_configs_load(name: str) -> None:
    config = Config.from_yaml(CONFIG_DIR / name)
    assert config.epochs > 0
    assert config.load_size >= config.crop_size


def test_cli_flags_override_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "base.yaml"
    Config(epochs=200, batch_size=4).to_yaml(yaml_path)

    parser = Config.add_cli_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--config", str(yaml_path), "--epochs", "5"])
    config = Config.from_cli(args)

    assert config.epochs == 5, "explicit flag wins"
    assert config.batch_size == 4, "unpassed flags keep the YAML value"


def test_cli_can_clear_a_nullable_int() -> None:
    parser = Config.add_cli_arguments(argparse.ArgumentParser())
    config = Config.from_cli(parser.parse_args(["--max-train-images", "12"]))
    assert config.max_train_images == 12


def test_cli_parses_booleans() -> None:
    parser = Config.add_cli_arguments(argparse.ArgumentParser())
    assert Config.from_cli(parser.parse_args(["--amp", "false"])).amp is False
    assert Config.from_cli(parser.parse_args(["--amp", "true"])).amp is True
