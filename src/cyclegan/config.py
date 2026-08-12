"""Typed training configuration, loadable from YAML and overridable from the CLI.

One dataclass is the single source of truth for hyperparameters. Every run
writes its resolved config next to the checkpoints, so a checkpoint is always
traceable to the settings that produced it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # --- data -------------------------------------------------------------
    data_root: str = "data/summer2winter_yosemite"
    load_size: int = 143
    """Size images are resized to before the random crop. ~1.12x crop_size."""
    crop_size: int = 128
    """Resolution the networks train at. Must be divisible by 4."""
    batch_size: int = 4
    num_workers: int = 4
    max_train_images: int | None = None
    """Cap images per domain. Useful for smoke tests, leave null for real runs."""

    # --- model ------------------------------------------------------------
    g_base_channels: int = 64
    d_base_channels: int = 64
    n_res_blocks: int = 6
    """6 for 128px inputs, 9 for 256px."""

    # --- optimisation -----------------------------------------------------
    epochs: int = 200
    """Total epochs. The learning rate is constant for the first half."""
    decay_start_epoch: int | None = None
    """Epoch where linear LR decay to zero begins. Defaults to epochs // 2."""
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    """Scales the identity term *relative to* lambda_cycle. 0 disables it."""
    buffer_capacity: int = 50
    amp: bool = True
    """Mixed precision. Ignored on CPU."""

    # --- run --------------------------------------------------------------
    output_dir: str = "runs/default"
    seed: int = 42
    sample_every: int = 1
    """Epochs between sample grids written to <output_dir>/samples/."""
    checkpoint_every: int = 5
    log_every: int = 50
    """Training steps between console/CSV loss rows."""
    device: str = "auto"
    """'auto', 'cpu', 'cuda', or an explicit device string like 'cuda:1'."""
    resume: str | None = None
    """Path to a checkpoint to continue from."""

    def __post_init__(self) -> None:
        if self.crop_size % 4:
            raise ValueError("crop_size must be divisible by 4 (two stride-2 stages)")
        if self.load_size < self.crop_size:
            raise ValueError("load_size must be >= crop_size")
        if self.decay_start_epoch is None:
            self.decay_start_epoch = max(1, self.epochs // 2)
        if not 0 < self.decay_start_epoch <= self.epochs:
            raise ValueError("decay_start_epoch must be in (0, epochs]")

    # -- (de)serialisation --------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=True), encoding="utf-8")

    # -- CLI ---------------------------------------------------------------
    @staticmethod
    def add_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add ``--field-name`` flags for every config field.

        Flags default to ``None`` so ``merge_cli`` can tell "not passed" from
        "passed a falsy value", and only explicit flags override the YAML.
        """
        parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
        for f in fields(Config):
            flag = "--" + f.name.replace("_", "-")
            if f.type in ("bool", bool):
                parser.add_argument(flag, dest=f.name, type=_parse_bool, default=None)
            elif "int" in str(f.type):
                parser.add_argument(flag, dest=f.name, type=_optional_int, default=None)
            elif "float" in str(f.type):
                parser.add_argument(flag, dest=f.name, type=float, default=None)
            else:
                parser.add_argument(flag, dest=f.name, type=str, default=None)
        return parser

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> Config:
        """Build a config from ``--config`` plus any explicitly passed flags."""
        base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {} if args.config else {}
        names = {f.name for f in fields(cls)}
        overrides = {k: v for k, v in vars(args).items() if k in names and v is not None}

        # ``--epochs 20`` against a 200-epoch config would otherwise inherit
        # decay_start_epoch: 100 and fail validation. The decay point is derived
        # from the run length, so shortening the run re-derives it -- unless the
        # caller pinned it explicitly on the same command line.
        if "epochs" in overrides and "decay_start_epoch" not in overrides:
            base.pop("decay_start_epoch", None)

        return cls.from_dict({**base, **overrides})


def _parse_bool(value: str) -> bool:
    if value.lower() in ("1", "true", "yes", "y", "on"):
        return True
    if value.lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def _optional_int(value: str) -> int | None:
    """Accept ``null``/``none`` so nullable int fields can be cleared from the CLI."""
    if value.lower() in ("null", "none", ""):
        return None
    return int(value)
