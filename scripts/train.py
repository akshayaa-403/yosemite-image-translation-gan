"""Train CycleGAN on summer2winter_yosemite.

    python scripts/train.py --config configs/yosemite_128.yaml
    python scripts/train.py --config configs/yosemite_128.yaml --epochs 50 --batch-size 2
    python scripts/train.py --config configs/smoke.yaml          # 2-epoch sanity run

Any config field can be overridden with a matching ``--flag``; the resolved
config is written to ``<output_dir>/config.yaml``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyclegan.config import Config  # noqa: E402
from cyclegan.data import build_dataloaders  # noqa: E402
from cyclegan.trainer import CycleGANTrainer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    Config.add_cli_arguments(parser)
    cfg = Config.from_cli(parser.parse_args())

    train_loader, test_loader = build_dataloaders(
        data_root=cfg.data_root,
        load_size=cfg.load_size,
        crop_size=cfg.crop_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_train_images=cfg.max_train_images,
        seed=cfg.seed,
    )

    trainer = CycleGANTrainer(cfg, train_loader, test_loader)
    trainer.train()

    metrics = trainer.evaluate_cycle_consistency(max_batches=50)
    if metrics:
        print("Cycle consistency on held-out data:")
        for key, value in sorted(metrics.items()):
            print(f"  {key:<8} {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
