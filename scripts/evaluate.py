"""Evaluate a checkpoint on the held-out test split.

    python scripts/evaluate.py --checkpoint runs/default/checkpoints/latest.pt
    python scripts/evaluate.py --checkpoint runs/.../latest.pt --fid   # needs torchmetrics

Reports cycle-reconstruction L1/PSNR/SSIM in both directions, and optionally
FID between translations and real images of the target domain. Writes a sample
grid and a ``metrics.json`` beside the checkpoint's run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from torch.utils.data import DataLoader  # noqa: E402

from cyclegan.data import UnpairedImageDataset, resolve_split_dirs  # noqa: E402
from cyclegan.inference import load_generator  # noqa: E402
from cyclegan.metrics import FIDAccumulator, cycle_metrics  # noqa: E402
from cyclegan.utils import resolve_device, save_image_grid  # noqa: E402


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=str, default=None, help="Defaults to the training value")
    parser.add_argument("--crop-size", type=int, default=None, help="Defaults to the training value")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None, help="Cap batches for a quick look")
    parser.add_argument("--fid", action="store_true", help="Also compute FID (needs torchmetrics)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <checkpoint>/../..")
    args = parser.parse_args()

    device = resolve_device(args.device)
    g_ab, cfg = load_generator(args.checkpoint, "summer2winter", device)
    g_ba, _ = load_generator(args.checkpoint, "winter2summer", device)

    data_root = args.data_root or cfg.get("data_root", "data/summer2winter_yosemite")
    crop_size = args.crop_size or int(cfg.get("crop_size", 128))
    test_a, test_b = resolve_split_dirs(data_root, "test")
    dataset = UnpairedImageDataset(test_a, test_b, crop_size, crop_size, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if args.fid:
        try:
            import torchmetrics.image.fid  # noqa: F401
        except ImportError:
            print("[warn] torchmetrics is not installed; skipping FID. pip install torchmetrics")
            args.fid = False
    fid_b = FIDAccumulator(device) if args.fid else None  # real winter vs. summer->winter
    fid_a = FIDAccumulator(device) if args.fid else None  # real summer vs. winter->summer

    sums: dict[str, float] = {}
    batches = 0
    first_grid: list[torch.Tensor] | None = None

    for i, batch in enumerate(loader):
        if args.max_batches is not None and i >= args.max_batches:
            break
        real_a = batch["A"].to(device)
        real_b = batch["B"].to(device)
        fake_b = g_ab(real_a)
        fake_a = g_ba(real_b)
        rec_a = g_ba(fake_b)
        rec_b = g_ab(fake_a)

        for prefix, real, rec in (("summer", real_a, rec_a), ("winter", real_b, rec_b)):
            for key, value in cycle_metrics(real, rec).items():
                name = f"cycle_{prefix}_{key}"
                sums[name] = sums.get(name, 0.0) + value
        batches += 1

        if fid_b is not None and fid_a is not None:
            fid_b.add_real(real_b)
            fid_b.add_fake(fake_b)
            fid_a.add_real(real_a)
            fid_a.add_fake(fake_a)

        if first_grid is None:
            first_grid = [
                torch.cat([real_a, fake_b, rec_a], dim=0).cpu(),
                torch.cat([real_b, fake_a, rec_b], dim=0).cpu(),
            ]

    if batches == 0:
        print("No test batches were produced -- is the data root correct?")
        return 1

    results = {k: round(v / batches, 4) for k, v in sorted(sums.items())}
    results["images_per_domain"] = batches * args.batch_size
    if fid_b is not None and fid_a is not None:
        results["fid_summer2winter"] = round(fid_b.compute(), 3)
        results["fid_winter2summer"] = round(fid_a.compute(), 3)

    output_dir = args.output_dir or args.checkpoint.parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if first_grid is not None:
        grid_path = output_dir / "eval_samples.png"
        save_image_grid(first_grid, grid_path, nrow=args.batch_size)
        print(f"Sample grid: {grid_path}")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Metrics:      {metrics_path}")
    for key, value in results.items():
        print(f"  {key:<24} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
