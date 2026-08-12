"""Populate the demo's sample gallery and the README's before/after strip.

    python scripts/make_demo_assets.py --checkpoint runs/yosemite_128/checkpoints/latest.pt

Copies a handful of held-out test photos into ``docs/samples/`` (with a
manifest the demo page reads), and renders ``assets/results_*.jpg`` strips of
real -> translated -> reconstructed for the README. Keeping this a script means
the published images always come from a specific checkpoint rather than being
hand-picked and stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyclegan.data import list_images, resolve_split_dirs  # noqa: E402
from cyclegan.inference import load_generator, translate_image  # noqa: E402


def build_strip(images: list[Image.Image], gap: int = 6, background: tuple = (18, 22, 28)) -> Image.Image:
    """Lay images out in a row with a small gutter."""
    height = max(i.height for i in images)
    width = sum(i.width for i in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), background)
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--count", type=int, default=6, help="Sample photos per domain")
    parser.add_argument("--size", type=int, default=None, help="Defaults to the training crop size")
    parser.add_argument("--samples-dir", type=Path, default=REPO_ROOT / "docs" / "samples")
    parser.add_argument("--assets-dir", type=Path, default=REPO_ROOT / "assets")
    args = parser.parse_args()

    g_ab, cfg = load_generator(args.checkpoint, "summer2winter", device="cpu")
    g_ba, _ = load_generator(args.checkpoint, "winter2summer", device="cpu")
    size = args.size or int(cfg.get("crop_size", 128))
    data_root = args.data_root or cfg.get("data_root", "data/summer2winter_yosemite")

    test_a, test_b = resolve_split_dirs(data_root, "test")
    picks = {
        "summer": list_images(test_a)[: args.count],
        "winter": list_images(test_b)[: args.count],
    }

    # -- gallery for the demo page -----------------------------------------
    args.samples_dir.mkdir(parents=True, exist_ok=True)
    for old in args.samples_dir.glob("*.jpg"):
        old.unlink()
    names: list[str] = []
    for domain, files in picks.items():
        for i, source in enumerate(files):
            name = f"{domain}_{i:02d}.jpg"
            with Image.open(source) as image:
                # Downscale: these ship in the repo and are only ever used as
                # model input at `size` px anyway.
                image.convert("RGB").resize((size * 2, size * 2), Image.LANCZOS).save(
                    args.samples_dir / name, quality=88
                )
            names.append(name)
    (args.samples_dir / "manifest.json").write_text(
        json.dumps({"images": names}, indent=2), encoding="utf-8"
    )
    print(f"docs/samples: {len(names)} photos + manifest.json")

    # -- README strips ------------------------------------------------------
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    for domain, forward, backward in (
        ("summer2winter", g_ab, g_ba),
        ("winter2summer", g_ba, g_ab),
    ):
        source_files = picks["summer" if domain == "summer2winter" else "winter"][:3]
        rows = []
        for path in source_files:
            with Image.open(path) as image:
                original = image.convert("RGB").resize((size, size), Image.BICUBIC)
            translated = translate_image(forward, original, size=size, keep_original_size=False)
            reconstructed = translate_image(backward, translated, size=size, keep_original_size=False)
            rows.append(build_strip([original, translated, reconstructed]))

        strip = Image.new(
            "RGB",
            (rows[0].width, sum(r.height for r in rows) + 6 * (len(rows) - 1)),
            (18, 22, 28),
        )
        y = 0
        for row in rows:
            strip.paste(row, (0, y))
            y += row.height + 6
        destination = args.assets_dir / f"results_{domain}.jpg"
        strip.save(destination, quality=92)
        print(f"{destination}: original | translated | reconstructed")

    print("\nReview the images, then commit docs/samples/ and assets/ if they look right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
