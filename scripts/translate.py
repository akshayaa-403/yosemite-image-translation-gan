"""Translate images with a trained checkpoint.

    # one image, summer -> winter
    python scripts/translate.py --checkpoint runs/default/checkpoints/latest.pt \
        --input photo.jpg --output winter.jpg

    # a whole folder, the other direction
    python scripts/translate.py --checkpoint runs/.../latest.pt \
        --input data/summer2winter_yosemite/testB --output out/ --direction winter2summer

    # side-by-side comparison images
    python scripts/translate.py --checkpoint runs/.../latest.pt --input testA --output out/ --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyclegan.data import IMAGE_SUFFIXES  # noqa: E402
from cyclegan.inference import DIRECTIONS, load_generator, translate_image  # noqa: E402


def side_by_side(before: Image.Image, after: Image.Image) -> Image.Image:
    """Glue two images horizontally, matching heights."""
    if after.size != before.size:
        after = after.resize(before.size, Image.BICUBIC)
    canvas = Image.new("RGB", (before.width * 2, before.height))
    canvas.paste(before, (0, 0))
    canvas.paste(after, (before.width, 0))
    return canvas


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise SystemExit(f"No images under {path}")
        return files
    raise SystemExit(f"No such file or directory: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--direction", choices=sorted(DIRECTIONS), default="summer2winter")
    parser.add_argument("--size", type=int, default=None, help="Defaults to the training crop size")
    parser.add_argument(
        "--keep-size",
        action="store_true",
        help="Resize the result back to the input's resolution",
    )
    parser.add_argument("--compare", action="store_true", help="Write before|after pairs")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model, cfg = load_generator(args.checkpoint, args.direction, args.device)
    size = args.size or int(cfg.get("crop_size", 128))
    inputs = collect_inputs(args.input)

    # A single input may write to a file path; multiple inputs need a directory.
    to_directory = len(inputs) > 1 or args.output.is_dir() or not args.output.suffix
    if to_directory:
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"{DIRECTIONS[args.direction][1]} | {len(inputs)} image(s) at {size}px")
    for path in inputs:
        with Image.open(path) as image:
            image = image.convert("RGB")
            result = translate_image(model, image, size=size, keep_original_size=args.keep_size)
            if args.compare:
                result = side_by_side(image, result)
        destination = args.output / f"{path.stem}_{args.direction}.jpg" if to_directory else args.output
        result.save(destination, quality=95)
        print(f"  {path.name} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
