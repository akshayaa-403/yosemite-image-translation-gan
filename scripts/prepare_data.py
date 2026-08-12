"""Fetch and verify the summer2winter_yosemite dataset.

Three ways in, in order of convenience:

    # 1. Kaggle API (needs ~/.kaggle/kaggle.json)
    python scripts/prepare_data.py --source kaggle

    # 2. A zip you downloaded by hand from
    #    https://www.kaggle.com/datasets/balraj98/summer2winter-yosemite
    python scripts/prepare_data.py --source zip --archive ~/Downloads/archive.zip

    # 3. Already extracted somewhere
    python scripts/prepare_data.py --source local --archive /path/to/extracted

All three end with the same normalised layout under ``--dest``::

    data/summer2winter_yosemite/{trainA,trainB,testA,testB}/

and print per-split image counts so a truncated download is obvious.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyclegan.data import IMAGE_SUFFIXES, resolve_split_dirs  # noqa: E402

KAGGLE_DATASET = "balraj98/summer2winter-yosemite"

# Counts in the canonical CycleGAN release. A mismatch is a warning, not an
# error -- mirrors occasionally differ by a file or two.
EXPECTED_COUNTS = {"trainA": 1231, "trainB": 962, "testA": 309, "testB": 238}

CANONICAL = ("trainA", "trainB", "testA", "testB")


def count_images(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def download_from_kaggle(work_dir: Path) -> Path:
    """Download and unzip the dataset with the Kaggle CLI."""
    work_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("kaggle") is None:
        raise SystemExit(
            "The 'kaggle' CLI is not on PATH. Install it with 'pip install kaggle', put your "
            "API token in ~/.kaggle/kaggle.json, or use --source zip with a manual download."
        )
    print(f"Downloading {KAGGLE_DATASET} into {work_dir} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(work_dir), "--unzip"],
        check=True,
    )
    return work_dir


def extract_zip(archive: Path, work_dir: Path) -> Path:
    """Extract a downloaded archive into a scratch directory."""
    if not archive.is_file():
        raise SystemExit(f"No such archive: {archive}")
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive.name} ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(work_dir)
    return work_dir


def find_dataset_root(search_root: Path) -> Path:
    """Find the directory that actually holds the split folders.

    Archives are inconsistent about nesting: the splits may sit at the top
    level or one or two directories down.
    """
    candidates = [search_root, *(p for p in search_root.rglob("*") if p.is_dir())]
    for candidate in candidates:
        names = {p.name for p in candidate.iterdir() if p.is_dir()}
        if {"trainA", "trainB"} <= names or {"summer", "winter"} <= names:
            return candidate
    raise SystemExit(
        f"Could not find trainA/trainB (or summer/winter) anywhere under {search_root}. "
        "Check that the download completed."
    )


def normalize_layout(source_root: Path, dest: Path, move: bool) -> None:
    """Copy or move each split into the canonical ``trainA/.../testB`` names."""
    dest.mkdir(parents=True, exist_ok=True)
    train_a, train_b = resolve_split_dirs(source_root, "train")
    test_a, test_b = resolve_split_dirs(source_root, "test")

    for name, src in zip(CANONICAL, (train_a, train_b, test_a, test_b), strict=True):
        target = dest / name
        if target.resolve() == src.resolve():
            continue
        if target.exists():
            print(f"  {name}: already present, leaving it alone")
            continue
        print(f"  {name}: {'moving' if move else 'copying'} {count_images(src)} images")
        if move:
            shutil.move(str(src), str(target))
        else:
            shutil.copytree(src, target)


def verify(dest: Path) -> bool:
    """Print counts per split and flag anything unexpected."""
    print(f"\nDataset at {dest}")
    ok = True
    for name in CANONICAL:
        directory = dest / name
        if not directory.is_dir():
            print(f"  {name:<7} MISSING")
            ok = False
            continue
        n = count_images(directory)
        expected = EXPECTED_COUNTS[name]
        flag = "" if n == expected else f"  (expected {expected})"
        if n == 0:
            ok = False
        print(f"  {name:<7} {n:>5} images{flag}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=("kaggle", "zip", "local"), default="kaggle")
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Zip file (--source zip) or already-extracted directory (--source local)",
    )
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "data" / "summer2winter_yosemite")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move instead of copy when normalising (saves ~200MB of disk)",
    )
    parser.add_argument("--verify-only", action="store_true", help="Just report counts under --dest")
    args = parser.parse_args()

    if args.verify_only:
        return 0 if verify(args.dest) else 1

    scratch = args.dest.parent / "_raw"
    if args.source == "kaggle":
        source_root = find_dataset_root(download_from_kaggle(scratch))
    elif args.source == "zip":
        if args.archive is None:
            parser.error("--source zip requires --archive path/to/archive.zip")
        source_root = find_dataset_root(extract_zip(args.archive, scratch))
    else:
        if args.archive is None:
            parser.error("--source local requires --archive path/to/extracted")
        source_root = find_dataset_root(args.archive)

    print(f"Found dataset root: {source_root}")
    normalize_layout(source_root, args.dest, move=args.move)

    if not verify(args.dest):
        print("\nSomething is missing -- see above.")
        return 1
    if scratch.is_dir() and not any(scratch.iterdir()):
        scratch.rmdir()
    print("\nReady. Train with:\n  python scripts/train.py --config configs/yosemite_128.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
