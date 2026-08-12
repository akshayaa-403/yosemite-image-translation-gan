"""Export trained generators to ONNX for the browser demo.

    python scripts/export_onnx.py --checkpoint runs/default/checkpoints/latest.pt

Writes ``docs/models/summer2winter.onnx``, ``docs/models/winter2summer.onnx``
and a ``manifest.json`` the demo page reads to discover what is available.

Size matters here: the models are served to a browser and committed to the
repo, and GitHub refuses files over 100MB. A 64-channel / 6-block generator is
about 7.8M parameters ~= 31MB in float32 per direction. ``--quantize`` runs
ONNX Runtime's dynamic int8 quantisation, which cuts that to roughly 8MB at
some cost in output fidelity; check the exported result before shipping it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cyclegan.inference import DIRECTIONS, load_generator  # noqa: E402

OPSET = 17

# torch.onnx prints status lines containing emoji. On a Windows console still
# defaulting to cp1252 that raises UnicodeEncodeError mid-export, so widen the
# stream before any exporting happens.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def export_direction(checkpoint: Path, direction: str, destination: Path, size: int) -> dict:
    """Export one generator and verify the ONNX graph reproduces PyTorch output."""
    model, cfg = load_generator(checkpoint, direction, device="cpu")
    example = torch.randn(1, 3, size, size)

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(destination),
        input_names=["image"],
        output_names=["translated"],
        opset_version=OPSET,
        do_constant_folding=True,
        # The TorchScript path emits the InstanceNorm/ConvTranspose graph that
        # onnxruntime-web's WASM backend handles reliably; the newer dynamo
        # exporter produces ops some browser builds still lack kernels for.
        dynamo=False,
        verbose=False,
    )

    info = {
        "direction": direction,
        "file": destination.name,
        "input_size": size,
        "opset": OPSET,
        "megabytes": round(destination.stat().st_size / 1e6, 1),
        "n_res_blocks": int(cfg.get("n_res_blocks", 6)),
        "base_channels": int(cfg.get("g_base_channels", 64)),
        "trained_epochs": int(cfg.get("epochs", 0)),
    }

    max_diff = verify(destination, model, example)
    if max_diff is not None:
        info["max_abs_error_vs_pytorch"] = round(max_diff, 6)
    return info


def verify(onnx_path: Path, model: torch.nn.Module, example: torch.Tensor) -> float | None:
    """Compare ONNX Runtime output against PyTorch. Returns the max abs diff."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [warn] onnxruntime not installed; skipping verification")
        return None

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"image": example.numpy()})[0]
    with torch.no_grad():
        torch_out = model(example).numpy()
    diff = float(np.abs(onnx_out - torch_out).max())
    status = "ok" if diff < 1e-3 else "SUSPICIOUS"
    print(f"  verified against PyTorch: max abs diff {diff:.2e} ({status})")
    return diff


def quantize(path: Path) -> Path:
    """Dynamic int8 quantisation, replacing the file in place."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("  [warn] onnxruntime not installed; skipping quantisation")
        return path

    before = path.stat().st_size / 1e6
    temp = path.with_suffix(".int8.onnx")
    quantize_dynamic(str(path), str(temp), weight_type=QuantType.QInt8)
    temp.replace(path)
    print(f"  quantised: {before:.1f}MB -> {path.stat().st_size / 1e6:.1f}MB")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs" / "models")
    parser.add_argument("--size", type=int, default=None, help="Defaults to the training crop size")
    parser.add_argument("--quantize", action="store_true", help="Shrink to int8 (~4x smaller)")
    parser.add_argument(
        "--direction",
        choices=[*sorted(DIRECTIONS), "both"],
        default="both",
    )
    args = parser.parse_args()

    _, cfg = load_generator(args.checkpoint, "summer2winter", device="cpu")
    size = args.size or int(cfg.get("crop_size", 128))
    directions = sorted(DIRECTIONS) if args.direction == "both" else [args.direction]

    entries = []
    for direction in directions:
        destination = args.output_dir / f"{direction}.onnx"
        print(f"Exporting {direction} at {size}px -> {destination}")
        info = export_direction(args.checkpoint, direction, destination, size)
        if args.quantize:
            quantize(destination)
            info["megabytes"] = round(destination.stat().st_size / 1e6, 1)
            info["quantized"] = "int8-dynamic"
        entries.append(info)

    manifest = args.output_dir / "manifest.json"
    manifest.write_text(
        json.dumps({"input_size": size, "models": entries}, indent=2), encoding="utf-8"
    )
    print(f"\nManifest: {manifest}")
    total = sum(e["megabytes"] for e in entries)
    print(f"Total payload the demo will download: {total:.1f}MB")
    if total > 90:
        print("[warn] That is close to GitHub's 100MB per-file limit -- consider --quantize.")
    print("Preview the demo locally with:\n  python -m http.server -d docs 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
