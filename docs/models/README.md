# Published models

The demo page loads whatever `manifest.json` in this folder describes. Generate
both files with:

```bash
python scripts/export_onnx.py --checkpoint runs/yosemite_128/checkpoints/latest.pt
```

That writes `summer2winter.onnx`, `winter2summer.onnx` and `manifest.json`.

Size is the thing to watch: a 64-channel / 6-block generator is ~31MB per
direction in float32, so both together are a ~62MB download for every visitor.
Options, in order of preference:

1. Train at `g_base_channels: 32` for a demo-sized model (~8MB each).
2. Export with `--quantize` for dynamic int8 (~4x smaller, some quality loss).
3. Export a single direction with `--direction summer2winter`.

GitHub rejects individual files over 100MB and warns above 50MB. If you need
large weights, enable Git LFS for `*.onnx` — the Pages workflow already checks
out LFS objects.
