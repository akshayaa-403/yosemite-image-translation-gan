# Yosemite CycleGAN — summer ↔ winter

A PyTorch CycleGAN that turns summer photographs of Yosemite into winter ones and
back, trained on **unpaired** images: two piles of photos with no correspondence
between them, and nothing in the data saying what any given summer scene should
look like under snow.

**[▶ Try the demo](https://akshayaa-403.github.io/yosemite-image-translation-gan/)** —
the generator runs in your browser via ONNX Runtime Web. Nothing is uploaded.

<!-- Regenerate with: python scripts/make_demo_assets.py --checkpoint runs/.../latest.pt -->
<!-- ![Summer to winter](assets/results_summer2winter.jpg) -->
<!-- Columns: original | translated | reconstructed after the round trip -->

---

## How it works

Two generators, `G_AB` (summer→winter) and `G_BA` (winter→summer), and two
discriminators that judge whether an image looks like a real photo of their
domain. Three things keep the translation honest:

| Term | What it does | Why it matters |
|---|---|---|
| Adversarial (LSGAN) | Makes output indistinguishable from real photos of the target season | Alone, it would happily output *any* convincing winter scene |
| Cycle consistency (λ=10) | `G_BA(G_AB(a))` must return `a` | This is what ties the output to *your* image; without it there is no supervision at all |
| Identity (λ=0.5·λ_cycle) | An image already in the target domain must pass through unchanged | Stops the generators tinting the sky and shifting the palette |

Architecture details that matter in practice, and that a naive implementation
usually gets wrong:

- **ResNet generator** with reflection padding on the 7×7 layers. Zero padding
  paints a dark frame around the output.
- **Instance normalisation**, not batch norm. CycleGAN trains at batch sizes of
  1–4, where batch statistics are noise.
- **70×70 PatchGAN discriminator** emitting a grid of scores rather than one
  number, which pushes the generator towards realistic *texture* (snow, foliage)
  instead of globally plausible mush. It also works at any input resolution.
- **Replay buffer** of 50 past fakes per domain, so the discriminator is not
  chasing only the generator's newest quirk.
- **Linear LR decay to zero** over the second half of training.

## Layout

```
src/cyclegan/         generator, discriminator, losses, data, buffer, trainer, metrics
scripts/              prepare_data · train · evaluate · translate · export_onnx · make_demo_assets
configs/              yosemite_128 (default) · yosemite_256 · smoke (2-minute CPU sanity run)
notebooks/            train_and_export.ipynb — dataset to published weights on Colab
docs/                 the GitHub Pages demo (static, no backend)
tests/                64 tests, CPU-only, seconds to run
```

## Setup

```bash
git clone https://github.com/akshayaa-403/yosemite-image-translation-gan.git
cd yosemite-image-translation-gan

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or: pip install -e ".[dev,export,metrics]"
```

### Data

The [summer2winter_yosemite dataset](https://www.kaggle.com/datasets/balraj98/summer2winter-yosemite)
(1231 + 962 train, 309 + 238 test, ~200MB):

```bash
python scripts/prepare_data.py --source kaggle                       # needs ~/.kaggle/kaggle.json
python scripts/prepare_data.py --source zip --archive ~/Downloads/archive.zip
python scripts/prepare_data.py --verify-only                         # print per-split counts
```

Either way you end up with `data/summer2winter_yosemite/{trainA,trainB,testA,testB}`,
where A is summer and B is winter. The loader also accepts the
`summer/ winter/ test_summer/ test_winter/` naming some re-uploads use.

## Training

```bash
python scripts/train.py --config configs/smoke.yaml            # 2 min on CPU, proves the pipeline
python scripts/train.py --config configs/yosemite_128.yaml     # the real thing
python scripts/train.py --config configs/yosemite_128.yaml --epochs 50 --batch-size 2
```

Any config field is overridable with a matching flag (`--lambda-cycle 5`). Each
run writes to `runs/<name>/`:

```
config.yaml                  the exact resolved settings that produced this run
losses.csv                   one row per logged step, for plotting
samples/epoch_XXXX.png       held-out translations: original | translated | reconstructed
checkpoints/latest.pt        resume with --resume runs/<name>/checkpoints/latest.pt
```

Rough cost at 128px, batch 4, on one T4: ~3 min/epoch, so ~10 hours for the
default 200 epochs. Recognisable seasonal change appears well before that —
around epoch 30–40 — so check the sample grids early rather than waiting.

## Evaluating

```bash
python scripts/evaluate.py --checkpoint runs/yosemite_128/checkpoints/latest.pt --fid
```

Unpaired translation has no ground truth, so there is no single accuracy number
to report. What this measures:

- **Cycle reconstruction** L1 / PSNR / SSIM. Useful as a *failure* signal — a
  collapse means the generators have stopped cooperating. Note that a perfect
  score is also what an identity mapping would get, so it cannot be read as
  translation quality on its own.
- **FID** against real images of the target domain (needs `torchmetrics`). The
  standard proxy for translation quality, but the test splits hold a few hundred
  images against the 50k that FID assumes — treat it as a relative signal
  between your own checkpoints, not a number to compare against papers.

## Translating images

```bash
python scripts/translate.py --checkpoint runs/.../latest.pt --input photo.jpg --output winter.jpg
python scripts/translate.py --checkpoint runs/.../latest.pt --input testA/ --output out/ --compare
python scripts/translate.py --checkpoint runs/.../latest.pt --input photo.jpg --output summer.jpg \
    --direction winter2summer
```

## The browser demo

`docs/` is a static page: **upload a landscape photo, get it back translated**, with
a draggable divider between the two. The generator runs client-side through ONNX
Runtime Web (WebGPU where available, WebAssembly otherwise) — nothing is uploaded,
which is also why the page needs no backend.

### Publishing weights

The page needs exported weights before it can translate anything. Easiest route is
the notebook, which does the whole chain on a free Colab GPU and hands you a zip to
unpack into the repo:

**[`notebooks/train_and_export.ipynb`](notebooks/train_and_export.ipynb)** — upload
your Kaggle token, Runtime → Run all, ~3 hours on a T4 for 60 epochs.

Or locally, if you have the dataset and a GPU:

```bash
python scripts/export_onnx.py --checkpoint runs/yosemite_128/checkpoints/latest.pt
python scripts/make_demo_assets.py --checkpoint runs/yosemite_128/checkpoints/latest.pt
python -m http.server -d docs 8000        # preview at http://localhost:8000
```

Enable Pages once (**Settings → Pages → Source: GitHub Actions**); pushes touching
`docs/` then deploy via `.github/workflows/pages.yml`.

Until weights exist the page still accepts an image and displays it, alongside a
note saying there is no model to run it through — it never fabricates an output.

### Payload size

Float32 weights are ~31MB per direction at the default 64 base channels, so both
directions are a ~62MB download for every visitor. Training with
`--g-base-channels 32` gives ~8MB each and still looks decent at 128px; failing
that, `export_onnx.py --quantize` cuts float32 to int8. See
[`docs/models/README.md`](docs/models/README.md).

### Accessibility

Light and dark themes (following the OS preference until you pick one), a skip
link, visible focus rings, a keyboard-operable comparison divider (`←`/`→`, `Home`,
`End`) exposed as an ARIA slider, labelled controls, `aria-live` status messages,
and support for `prefers-reduced-motion` and `prefers-contrast: more`. Both
palettes meet WCAG AA contrast for body and secondary text.

## Tests

```bash
pytest                 # 64 tests, CPU-only
ruff check src scripts tests
```

The suite trains a tiny model on synthetic images for two epochs, so it covers
the loop end to end — including regression tests for the discriminator-gradient
and LR-schedule bugs described below.

## Notes on the earlier version of this project

This was a Colab notebook flattened into modules. The refactor fixed real
correctness bugs, kept here as a record of what to watch for:

- One discriminator's optimiser never had `zero_grad()` called, so its gradients
  accumulated across every step of the entire run.
- Inputs stayed in `[0, 1]` while the generator emitted `[-1, 1]` through
  `tanh`, making half the output range unreachable.
- "Epochs" were single iterations: one batch per "epoch", with the loader
  iterator reset on a modulo that did not match its length.
- The discriminator ended in a fixed 8×8 kernel that silently assumed 128px
  input, and produced one score per image rather than a patch grid.
- Batch norm throughout, at batch sizes where it does not work.
- No identity loss, no replay buffer, no LR decay, no resume, and a data loader
  hard-coded to a Google Drive path.

## References

- Zhu et al., [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (2017)
- Shrivastava et al., [Learning from Simulated and Unsupervised Images](https://arxiv.org/abs/1612.07828) (2017) — the replay buffer
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)

## License

MIT — see [LICENSE](LICENSE).
