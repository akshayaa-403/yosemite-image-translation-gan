# CycleGAN: Yosemite Summer ↔ Winter Image Translation

**A clean, production-ready PyTorch implementation of CycleGAN for unpaired image-to-image translation.**

Turn Yosemite summer photos into winter scenes (and vice versa) using only unpaired images. No paired training data required.

Built with **modular, well-documented code** that is easy to train, monitor, extend, and deploy.

![CycleGAN Demo](https://github.com/akshayaa-403/yosemite-image-translation-gan/blob/main/samples/sample-000100.png)  
*(Example output after training: top row = summer → winter, bottom row = winter → summer)*

---

## Features

- **Full CycleGAN** implementation (generators + discriminators + cycle consistency + identity loss)
- **Modular architecture** (`src/models/`, `src/losses/`, `src/data/`, `src/utils/`)
- **Production-ready** training script with checkpoints, sample generation, and loss tracking
- **Multiple deployment options** (Google Colab, Hugging Face Spaces, Streamlit, AWS SageMaker, Docker, etc.)
- **Real-time monitoring** with TensorBoard
- **Custom dataset support** – just drop in your own domain folders
- **MIT License** – free to use, modify, and ship

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (GPU strongly recommended)
- 8 GB+ RAM

```bash
git clone https://github.com/akshayaa-403/yosemite-image-translation-gan.git
cd yosemite-image-translation-gan

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
```

---

## Dataset Structure

Organize your images like this:

```
data/
├── summer/
│   └── data/
│       ├── photo1.jpg
│       ├── photo2.jpg
│       └── ...
├── winter/
│   └── data/
│       ├── photo1.jpg
│       └── ...
├── test_summer/
│   └── data/
│       ├── test1.jpg
│       └── ...
└── test_winter/
    └── data/
        ├── test1.jpg
        └── ...
```

---

## Training

### Basic command
```bash
python scripts/train.py --data_dir ./data
```

### Custom hyperparameters
```bash
python scripts/train.py \
  --data_dir ./data \
  --num_epochs 4000 \
  --batch_size 16 \
  --image_size 256 \
  --lr 0.0002 \
  --n_res_blocks 9
```

**All available arguments** are listed in `scripts/train.py`.

### What happens during training?
- Models are automatically moved to GPU if available.
- Discriminator and generator losses are printed every 10 epochs.
- Sample translations are saved every **100 epochs** (`samples/sample-XXXXXX.png`).
- Full model checkpoints are saved every **1000 epochs** (`checkpoints_cyclegan/checkpoint_epoch_XXXX.pth`).

---

## Performance Benchmarks

| Platform       | GPU      | Batch Size | Time per Epoch | Full Training (4000 epochs) | Approx. Cost |
|----------------|----------|------------|----------------|-----------------------------|--------------|
| Google Colab   | T4       | 16         | ~120s          | ~133 hours                  | Free         |
| Lambda Labs    | V100     | 16         | ~45s           | ~50 hours                   | ~$350        |
| AWS SageMaker  | V100     | 16         | ~45s           | ~50 hours                   | ~$155        |
| Local          | RTX 3090 | 32         | ~20s           | ~22 hours                   | Hardware cost only |

---

## Results

During training you will see:

- **Console logs** showing discriminator and generator losses
- **Sample grids** every 100 epochs (real vs. translated images)
- **Checkpoints** every 1000 epochs (ready for inference or further training)

Example output image:  
![Sample Output](https://github.com/akshayaa-403/yosemite-image-translation-gan/blob/main/samples/sample-000100.png)

---

## Resources

- Original [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
- [PyTorch Documentation](https://pytorch.org)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [Streamlit Documentation](https://docs.streamlit.io)

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
