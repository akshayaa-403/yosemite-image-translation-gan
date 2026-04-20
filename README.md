# CycleGAN: Yosemite Summer-to-Winter Image Translation

A production-ready PyTorch implementation of CycleGAN for unpaired image-to-image translation. This project translates summer Yosemite images to winter and vice versa.

## Project Structure

```
yosemite-image-translation-gan/
├── src/                           # Main source code
│   ├── __init__.py
│   ├── data/                      # Data loading
│   │   ├── __init__.py
│   │   └── loader.py              # DataLoader utilities
│   ├── models/                    # Neural network models
│   │   ├── __init__.py
│   │   ├── discriminator.py       # Discriminator architecture
│   │   └── generator.py           # Generator architecture
│   ├── losses/                    # Loss functions
│   │   ├── __init__.py
│   │   └── losses.py              # Adversarial and cycle consistency losses
│   └── utils/                     # Helper functions
│       ├── __init__.py
│       └── helpers.py             # Image scaling, sampling, checkpointing
├── scripts/
│   └── train.py                   # Main training script
├── requirements.txt               # Dependencies
├── LICENSE
└── README.md
```

## Code Outputs and Results

The refactored CycleGAN code produces the following outputs during training:

### 1. **Console Output (Training Progress)**
```
Models moved to GPU.
Starting training...
Epoch [   10/  4000] | d_X_loss: 0.5234 | d_Y_loss: 0.4891 | g_total_loss: 2.1456
Epoch [   20/  4000] | d_X_loss: 0.4123 | d_Y_loss: 0.3956 | g_total_loss: 1.8923
...
Training complete!
```

**What it shows:**
- Discriminator losses for X and Y domains (lower is better)
- Generator total loss including adversarial + cycle consistency (lower is better)
- Updates every 10 epochs

### 2. **Sample Images (Every 100 Epochs)**
Saves to `./samples/sample-000100.png`, `sample-000200.png`, etc.

**Contains:**
- Top half: Real summer images → Generated winter images (X→Y translation)
- Bottom half: Real winter images → Generated summer images (Y→X translation)

### 3. **Model Checkpoints (Every 1000 Epochs)**
Saves to `./checkpoints_cyclegan/checkpoint_epoch_1000.pth`, etc.

**Contains:**
- G_XtoY state dict (Summer → Winter generator)
- G_YtoX state dict (Winter → Summer generator)
- D_X state dict (X domain discriminator)
- D_Y state dict (Y domain discriminator)

### 4. **Loss Tracking Data**
Returns list of tuples: `(d_x_loss, d_y_loss, g_total_loss)` for plotting

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (GPU recommended, CPU supported)
- 8GB+ RAM

### Step 1: Clone Repository
```bash
git clone https://github.com/akshayaa-403/yosemite-image-translation-gan.git
cd yosemite-image-translation-gan
```

### Step 2: Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Running Locally

### Dataset Structure
Organize your images in this directory structure:

```
data/
├── summer/
│   ├── data/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
├── winter/
│   ├── data/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
├── test_summer/
│   ├── data/
│   │   ├── test1.jpg
│   │   └── ...
└── test_winter/
    ├── data/
        ├── test1.jpg
        └── ...
```

### Training

**Basic training (4000 epochs, default parameters):**
```bash
python scripts/train.py --data_dir ./data
```

**Custom parameters:**
```bash
python scripts/train.py \
  --data_dir ./data \
  --num_epochs 2000 \
  --batch_size 32 \
  --image_size 256 \
  --g_conv_dim 128 \
  --lr 0.0001
```

**Available arguments:**
- `--data_dir`: Path to data directory
- `--image_size`: Image size (default: 128)
- `--batch_size`: Batch size (default: 16)
- `--num_epochs`: Number of epochs (default: 4000)
- `--g_conv_dim`: Generator conv dimension (default: 64)
- `--d_conv_dim`: Discriminator conv dimension (default: 64)
- `--n_res_blocks`: Residual blocks in generator (default: 6)
- `--lr`: Learning rate (default: 0.0002)
- `--beta1`: Adam beta1 (default: 0.5)
- `--beta2`: Adam beta2 (default: 0.999)

### Monitoring Training

**Console output location:** Terminal where you ran the command

**Outputs saved in:**
- `./samples/`: Generated images every 100 epochs
- `./checkpoints_cyclegan/`: Model checkpoints every 1000 epochs

---

## Hosting & Deployment Options

### Option 1: Google Colab (Recommended for Beginners) ⭐

**Advantages:**
- Free GPU access (Tesla T4/P100)
- Easy to share
- Pre-installed dependencies
- Can train 24/7 with Pro subscription

**Steps:**

1. **Create Colab Notebook:**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Create new notebook

2. **Setup Code Cell 1 - Clone & Install:**
   ```python
   !git clone https://github.com/YOUR_USERNAME/yosemite-image-translation-gan.git
   %cd yosemite-image-translation-gan
   !pip install -r requirements.txt
   ```

3. **Setup Code Cell 2 - Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Training Code Cell 3:**
   ```python
   !python scripts/train.py \
     --data_dir /content/drive/MyDrive/summer2winter_yosemite \
     --num_epochs 4000 \
     --batch_size 16
   ```

5. **Share:** Click "Share" → Anyone with link → Share the Colab URL

**Estimated Training Time:** ~50 hours on free T4 GPU
**Cost:** Free (or $9.99/month for Pro with unlimited GPU)

---

### Option 2: Hugging Face Spaces (Best for Live Demo) ⭐⭐

**Advantages:**
- Free hosting
- Live demo interface
- Easy Git integration
- Works with pre-trained models
- No credit card needed

**Steps:**

1. **Create Hugging Face Account** → [huggingface.co/spaces](https://huggingface.co/spaces)

2. **Create New Space:**
   - Click "Create new Space"
   - Name: `yosemite-cyclegan`
   - License: Choose (OpenRAIL Licenses recommended)
   - Space type: Gradio
   - Visibility: Public

3. **Upload Pre-trained Weights:**
   - Go to [huggingface.co/models](https://huggingface.co/models)
   - Upload your trained checkpoint files (.pth)

4. **Create `app.py` in Space:**
   ```python
   import gradio as gr
   import torch
   from PIL import Image
   import numpy as np
   from src.models import CycleGenerator
   from src.utils import scale
   
   # Load models
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   G_XtoY = CycleGenerator().to(device)
   G_YtoX = CycleGenerator().to(device)
   
   # Load pre-trained weights (upload to Hugging Face)
   # G_XtoY.load_state_dict(torch.load('G_XtoY.pth', map_location=device))
   # G_YtoX.load_state_dict(torch.load('G_YtoX.pth', map_location=device))
   
   G_XtoY.eval()
   G_YtoX.eval()
   
   def translate_to_winter(image):
       """Translate summer image to winter"""
       if image is None:
           return None
       
       # Convert to tensor
       img_array = np.array(image).astype(np.float32) / 255.0
       img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
       img_tensor = img_tensor.to(device)
       
       # Scale to [-1, 1]
       img_tensor = scale(img_tensor)
       
       # Translate
       with torch.no_grad():
           output = G_XtoY(img_tensor)
       
       # Convert back to image
       output = (output.squeeze(0).permute(1, 2, 0) + 1) / 2
       output = (output.cpu().numpy() * 255).astype(np.uint8)
       
       return Image.fromarray(output)
   
   def translate_to_summer(image):
       """Translate winter image to summer"""
       if image is None:
           return None
       
       img_array = np.array(image).astype(np.float32) / 255.0
       img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
       img_tensor = img_tensor.to(device)
       
       img_tensor = scale(img_tensor)
       
       with torch.no_grad():
           output = G_YtoX(img_tensor)
       
       output = (output.squeeze(0).permute(1, 2, 0) + 1) / 2
       output = (output.cpu().numpy() * 255).astype(np.uint8)
       
       return Image.fromarray(output)
   
   # Create interface
   with gr.Blocks(title="Yosemite CycleGAN") as demo:
       gr.Markdown("# 🏔️ Yosemite Image Translation with CycleGAN")
       gr.Markdown("Translate summer Yosemite images to winter and vice versa!")
       
       with gr.Tabs():
           with gr.TabItem("Summer → Winter"):
               with gr.Row():
                   input_summer = gr.Image(label="Upload Summer Image", type="pil")
                   output_winter = gr.Image(label="Winter Translation")
               btn_to_winter = gr.Button("Translate to Winter")
               btn_to_winter.click(translate_to_winter, inputs=input_summer, outputs=output_winter)
           
           with gr.TabItem("Winter → Summer"):
               with gr.Row():
                   input_winter = gr.Image(label="Upload Winter Image", type="pil")
                   output_summer = gr.Image(label="Summer Translation")
               btn_to_summer = gr.Button("Translate to Summer")
               btn_to_summer.click(translate_to_summer, inputs=input_winter, outputs=output_summer)
   
   demo.launch()
   ```

5. **Create `requirements.txt` in Space:**
   ```
   torch
   torchvision
   matplotlib
   numpy
   scikit-image
   gradio
   pillow
   ```

6. **Push to Space via Git:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/yosemite-cyclegan
   cd yosemite-cyclegan
   # Copy your src/ folder and app.py here
   git add .
   git commit -m "Initial commit"
   git push
   ```

**Live Demo URL:** `https://huggingface.co/spaces/YOUR_USERNAME/yosemite-cyclegan`
**Cost:** FREE

---

### Option 3: Streamlit Cloud (Free Web App)

**Advantages:**
- Free hosting
- Beautiful UI
- Easy to develop
- Auto-deploy from GitHub

**Steps:**

1. **Create `streamlit_app.py`:**
   ```python
   import streamlit as st
   import torch
   from PIL import Image
   import numpy as np
   from src.models import CycleGenerator
   from src.utils import scale
   
   st.set_page_config(page_title="Yosemite CycleGAN", layout="wide")
   
   st.title("🏔️ Yosemite Summer ↔ Winter Translation")
   st.write("Transform Yosemite images between seasons using CycleGAN")
   
   @st.cache_resource
   def load_models():
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       G_XtoY = CycleGenerator().to(device)
       G_YtoX = CycleGenerator().to(device)
       # Load pre-trained weights
       # G_XtoY.load_state_dict(torch.load('weights.pth'))
       return G_XtoY, G_YtoX, device
   
   G_XtoY, G_YtoX, device = load_models()
   
   col1, col2 = st.columns(2)
   
   with col1:
       st.subheader("Summer → Winter")
       summer_img = st.file_uploader("Upload summer image", type=['jpg', 'png'], key='summer')
       if summer_img:
           img = Image.open(summer_img).resize((256, 256))
           st.image(img, caption="Input")
           
           if st.button("Translate to Winter"):
               img_array = np.array(img).astype(np.float32) / 255.0
               img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
               img_tensor = scale(img_tensor.to(device))
               
               with torch.no_grad():
                   output = G_XtoY(img_tensor)
               
               output = (output.squeeze(0).permute(1, 2, 0) + 1) / 2
               output = (output.cpu().numpy() * 255).astype(np.uint8)
               st.image(output, caption="Winter Version")
   
   with col2:
       st.subheader("Winter → Summer")
       winter_img = st.file_uploader("Upload winter image", type=['jpg', 'png'], key='winter')
       if winter_img:
           img = Image.open(winter_img).resize((256, 256))
           st.image(img, caption="Input")
           
           if st.button("Translate to Summer"):
               img_array = np.array(img).astype(np.float32) / 255.0
               img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
               img_tensor = scale(img_tensor.to(device))
               
               with torch.no_grad():
                   output = G_YtoX(img_tensor)
               
               output = (output.squeeze(0).permute(1, 2, 0) + 1) / 2
               output = (output.cpu().numpy() * 255).astype(np.uint8)
               st.image(output, caption="Summer Version")
   ```

2. **Deploy to Streamlit Cloud:**
   - Push code to GitHub
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub repo
   - Select file: `streamlit_app.py`

**Live App URL:** Auto-generated
**Cost:** FREE

---

### Option 4: AWS SageMaker (Production-Grade)

**Advantages:**
- Scalable infrastructure
- High-end GPUs (V100, A100)
- Easy to productionize
- Auto-scaling support

**Steps:**

1. **Login to AWS** → Amazon SageMaker → Create notebook instance

2. **Configure Instance:**
   - Instance name: `yosemite-cyclegan`
   - Instance type: `ml.p3.2xlarge` (1 V100 GPU, $3.06/hr)
   - Volume size: 50GB
   - Role: Create new role

3. **In Terminal:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/yosemite-image-translation-gan.git
   cd yosemite-image-translation-gan
   pip install -r requirements.txt
   ```

4. **Run Training in Background:**
   ```bash
   nohup python scripts/train.py --data_dir ./data --num_epochs 4000 > training.log 2>&1 &
   ```

5. **Monitor Progress:**
   ```bash
   tail -f training.log
   ```

6. **Save to S3:**
   ```python
   import boto3
   s3 = boto3.client('s3')
   s3.upload_file('checkpoints_cyclegan/checkpoint_epoch_4000.pth', 
                  'my-bucket', 'cyclegan/checkpoint.pth')
   ```

**Cost:** $1.35-$3.06/hour depending on instance

---

### Option 5: Lambda Labs (Affordable GPU)

**Advantages:**
- Affordable GPU options
- Simple setup
- SSH access
- Pay-as-you-go

**Steps:**

1. Go to [lambdalabs.com](https://www.lambdalabs.com/service/gpu-cloud)
2. Create account → Select GPU (A100: $1.51/hr, V100: $0.49/hr)
3. Launch instance → Get SSH credentials
4. SSH in:
   ```bash
   ssh -i lambda_key.pem ubuntu@your-instance-ip
   ```

5. Setup and train:
   ```bash
   git clone https://github.com/YOUR_USERNAME/yosemite-image-translation-gan.git
   cd yosemite-image-translation-gan
   pip install -r requirements.txt
   python scripts/train.py --data_dir ./data
   ```

**Cost:** $0.49-$1.51/hour

---

### Option 6: Docker Container (Any Cloud)

Deploy to DigitalOcean, Azure, GCP, or Kubernetes

1. **Create Dockerfile:**
   ```dockerfile
   FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04
   WORKDIR /workspace
   COPY . .
   RUN pip install -r requirements.txt
   ENTRYPOINT ["python", "scripts/train.py"]
   ```

2. **Build & test locally:**
   ```bash
   docker build -t cyclegan .
   docker run --gpus all -v $(pwd)/data:/data cyclegan --data_dir /data
   ```

3. **Push to Docker Hub:**
   ```bash
   docker tag cyclegan:latest your-username/cyclegan:latest
   docker push your-username/cyclegan:latest
   ```

4. **Deploy to DigitalOcean App Platform:**
   - Connect GitHub
   - Build from Docker Hub image
   - Allocate 2GB+ RAM + GPU

**Cost:** Starting at $10/month (DigitalOcean)

---

## Performance Benchmarks

| Platform | GPU | Batch Size | Epoch Time | 4000 Epochs | Cost/Month |
|----------|-----|-----------|-----------|-----------|----------|
| Google Colab | T4 | 16 | ~120s | ~133 hrs | FREE |
| Lambda Labs | V100 | 16 | ~45s | ~50 hrs | $350 |
| AWS SageMaker | V100 | 16 | ~45s | $155 | $155 |
| Hugging Face | CPU | 4 | ~10min | Not practical | FREE |
| Local (RTX 3090) | RTX 3090 | 32 | ~20s | ~22 hrs | One-time cost |

---

## Recommended Setup for Different Use Cases

### 🎓 Learning & Experimentation
**Use:** Google Colab
- Free GPU
- Easy sharing
- Great for learning

### 🎨 Live Demo for Users
**Use:** Hugging Face Spaces
- Free hosting
- Beautiful interface
- No infrastructure needed
- Perfect for showcasing

### 🚀 Production Training
**Use:** AWS SageMaker or Lambda Labs
- High-end GPUs
- Professional support
- Easy scaling

### 💻 Custom Web App
**Use:** Streamlit Cloud
- Beautiful interface
- Easy to develop
- Auto-deploy from GitHub

---

## Quick Start: 5-Minute Setup

**If you just want to see it work:**

1. Open [Google Colab](https://colab.research.google.com)
2. Copy-paste this into a cell:
   ```python
   !git clone https://github.com/akshayaa-403/yosemite-image-translation-gan.git
   %cd yosemite-image-translation-gan
   !pip install -r requirements.txt
   
   # Download sample data
   !mkdir -p data/summer/data data/winter/data data/test_summer/data data/test_winter/data
   
   # Download sample images (replace with your own)
   !wget -O data/summer/data/sample.jpg https://example.com/sample.jpg
   
   # Run training
   !python scripts/train.py --data_dir ./data --num_epochs 100 --batch_size 8
   ```
3. Click Run!

---

## Real-Time Monitoring

**View training progress with TensorBoard:**

Modify `scripts/train.py` to add:
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# In training loop:
writer.add_scalar('Loss/discriminator_x', d_x_loss.item(), epoch)
writer.add_scalar('Loss/discriminator_y', d_y_loss.item(), epoch)
writer.add_scalar('Loss/generator', g_total_loss.item(), epoch)
```

Then run:
```bash
tensorboard --logdir=runs --port=6006
```

Access at `http://localhost:6006`

---

## Resources

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
- [PyTorch Documentation](https://pytorch.org)
- [Hugging Face Spaces Guide](https://huggingface.co/docs/hub/spaces)
- [Streamlit Documentation](https://docs.streamlit.io)
- [AWS SageMaker Docs](https://docs.aws.amazon.com/sagemaker/)
- [Google Colab Guide](https://colab.research.google.com)

---

## License

See LICENSE file for details.
# yosemite-image-translation-gan