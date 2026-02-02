# LingBot-World L4 Setup

Patches and setup guide for running [LingBot-World](https://github.com/robbyant/lingbot-world) on a single NVIDIA L4 GPU (24GB VRAM).

## Background

I initially tried running LingBot-World on **Nautilus** (academic Kubernetes cluster) but hit persistent memory issues with pod scheduling and CUDA OOM errors in the containerized environment.

Pivoted to **Google Cloud Platform** using **$300 in free credits**. While **A100/H100** GPUs would be ideal (the 35GB models would fit in memory), my quota requests were denied. The **L4 (24GB)** was the only approved option, which required the memory optimization patches documented here.

See [GPU_SETUP.md](GPU_SETUP.md) for the full journey and detailed setup instructions.

## The Problem

LingBot-World A14B has two 14B parameter models (~35GB each in bfloat16). This exceeds the L4's 24GB VRAM.

## The Solution

We use [Hugging Face Accelerate](https://github.com/huggingface/accelerate) for **layer-wise CPU offloading** - layers are moved to GPU only during computation, then back to CPU.

## Quick Start

### 1. Clone LingBot-World

```bash
git clone https://github.com/robbyant/lingbot-world.git
cd lingbot-world
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install accelerate
```

### 3. Download Models

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./lingbot-world-base-cam
```

### 4. Apply Patches

```bash
# Copy the patched file
cp /path/to/this/repo/patches/image2video_patched.py wan/image2video.py
```

Or run the setup script:

```bash
./setup_l4.sh
```

### 5. Generate

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate.py \
  --task i2v-A14B \
  --ckpt_dir ./lingbot-world-base-cam \
  --prompt "your prompt here" \
  --save_file outputs/video.mp4 \
  --frame_num 9 \
  --size 480*832 \
  --sample_steps 12 \
  --sample_guide_scale 5.5 \
  --convert_model_dtype \
  --t5_cpu \
  --offload_model True
```

## What's Included

| File | Description |
|------|-------------|
| `patches/image2video_patched.py` | Pre-patched file with all L4 fixes |
| `setup_l4.sh` | Automated setup script |
| `GPU_SETUP.md` | Full detailed setup guide |

## Performance

On L4 (24GB) with layer-wise offloading:
- Model loading: ~11 minutes
- Generation (12 steps): ~6 minutes
- Total: ~17-20 minutes per video

## What's NOT Included

- Model checkpoints (140GB+) - download from [HuggingFace](https://huggingface.co/robbyant/lingbot-world-base-cam)
- Real-time interactive mode - not released yet by Robbyant

## Current Limitations

**This repo provides single-GPU workarounds.** The official LingBot-World is designed for multi-GPU inference:
```bash
torchrun --nproc_per_node=8 generate.py --dit_fsdp --t5_fsdp --ulysses_size 8 ...
```

Our patches enable running on a single 24GB GPU through layer-wise CPU offloading (slower but works).

**Released models:**
- **LingBot-World-Base (Cam)** - Camera trajectory control (pre-defined paths)

**NOT yet released:**
- **LingBot-World-Base (Act)** - WASD keyboard control
- **LingBot-World-Fast** - Real-time 16 FPS interactive mode

## License

Patches are provided under MIT license. LingBot-World itself is Apache 2.0.
