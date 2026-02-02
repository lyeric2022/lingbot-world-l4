# LingBot-World L4 Patches

Patches to run [LingBot-World](https://github.com/robbyant/lingbot-world) on a **single 24GB GPU** (L4/RTX 4090/A10).

> For installation, model downloads, and usage, see the [official repo](https://github.com/robbyant/lingbot-world).

## The Problem

LingBot-World A14B uses two 14B parameter models (~35GB each in bfloat16). This exceeds 24GB VRAM.

## The Solution

[HuggingFace Accelerate](https://github.com/huggingface/accelerate) **layer-wise CPU offloading** - layers shuttle to GPU only during computation, then back to CPU.

## Quick Start

```bash
# 1. Clone official repo and install deps (see official README)
git clone https://github.com/robbyant/lingbot-world.git
cd lingbot-world
pip install -r requirements.txt
pip install accelerate

# 2. Apply patch
cp /path/to/this/repo/patches/image2video_patched.py wan/image2video.py

# 3. Run with memory flags
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate.py \
  --task i2v-A14B \
  --ckpt_dir ./lingbot-world-base-cam \
  --prompt "your prompt" \
  --image your_image.jpg \
  --save_file outputs/video.mp4 \
  --frame_num 9 \
  --size 480*832 \
  --sample_steps 12 \
  --sample_guide_scale 5.5 \
  --convert_model_dtype \
  --t5_cpu \
  --offload_model True
```

## What's Here

| File | Description |
|------|-------------|
| `patches/image2video_patched.py` | Patched file with Accelerate offloading |
| `setup_l4.sh` | Automated patch script |
| `GPU_SETUP.md` | Full GCP L4 setup guide |

## Performance (L4 24GB)

| Phase | Time |
|-------|------|
| Model loading | ~11 min |
| Generation (12 steps) | ~6 min |
| **Total** | **~17-20 min** |

Frame limit: 9-13 frames tested working, 17+ causes OOM.

## Background

Tried Nautilus (academic K8s) first - hit memory issues. Pivoted to GCP with $300 free credits. A100/H100 quota denied, L4 approved instantly. See [GPU_SETUP.md](GPU_SETUP.md) for the full journey.

## License

Patches: MIT. LingBot-World: [Apache 2.0](https://github.com/robbyant/lingbot-world/blob/main/LICENSE.txt).
