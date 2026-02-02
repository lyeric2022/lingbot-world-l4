# LingBot-World L4 Patches

Patches to run [LingBot-World](https://github.com/robbyant/lingbot-world) on a **single 24GB GPU** (L4/RTX 4090/A10).

> For installation, model downloads, and official usage, see the [original repo](https://github.com/robbyant/lingbot-world).

## Background

I was very impressed with Google's Genie 3 and wanted to try it out badly, but I also I didn't want to pay the $249.99/mo ultra subscription. I'm also chronically on Twitter, and saw that LingBot-World was a viable open-source option. 

Initially I tried using **Nautilus** (academic Kubernetes cluster) but encountered persistent CUDA OOM errors in the containerized environment. Enraged, I pivoted to **Google Cloud Platform** ($300 trial credits given on new acc sign-up). Requested A100/H100 quota (the 35GB models would fit comfortably), but was denied- likely due to new account. Luckily, the **L4 (24GB)** was approved <1 min, which led to the memory optimization work documented here.

The official LingBot-World expects **8× GPUs with FSDP**. This repo enables running on a **single 24GB GPU** through layer-wise CPU offloading - slower but functional.

See [GPU_SETUP.md](GPU_SETUP.md) for the complete setup journey from VM creation to video generation.

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

**Frame limit:** 9 frames max. Tested 13 and 17 frames - both OOM. Frame count must be `4n+1` (5, 9, 13, 17...), so 9 is the ceiling for 24GB GPUs.

## License

Patches: MIT. LingBot-World: [Apache 2.0](https://github.com/robbyant/lingbot-world/blob/main/LICENSE.txt).
