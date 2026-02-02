# LingBot-World GPU Setup Guide

This guide documents a fully working setup for running LingBot-World image-to-video generation on a single NVIDIA L4 GPU using Google Cloud Compute Engine.

---

## Setup Overview

Using **GCP** with **$300 free trial credits** on signup.

### GPU Selection

| GPU | VRAM | GCP Availability | Status |
|-----|------|------------------|--------|
| **H100** | 80GB | Limited | Quota denied |
| **A100** | 40-80GB | Limited | Quota denied |
| **L4** | 24GB | Generally available | Approved (<1 min) |

### Cost

- `g2-standard-32` + L4: ~$2.50/hour
- $300 free credits = ~120 hours of compute

### System Configuration

- **OS:** Debian 12 (Bookworm)
- **GPU:** NVIDIA L4 (24GB VRAM)
- **Machine:** g2-standard-32 (32 vCPU, 128GB RAM)

---

## The Memory Challenge

LingBot A14B uses **two separate 14-billion parameter models**:
- `low_noise_model` - used for low noise timesteps
- `high_noise_model` - used for high noise timesteps

Each model is 14B parameters and requires **~28GB VRAM** when loaded in bfloat16 (~56GB total for both). This far exceeds the L4's 24GB.

**The Solution:** We use **Hugging Face Accelerate's `dispatch_model`** to enable **layer-wise CPU offloading**. Instead of loading the entire model to GPU, it:

1. Analyzes available GPU/CPU memory
2. Splits model layers between GPU and CPU
3. During forward pass: moves each layer to GPU → computes → moves back
4. This allows running models **larger than VRAM** (slower, but works)

---

## 1. Create GPU VM

> **Note:** I created the VM via GCP Console UI, not CLI. The gcloud command below is the equivalent for automation.

**Via Console (what I did):**
1. Go to Compute Engine → VM instances → Create
2. Select region with L4 availability (e.g., `us-central1-c`)
3. Machine type: `g2-standard-32` (32 vCPU, 128GB RAM)
4. GPU: Add NVIDIA L4
5. Boot disk: Debian 12, **200GB** (critical - models are 140GB+)
6. Allow SSH + HTTP/HTTPS

**Via gcloud CLI (equivalent):**
```bash
gcloud compute instances create lingbot-vm \
  --zone=us-central1-c \
  --machine-type=g2-standard-32 \
  --accelerator=type=nvidia-l4,count=1 \
  --boot-disk-size=200GB \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --maintenance-policy=TERMINATE
```

---

## 2. Connect

### Generate SSH Key (if needed)

```bash
# On your local machine
ssh-keygen -t ed25519 -C "your_email@example.com"
```

### Add Key to VM

Option A: Via GCP Console
1. Go to Compute Engine → VM instances
2. Click your instance → Edit
3. Under "SSH Keys", add your public key (`~/.ssh/id_ed25519.pub`)

Option B: Via gcloud CLI
```bash
gcloud compute ssh USERNAME@INSTANCE_NAME --zone=us-central1-c
```
(This auto-generates and adds keys)

### Connect

```bash
ssh USERNAME@VM_EXTERNAL_IP
```

---

## 3. Resize Disk (if needed)

```bash
sudo apt update
sudo apt install -y cloud-guest-utils
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
df -h /
```

---

## 4. Install NVIDIA Drivers + CUDA

```bash
sudo apt install -y nvidia-driver nvidia-cuda-toolkit
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
nvcc --version
```

Set CUDA environment (add to `~/.bashrc`):

```bash
echo 'export CUDA_HOME=/usr' >> ~/.bashrc
echo 'export PATH=/usr/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 5. Install Python + venv

```bash
sudo apt install -y python3 python3-venv python3-pip
```

---

## 6. Clone LingBot

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/robbyant/lingbot-world.git
cd lingbot-world
```

---

## 7. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

---

## 8. Install Dependencies

```bash
pip install -r requirements.txt
pip install packaging wheel ninja psutil accelerate
```

---

## 9. Install Flash-Attn

```bash
python -m pip install flash-attn --no-build-isolation
```

> If build fails, you can skip this step - the model will run slower but still works.

---

## 10. Download Models

```bash
pip install huggingface-hub hf-xet

hf download robbyant/lingbot-world-base-cam \
  --local-dir /home/$USER/lingbot-world-base-cam
```

---

## 11. Environment Variable

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Add to `~/.bashrc` for persistence:
```bash
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
```

---

## 12. Apply Code Patches (REQUIRED for 24GB GPUs)

The A14B model is too large to fit entirely in 24GB VRAM. We must patch the code to use **layer-wise CPU offloading** via Accelerate.

### Patch 1: Add Accelerate Imports

Run this to add the necessary imports:

```bash
python3 << 'EOF'
filepath = '/home/$USER/projects/lingbot-world/wan/image2video.py'

with open(filepath, 'r') as f:
    content = f.read()

# Add accelerate import
if 'from accelerate import' not in content:
    old_imports = 'from .modules.model import WanModel'
    new_imports = '''from .modules.model import WanModel
from accelerate import cpu_offload_with_hook, dispatch_model, infer_auto_device_map'''
    content = content.replace(old_imports, new_imports)
    with open(filepath, 'w') as f:
        f.write(content)
    print('Imports added!')
else:
    print('Imports already present')
EOF
```

### Patch 2: Modify Model Loading

Load models to CPU with low memory usage:

```bash
python3 << 'EOF'
filepath = '/home/$USER/projects/lingbot-world/wan/image2video.py'

with open(filepath, 'r') as f:
    content = f.read()

# Modify low_noise_model loading
old_load1 = '''        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint, torch_dtype=torch.bfloat16)'''
new_load1 = '''        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map='cpu')'''
content = content.replace(old_load1, new_load1)

# Modify high_noise_model loading
old_load2 = '''        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint, torch_dtype=torch.bfloat16)'''
new_load2 = '''        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map='cpu')'''
content = content.replace(old_load2, new_load2)

with open(filepath, 'w') as f:
    f.write(content)
print('Model loading patched!')
EOF
```

### Patch 3: Enable Layer-wise GPU Execution

Replace `_prepare_model_for_timestep` function with accelerate-enabled version:

```bash
python3 << 'EOF'
import re

filepath = '/home/$USER/projects/lingbot-world/wan/image2video.py'

with open(filepath, 'r') as f:
    content = f.read()

# Find and replace the entire _prepare_model_for_timestep function
old_func_pattern = r'(    def _prepare_model_for_timestep\(self, t, boundary, offload_model\):.*?)(\n    def generate\(self,)'

new_func = '''    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.
        Uses accelerate dispatch_model for layer-wise GPU execution on memory-constrained GPUs.
        """
        import gc
        
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        
        if offload_model or self.init_on_cpu:
            # First ensure everything is on CPU and memory is cleared
            for model_name in ['low_noise_model', 'high_noise_model']:
                if hasattr(self, model_name):
                    try:
                        model = getattr(self, model_name)
                        # Remove any dispatch hooks if present
                        if hasattr(model, '_hf_hook'):
                            from accelerate.hooks import remove_hook_from_module
                            remove_hook_from_module(model, recurse=True)
                        if next(model.parameters()).device.type == 'cuda':
                            model.to('cpu')
                    except Exception:
                        pass
            
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Use accelerate dispatch_model for layer-wise execution
            required_model = getattr(self, required_model_name)
            try:
                max_memory = {0: "20GiB", "cpu": "100GiB"}
                device_map = infer_auto_device_map(
                    required_model, 
                    max_memory=max_memory,
                    no_split_module_classes=["WanAttentionBlock"]
                )
                required_model = dispatch_model(required_model, device_map=device_map)
                setattr(self, required_model_name, required_model)
            except Exception as e:
                # Fallback to simple GPU load if dispatch fails
                print(f"dispatch_model failed: {e}, trying simple load...")
                required_model.to(self.device)
        
        return getattr(self, required_model_name)

'''

match = re.search(old_func_pattern, content, re.DOTALL)
if match:
    content = content[:match.start(1)] + new_func + match.group(2) + content[match.end():]
    with open(filepath, 'w') as f:
        f.write(content)
    print('Function replaced with accelerate-enabled version!')
else:
    print('Could not find function to replace - may need manual patching')
EOF
```

### Patch 4: Fix Cleanup Code for Meta Tensors

When using `dispatch_model`, the model parameters become "meta tensors" (placeholders). The original cleanup code crashes when trying to move them to CPU. This patch fixes that:

```bash
python3 << 'EOF'
filepath = '/home/$USER/projects/lingbot-world/wan/image2video.py'

with open(filepath, 'r') as f:
    content = f.read()

old_cleanup = '''            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()'''

new_cleanup = '''            if offload_model:
                # Handle dispatched models (meta tensors) gracefully
                for model_name in ['low_noise_model', 'high_noise_model']:
                    try:
                        model = getattr(self, model_name)
                        if hasattr(model, '_hf_hook'):
                            from accelerate.hooks import remove_hook_from_module
                            remove_hook_from_module(model, recurse=True)
                        # Only move to CPU if not on meta device
                        if not any(p.device.type == 'meta' for p in model.parameters()):
                            model.cpu()
                    except Exception:
                        pass
                torch.cuda.empty_cache()'''

if old_cleanup in content:
    content = content.replace(old_cleanup, new_cleanup)
    with open(filepath, 'w') as f:
        f.write(content)
    print('Cleanup code fixed!')
else:
    print('Cleanup code already patched or not found')
EOF
```

**Why this is needed:** Without this patch, generation completes successfully but crashes during cleanup with `NotImplementedError: Cannot copy out of meta tensor; no data!`

---

## 13. Run Generation

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate.py \
  --task i2v-A14B \
  --ckpt_dir /home/$USER/lingbot-world-base-cam \
  --prompt "a small robot exploring an abandoned sci-fi city, cinematic lighting, slow camera movement" \
  --save_file outputs/test1.mp4 \
  --frame_num 9 \
  --size 480*832 \
  --sample_steps 12 \
  --sample_guide_scale 5.5 \
  --convert_model_dtype \
  --t5_cpu \
  --offload_model True
```

---

## 14. Output

When generation completes successfully, you'll see:

```
12/12 [06:18<00:00, 31.58s/it]
[INFO] Video saved to outputs/test1.mp4
```

Video saved to:

```
~/projects/lingbot-world/outputs/test1.mp4
```

### Download to Local Machine

```bash
# Run on your LOCAL machine (not the server)
scp USERNAME@VM_EXTERNAL_IP:~/projects/lingbot-world/outputs/test1.mp4 ~/Downloads/
```

### Video Specs

- **9 frames** at 16 FPS = ~0.5 second video
- **480×832 resolution** (portrait)
- Format: MP4

---

## 15. Monitor GPU Usage

In a separate terminal, watch GPU memory and utilization:

```bash
watch -n 1 nvidia-smi
```

Expected behavior during generation:
- VRAM usage climbs during model loading
- Utilization spikes during diffusion steps
- Memory oscillates as layers shuttle between CPU↔GPU

---

## 16. Experimenting with Parameters

Once working, you can adjust:

| Parameter | Effect | L4 Safe Range |
|-----------|--------|---------------|
| `--frame_num` | More frames = longer video | 5-9 (max 9 tested) |
| `--sample_steps` | Higher = better quality, slower | 8-20 |
| `--size` | Resolution (WxH or HxW) | 480*832, 832*480 |
| `--sample_guide_scale` | Prompt adherence (higher = stronger) | 5.0-7.5 |
| `--image` | Input image path | Any JPG/PNG |
| `--action_path` | Camera trajectory folder (optional) | See below |

### Camera Control (Optional)

The `examples/` folder contains pre-defined camera trajectories you can use:

```bash
--action_path examples/00
```

Each trajectory folder contains:
- `intrinsics.npy` - Shape [num_frames, 4]: [fx, fy, cx, cy]
- `poses.npy` - Shape [num_frames, 4, 4]: transformation matrices (OpenCV coords)

You can extract camera paths from existing videos using [ViPE](https://github.com/robbyant/vipe) or create custom trajectories.

**Example: Higher quality (more steps):**
```bash
python generate.py \
  --task i2v-A14B \
  --ckpt_dir /home/$USER/lingbot-world-base-cam \
  --prompt "your prompt here" \
  --image /path/to/your/image.jpg \
  --save_file outputs/video2.mp4 \
  --frame_num 9 \
  --size 480*832 \
  --sample_steps 20 \
  --sample_guide_scale 6.5 \
  --convert_model_dtype \
  --t5_cpu \
  --offload_model True
```

---

## 17. How Layer-wise Offloading Works (Technical Details)

```
┌─────────────────────────────────────────────────────────────┐
│                    Traditional Loading                       │
│                                                              │
│  GPU [████████████████████████████] 28GB model → OOM!       │
│      [        24GB available      ]                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                Layer-wise with Accelerate                    │
│                                                              │
│  CPU [████████████████████████████] Most layers stay here   │
│                                                              │
│  GPU [████████]  Only active layers loaded temporarily      │
│      [ 20GB  ]   Computes → moves back to CPU → next layer  │
└─────────────────────────────────────────────────────────────┘
```

**Trade-off:** Slower inference (layers shuttle between CPU↔GPU) but runs on smaller GPUs.

---

## 18. Common Issues

### CUDA Out of Memory

Even with patches, if OOM persists:
- Reduce `--frame_num` (try 5)
- Reduce `--sample_steps` (try 8)
- Use smaller `--size` (try 480*832)

### dispatch_model Fails

If you see "dispatch_model failed", the fallback will try simple loading. You may need to reduce parameters further.

### Flash-Attn Fails to Build

You can run without it (slower) - just skip the flash-attn install step.

### "Cannot copy out of meta tensor" Error

This happens if Patch 4 wasn't applied. The generation actually completed, but cleanup failed. Apply Patch 4 and re-run.

---

## 19. Required Flags

LingBot A14B **requires** these flags on 24GB GPUs:

| Flag | Purpose |
|------|---------|
| `--t5_cpu` | Keep T5 text encoder on CPU |
| `--offload_model True` | Enable model offloading between timesteps |
| `--convert_model_dtype` | Use bfloat16 to reduce memory |

---

## 20. Verified Working Hardware

| GPU | VRAM | Notes |
|-----|------|-------|
| NVIDIA L4 | 24GB | Requires all patches, layer-wise offload |
| A100 | 40GB+ | May work with basic offload flags |
| H100 | 80GB | Should work without patches |

---

## 21. Performance Expectations

On L4 with layer-wise offloading (tested):

| Phase | Time |
|-------|------|
| Load low_noise_model (8 shards) | ~6 min |
| Load high_noise_model (8 shards) | ~5 min |
| Generation (12 steps × ~30s each) | ~6 min |
| **Total** | **~17-20 min** |

- Higher RAM helps (128GB recommended for smooth operation)
- SSD helps with model loading speed
- First run may be slower due to dispatch_model setup

---

## 22. Known Limitations

Even with all optimizations:
- This model **barely fits** on 24GB VRAM
- **Max 9 frames** - tested 13 and 17, both OOM
- Frame count must be `4n+1` (5, 9, 13, 17...), so 9 is the ceiling
- You are at the **edge of consumer VRAM physics**

This setup is a workaround, not the intended deployment. The official LingBot-World expects 8× A100 GPUs with FSDP.

---

## 23. License

Follow upstream LingBot license.

---

# Summary: What We Built

You've successfully:

- ✅ Provisioned GPU infrastructure on GCP (g2-standard-32 + L4)
- ✅ Installed CUDA drivers and toolkit
- ✅ Built the PyTorch + Flash-Attention stack
- ✅ Downloaded 140GB+ of model checkpoints
- ✅ Diagnosed VRAM constraints (28GB model vs 24GB GPU)
- ✅ Implemented 4 code patches for layer-wise CPU offloading
- ✅ Fixed meta tensor cleanup issues
- ✅ Generated video successfully

**The 4 Patches:**
1. Add Accelerate imports
2. Modify model loading (`device_map='cpu'`)
3. Replace `_prepare_model_for_timestep` with dispatch_model version
4. Fix cleanup code for meta tensors

This is real ML infrastructure engineering - running a model that "shouldn't" fit on your hardware through intelligent memory management.

---

# Quick Reference: All Patches in One Script

Save this as `apply_patches.sh` in the `lingbot-world` directory and run once after cloning:

```bash
#!/bin/bash
set -e

cd ~/projects/lingbot-world
FILEPATH="wan/image2video.py"

echo "Applying LingBot L4 patches..."

python3 << 'PATCH_EOF'
filepath = 'wan/image2video.py'

with open(filepath, 'r') as f:
    content = f.read()

# Patch 1: Add imports
if 'from accelerate import' not in content:
    content = content.replace(
        'from .modules.model import WanModel',
        '''from .modules.model import WanModel
from accelerate import cpu_offload_with_hook, dispatch_model, infer_auto_device_map'''
    )
    print("✓ Patch 1: Imports added")

# Patch 2: Model loading - low_noise_model
if 'low_cpu_mem_usage=True' not in content:
    content = content.replace(
        'subfolder=config.low_noise_checkpoint, torch_dtype=torch.bfloat16)',
        'subfolder=config.low_noise_checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cpu")'
    )
    content = content.replace(
        'subfolder=config.high_noise_checkpoint, torch_dtype=torch.bfloat16)',
        'subfolder=config.high_noise_checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cpu")'
    )
    print("✓ Patch 2: Model loading modified")

# Patch 4: Fix cleanup code
old_cleanup = '''            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()'''

new_cleanup = '''            if offload_model:
                # Handle dispatched models (meta tensors) gracefully
                for model_name in ['low_noise_model', 'high_noise_model']:
                    try:
                        model = getattr(self, model_name)
                        if hasattr(model, '_hf_hook'):
                            from accelerate.hooks import remove_hook_from_module
                            remove_hook_from_module(model, recurse=True)
                        if not any(p.device.type == 'meta' for p in model.parameters()):
                            model.cpu()
                    except Exception:
                        pass
                torch.cuda.empty_cache()'''

if old_cleanup in content:
    content = content.replace(old_cleanup, new_cleanup)
    print("✓ Patch 4: Cleanup code fixed")

with open(filepath, 'w') as f:
    f.write(content)

print("\nPatches 1, 2, 4 applied!")
print("NOTE: Patch 3 (function replacement) must be applied manually - see guide.")
PATCH_EOF

echo ""
echo "Done! Apply Patch 3 manually from the guide for _prepare_model_for_timestep."
```
