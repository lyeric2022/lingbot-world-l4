#!/bin/bash
# LingBot-World L4 Setup Script
# Applies patches for running on 24GB GPUs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "LingBot-World L4 Setup"
echo "============================================"
echo ""

# Check if we're in lingbot-world directory
if [ ! -f "generate.py" ] || [ ! -d "wan" ]; then
    echo "ERROR: Run this script from the lingbot-world directory"
    echo ""
    echo "Usage:"
    echo "  cd /path/to/lingbot-world"
    echo "  $0"
    exit 1
fi

# Check if patched file exists
PATCHED_FILE="$SCRIPT_DIR/patches/image2video_patched.py"
if [ ! -f "$PATCHED_FILE" ]; then
    echo "ERROR: Patched file not found at $PATCHED_FILE"
    exit 1
fi

# Backup original
if [ ! -f "wan/image2video.py.original" ]; then
    echo "Backing up original image2video.py..."
    cp wan/image2video.py wan/image2video.py.original
fi

# Apply patch
echo "Applying L4 patches..."
cp "$PATCHED_FILE" wan/image2video.py

# Check if accelerate is installed
if ! python -c "import accelerate" 2>/dev/null; then
    echo ""
    echo "Installing accelerate..."
    pip install accelerate
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To generate video:"
echo ""
echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate.py \\"
echo "  --task i2v-A14B \\"
echo "  --ckpt_dir ./lingbot-world-base-cam \\"
echo "  --prompt \"your prompt here\" \\"
echo "  --save_file outputs/video.mp4 \\"
echo "  --frame_num 9 \\"
echo "  --size 480*832 \\"
echo "  --sample_steps 12 \\"
echo "  --sample_guide_scale 5.5 \\"
echo "  --convert_model_dtype \\"
echo "  --t5_cpu \\"
echo "  --offload_model True"
echo ""
