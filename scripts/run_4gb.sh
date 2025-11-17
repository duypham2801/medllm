#!/bin/bash

# Training script for PerceptionGPT on 4GB GPU
# Optimized configuration with DeepSpeed ZeRO-3 offloading

set -e  # Exit on error

echo "========================================="
echo "PerceptionGPT Training on 4GB GPU"
echo "========================================="

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA may not be available."
    exit 1
fi

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# Warn about 4GB limitation
read -p "WARNING: 4GB GPU is very limited. Training will be SLOW. Continue? (y/n): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "Working directory: $PROJECT_ROOT"
echo "Config: config/training_configs/perception_1gpu_4gb_lora.py"
echo ""

# Activate conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Conda environment: $CONDA_DEFAULT_ENV"
else
    echo "WARNING: No conda environment detected"
    echo "Activate environment first: conda activate llm"
    exit 1
fi

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0  # Use only first GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Reduce fragmentation
export OMP_NUM_THREADS=4  # CPU threads for offloading

echo ""
echo "Environment variables:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""

# Run training with DeepSpeed
echo "Starting training..."
echo "Monitor GPU usage with: watch -n 0.5 nvidia-smi"
echo ""
echo "========================================="
echo ""

# Use DeepSpeed launcher for single GPU
deepspeed --num_gpus=1 \
    --master_port 29500 \
    mllm/pipeline/finetune.py \
    config/training_configs/perception_1gpu_4gb_lora.py \
    "$@"

echo ""
echo "========================================="
echo "Training completed!"
echo "Check results in: ./exp/perceptionGPT_4gb/"
echo "========================================="
