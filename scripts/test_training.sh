#!/bin/bash

# Quick test script without DeepSpeed
# Tests if training pipeline works with dummy data

set -e

echo "========================================="
echo "Quick Training Test (Dummy Data)"
echo "========================================="

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Check conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Conda environment: $CONDA_DEFAULT_ENV"
else
    echo "WARNING: No conda environment detected"
    echo "Activate environment first: conda activate llm"
    exit 1
fi

echo ""
echo "This will test the training pipeline with:"
echo "  - Dummy dataset (10 samples, no real data needed)"
echo "  - 3 training steps only"
echo "  - Auto-download small model if needed"
echo ""

read -p "Continue? (y/n): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting test training..."
echo "========================================="
echo ""

# Run without DeepSpeed for simpler testing
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Simple python call (no DeepSpeed)
python mllm/pipeline/finetune.py \
    config/training_configs/test_3b_dummy.py \
    --local_rank=-1

echo ""
echo "========================================="
echo "Test completed!"
echo "If you see training loss printed, it means the pipeline works!"
echo "========================================="
