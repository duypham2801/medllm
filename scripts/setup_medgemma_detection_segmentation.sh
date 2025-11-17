#!/bin/bash

# Setup script for MedGemma Detection + Segmentation
# Converts YOLO dataset to MedGemma format and starts training

set -e

echo "========================================="
echo "MedGemma Detection + Segmentation Setup"
echo "========================================="

# Configuration
GPU_MEMORY=${1:-"16gb"}  # Options: 4gb, 16gb
CATEGORIES=${2:-""}       # Optional: specific categories
DATA_ROOT="dataset"
OUTPUT_ROOT="data_medgemma"

echo "Configuration:"
echo "  GPU Memory: $GPU_MEMORY"
echo "  Categories: ${CATEGORIES:-'All'}"
echo "  Data Root: $DATA_ROOT"
echo "  Output Root: $OUTPUT_ROOT"
echo ""

# Step 1: Convert YOLO dataset to MedGemma format
echo "[1/4] Converting YOLO dataset to MedGemma format..."

if [ -n "$CATEGORIES" ]; then
    python scripts/convert_yolo_to_medgemma.py \
        --dataset_root $DATA_ROOT \
        --output_root $OUTPUT_ROOT \
        --categories $CATEGORIES
else
    python scripts/convert_yolo_to_medgemma.py \
        --dataset_root $DATA_ROOT \
        --output_root $OUTPUT_ROOT
fi

# Check if conversion was successful
if [ ! -f "$OUTPUT_ROOT/medical_detection_segmentation_all.jsonl" ]; then
    echo "ERROR: Dataset conversion failed!"
    echo "Check the error messages above."
    exit 1
fi

echo "✓ Dataset conversion completed!"
echo "  Output: $OUTPUT_ROOT/medical_detection_segmentation_all.jsonl"

# Count samples
NUM_SAMPLES=$(wc -l < "$OUTPUT_ROOT/medical_detection_segmentation_all.jsonl")
echo "  Total samples: $NUM_SAMPLES"
echo ""

# Step 2: Skip dataset registration (avoid circular import)
echo "[2/4] Skipping dataset registration to avoid circular import..."

echo "✓ Dataset files will be imported directly during training"
echo "✓ No circular import issues expected"

echo ""

# Step 3: Verify dataset loading
echo "[3/4] Verifying dataset loading..."

# Run comprehensive test
bash scripts/test_medgemma_detection_segmentation_simple.py

if [ $? -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "❌ Some tests failed!"
    echo "Please check the error messages above."
    exit 1
fi

echo ""

# Step 4: Choose training config
echo "[4/4] Preparing training..."

if [ "$GPU_MEMORY" = "4gb" ]; then
    CONFIG_FILE="config/training_configs/medgemma_detection_segmentation_4gb.py"
    echo "Using 4GB configuration"
elif [ "$GPU_MEMORY" = "16gb" ]; then
    CONFIG_FILE="config/training_configs/medgemma_detection_segmentation_direct_16gb.py"
    echo "Using 16GB direct configuration (no dataset registration)"
else
    echo "ERROR: Invalid GPU memory option: $GPU_MEMORY"
    echo "Options: 4gb, 16gb"
    exit 1
fi

echo "Config file: $CONFIG_FILE"

# Check if required files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -la config/training_configs/medgemma_detection_segmentation*
    exit 1
fi

# Check if models are downloaded
if [ ! -d "ckpt/medgemma-4b-it" ]; then
    echo "WARNING: MedGemma model not found!"
    echo "Please run: bash scripts/download_medgemma.sh"
    echo ""
fi

if [ ! -d "ckpt/flare25-medgemma" ]; then
    echo "WARNING: FLARE25 adapters not found!"
    echo "Please run: bash scripts/download_medgemma.sh"
    echo ""
fi

echo ""
echo "========================================="
echo "Setup completed successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Download models (if not done):"
echo "   bash scripts/download_medgemma.sh"
echo ""
echo "2. Quick test:"
echo "   bash scripts/quick_train_medgemma.py"
echo ""
echo "3. Start training:"
echo "   python mllm/pipeline/finetune.py $CONFIG_FILE"
echo ""
echo "Dataset info:"
echo "  - Format: Detection + Segmentation"
echo "  - Samples: $NUM_SAMPLES"
echo "  - Modality: Endoscopy (GI conditions)"
echo "  - Categories: Viem_thuc_quan, Viem_da_day_HP_am, Viem_da_day_HP_duong, Ung_thu_thuc_quan, Ung_thu_da_day, Loet_hoanh_tao_trang"
echo ""

# Ask user if they want to start quick test now
read -p "Do you want to run quick test now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running quick test..."
    bash scripts/quick_train_medgemma.py
    if [ $? -eq 0 ]; then
        echo ""
        read -p "Test passed! Start training now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Starting training..."
            python mllm/pipeline/finetune.py $CONFIG_FILE
        fi
    else
        echo "Test failed. Please fix issues before training."
    fi
fi