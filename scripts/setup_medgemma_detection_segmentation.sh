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

# Step 2: Update dataset registration
echo "[2/4] Updating dataset registration..."

# Import the new dataset class
python -c "
import sys
from pathlib import Path

# Add to mllm/dataset/single_image_dataset/__init__.py
init_file = Path('mllm/dataset/single_image_dataset/__init__.py')
content = init_file.read_text()

# Check if already imported
if 'MedicalDetectionSegmentationDataset' not in content:
    # Add import at the end
    content += '\nfrom .medical_detection_segmentation_dataset import MedicalDetectionSegmentationDataset, MedicalDetectionSegmentationMultiDataset\n'
    init_file.write_text(content)
    print('✓ Dataset class imported')
else:
    print('✓ Dataset class already imported')

# Register in root.py
root_file = Path('mllm/dataset/root.py')
root_content = root_file.read_text()

# Check if already registered
if 'MedicalDetectionSegmentationDataset' not in root_content:
    # Add registration
    if 'DATASETS = {' in root_content:
        root_content = root_content.replace(
            'DATASETS = {',
            'DATASETS = {\n    \"MedicalDetectionSegmentationDataset\": MedicalDetectionSegmentationDataset,\n    \"MedicalDetectionSegmentationMultiDataset\": MedicalDetectionSegmentationMultiDataset,'
        )
    else:
        # Add after imports
        root_content += '\n\nfrom .single_image_dataset.medical_detection_segmentation_dataset import MedicalDetectionSegmentationDataset, MedicalDetectionSegmentationMultiDataset\n\nDATASETS = {\n    \"MedicalDetectionSegmentationDataset\": MedicalDetectionSegmentationDataset,\n    \"MedicalDetectionSegmentationMultiDataset\": MedicalDetectionSegmentationMultiDataset,\n}'

    root_file.write_text(root_content)
    print('✓ Dataset registered')
else:
    print('✓ Dataset already registered')
"

echo ""

# Step 3: Verify dataset loading
echo "[3/4] Verifying dataset loading..."

python -c "
from mllm.dataset import prepare_data
from mllm.config import prepare_args

try:
    cfg = prepare_args(['config/training_configs/medgemma_detection_segmentation_${GPU_MEMORY}.py'])
    data = prepare_data(cfg.model_args, cfg.data_args)
    print(f'✓ Dataset loaded successfully: {type(data).__name__}')
    print(f'✓ Length: {len(data)} samples')
except Exception as e:
    print(f'✗ Dataset loading failed: {e}')
    exit(1)
"

echo ""

# Step 4: Choose training config
echo "[4/4] Preparing training..."

if [ "$GPU_MEMORY" = "4gb" ]; then
    CONFIG_FILE="config/training_configs/medgemma_detection_segmentation_4gb.py"
    TRAIN_SCRIPT="scripts/train_medgemma_4gb.sh"
    echo "Using 4GB configuration"
elif [ "$GPU_MEMORY" = "16gb" ]; then
    CONFIG_FILE="config/training_configs/medgemma_detection_segmentation_16gb.py"
    TRAIN_SCRIPT="scripts/train_medgemma_16gb.sh"
    echo "Using 16GB configuration"
else
    echo "ERROR: Invalid GPU memory option: $GPU_MEMORY"
    echo "Options: 4gb, 16gb"
    exit 1
fi

echo "Config file: $CONFIG_FILE"

# Check if required files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
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
echo "2. Test the configuration:"
echo "   python scripts/test_config.py $CONFIG_FILE"
echo ""
echo "3. Start training:"
echo "   python mllm/pipeline/finetune.py $CONFIG_FILE"
echo ""
echo "Or use the training script:"
echo "   bash $TRAIN_SCRIPT"
echo ""
echo "Dataset info:"
echo "  - Format: Detection + Segmentation"
echo "  - Samples: $NUM_SAMPLES"
echo "  - Modalities: X-ray, Dermatology, Endoscopy, CT"
echo "  - Categories: ${CATEGORIES:-'All disease categories'}"
echo ""

# Ask user if they want to start training now
read -p "Do you want to start training now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting training..."
    python mllm/pipeline/finetune.py $CONFIG_FILE
fi