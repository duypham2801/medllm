#!/bin/bash
# Test MedGemma on GTX 1650 4GB GPU
# This script runs a quick 3-step training test with dummy data

set -e  # Exit on error

echo "========================================="
echo "MedGemma 4GB GPU Test Script"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo -e "${YELLOW}⚠ Conda environment not activated${NC}"
    echo "Activating 'llm' environment..."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate llm
else
    echo -e "${GREEN}✓ Conda environment: ${CONDA_DEFAULT_ENV}${NC}"
fi

# Check GPU
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo -e "${YELLOW}⚠ No GPU detected. Training will run on CPU (very slow).${NC}"
    read -p "Continue? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check config file
CONFIG="config/training_configs/medgemma_4gb_test.py"
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}✗ Config file not found: $CONFIG${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Config file found: $CONFIG${NC}"

# Check if models are downloaded
echo ""
echo "Checking model files..."

BASE_MODEL="ckpt/medgemma-4b-it"
ADAPTER="ckpt/flare25-medgemma"

if [ ! -d "$BASE_MODEL" ]; then
    echo -e "${YELLOW}⚠ Base model not found at: $BASE_MODEL${NC}"
    echo "Please run: bash scripts/download_medgemma.sh"
    read -p "Download now? (y/N): " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        bash scripts/download_medgemma.sh
    else
        echo "Exiting. Please download models first."
        exit 1
    fi
fi

if [ ! -d "$ADAPTER" ]; then
    echo -e "${YELLOW}⚠ LoRA adapters not found at: $ADAPTER${NC}"
    echo "The script will download from HuggingFace automatically."
else
    echo -e "${GREEN}✓ Models found${NC}"
fi

# Display test info
echo ""
echo "========================================="
echo "Test Configuration"
echo "========================================="
echo "Config: $CONFIG"
echo "GPU: GTX 1650 4GB (or similar)"
echo "Quantization: 4-bit (NF4)"
echo "LoRA rank: 16"
echo "Batch size: 1"
echo "Steps: 3"
echo "Dataset: Dummy (10 samples)"
echo ""

# Confirm
read -p "Start test? (Y/n): " confirm
if [[ $confirm =~ ^[Nn]$ ]]; then
    echo "Test cancelled."
    exit 0
fi

echo ""
echo "========================================="
echo "Starting Test..."
echo "========================================="
echo ""

# Run test
python mllm/pipeline/finetune.py "$CONFIG" --local_rank=-1 2>&1 | tee exp/medgemma_4gb_test.log

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}✓ Test Completed Successfully!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""
    echo "Log saved to: exp/medgemma_4gb_test.log"
    echo ""
    echo "Next steps:"
    echo "1. Review the log for memory usage and performance"
    echo "2. If successful, try training on T4 16GB for better performance"
    echo "3. Add your medical imaging dataset to start real training"
else
    echo ""
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}✗ Test Failed${NC}"
    echo -e "${RED}=========================================${NC}"
    echo ""
    echo "Common issues:"
    echo "1. Out of Memory (OOM)"
    echo "   Solution: This model is at the limit for 4GB GPU"
    echo "   - Try reducing image_token_len to 64 in config"
    echo "   - Try CPU training (very slow): CUDA_VISIBLE_DEVICES=\"\" python ..."
    echo ""
    echo "2. Model download failed"
    echo "   Solution: Run bash scripts/download_medgemma.sh"
    echo ""
    echo "3. Import errors"
    echo "   Solution: pip install transformers peft bitsandbytes"
    echo ""
    echo "Check log: exp/medgemma_4gb_test.log"
    exit 1
fi
