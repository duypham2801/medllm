#!/bin/bash
# Test MedGemma on T4 16GB GPU (or similar)
# This script runs a quick 3-step training test with dummy data

set -e

echo "========================================="
echo "MedGemma 16GB GPU Test Script"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Activate conda
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo -e "${YELLOW}⚠ Activating llm environment...${NC}"
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate llm
fi

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); mem=torch.cuda.get_device_properties(0).total_memory/(1024**3); print(f'VRAM: {mem:.1f}GB')"

# Run test
CONFIG="config/training_configs/medgemma_16gb_medical.py"
echo -e "${GREEN}Running test with config: $CONFIG${NC}"
echo ""

python mllm/pipeline/finetune.py "$CONFIG" --local_rank=-1 2>&1 | tee exp/medgemma_16gb_test.log

echo ""
echo -e "${GREEN}✓ Test complete. Log: exp/medgemma_16gb_test.log${NC}"
