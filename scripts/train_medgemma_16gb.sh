#!/bin/bash
# Full MedGemma training script for medical imaging fine-tuning
# Use this for actual training, not just testing

set -e

echo "========================================="
echo "MedGemma Medical Fine-tuning Script"
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

# Run full training
CONFIG="config/training_configs/medgemma_16gb_medical.py"
echo -e "${GREEN}Starting full training with config: $CONFIG${NC}"
echo ""

# Remove --max_steps 3 if it exists for actual training
python mllm/pipeline/finetune.py "$CONFIG" --local_rank=-1 2>&1 | tee exp/medgemma_full_training.log

echo ""
echo -e "${GREEN}✓ Training complete! Log: exp/medgemma_full_training.log${NC}"