#!/bin/bash
# Download MedGemma models
# This script downloads:
# 1. Base google/medgemma-4b-it model (~8GB)
# 2. FLARE25 LoRA adapters from leoyinn/flare25-medgemma (~130MB)

set -e  # Exit on error

echo "========================================="
echo "MedGemma Model Download Script"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}Error: huggingface-cli not found${NC}"
    echo "Please install it with: pip install huggingface-hub"
    exit 1
fi

echo -e "${GREEN}✓ huggingface-cli found${NC}"
echo ""

# Create checkpoint directory
CKPT_DIR="ckpt"
mkdir -p "$CKPT_DIR"
echo -e "${GREEN}✓ Created checkpoint directory: $CKPT_DIR/${NC}"
echo ""

# ============================================================================
# Download Base Model: google/medgemma-4b-it
# ============================================================================
echo "========================================="
echo "1. Downloading Base Model"
echo "========================================="
echo "Model: google/medgemma-4b-it"
echo "Size: ~8GB"
echo "Location: $CKPT_DIR/medgemma-4b-it/"
echo ""

BASE_MODEL_PATH="$CKPT_DIR/medgemma-4b-it"

if [ -d "$BASE_MODEL_PATH" ] && [ "$(ls -A $BASE_MODEL_PATH)" ]; then
    echo -e "${YELLOW}⚠ Base model already exists at $BASE_MODEL_PATH${NC}"
    read -p "Re-download? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "Skipping base model download."
    else
        echo "Downloading base model..."
        huggingface-cli download google/medgemma-4b-it \
            --local-dir "$BASE_MODEL_PATH" \
            --local-dir-use-symlinks False
        echo -e "${GREEN}✓ Base model downloaded${NC}"
    fi
else
    echo "Downloading base model (this may take several minutes)..."
    echo ""
    echo -e "${YELLOW}⚠ IMPORTANT: You must accept Health AI Developer Foundation's terms of use${NC}"
    echo "Visit: https://huggingface.co/google/medgemma-4b-it"
    echo "Click 'Agree and access repository' if you haven't already"
    echo ""
    read -p "Press Enter to continue after accepting terms..."

    huggingface-cli download google/medgemma-4b-it \
        --local-dir "$BASE_MODEL_PATH" \
        --local-dir-use-symlinks False

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Base model downloaded successfully${NC}"
    else
        echo -e "${RED}✗ Base model download failed${NC}"
        echo "Please check:"
        echo "1. You have accepted the model terms at https://huggingface.co/google/medgemma-4b-it"
        echo "2. You are logged in: huggingface-cli login"
        exit 1
    fi
fi

echo ""

# ============================================================================
# Download LoRA Adapters: leoyinn/flare25-medgemma
# ============================================================================
echo "========================================="
echo "2. Downloading FLARE25 LoRA Adapters"
echo "========================================="
echo "Model: leoyinn/flare25-medgemma"
echo "Size: ~130MB"
echo "Location: $CKPT_DIR/flare25-medgemma/"
echo ""

ADAPTER_PATH="$CKPT_DIR/flare25-medgemma"

if [ -d "$ADAPTER_PATH" ] && [ "$(ls -A $ADAPTER_PATH)" ]; then
    echo -e "${YELLOW}⚠ LoRA adapters already exist at $ADAPTER_PATH${NC}"
    read -p "Re-download? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "Skipping LoRA adapter download."
    else
        echo "Downloading LoRA adapters..."
        huggingface-cli download leoyinn/flare25-medgemma \
            --local-dir "$ADAPTER_PATH" \
            --local-dir-use-symlinks False
        echo -e "${GREEN}✓ LoRA adapters downloaded${NC}"
    fi
else
    echo "Downloading LoRA adapters..."
    huggingface-cli download leoyinn/flare25-medgemma \
        --local-dir "$ADAPTER_PATH" \
        --local-dir-use-symlinks False

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ LoRA adapters downloaded successfully${NC}"
    else
        echo -e "${RED}✗ LoRA adapter download failed${NC}"
        exit 1
    fi
fi

echo ""

# ============================================================================
# Verify Downloads
# ============================================================================
echo "========================================="
echo "3. Verifying Downloads"
echo "========================================="

# Check base model files
if [ -f "$BASE_MODEL_PATH/config.json" ]; then
    echo -e "${GREEN}✓ Base model config.json found${NC}"
else
    echo -e "${RED}✗ Base model config.json not found${NC}"
fi

if [ -f "$BASE_MODEL_PATH/tokenizer.json" ]; then
    echo -e "${GREEN}✓ Base model tokenizer.json found${NC}"
else
    echo -e "${RED}✗ Base model tokenizer.json not found${NC}"
fi

# Check adapter files
if [ -f "$ADAPTER_PATH/adapter_config.json" ]; then
    echo -e "${GREEN}✓ Adapter config found${NC}"
else
    echo -e "${RED}✗ Adapter config not found${NC}"
fi

if [ -f "$ADAPTER_PATH/adapter_model.safetensors" ]; then
    echo -e "${GREEN}✓ Adapter weights found${NC}"
else
    echo -e "${RED}✗ Adapter weights not found${NC}"
fi

echo ""

# ============================================================================
# Display Summary
# ============================================================================
echo "========================================="
echo "Download Summary"
echo "========================================="
echo ""
echo "Base Model Location:"
echo "  $BASE_MODEL_PATH"
echo ""
echo "LoRA Adapters Location:"
echo "  $ADAPTER_PATH"
echo ""
echo "Total Disk Usage:"
du -sh "$CKPT_DIR"
echo ""

# ============================================================================
# Update Configs
# ============================================================================
echo "========================================="
echo "4. Configuration"
echo "========================================="
echo ""
echo "To use these models, your config files should have:"
echo ""
echo "  model_name_or_path = \"$BASE_MODEL_PATH\""
echo "  adapter_path = \"$ADAPTER_PATH\""
echo ""
echo "Example configs:"
echo "  - config/training_configs/medgemma_4gb_test.py (GTX 1650 4GB)"
echo "  - config/training_configs/medgemma_16gb_medical.py (T4 16GB)"
echo ""

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}✓ Download Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Test model loading: bash scripts/test_medgemma_4gb.sh"
echo "2. Start training: python mllm/pipeline/finetune.py config/training_configs/medgemma_4gb_test.py"
echo ""
