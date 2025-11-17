#!/bin/bash

# Download script for PerceptionGPT data and models
# This script helps download required datasets and pretrained models

set -e  # Exit on error

echo "========================================="
echo "PerceptionGPT Data & Model Download"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}Project root: $PROJECT_ROOT${NC}"

# ====================
# 1. Create directories
# ====================
echo ""
echo "[1/4] Creating directory structure..."
mkdir -p data
mkdir -p ckpt
echo -e "${GREEN}✓ Directories created${NC}"

# ====================
# 2. Download LLaVA checkpoint
# ====================
echo ""
echo "[2/4] Downloading LLaVA-v1.5-7B checkpoint..."
echo ""
echo -e "${YELLOW}LLaVA checkpoint options:${NC}"
echo "1. LLaVA-v1.5-7B (Recommended for 4GB GPU with 8-bit loading)"
echo "2. Skip - I already have the checkpoint"
echo ""
read -p "Enter your choice (1-2): " llava_choice

case $llava_choice in
    1)
        echo ""
        echo -e "${YELLOW}Downloading LLaVA-v1.5-7B from HuggingFace...${NC}"
        echo "This will download ~13GB of data. This may take a while..."
        echo ""

        # Check if git-lfs is installed
        if ! command -v git-lfs &> /dev/null; then
            echo -e "${RED}ERROR: git-lfs is not installed.${NC}"
            echo "Please install git-lfs first:"
            echo "  sudo apt-get install git-lfs"
            echo "  git lfs install"
            exit 1
        fi

        # Clone LLaVA model
        if [ ! -d "ckpt/llava-v1.5-7b" ]; then
            cd ckpt
            git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
            cd ..
            echo -e "${GREEN}✓ LLaVA-v1.5-7b downloaded${NC}"
        else
            echo -e "${GREEN}✓ LLaVA-v1.5-7b already exists${NC}"
        fi
        ;;
    2)
        echo -e "${YELLOW}⚠ Skipping LLaVA download${NC}"
        echo "Make sure to update the model path in config file:"
        echo "  config/training_configs/perception_1gpu_4gb_lora.py"
        echo "  Line: model_name_or_path='YOUR_PATH_HERE'"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# ====================
# 3. Download RefCOCO annotations
# ====================
echo ""
echo "[3/4] Downloading RefCOCO dataset annotations..."
echo ""
echo -e "${YELLOW}RefCOCO annotation options:${NC}"
echo "1. Download from Google Drive (recommended - official annotations)"
echo "2. Skip - I will download manually"
echo ""
read -p "Enter your choice (1-2): " refcoco_choice

case $refcoco_choice in
    1)
        echo ""
        echo -e "${YELLOW}To download the official annotations:${NC}"
        echo "1. Visit: https://drive.google.com/file/d/1CNLu1zJKPtliQEYCZlZ8ykH00ppInnyN/view"
        echo "2. Download the ZIP file (annotations only)"
        echo "3. Extract to: $PROJECT_ROOT/data/"
        echo ""
        echo "The extracted files should look like:"
        echo "  data/"
        echo "    ├── blip_laion_cc_sbu_558k.jsonl"
        echo "    ├── CAP_coco2014_train.jsonl"
        echo "    ├── CWB_flickr30k_train.jsonl"
        echo "    └── ..."
        echo ""
        read -p "Press Enter after you've downloaded and extracted the files..."

        # Verify some key files exist
        if [ -f "data/CWB_flickr30k_train.jsonl" ]; then
            echo -e "${GREEN}✓ Annotation files found${NC}"
        else
            echo -e "${YELLOW}⚠ Warning: Could not find expected annotation files${NC}"
            echo "Make sure you extracted the ZIP to the data/ directory"
        fi
        ;;
    2)
        echo -e "${YELLOW}⚠ Skipping annotation download${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# ====================
# 4. Download RefCOCO images
# ====================
echo ""
echo "[4/4] Downloading COCO images for RefCOCO..."
echo ""
echo -e "${YELLOW}COCO image options:${NC}"
echo "RefCOCO uses COCO 2014 train/val images (~40GB total)"
echo ""
echo "1. Download train2014 images (~13GB) - Minimum for training"
echo "2. Download train2014 + val2014 (~19GB) - Recommended"
echo "3. Skip - I will download manually"
echo ""
read -p "Enter your choice (1-3): " coco_choice

case $coco_choice in
    1)
        echo ""
        echo -e "${YELLOW}Downloading COCO train2014 images...${NC}"
        mkdir -p data/coco
        cd data/coco

        if [ ! -f "train2014.zip" ]; then
            wget http://images.cocodataset.org/zips/train2014.zip
        fi

        if [ ! -d "train2014" ]; then
            unzip -q train2014.zip
            echo -e "${GREEN}✓ train2014 extracted${NC}"
        fi

        cd ../..
        ;;
    2)
        echo ""
        echo -e "${YELLOW}Downloading COCO train2014 + val2014 images...${NC}"
        mkdir -p data/coco
        cd data/coco

        # Download train2014
        if [ ! -f "train2014.zip" ]; then
            wget http://images.cocodataset.org/zips/train2014.zip
        fi
        if [ ! -d "train2014" ]; then
            unzip -q train2014.zip
            echo -e "${GREEN}✓ train2014 extracted${NC}"
        fi

        # Download val2014
        if [ ! -f "val2014.zip" ]; then
            wget http://images.cocodataset.org/zips/val2014.zip
        fi
        if [ ! -d "val2014" ]; then
            unzip -q val2014.zip
            echo -e "${GREEN}✓ val2014 extracted${NC}"
        fi

        cd ../..
        ;;
    3)
        echo -e "${YELLOW}⚠ Skipping COCO images download${NC}"
        echo ""
        echo "To download manually:"
        echo "  mkdir -p data/coco"
        echo "  cd data/coco"
        echo "  wget http://images.cocodataset.org/zips/train2014.zip"
        echo "  wget http://images.cocodataset.org/zips/val2014.zip"
        echo "  unzip train2014.zip"
        echo "  unzip val2014.zip"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# ====================
# Summary
# ====================
echo ""
echo "========================================="
echo "Download Summary"
echo "========================================="
echo ""
echo "Directory structure:"
tree -L 2 -d ckpt data 2>/dev/null || ls -R ckpt data
echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Update image paths in dataset configs if needed"
echo "   config/_base_/dataset/DEFAULT_TRAIN_DATASET.py"
echo "2. Run test script: python scripts/test_setup.py"
echo "3. Start training: bash scripts/run_4gb.sh"
echo ""
echo "========================================="
