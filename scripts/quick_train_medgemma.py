#!/usr/bin/env python3
"""
Quick training script for MedGemma detection+segmentation
Bypasses complex dataset registration
"""

import json
import sys
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Add to path
sys.path.append('.')

class SimpleMedicalDataset(Dataset):
    """Simple dataset that directly loads JSONL without registration"""

    def __init__(self, jsonl_file, image_root, image_size=896):
        self.jsonl_file = Path(jsonl_file)
        self.image_root = Path(image_root)
        self.image_size = image_size

        # Load all data
        self.data = []
        print(f"Loading dataset from {jsonl_file}...")

        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        sample = json.loads(line.strip())
                        self.data.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"Error at line {line_num + 1}: {e}")

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load image
        image_path = self.image_root / sample['image']
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
            image = np.array(image, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

        # Return formatted data
        return {
            'image': image,
            'conversations': sample['conversations'],
            'boxes': sample.get('boxes', []),
            'labels': sample.get('labels', []),
            'category': sample.get('category', 'Unknown'),
            'modality': sample.get('modality', 'Unknown'),
            'image_path': str(sample['image'])
        }

def test_dataset():
    """Test dataset loading"""
    print("🧪 Testing Simple Medical Dataset...")

    jsonl_file = 'data_medgemma/medical_detection_segmentation_all.jsonl'
    image_root = 'dataset'

    if not Path(jsonl_file).exists():
        print(f"❌ Dataset file not found: {jsonl_file}")
        return False

    try:
        # Create dataset
        dataset = SimpleMedicalDataset(jsonl_file, image_root)
        print(f"✅ Dataset created: {len(dataset)} samples")

        # Test first sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample loaded:")
            print(f"   Image shape: {sample['image'].shape}")
            print(f"   Category: {sample['category']}")
            print(f"   Modality: {sample['modality']}")
            print(f"   Boxes: {len(sample['boxes'])}")
            print(f"   Labels: {len(sample['labels'])}")
            print(f"   Conversation: {sample['conversations'][0]['value'][:50]}...")

        return True

    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def check_requirements():
    """Check if requirements are met"""
    print("🔍 Checking requirements...")

    # Check dataset file
    jsonl_file = Path('data_medgemma/medical_detection_segmentation_all.jsonl')
    if jsonl_file.exists():
        print(f"✅ Dataset file exists: {jsonl_file}")
        with open(jsonl_file, 'r') as f:
            sample_count = sum(1 for line in f if line.strip())
        print(f"   Samples: {sample_count}")
    else:
        print(f"❌ Dataset file not found: {jsonl_file}")
        return False

    # Check image root
    image_root = Path('dataset')
    if image_root.exists():
        print(f"✅ Image root exists: {image_root}")
    else:
        print(f"❌ Image root not found: {image_root}")
        return False

    # Check model
    model_path = Path('ckpt/medgemma-4b-it')
    if model_path.exists():
        print(f"✅ Model exists: {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
        print("   Run: bash scripts/download_medgemma.sh")
        return False

    return True

def main():
    print("🚀 Quick MedGemma Detection + Segmentation Setup")
    print("=" * 60)

    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        return 1

    if not test_dataset():
        print("\n❌ Dataset test failed.")
        return 1

    print("\n" + "=" * 60)
    print("✅ All checks passed!")
    print("\nNext steps:")
    print("1. Activate conda environment: conda activate perceptiongpt")
    print("2. Start training:")
    print("   python mllm/pipeline/finetune.py config/training_configs/medgemma_detection_segmentation_direct_16gb.py")
    print("3. Monitor training:")
    print("   tail -f exp/medgemma_detection_seg_direct_16gb/logs/training.log")

    return 0

if __name__ == "__main__":
    sys.exit(main())