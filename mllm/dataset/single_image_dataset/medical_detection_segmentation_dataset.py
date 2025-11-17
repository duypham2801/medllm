"""
Medical Detection + Segmentation Dataset for MedGemma

Supports detection and segmentation tasks on medical images with multiple modalities.
Dataset format:
- JSONL files with image paths, conversations, bounding boxes, and masks
- Supports COCO format and custom JSONL format
- Handles multiple medical modalities (CT, MRI, X-ray, etc.)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class MedicalDetectionSegmentationDataset(Dataset):
    """
    Medical Detection + Segmentation Dataset for MedGemma

    Supports both detection (bounding boxes) and segmentation (masks) tasks
    on medical images with various modalities and disease categories.
    """

    def __init__(
        self,
        data_file: str,
        image_root: str,
        modality: Optional[str] = None,
        categories: Optional[List[str]] = None,
        image_size: int = 448,
        max_num_instances: int = 10,
        mask_size: int = 224,
        **kwargs
    ):
        super().__init__(data_file, image_size=image_size, **kwargs)

        self.data_file = Path(data_file)
        self.image_root = Path(image_root)
        self.modality = modality
        self.categories = categories
        self.max_num_instances = max_num_instances
        self.mask_size = mask_size

        # Load data
        self.data = self._load_data()

        # Statistics
        self._print_dataset_info()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load JSONL data file"""
        data = []

        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        print(f"Loading data from: {self.data_file}")

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    if line.strip():
                        sample = json.loads(line.strip())

                        # Filter by modality if specified
                        if self.modality and sample.get('modality') != self.modality:
                            continue

                        # Filter by categories if specified
                        if self.categories and sample.get('category') not in self.categories:
                            continue

                        # Validate required fields
                        if not self._validate_sample(sample):
                            print(f"Warning: Invalid sample at line {line_num + 1}")
                            continue

                        data.append(sample)

                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error at line {line_num + 1}: {e}")
                    continue

        print(f"Loaded {len(data)} valid samples")
        return data

    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate sample has required fields"""
        required_fields = ['image', 'conversations']

        for field in required_fields:
            if field not in sample:
                return False

        # Check if image exists
        image_path = self.image_root / sample['image']
        if not image_path.exists():
            return False

        # Validate conversation format
        conversations = sample['conversations']
        if not isinstance(conversations, list) or len(conversations) < 2:
            return False

        # Check for detection/segmentation data
        has_boxes = 'boxes' in sample and isinstance(sample['boxes'], list)
        has_masks = 'masks' in sample and isinstance(sample['masks'], list)

        return has_boxes or has_masks

    def _print_dataset_info(self):
        """Print dataset statistics"""
        if not self.data:
            print("Empty dataset")
            return

        # Count modalities
        modalities = {}
        categories = {}
        split_distribution = {}

        for sample in self.data:
            modality = sample.get('modality', 'Unknown')
            category = sample.get('category', 'Unknown')
            split = sample.get('dataset_split', 'Unknown')

            modalities[modality] = modalities.get(modality, 0) + 1
            categories[category] = categories.get(category, 0) + 1
            split_distribution[split] = split_distribution.get(split, 0) + 1

        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Modalities: {dict(modalities)}")
        print(f"  Categories: {dict(categories)}")
        print(f"  Splits: {dict(split_distribution)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get one sample"""
        sample = self.data[idx].copy()

        # Load and process image
        image_path = self.image_root / sample['image']
        image = self._load_image(image_path)

        # Process conversations
        conversations = sample['conversations']

        # Process detection data (boxes)
        boxes = sample.get('boxes', [])
        labels = sample.get('labels', [])

        # Process segmentation data (masks)
        masks = sample.get('masks', [])

        # Prepare the final sample
        result = {
            'id': idx,
            'image': image,
            'conversations': conversations,
            'modality': sample.get('modality', 'Unknown'),
            'category': sample.get('category', 'Unknown'),
            'original_image_path': str(sample['image']),
        }

        # Add detection data
        if boxes and labels:
            # Normalize boxes to [0, 1] range
            img_h, img_w = image.shape[:2]
            normalized_boxes = []

            for box in boxes:
                if isinstance(box, list) and len(box) == 4:
                    x, y, w, h = box
                    # Convert to normalized coordinates
                    x_norm = x / img_w
                    y_norm = y / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    normalized_boxes.append([x_norm, y_norm, w_norm, h_norm])

            if normalized_boxes:
                result['boxes'] = normalized_boxes
                result['labels'] = labels

        # Add segmentation data
        if masks:
            processed_masks = []

            for mask_path in masks:
                # Mask paths might be relative to dataset root or absolute
                mask_full_path = Path(mask_path)
                if not mask_full_path.is_absolute():
                    # Try relative to image_root first
                    mask_full_path = self.image_root / mask_path
                    if not mask_full_path.exists():
                        # Try relative to current working directory
                        mask_full_path = Path(mask_path)

                if mask_full_path.exists():
                    mask = self._load_mask(mask_full_path)
                    if mask is not None:
                        processed_masks.append(mask)

            if processed_masks:
                result['masks'] = processed_masks

        # Add metadata
        result.update({
            'dataset_split': sample.get('dataset_split', 'unknown'),
            'has_detection': len(boxes) > 0,
            'has_segmentation': len(masks) > 0,
            'num_instances': max(len(boxes), len(masks))
        })

        return result

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess image"""
        try:
            image = Image.open(image_path).convert('RGB')

            # Resize to target size
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)

            # Convert to numpy array and normalize
            image = np.array(image, dtype=np.float32) / 255.0

            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

    def _load_mask(self, mask_path: Path) -> Optional[np.ndarray]:
        """Load and preprocess mask"""
        try:
            mask = Image.open(mask_path).convert('L')

            # Resize to target mask size
            mask = mask.resize((self.mask_size, self.mask_size), Image.NEAREST)

            # Convert to binary mask (0 or 1)
            mask = np.array(mask, dtype=np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)

            return mask
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return None

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get sample information without loading the image"""
        sample = self.data[idx]
        return {
            'id': idx,
            'image_path': str(sample['image']),
            'modality': sample.get('modality', 'Unknown'),
            'category': sample.get('category', 'Unknown'),
            'has_detection': 'boxes' in sample,
            'has_segmentation': 'masks' in sample,
            'num_boxes': len(sample.get('boxes', [])),
            'num_masks': len(sample.get('masks', [])),
            'dataset_split': sample.get('dataset_split', 'unknown')
        }


class MedicalDetectionSegmentationMultiDataset(Dataset):
    """
    Combine multiple medical detection+segmentation datasets
    """

    def __init__(self, datasets: List[MedicalDetectionSegmentationDataset]):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumsum = np.cumsum([0] + self.lengths)
        self.total_length = sum(self.lengths)

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumsum, idx, side='right') - 1
        sample_idx = idx - self.cumsum[dataset_idx]

        sample = self.datasets[dataset_idx][sample_idx]
        sample['dataset_idx'] = dataset_idx
        sample['global_idx'] = idx

        return sample

    def get_dataset_info(self) -> List[Dict[str, Any]]:
        """Get information about all datasets"""
        info = []
        for i, dataset in enumerate(self.datasets):
            info.append({
                'dataset_idx': i,
                'length': len(dataset),
                'data_file': str(dataset.data_file),
                'image_root': str(dataset.image_root),
                'modality': dataset.modality,
                'categories': dataset.categories
            })
        return info