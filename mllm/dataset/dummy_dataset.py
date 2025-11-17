"""
Dummy dataset for testing without real data
Creates synthetic samples to verify code functionality
"""

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from mllm.dataset.root import DATASETS


@DATASETS.register_module()
class DummyDataset(Dataset):
    """
    Minimal dummy dataset for testing training pipeline
    Creates fake images and simple text for debugging
    """

    def __init__(self, num_samples=10, image_size=224, **kwargs):
        """
        Args:
            num_samples: Number of dummy samples to generate
            image_size: Size of dummy images
        """
        self.num_samples = num_samples
        self.image_size = image_size
        print(f"[DummyDataset] Created with {num_samples} samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a dummy sample with:
        - image: Random RGB image
        - conversations: Simple Q&A format
        """
        # Create random RGB image
        image_array = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, mode='RGB')

        # Create simple conversation
        # Format similar to LLaVA training data
        conversations = [
            {
                "from": "human",
                "value": f"<image>\nDescribe this image {idx}."
            },
            {
                "from": "gpt",
                "value": f"This is a test image number {idx}. It contains random patterns for testing purposes."
            }
        ]

        # Return in expected format
        return {
            'image': image,
            'conversations': conversations,
            'id': f'dummy_{idx}',
        }

    def __repr__(self):
        return f"DummyDataset(num_samples={self.num_samples})"


@DATASETS.register_module()
class DummyDatasetWithBoxes(Dataset):
    """
    Dummy dataset with bounding boxes for REC/RES tasks
    """

    def __init__(self, num_samples=10, image_size=224, **kwargs):
        self.num_samples = num_samples
        self.image_size = image_size
        print(f"[DummyDatasetWithBoxes] Created with {num_samples} samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random image
        image_array = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, mode='RGB')

        # Random bounding box (normalized coordinates)
        x1, y1 = np.random.rand(2) * 0.5  # Top-left in first half
        x2, y2 = x1 + np.random.rand() * 0.4, y1 + np.random.rand() * 0.4  # Width/height up to 0.4
        bbox = [x1, y1, x2, y2]

        # Conversation with box
        conversations = [
            {
                "from": "human",
                "value": f"<image>\nWhere is the object {idx}?"
            },
            {
                "from": "gpt",
                "value": f"The object is at <box>{bbox}</box>."
            }
        ]

        return {
            'image': image,
            'conversations': conversations,
            'id': f'dummy_box_{idx}',
            'boxes': [bbox],
        }
