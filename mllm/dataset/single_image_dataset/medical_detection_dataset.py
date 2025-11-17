"""
Medical Detection Dataset
Supports COCO format for object detection tasks (lesions, tumors, organs, etc.)
"""

import json
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from mllm.dataset.root import DATASETS


@DATASETS.register_module()
class MedicalDetectionDataset(Dataset):
    """
    Medical imaging detection dataset in COCO format.

    Supports detection tasks for:
    - Lung nodules
    - Tumors
    - Lesions
    - Organs
    - Calcifications
    - Other medical findings

    Args:
        ann_file (str): Path to COCO annotation JSON file
        img_prefix (str): Prefix path to images
        modality (str): Imaging modality (CT, MRI, X-ray, etc.)
        image_size (int): Target image size (default: 448 for MedGemma)
        max_objects (int): Maximum number of objects per image (default: 20)
        task_type (str): 'detection' or 'grounding' (default: 'detection')
    """

    def __init__(
        self,
        ann_file,
        img_prefix='',
        modality='CT',
        image_size=448,
        max_objects=20,
        task_type='detection',
        **kwargs
    ):
        super().__init__()

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.modality = modality
        self.image_size = image_size
        self.max_objects = max_objects
        self.task_type = task_type

        # Load annotations
        print(f'[MedicalDetectionDataset] Loading annotations from {ann_file}')
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # Build indices
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Filter images that have annotations
        self.valid_images = [
            img for img in self.images
            if img['id'] in self.img_to_anns
        ]

        print(f'[MedicalDetectionDataset] Loaded {len(self.valid_images)} images with annotations')
        print(f'[MedicalDetectionDataset] Categories: {list(self.categories.values())}')
        print(f'[MedicalDetectionDataset] Total annotations: {len(self.annotations)}')

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        # Get image info
        img_info = self.valid_images[idx]
        img_id = img_info['id']

        # Load image
        img_path = os.path.join(self.img_prefix, img_info['file_name'])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'Error loading image {img_path}: {e}')
            # Return a dummy sample
            return self._get_dummy_sample()

        # Get annotations for this image
        anns = self.img_to_anns[img_id]

        # Extract boxes and labels
        boxes = []
        labels = []
        for ann in anns[:self.max_objects]:  # Limit to max_objects
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox

            # Convert to xyxy format
            box = [x, y, x + w, y + h]
            boxes.append(box)

            # Get category name
            cat_id = ann['category_id']
            cat_name = self.categories[cat_id]
            labels.append(cat_name)

        # Create conversation
        question = self._generate_question(img_info)
        answer = self._generate_answer(boxes, labels, img_info)

        conversations = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer}
        ]

        return {
            'image': image,
            'conversations': conversations,
            'boxes': boxes,
            'labels': labels,
            'modality': img_info.get('modality', self.modality),
            'patient_id': img_info.get('patient_id', 'unknown'),
            'image_id': img_id,
        }

    def _generate_question(self, img_info):
        """Generate question based on modality and task"""
        modality = img_info.get('modality', self.modality)

        questions = [
            f"<image>\nDetect all abnormalities in this {modality} image.",
            f"<image>\nIdentify and locate all lesions in this {modality} scan.",
            f"<image>\nWhat abnormal findings do you see in this {modality} image?",
            f"<image>\nLocate all regions of interest in this {modality} scan.",
        ]

        # Use hash of image_id to get consistent question for same image
        img_id = img_info['id']
        return questions[img_id % len(questions)]

    def _generate_answer(self, boxes, labels, img_info):
        """Generate answer with detected objects"""
        if len(boxes) == 0:
            return "No abnormalities detected."

        # Format: "Found N abnormalities: lesion at [x1,y1,x2,y2], tumor at [x1,y1,x2,y2]"
        findings = []
        for label, box in zip(labels, boxes):
            box_str = f"[{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]"
            findings.append(f"{label} at {box_str}")

        num_findings = len(findings)
        findings_text = ", ".join(findings)

        return f"Found {num_findings} finding(s): {findings_text}."

    def _get_dummy_sample(self):
        """Return dummy sample in case of error"""
        from PIL import Image
        import numpy as np

        # Create black image
        dummy_image = Image.fromarray(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        return {
            'image': dummy_image,
            'conversations': [
                {"from": "human", "value": "<image>\nDetect all abnormalities."},
                {"from": "gpt", "value": "Error loading image."}
            ],
            'boxes': [],
            'labels': [],
            'modality': self.modality,
            'patient_id': 'error',
            'image_id': -1,
        }


@DATASETS.register_module()
class MedicalDetectionJSONLDataset(Dataset):
    """
    Medical detection dataset using JSONL format (one JSON object per line).
    More flexible than COCO for rapid prototyping.

    JSONL format:
    {"image": "path/to/img.png", "boxes": [[x1,y1,x2,y2], ...], "labels": ["nodule", ...], ...}

    Args:
        data_file (str): Path to JSONL file
        image_root (str): Root directory for images
        modality (str): Default modality if not in data
        image_size (int): Target image size
    """

    def __init__(
        self,
        data_file,
        image_root='',
        modality='CT',
        image_size=448,
        **kwargs
    ):
        super().__init__()

        self.data_file = data_file
        self.image_root = image_root
        self.modality = modality
        self.image_size = image_size

        # Load data
        print(f'[MedicalDetectionJSONLDataset] Loading from {data_file}')
        self.samples = []
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f'[MedicalDetectionJSONLDataset] Loaded {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img_path = os.path.join(self.image_root, sample['image'])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'Error loading image {img_path}: {e}')
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

        return {
            'image': image,
            'conversations': sample.get('conversations', []),
            'boxes': sample.get('boxes', []),
            'labels': sample.get('labels', []),
            'modality': sample.get('modality', self.modality),
            'patient_id': sample.get('patient_id', 'unknown'),
        }
