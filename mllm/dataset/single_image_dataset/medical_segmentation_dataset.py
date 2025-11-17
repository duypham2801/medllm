"""
Medical Segmentation Dataset
Supports COCO format + mask files for segmentation tasks
"""

import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from mllm.dataset.root import DATASETS


@DATASETS.register_module()
class MedicalSegmentationDataset(Dataset):
    """
    Medical imaging segmentation dataset.

    Supports segmentation tasks for:
    - Tumor segmentation
    - Organ segmentation
    - Lesion segmentation
    - Multi-class tissue segmentation

    Args:
        ann_file (str): Path to COCO annotation JSON file
        img_prefix (str): Prefix path to images
        mask_prefix (str): Prefix path to mask images (optional)
        modality (str): Imaging modality (CT, MRI, X-ray, etc.)
        image_size (int): Target image size (default: 448)
        use_polygon (bool): Use polygon annotations instead of mask files
    """

    def __init__(
        self,
        ann_file,
        img_prefix='',
        mask_prefix=None,
        modality='MRI',
        image_size=448,
        use_polygon=False,
        **kwargs
    ):
        super().__init__()

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.mask_prefix = mask_prefix
        self.modality = modality
        self.image_size = image_size
        self.use_polygon = use_polygon

        # Load annotations
        print(f'[MedicalSegmentationDataset] Loading annotations from {ann_file}')
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

        # Filter images with annotations
        self.valid_images = [
            img for img in self.images
            if img['id'] in self.img_to_anns
        ]

        print(f'[MedicalSegmentationDataset] Loaded {len(self.valid_images)} images')
        print(f'[MedicalSegmentationDataset] Categories: {list(self.categories.values())}')

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
            return self._get_dummy_sample()

        # Get annotations
        anns = self.img_to_anns[img_id]

        # Process segmentation masks
        masks = []
        labels = []
        boxes = []

        for ann in anns:
            cat_id = ann['category_id']
            cat_name = self.categories[cat_id]
            labels.append(cat_name)

            # Get bounding box
            bbox = ann['bbox']
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])

            # Get mask
            if 'mask_file' in ann and self.mask_prefix:
                # Load external mask file
                mask_path = os.path.join(self.mask_prefix, ann['mask_file'])
                try:
                    mask = np.array(Image.open(mask_path))
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]  # Take first channel
                    masks.append(mask)
                except Exception as e:
                    print(f'Error loading mask {mask_path}: {e}')
                    # Create empty mask
                    masks.append(np.zeros((img_info['height'], img_info['width']), dtype=np.uint8))

            elif 'segmentation' in ann and self.use_polygon:
                # Convert polygon to mask
                mask = self._polygon_to_mask(
                    ann['segmentation'],
                    img_info['height'],
                    img_info['width']
                )
                masks.append(mask)

            else:
                # No mask available
                masks.append(np.zeros((img_info['height'], img_info['width']), dtype=np.uint8))

        # Create conversation
        question = self._generate_question(img_info, labels)
        answer = self._generate_answer(labels, img_info)

        conversations = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer}
        ]

        return {
            'image': image,
            'conversations': conversations,
            'masks': masks,  # List of numpy arrays
            'labels': labels,
            'boxes': boxes,
            'modality': img_info.get('modality', self.modality),
            'patient_id': img_info.get('patient_id', 'unknown'),
            'image_id': img_id,
        }

    def _polygon_to_mask(self, segmentation, height, width):
        """Convert polygon coordinates to binary mask"""
        from PIL import Image, ImageDraw

        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for polygon in segmentation:
            # Polygon format: [x1, y1, x2, y2, ..., xn, yn]
            poly_coords = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(poly_coords, outline=1, fill=1)

        return np.array(mask)

    def _generate_question(self, img_info, labels):
        """Generate question for segmentation task"""
        modality = img_info.get('modality', self.modality)

        if len(labels) == 1:
            label = labels[0]
            questions = [
                f"<image>\nSegment the {label} in this {modality} image.",
                f"<image>\nDelineate the {label} region in this {modality} scan.",
                f"<image>\nOutline the {label} area in this {modality} image.",
            ]
        else:
            questions = [
                f"<image>\nSegment all regions of interest in this {modality} image.",
                f"<image>\nDelineate all anatomical structures in this {modality} scan.",
                f"<image>\nIdentify and segment all findings in this {modality} image.",
            ]

        img_id = img_info['id']
        return questions[img_id % len(questions)]

    def _generate_answer(self, labels, img_info):
        """Generate answer for segmentation task"""
        if len(labels) == 0:
            return "No regions to segment."

        if len(labels) == 1:
            return f"The {labels[0]} region has been segmented."
        else:
            regions = ", ".join(labels[:-1]) + f" and {labels[-1]}"
            return f"Segmented {len(labels)} regions: {regions}."

    def _get_dummy_sample(self):
        """Return dummy sample in case of error"""
        dummy_image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        dummy_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        return {
            'image': dummy_image,
            'conversations': [
                {"from": "human", "value": "<image>\nSegment the region."},
                {"from": "gpt", "value": "Error loading data."}
            ],
            'masks': [dummy_mask],
            'labels': [],
            'boxes': [],
            'modality': self.modality,
            'patient_id': 'error',
            'image_id': -1,
        }


@DATASETS.register_module()
class MedicalSegmentationJSONLDataset(Dataset):
    """
    Medical segmentation dataset using JSONL format.

    JSONL format:
    {"image": "path/to/img.png", "mask": "path/to/mask.png", "category": "tumor", ...}

    Args:
        data_file (str): Path to JSONL file
        image_root (str): Root directory for images and masks
        modality (str): Default modality
        image_size (int): Target image size
    """

    def __init__(
        self,
        data_file,
        image_root='',
        modality='MRI',
        image_size=448,
        **kwargs
    ):
        super().__init__()

        self.data_file = data_file
        self.image_root = image_root
        self.modality = modality
        self.image_size = image_size

        # Load data
        print(f'[MedicalSegmentationJSONLDataset] Loading from {data_file}')
        self.samples = []
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f'[MedicalSegmentationJSONLDataset] Loaded {len(self.samples)} samples')

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

        # Load mask(s)
        masks = []
        if 'mask' in sample:
            # Single mask file
            mask_path = os.path.join(self.image_root, sample['mask'])
            try:
                mask = np.array(Image.open(mask_path))
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                masks.append(mask)
            except Exception as e:
                print(f'Error loading mask {mask_path}: {e}')
                masks.append(np.zeros((self.image_size, self.image_size), dtype=np.uint8))

        elif 'masks' in sample:
            # Multiple mask files
            for mask_file in sample['masks']:
                mask_path = os.path.join(self.image_root, mask_file)
                try:
                    mask = np.array(Image.open(mask_path))
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                    masks.append(mask)
                except Exception as e:
                    print(f'Error loading mask {mask_path}: {e}')

        return {
            'image': image,
            'conversations': sample.get('conversations', []),
            'masks': masks,
            'labels': [sample.get('category', 'unknown')] if 'category' in sample else sample.get('labels', []),
            'modality': sample.get('modality', self.modality),
            'patient_id': sample.get('patient_id', 'unknown'),
        }
