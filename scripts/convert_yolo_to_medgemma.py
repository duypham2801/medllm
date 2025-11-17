#!/usr/bin/env python3
"""
Convert YOLO segmentation dataset to MedGemma format for detection + segmentation tasks

Dataset structure expected:
dataset/
├── 2_VTQ/
│   ├── train/
│   │   ├── images/
│   │   ├── labels/  # YOLO format .txt files
│   │   └── masks/   # Optional: existing mask images
│   ├── val/
│   └── test/
└── ... (other disease categories)

Output format:
- JSONL files for MedGemma training
- Support for both detection (boxes) and segmentation (masks)
- Medical modality and disease category metadata
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import shutil
from PIL import Image
import numpy as np

class YOLOToMedGemmaConverter:
    def __init__(self, dataset_root: str, output_root: str):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Disease category mapping - Vietnamese medical conditions
        self.disease_mapping = {
            '2_VTQ': 'Viem_thuc_quan',          # Viêm thực quản
            '3_VDDHPA': 'Viem_da_day_HP_am',    # Viêm dạ dày HP âm tính
            '4_VDDHPD': 'Viem_da_day_HP_duong', # Viêm dạ dày HP dương tính
            '5_UTTQ': 'Ung_thu_thuc_quan',     # Ung thư thực quản
            '6_UTDD': 'Ung_thu_da_day',        # Ung thư dạ dày
            '7_LHTT': 'Loet_hoanh_tao_trang'   # Loét hoành tá tràng
        }

        # Modality mapping based on disease categories (Endoscopy for GI conditions)
        self.modality_mapping = {
            '2_VTQ': 'Endoscopy',
            '3_VDDHPA': 'Endoscopy',
            '4_VDDHPD': 'Endoscopy',
            '5_UTTQ': 'Endoscopy',
            '6_UTDD': 'Endoscopy',
            '7_LHTT': 'Endoscopy'
        }

    def yolo_polygon_to_points(self, yolo_line: str, img_width: int, img_height: int) -> List[List[float]]:
        """Convert YOLO polygon format to list of points"""
        parts = list(map(float, yolo_line.strip().split()))
        if len(parts) < 3:  # class_id + at least 2 points (x1,y1,x2,y2)
            return []

        class_id = int(parts[0])
        points = []

        # Convert normalized coordinates back to absolute
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                x = parts[i] * img_width
                y = parts[i + 1] * img_height
                points.append([x, y])

        return points

    def polygon_to_bbox(self, points: List[List[float]]) -> List[float]:
        """Convert polygon points to bounding box [x, y, width, height]"""
        if not points:
            return []

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def create_mask_from_polygon(self, points: List[List[float]], img_width: int, img_height: int) -> np.ndarray:
        """Create binary mask from polygon points"""
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        if len(points) < 3:  # Need at least 3 points for polygon
            return mask

        # Convert points to integer coordinates
        pts = np.array(points, dtype=np.int32)

        # Fill polygon
        from PIL import Image, ImageDraw
        img = Image.new('L', (img_width, img_height), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(pts.flatten().tolist(), fill=255)

        return np.array(img)

    def load_image_info(self, image_path: Path) -> Tuple[int, int]:
        """Get image dimensions"""
        try:
            with Image.open(image_path) as img:
                return img.width, img.height
        except Exception as e:
            print(f"Warning: Could not read {image_path}: {e}")
            return 0, 0

    def convert_category(self, category_name: str, split: str) -> str:
        """Convert one disease category and split to MedGemma format"""
        category_path = self.dataset_root / category_name / split
        if not category_path.exists():
            print(f"Warning: {category_path} does not exist")
            return ""

        images_dir = category_path / "images"
        labels_dir = category_path / "labels"
        masks_dir = category_path / "masks"

        if not images_dir.exists():
            print(f"Warning: Images directory {images_dir} does not exist")
            return ""

        # Get disease name and modality
        disease_name = self.disease_mapping.get(category_name, category_name)
        modality = self.modality_mapping.get(category_name, 'Medical')

        # Prepare output
        output_samples = []
        mask_output_dir = self.output_root / "masks" / category_name / split
        mask_output_dir.mkdir(parents=True, exist_ok=True)

        # Process each image
        image_files = list(images_dir.glob("*"))  # Get all files
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

        print(f"Processing {len(image_files)} images in {category_name}/{split}")

        for img_path in image_files:
            # Get corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            # Get image dimensions
            img_width, img_height = self.load_image_info(img_path)
            if img_width == 0 or img_height == 0:
                continue

            # Read YOLO labels
            try:
                with open(label_path, 'r') as f:
                    yolo_lines = f.readlines()
            except Exception as e:
                print(f"Warning: Could not read {label_path}: {e}")
                continue

            # Process each annotation
            all_boxes = []
            all_labels = []
            all_masks = []

            for line_num, yolo_line in enumerate(yolo_lines):
                yolo_line = yolo_line.strip()
                if not yolo_line:
                    continue

                # Convert polygon to points
                points = self.yolo_polygon_to_points(yolo_line, img_width, img_height)
                if not points:
                    continue

                # Get bounding box
                bbox = self.polygon_to_bbox(points)
                if not bbox:
                    continue

                all_boxes.append(bbox)

                # Use disease name as label (you can modify this for multi-class)
                all_labels.append(disease_name)

                # Create and save mask
                mask_array = self.create_mask_from_polygon(points, img_width, img_height)
                mask_filename = f"{img_path.stem}_mask_{line_num}.png"
                mask_path = mask_output_dir / mask_filename

                # Save mask as image
                mask_img = Image.fromarray(mask_array)
                mask_img.save(mask_path)

                # Relative path for JSONL
                all_masks.append(f"masks/{category_name}/{split}/{mask_filename}")

            if not all_boxes:  # Skip if no valid annotations
                continue

            # Format boxes as [x1,y1,x2,y2] normalized to [0,1] range
            formatted_boxes = []
            for box in all_boxes:
                x, y, w, h = box
                x1, y1 = x / img_width, y / img_height
                x2, y2 = (x + w) / img_width, (y + h) / img_height
                formatted_boxes.append(f"[{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}]")

            box_str = ";".join(formatted_boxes)

            # Create conversation with PerceptionGPT/MedGemma format using <boxes> tags
            import random
            use_medical_template = random.random() > 0.5  # 50% chance

            if use_medical_template:
                # Use GI endoscopy-specific templates
                if 'Ung_thu' in disease_name:  # Cancer cases
                    medical_templates = [
                        f"<image>\nDetect the {disease_name} lesion in this endoscopy.",
                        f"<image>\nLocate cancerous tissue indicating {disease_name}.",
                        f"<image>\nIdentify tumor regions associated with {disease_name}.",
                        f"<image>\nFind malignant lesions of {disease_name} in this endoscopic image.",
                        f"<image>\nWhere are the {disease_name} tumors visible in this endoscopy?"
                    ]
                elif 'Viem' in disease_name:  # Inflammation cases
                    medical_templates = [
                        f"<image>\nDetect inflammation indicating {disease_name}.",
                        f"<image>\nLocate the {disease_name} regions in this endoscopy.",
                        f"<image>\nIdentify inflamed tissue showing {disease_name}.",
                        f"<image>\nFind areas affected by {disease_name} in this endoscopic image.",
                        f"<image>\nWhere are the signs of {disease_name} visible?"
                    ]
                elif 'Loet' in disease_name:  # Ulcer cases
                    medical_templates = [
                        f"<image>\nDetect ulcerative lesions of {disease_name}.",
                        f"<image>\nLocate ulcerated areas indicating {disease_name}.",
                        f"<image>\nIdentify ulceration in this endoscopy showing {disease_name}.",
                        f"<image>\nFind ulcer regions characteristic of {disease_name}.",
                        f"<image>\nWhere are the ulcers from {disease_name} located?"
                    ]
                else:  # General GI templates
                    medical_templates = [
                        f"<image>\nDetect the {disease_name} region in this endoscopy.",
                        f"<image>\nLocate {disease_name} abnormalities in this GI image.",
                        f"<image>\nIdentify the {disease_name} in this endoscopic scan.",
                        f"<image>\nFind the {disease_name} lesion in this medical image.",
                        f"<image>\nPoint out the {disease_name} area in this endoscopy."
                    ]
                human_msg = random.choice(medical_templates)
            else:
                # Use standard templates
                if len(formatted_boxes) == 1:
                    human_msg = f"<image>\nWhere is the {disease_name} in this {modality.lower()} image?"
                else:
                    human_msg = f"<image>\nWhere are all instances of {disease_name} in this {modality.lower()} image?"

            # Format response based on number of instances
            if len(formatted_boxes) == 1:
                gpt_msg = f"The {disease_name} is located at <obj_vis_s><boxes><obj_vis_e>."
            else:
                gpt_msg = f"Found {len(all_boxes)} instances of {disease_name} at <obj_vis_s><boxes><obj_vis_e>."

            # Replace <boxes> placeholder with actual coordinates in the final messages
            if "<boxes>" in human_msg:
                human_msg = human_msg.replace("<boxes>", box_str)
            if "<boxes>" in gpt_msg:
                gpt_msg = gpt_msg.replace("<boxes>", box_str)

            conversations = [
                {"from": "human", "value": human_msg},
                {"from": "gpt", "value": gpt_msg}
            ]

            # Create sample
            sample = {
                "image": str(img_path.relative_to(self.dataset_root)),
                "conversations": conversations,
                "boxes": all_boxes,  # Original absolute coordinates for processing
                "formatted_boxes": formatted_boxes,  # Normalized coordinates for model
                "labels": all_labels,
                "masks": all_masks,
                "category": disease_name,
                "modality": modality,
                "dataset_split": split,
                "original_category": category_name
            }

            output_samples.append(sample)

        # Save to JSONL
        output_file = self.output_root / f"{category_name}_{split}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in output_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"Saved {len(output_samples)} samples to {output_file}")
        return str(output_file)

    def convert_all(self, categories: Optional[List[str]] = None):
        """Convert all categories or specified ones"""
        if categories is None:
            categories = [d.name for d in self.dataset_root.iterdir() if d.is_dir() and d.name != '__pycache__']

        all_files = []
        for category in categories:
            if category not in self.disease_mapping:
                print(f"Warning: Unknown category {category}, adding to mapping")
                self.disease_mapping[category] = category
                self.modality_mapping[category] = 'Medical'

            for split in ['train', 'val', 'test']:
                output_file = self.convert_category(category, split)
                if output_file:
                    all_files.append(output_file)

        # Create master dataset file
        master_file = self.output_root / "medical_detection_segmentation_all.jsonl"
        all_samples = []

        for file_path in all_files:
            if file_path and Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_samples.append(json.loads(line))

        # Shuffle and save
        import random
        random.shuffle(all_samples)

        with open(master_file, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"\nConversion completed!")
        print(f"Total samples: {len(all_samples)}")
        print(f"Master dataset saved to: {master_file}")
        print(f"Masks saved to: {self.output_root}/masks/")

        # Print statistics
        from collections import Counter
        modality_counts = Counter(s['modality'] for s in all_samples)
        category_counts = Counter(s['category'] for s in all_samples)

        print(f"\nModality distribution:")
        for modality, count in modality_counts.items():
            print(f"  {modality}: {count}")

        print(f"\nCategory distribution:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO segmentation dataset to MedGemma format")
    parser.add_argument("--dataset_root", type=str, default="dataset",
                       help="Root directory of YOLO dataset")
    parser.add_argument("--output_root", type=str, default="data_medgemma",
                       help="Output directory for MedGemma dataset")
    parser.add_argument("--categories", nargs="+", default=None,
                       help="Specific categories to convert (default: all)")

    args = parser.parse_args()

    converter = YOLOToMedGemmaConverter(args.dataset_root, args.output_root)
    converter.convert_all(args.categories)

if __name__ == "__main__":
    main()