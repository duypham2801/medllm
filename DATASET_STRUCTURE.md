# Dataset Structure Guide for Detection & Segmentation

**For**: Medical imaging detection and segmentation tasks with MedGemma/PerceptionGPT

---

## Overview

This guide covers dataset structure for:
1. **Object Detection** - Bounding boxes for lesions, tumors, organs
2. **Instance Segmentation** - Pixel-level masks for regions of interest
3. **Semantic Segmentation** - Full image segmentation
4. **Detection + Segmentation** - Combined tasks with both boxes and masks

Compatible with both **MedGemma** and **PerceptionGPT** models.

---

## Quick Reference

### Annotation Format

| Task | Format | File Type | Example |
|------|--------|-----------|---------|
| **Detection** | COCO JSON | `.json` | Bounding boxes with class labels |
| **Segmentation** | COCO + Masks | `.json` + `.png` | Polygons or RLE masks |
| **Mixed** | Hybrid | `.jsonl` | Both boxes and masks per image |
| **YOLO → MedGemma** | Conversion | `.jsonl` + `.png` | Convert from YOLO segmentation |

---

## YOLO Segmentation → MedGemma Conversion

### Input Structure (YOLO Format)
```
dataset/
├── 2_VTQ/                    # Viêm thực quản (Esophagitis)
│   ├── train/
│   │   ├── images/           # Original images
│   │   │   ├── 10033.jpg
│   │   │   └── ...
│   │   ├── labels/           # YOLO format .txt files
│   │   │   ├── 10033.txt     # Class_id x1 y1 x2 y2 ... xn yn
│   │   │   └── ...
│   │   └── masks/            # Optional existing masks
│   │       └── ...
│   ├── val/
│   └── test/
├── 3_VDDHPA/                 # Viêm dạ dày HP âm (Gastritis HP Negative)
├── 4_VDDHPD/                 # Viêm dạ dày HP dương (Gastritis HP Positive)
├── 5_UTTQ/                   # Ung thư thực quản (Esophageal Cancer)
├── 6_UTDD/                   # Ung thư dạ dày (Stomach Cancer)
└── 7_LHTT/                   # Loét hoành tá tràng (Esophageal Ulcer)
```

### YOLO Label Format
```
# 10033.txt
0 0.577 0.529 0.620 0.611 0.752 0.629 ...  # Polygon points (normalized)
```

### Converted Output (MedGemma Format)
```
data_medgemma/
├── medical_detection_segmentation_all.jsonl  # Master dataset
├── 2_VTQ_train.jsonl                         # Per-category split
├── 2_VTQ_val.jsonl
├── 2_VTQ_test.jsonl
└── masks/
    ├── 2_VTQ/
    │   ├── train/
    │   │   ├── 10033_mask_0.png              # Generated masks
    │   │   └── ...
    │   └── val/
    └── ...
```

### JSONL Format (Detection + Segmentation)
```json
{
  "image": "2_VTQ/train/images/10033.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nDetect inflammation indicating Viem_thuc_quan in this endoscopy."
    },
    {
      "from": "gpt",
      "value": "The Viem_thuc_quan is located at <obj_vis_s>[0.577,0.529,0.620,0.611]<obj_vis_e>."
    }
  ],
  "boxes": [[146.7, 118.1, 158.0, 139.6]],          // [x, y, width, height]
  "formatted_boxes": ["[0.577,0.529,0.620,0.611]"], // Normalized [x1,y1,x2,y2]
  "labels": ["Viem_thuc_quan"],
  "masks": ["masks/2_VTQ/train/10033_mask_0.png"], // Mask files
  "category": "Viem_thuc_quan",
  "modality": "Endoscopy",
  "dataset_split": "train",
  "original_category": "2_VTQ"
}
```

### Conversion Command
```bash
# Convert all categories
python scripts/convert_yolo_to_medgemma.py \
    --dataset_root dataset \
    --output_root data_medgemma

# Convert specific categories only
python scripts/convert_yolo_to_medgemma.py \
    --dataset_root dataset \
    --output_root data_medgemma \
    --categories 2_VTQ 3_VDDHPA

# One-click setup
bash scripts/setup_medgemma_detection_segmentation.sh 4gb  # or 16gb
```

---

## Directory Structure

### Recommended Layout

```
your_medical_dataset/
├── images/
│   ├── train/
│   │   ├── patient_001_ct_001.png
│   │   ├── patient_001_ct_002.png
│   │   ├── patient_002_mri_001.png
│   │   └── ...
│   ├── val/
│   │   ├── patient_050_ct_001.png
│   │   └── ...
│   └── test/
│       └── ...
│
├── annotations/
│   ├── train_detection.json          # COCO format for detection
│   ├── train_segmentation.json       # COCO format for segmentation
│   ├── val_detection.json
│   ├── val_segmentation.json
│   └── metadata.json                 # Dataset metadata
│
├── masks/                            # Optional: pre-rendered masks
│   ├── train/
│   │   ├── patient_001_ct_001_mask.png
│   │   └── ...
│   └── val/
│
└── dataset_config.yaml               # Dataset configuration
```

---

## 1. Detection Format (COCO-style)

### Annotation JSON Structure

```json
{
  "info": {
    "description": "Lung Nodule Detection Dataset",
    "version": "1.0",
    "year": 2024,
    "date_created": "2024-11-16"
  },

  "images": [
    {
      "id": 1,
      "file_name": "patient_001_ct_001.png",
      "width": 512,
      "height": 512,
      "modality": "CT",
      "patient_id": "patient_001",
      "slice_idx": 1
    },
    {
      "id": 2,
      "file_name": "patient_001_ct_002.png",
      "width": 512,
      "height": 512,
      "modality": "CT",
      "patient_id": "patient_001",
      "slice_idx": 2
    }
  ],

  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [120, 150, 80, 75],  // [x, y, width, height]
      "area": 6000,
      "iscrowd": 0,
      "confidence": 0.95,           // Optional: annotation confidence
      "severity": "moderate"        // Optional: medical metadata
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 2,
      "bbox": [300, 200, 50, 50],
      "area": 2500,
      "iscrowd": 0
    }
  ],

  "categories": [
    {
      "id": 1,
      "name": "nodule",
      "supercategory": "lesion",
      "color": [255, 0, 0]           // RGB color for visualization
    },
    {
      "id": 2,
      "name": "mass",
      "supercategory": "lesion",
      "color": [0, 255, 0]
    },
    {
      "id": 3,
      "name": "calcification",
      "supercategory": "finding",
      "color": [0, 0, 255]
    }
  ]
}
```

### Key Fields Explained

- **bbox**: `[x, y, width, height]` in pixels (top-left corner)
- **area**: Bounding box area (for sorting/filtering)
- **iscrowd**: 0 for single object, 1 for crowd/multiple objects
- **category_id**: Links to categories list
- **modality**: Medical imaging modality (CT, MRI, X-ray, etc.)

---

## 2. Segmentation Format (COCO + Masks)

### Annotation JSON Structure

```json
{
  "info": { /* same as detection */ },
  "images": [ /* same as detection */ ],

  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,

      // Option 1: Polygon format (preferred for irregular shapes)
      "segmentation": [
        [120, 150, 140, 145, 160, 150, 180, 160, 190, 180,
         185, 200, 170, 210, 150, 208, 130, 200, 120, 180]
      ],

      "bbox": [120, 145, 70, 65],  // Bounding box around mask
      "area": 3500,
      "iscrowd": 0
    },

    {
      "id": 2,
      "image_id": 1,
      "category_id": 2,

      // Option 2: RLE (Run-Length Encoding) for dense masks
      "segmentation": {
        "counts": [272, 2, 4, 4, 1, 9, ...],
        "size": [512, 512]
      },

      "bbox": [300, 200, 50, 50],
      "area": 2000,
      "iscrowd": 0
    }
  ],

  "categories": [ /* same as detection */ ]
}
```

### Segmentation Formats

#### A. Polygon Format (Recommended)
```python
# List of [x1, y1, x2, y2, ..., xn, yn] coordinates
"segmentation": [
    [x1, y1, x2, y2, x3, y3, ..., xn, yn]
]

# Multiple polygons for disconnected regions
"segmentation": [
    [x1, y1, ..., xn, yn],  # First region
    [x1, y1, ..., xm, ym]   # Second region
]
```

#### B. RLE Format (For Dense Masks)
```python
from pycocotools import mask as mask_utils

# Convert binary mask to RLE
rle = mask_utils.encode(np.asfortranarray(binary_mask))

"segmentation": {
    "counts": rle['counts'],
    "size": [height, width]
}
```

#### C. External Mask Files
```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "mask_file": "masks/train/patient_001_ct_001_mask.png",
  "bbox": [120, 150, 80, 75]
}
```

Mask file format:
- **PNG/TIFF**: Single-channel, pixel value = class ID
- **Multi-class**: Each pixel value represents category_id
- **Binary**: 0 = background, 255 = foreground

---

## 3. JSONL Format (PerceptionGPT Native)

### Line-by-Line Format

Each line is a JSON object for one image:

```jsonl
{"image": "images/train/patient_001_ct_001.png", "conversations": [{"from": "human", "value": "Detect all nodules in this CT scan."}, {"from": "gpt", "value": "There are 2 nodules: <nodule> at [120, 150, 200, 225], <mass> at [300, 200, 350, 250]."}], "boxes": [[120, 150, 200, 225], [300, 200, 350, 250]], "labels": ["nodule", "mass"], "modality": "CT"}

{"image": "images/train/patient_002_mri_001.png", "conversations": [{"from": "human", "value": "Segment the tumor region."}, {"from": "gpt", "value": "The tumor is located at <region>."}], "mask": "masks/train/patient_002_mri_001_mask.png", "category": "tumor", "modality": "MRI"}
```

### JSONL Field Definitions

```python
{
    # Required
    "image": "path/to/image.png",           # Relative to dataset root
    "conversations": [                       # Conversation format
        {
            "from": "human",
            "value": "What abnormalities do you see?"
        },
        {
            "from": "gpt",
            "value": "I see a nodule at <box>."
        }
    ],

    # Detection fields
    "boxes": [[x1, y1, x2, y2], ...],       # Bounding boxes (xyxy format)
    "boxes_xywh": [[x, y, w, h], ...],      # Alternative: xywh format
    "labels": ["nodule", "mass"],            # Class labels for each box

    # Segmentation fields
    "mask": "path/to/mask.png",              # Path to mask file
    "mask_rle": {"counts": [...], "size": [h, w]},  # Or RLE encoding
    "polygon": [[[x1,y1,...]], ...],         # Or polygon coordinates

    # Metadata
    "modality": "CT",                        # Imaging modality
    "patient_id": "patient_001",
    "slice_idx": 1,
    "category": "lesion",
    "severity": "moderate",

    # Optional
    "text": "Free-form description",
    "qa_pairs": [{"q": "...", "a": "..."}]
}
```

---

## 4. Example Datasets

### A. Detection: Lung Nodule Dataset

```python
# File: data/lung_nodules_train.jsonl

{"image": "images/ct_001.png", "conversations": [{"from": "human", "value": "<image>\nDetect all lung nodules."}, {"from": "gpt", "value": "Found 2 nodules: nodule at [150, 200, 180, 230], nodule at [320, 150, 340, 170]."}], "boxes": [[150, 200, 180, 230], [320, 150, 340, 170]], "labels": ["nodule", "nodule"], "modality": "CT", "patient_id": "P001"}

{"image": "images/ct_002.png", "conversations": [{"from": "human", "value": "<image>\nDetect all lung nodules."}, {"from": "gpt", "value": "Found 1 mass: mass at [200, 250, 250, 300]."}], "boxes": [[200, 250, 250, 300]], "labels": ["mass"], "modality": "CT", "patient_id": "P002"}
```

### B. Segmentation: Tumor Segmentation

```python
# File: data/tumor_seg_train.jsonl

{"image": "images/mri_001.png", "conversations": [{"from": "human", "value": "<image>\nSegment the tumor region."}, {"from": "gpt", "value": "The tumor region is segmented."}], "mask": "masks/mri_001_tumor.png", "category": "glioblastoma", "modality": "MRI", "patient_id": "P001", "severity": "grade_4"}

{"image": "images/mri_002.png", "conversations": [{"from": "human", "value": "<image>\nSegment the tumor region."}, {"from": "gpt", "value": "The tumor region is segmented."}], "polygon": [[[120, 150, 140, 145, 160, 150, 180, 160, 190, 180, 185, 200, 170, 210, 150, 208, 130, 200, 120, 180]]], "category": "meningioma", "modality": "MRI", "patient_id": "P002"}
```

### C. Mixed: Detection + Segmentation

```python
# File: data/mixed_train.jsonl

{"image": "images/ct_001.png", "conversations": [{"from": "human", "value": "<image>\nDetect and segment all lesions."}, {"from": "gpt", "value": "Found 2 lesions with segmentation masks."}], "boxes": [[150, 200, 230, 280], [320, 150, 380, 210]], "labels": ["nodule", "mass"], "masks": ["masks/ct_001_nodule.png", "masks/ct_001_mass.png"], "modality": "CT"}
```

---

## 5. Medical Imaging Specific Fields

### Modality-Specific Metadata

```json
{
  // CT Scan
  "modality": "CT",
  "window_center": 40,
  "window_width": 400,
  "slice_thickness": 1.5,
  "kvp": 120,
  "exposure": 200
}

{
  // MRI
  "modality": "MRI",
  "sequence": "T1-weighted",
  "te": 10,              // Echo time (ms)
  "tr": 500,             // Repetition time (ms)
  "flip_angle": 90
}

{
  // X-ray
  "modality": "X-ray",
  "view": "PA",          // PA (Posterior-Anterior) or AP
  "body_part": "chest",
  "kvp": 120,
  "mas": 3.2
}

{
  // Pathology
  "modality": "Pathology",
  "stain": "H&E",        // Hematoxylin & Eosin
  "magnification": "40x",
  "tissue_type": "lung"
}
```

### Clinical Annotations

```json
{
  "clinical_info": {
    "diagnosis": "Non-small cell lung cancer",
    "stage": "T2N1M0",
    "grade": "moderately differentiated",
    "size_mm": 25,
    "location": "right upper lobe",
    "characteristics": ["spiculated", "irregular_border"]
  },

  "radiologist_annotations": {
    "confidence": 0.95,
    "difficulty": "moderate",
    "review_time_sec": 45,
    "annotator_id": "radiologist_001",
    "date": "2024-11-16"
  }
}
```

---

## 6. Dataset Configuration File

### dataset_config.yaml

```yaml
name: "Medical Imaging Detection & Segmentation Dataset"
version: "1.0"
description: "Multi-modal medical imaging dataset for lesion detection and segmentation"

# Dataset splits
splits:
  train:
    images: "images/train/"
    annotations: "annotations/train_coco.json"
    masks: "masks/train/"
    size: 5000

  val:
    images: "images/val/"
    annotations: "annotations/val_coco.json"
    masks: "masks/val/"
    size: 1000

  test:
    images: "images/test/"
    annotations: "annotations/test_coco.json"
    masks: "masks/test/"
    size: 1500

# Modalities
modalities:
  - CT
  - MRI
  - X-ray
  - Ultrasound
  - Pathology

# Categories (for detection/segmentation)
categories:
  - id: 1
    name: "nodule"
    supercategory: "lesion"
    color: [255, 0, 0]

  - id: 2
    name: "mass"
    supercategory: "lesion"
    color: [0, 255, 0]

  - id: 3
    name: "tumor"
    supercategory: "lesion"
    color: [0, 0, 255]

  - id: 4
    name: "calcification"
    supercategory: "finding"
    color: [255, 255, 0]

# Image preprocessing
image_config:
  target_size: [448, 448]    # For MedGemma
  normalize: true
  normalization_method: "hounsfield"  # For CT
  augmentation:
    - "random_flip"
    - "random_rotation"
    - "random_brightness"

# Task types
tasks:
  - "detection"
  - "segmentation"
  - "classification"
  - "report_generation"
```

---

## 7. Validation Script

### validate_dataset.py

```python
import json
from pathlib import Path
from PIL import Image

def validate_coco_dataset(annotation_file, image_dir):
    """
    Validate COCO format dataset
    """
    with open(annotation_file) as f:
        data = json.load(f)

    errors = []

    # Check required keys
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")

    # Validate images
    image_ids = set()
    for img in data['images']:
        # Check file exists
        img_path = Path(image_dir) / img['file_name']
        if not img_path.exists():
            errors.append(f"Image not found: {img_path}")

        # Check dimensions
        try:
            with Image.open(img_path) as pil_img:
                if (pil_img.width, pil_img.height) != (img['width'], img['height']):
                    errors.append(f"Size mismatch for {img['file_name']}")
        except Exception as e:
            errors.append(f"Cannot open {img_path}: {e}")

        image_ids.add(img['id'])

    # Validate annotations
    category_ids = {cat['id'] for cat in data['categories']}

    for ann in data['annotations']:
        # Check image_id exists
        if ann['image_id'] not in image_ids:
            errors.append(f"Invalid image_id in annotation {ann['id']}")

        # Check category_id exists
        if ann['category_id'] not in category_ids:
            errors.append(f"Invalid category_id in annotation {ann['id']}")

        # Validate bbox
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                errors.append(f"Invalid bbox in annotation {ann['id']}")

    if errors:
        print(f"Found {len(errors)} errors:")
        for err in errors[:10]:  # Show first 10
            print(f"  - {err}")
    else:
        print("✓ Dataset validation passed!")

    return len(errors) == 0

# Usage
validate_coco_dataset(
    'annotations/train_detection.json',
    'images/train/'
)
```

---

## 8. Conversion Scripts

### Convert to PerceptionGPT JSONL

```python
import json
from pathlib import Path

def coco_to_jsonl(coco_file, output_file, image_root="images/train/"):
    """
    Convert COCO format to PerceptionGPT JSONL format
    """
    with open(coco_file) as f:
        coco = json.load(f)

    # Build category map
    cat_map = {cat['id']: cat['name'] for cat in coco['categories']}

    # Build image map
    img_map = {img['id']: img for img in coco['images']}

    # Group annotations by image
    img_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    # Convert to JSONL
    with open(output_file, 'w') as f:
        for img_id, anns in img_anns.items():
            img_info = img_map[img_id]

            # Extract boxes and labels
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x+w, y+h])  # Convert to xyxy
                labels.append(cat_map[ann['category_id']])

            # Create conversation
            conv = [
                {
                    "from": "human",
                    "value": f"<image>\nDetect all lesions in this {img_info.get('modality', 'medical')} image."
                },
                {
                    "from": "gpt",
                    "value": f"Found {len(boxes)} lesions: " + ", ".join(
                        f"{label} at {box}" for label, box in zip(labels, boxes)
                    )
                }
            ]

            # Create JSONL entry
            entry = {
                "image": str(Path(image_root) / img_info['file_name']),
                "conversations": conv,
                "boxes": boxes,
                "labels": labels,
                "modality": img_info.get('modality', 'unknown'),
                "patient_id": img_info.get('patient_id', 'unknown')
            }

            f.write(json.dumps(entry) + '\n')

# Usage
coco_to_jsonl(
    'annotations/train_detection.json',
    'data/train_detection.jsonl',
    'images/train/'
)
```

---

## 9. Quick Start Examples

### Example 1: Simple Detection Dataset

```bash
# Directory structure
my_dataset/
├── images/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── annotations.json  # COCO format

# Load in config
data_args = dict(
    train=dict(
        type='COCODetectionDataset',
        ann_file='my_dataset/annotations.json',
        img_prefix='my_dataset/images/',
        modality='CT'
    )
)
```

### Example 2: Segmentation Dataset

```bash
# Directory structure
my_dataset/
├── images/
├── masks/         # PNG masks
└── annotations.json

# Load in config
data_args = dict(
    train=dict(
        type='COCOSegmentationDataset',
        ann_file='my_dataset/annotations.json',
        img_prefix='my_dataset/images/',
        mask_prefix='my_dataset/masks/',
        modality='MRI'
    )
)
```

### Example 3: JSONL Format

```bash
# Single file format
my_dataset/
├── images/
├── masks/
└── train.jsonl    # All annotations in one file

# Load in config
data_args = dict(
    train=dict(
        type='JSONLDataset',
        data_file='my_dataset/train.jsonl',
        image_root='my_dataset/',
    )
)
```

---

## 10. Best Practices

### ✅ DO:
- Use consistent naming conventions
- Include modality in metadata
- Validate annotations before training
- Use external mask files for large datasets
- Include clinical metadata when available
- Version your datasets
- Split by patient (not by image) to avoid data leakage

### ❌ DON'T:
- Mix different coordinate systems (xyxy vs xywh)
- Use absolute paths in annotations
- Include test patients in training set
- Forget to normalize/preprocess medical images
- Ignore class imbalance
- Skip data augmentation

---

## Summary

### Format Recommendations

| Use Case | Recommended Format | Why |
|----------|-------------------|-----|
| **Detection only** | COCO JSON | Standard, well-supported |
| **Segmentation only** | COCO + Masks | Efficient for dense masks |
| **Both** | JSONL | Flexible, easy to extend |
| **Large scale** | COCO + External masks | Better for >10K images |
| **Rapid prototyping** | JSONL | Quick to create/modify |

### Next Steps

1. **Prepare your data** in one of the formats above
2. **Validate** using provided scripts
3. **Create dataset class** (see DATASET_CLASSES.md - coming next)
4. **Update config** with your dataset path
5. **Start training!**

---

**Last Updated**: November 16, 2025
**Related Docs**:
- `MEDGEMMA.md` - MedGemma integration guide
- `DATASET_CLASSES.md` - Example dataset implementations (next)
