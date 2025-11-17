# Quick Start: MedGemma + Detection/Segmentation

**Status**: Downloads in progress... ⏳
- ✅ LoRA adapters downloaded (126MB)
- ⏳ Base model downloading (~8GB) - ETA: ~5-10 minutes

---

## What's Ready

### 1. ✅ MedGemma Integration
- Model classes: `MedGemmaPerception`
- Builder: `build_medgemma.py`
- Configs: 4GB test + 16GB production
- Documentation: `MEDGEMMA.md`

### 2. ✅ Dataset Classes for Medical Detection/Segmentation

Created 4 new dataset classes:

#### Detection Datasets
1. **MedicalDetectionDataset** - COCO format
2. **MedicalDetectionJSONLDataset** - JSONL format (simpler)

#### Segmentation Datasets
3. **MedicalSegmentationDataset** - COCO + masks
4. **MedicalSegmentationJSONLDataset** - JSONL + masks

### 3. ✅ Documentation
- `DATASET_STRUCTURE.md` - Complete dataset format guide
- `MEDGEMMA.md` - MedGemma integration guide
- This file - Quick start

---

## Once Download Completes

### Test Model Loading (Quick)

```bash
conda activate llm

# Test imports
python -c "
from mllm.models.medgemma import MedGemmaPerception
from mllm.models.builder.build_medgemma import load_pretrained_medgemma
print('✓ MedGemma imports successful')
"

# Test model loading
bash scripts/test_medgemma_4gb.sh
```

Expected output:
```
✓ Config loaded
✓ Model loaded: MedGemmaPerception
✓ LoRA adapters loaded
trainable params: XXX || all params: 4B || trainable%: X.XX
[DummyDataset] Created with 10 samples
Step 1/3 ...
```

---

## Dataset Examples

### Example 1: Detection (COCO Format)

**Directory Structure:**
```
my_lung_nodules/
├── images/
│   ├── ct_001.png
│   ├── ct_002.png
│   └── ...
└── annotations_coco.json
```

**Config:**
```python
data_args = dict(
    train=dict(
        type='MedicalDetectionDataset',
        ann_file='my_lung_nodules/annotations_coco.json',
        img_prefix='my_lung_nodules/images/',
        modality='CT',
        image_size=448
    ),
    ...
)
```

**COCO JSON Format:**
```json
{
  "images": [
    {"id": 1, "file_name": "ct_001.png", "width": 512, "height": 512}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [120, 150, 80, 75],  // [x, y, width, height]
      "area": 6000
    }
  ],
  "categories": [
    {"id": 1, "name": "nodule"}
  ]
}
```

### Example 2: Detection (JSONL Format - Simpler!)

**Directory Structure:**
```
my_tumors/
├── images/
│   ├── mri_001.png
│   └── ...
└── train.jsonl
```

**JSONL File (train.jsonl):**
```jsonl
{"image": "images/mri_001.png", "conversations": [{"from": "human", "value": "<image>\nDetect tumors."}, {"from": "gpt", "value": "Found 1 tumor at [200, 250, 280, 320]."}], "boxes": [[200, 250, 280, 320]], "labels": ["tumor"], "modality": "MRI"}
{"image": "images/mri_002.png", "conversations": [{"from": "human", "value": "<image>\nDetect tumors."}, {"from": "gpt", "value": "Found 2 tumors."}], "boxes": [[150, 200, 200, 250], [300, 350, 350, 400]], "labels": ["tumor", "tumor"], "modality": "MRI"}
```

**Config:**
```python
data_args = dict(
    train=dict(
        type='MedicalDetectionJSONLDataset',
        data_file='my_tumors/train.jsonl',
        image_root='my_tumors/',
        modality='MRI'
    ),
    ...
)
```

### Example 3: Segmentation

**Directory Structure:**
```
my_tumor_seg/
├── images/
│   ├── img_001.png
│   └── ...
├── masks/
│   ├── img_001_mask.png  // Binary mask: 0=background, 255=tumor
│   └── ...
└── train.jsonl
```

**JSONL File:**
```jsonl
{"image": "images/img_001.png", "mask": "masks/img_001_mask.png", "conversations": [{"from": "human", "value": "<image>\nSegment the tumor."}, {"from": "gpt", "value": "Tumor region segmented."}], "category": "glioblastoma", "modality": "MRI"}
```

**Config:**
```python
data_args = dict(
    train=dict(
        type='MedicalSegmentationJSONLDataset',
        data_file='my_tumor_seg/train.jsonl',
        image_root='my_tumor_seg/',
        modality='MRI'
    ),
    ...
)
```

---

## Full Training Pipeline

### Step 1: Prepare Your Data

Choose format based on your use case:

| Format | Best For | Complexity |
|--------|----------|------------|
| **JSONL** | Quick prototyping, <10K images | ⭐ Easy |
| **COCO** | Large datasets, standard benchmarks | ⭐⭐ Medium |
| **COCO + External Masks** | Dense segmentation, >10K images | ⭐⭐⭐ Complex |

**Recommendation**: Start with JSONL format!

### Step 2: Create Config

Copy and modify:

**For 4GB GPU (test):**
```bash
cp config/training_configs/medgemma_4gb_test.py config/training_configs/my_medical_4gb.py
```

**For 16GB GPU (production):**
```bash
cp config/training_configs/medgemma_16gb_medical.py config/training_configs/my_medical_16gb.py
```

**Edit data section:**
```python
data_args = dict(
    train=dict(
        type='MedicalDetectionJSONLDataset',  # or other dataset type
        data_file='path/to/your/train.jsonl',
        image_root='path/to/your/images/',
        modality='CT',  # or MRI, X-ray, etc.
    ),
    validation=dict(
        type='MedicalDetectionJSONLDataset',
        data_file='path/to/your/val.jsonl',
        image_root='path/to/your/images/',
        modality='CT',
    ),
    ...
)
```

### Step 3: Train

**4GB GPU:**
```bash
conda activate llm
python mllm/pipeline/finetune.py config/training_configs/my_medical_4gb.py
```

**16GB GPU:**
```bash
conda activate llm
python mllm/pipeline/finetune.py config/training_configs/my_medical_16gb.py
```

### Step 4: Monitor

```bash
# In another terminal
watch -n 1 nvidia-smi

# Or check logs
tail -f exp/medgemma_*/logs/training.log
```

---

## Quick Tests to Run After Download

### Test 1: Model Imports ✅
```bash
python -c "from mllm.models.medgemma import MedGemmaPerception; print('OK')"
```

### Test 2: Config Loading ✅
```bash
python -c "
from mllm.config import prepare_args
cfg = prepare_args(['config/training_configs/medgemma_4gb_test.py'])
print(f'Model type: {cfg.model_args.type}')
"
```

### Test 3: Dataset Loading ✅
```bash
python -c "
from mllm.dataset import MedicalDetectionJSONLDataset
dataset = MedicalDetectionJSONLDataset(
    data_file='path/to/your/data.jsonl',
    image_root='path/to/images/'
)
print(f'Loaded {len(dataset)} samples')
"
```

### Test 4: Full Training (3 steps) ✅
```bash
bash scripts/test_medgemma_4gb.sh
```

---

## Troubleshooting

### Downloads Not Working?

```bash
# Login to HuggingFace
huggingface-cli login

# Accept terms at:
# https://huggingface.co/google/medgemma-4b-it

# Retry download
bash scripts/download_medgemma.sh
```

### Out of Memory (4GB GPU)?

The 4GB config is already optimized, but if still OOM:

```python
# In your config, reduce further:
image_token_len = 64  # Instead of 128
max_length = 256      # Instead of 512
load_in_4bit = True   # Already enabled
```

Or use CPU (slow):
```bash
CUDA_VISIBLE_DEVICES="" python mllm/pipeline/finetune.py ...
```

### Dataset Not Loading?

Check:
1. File paths are correct (relative to dataset root)
2. Images exist and are readable (PIL can open them)
3. JSONL syntax is valid (each line is valid JSON)
4. Dataset class is registered (check imports in `__init__.py`)

Validate with:
```python
python DATASET_STRUCTURE.md  # See validation script
```

---

## Next Steps

1. ✅ Wait for downloads to complete (~5-10 min)
2. ✅ Test model loading: `bash scripts/test_medgemma_4gb.sh`
3. ✅ Prepare your medical dataset (see `DATASET_STRUCTURE.md`)
4. ✅ Create config for your data
5. ✅ Start training!

---

## Files Created

### Models & Builders
- `mllm/models/medgemma/medgemma_perception.py` - Main model
- `mllm/models/builder/build_medgemma.py` - Model loader

### Datasets
- `mllm/dataset/single_image_dataset/medical_detection_dataset.py`
- `mllm/dataset/single_image_dataset/medical_segmentation_dataset.py`

### Configs
- `config/training_configs/medgemma_4gb_test.py` - For GTX 1650 4GB
- `config/training_configs/medgemma_16gb_medical.py` - For T4 16GB

### Scripts
- `scripts/download_medgemma.sh` - Download models
- `scripts/test_medgemma_4gb.sh` - Quick test (4GB)
- `scripts/test_medgemma_16gb.sh` - Quick test (16GB)

### Documentation
- `MEDGEMMA.md` - Full MedGemma guide
- `DATASET_STRUCTURE.md` - Dataset formats guide
- `QUICK_START.md` - This file

---

## Support

**Questions? Issues?**
1. Check `MEDGEMMA.md` for detailed docs
2. Check `DATASET_STRUCTURE.md` for data formats
3. Review example configs in `config/training_configs/`
4. Check existing dataset classes for examples

**Common Issues:**
- Model download: Accept terms at https://huggingface.co/google/medgemma-4b-it
- OOM on 4GB: Use T4 16GB or cloud GPU
- Dataset errors: Validate format with scripts in `DATASET_STRUCTURE.md`

---

**Status**: Downloads in progress... Check back in 5-10 minutes!

Run `ls -lh ckpt/medgemma-4b-it/` to see download progress.
