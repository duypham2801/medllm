# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PerceptionGPT** is a multimodal large language model (MLLM) that fuses visual perception into LLM. The project is built on LLaVA and Shikra, focusing on visual grounding tasks including referring expression comprehension (REC), referring expression segmentation (RES), and visual question answering with spatial reasoning.

**Medical Adaptation**: This fork has been extended with **MedGemma** integration for medical imaging tasks (CT, MRI, X-ray, Ultrasound, Pathology, Endoscopy, Fundus photography) and optimizations for low-memory (4GB) GPU training.

**Paper**: [PerceptionGPT: Effectively Fusing Visual Perception into LLM](https://arxiv.org/abs/2311.06612)

## Development Commands and Workflows

### Environment Setup
```bash
# Create and activate conda environment
conda create -n perceptiongpt python=3.10
conda activate perceptiongpt

# Complete automated installation (recommended)
bash scripts/install_packages.sh

# Manual installation with pinned versions
pip install -r requirements_fixed.txt

# Verify installation
python scripts/test_setup.py
```

### Model Downloads
```bash
# Download LLaVA checkpoint for PerceptionGPT
bash scripts/download_data.sh

# Download MedGemma base model + FLARE25 adapters
bash scripts/download_medgemma.sh

# Check download status
ls -lh ckpt/
```

### Training Commands

#### PerceptionGPT (Visual Grounding)
```bash
# Multi-GPU training (8 GPUs)
bash scripts/run.sh

# Single GPU training optimized for 4GB
bash scripts/run_4gb.sh

# Direct training with specific config
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 30005 \
    mllm/pipeline/finetune.py config/training_configs/shikra3_rec3_mask_box_cls_refcoco_all.py

# Custom config training
python mllm/pipeline/finetune.py config/training_configs/your_custom_config.py
```

#### MedGemma (Medical Imaging)
```bash
# Test training on 4GB GPU (GTX 1650)
bash scripts/test_medgemma_4gb.sh

# Test training on 16GB GPU (T4, RTX)
bash scripts/test_medgemma_16gb.sh

# Full medical training (16GB GPU)
bash scripts/train_medgemma_16gb.sh

# Direct MedGemma training
python mllm/pipeline/finetune.py config/training_configs/medgemma_4gb_test.py
```

### Development and Debugging
```bash
# Test configuration loading
python scripts/test_config.py

# Run quick training test (3 steps)
bash scripts/test_training.sh

# Monitor GPU usage during development
watch -n 1 nvidia-smi

# Check training logs
tail -f exp/*/logs/training.log

# Validate dataset format
python -c "
from mllm.dataset import prepare_data
from mllm.config import prepare_args
cfg = prepare_args(['your_config.py'])
data = prepare_data(cfg.model_args, cfg.data_args)
print(f'Dataset loaded: {len(data)} samples')
"
```

### Configuration Files

Training configurations follow a hierarchical structure using Python-based configs (similar to mmengine/mmdetection):

- **Main config**: `config/training_configs/shikra3_rec3_mask_box_cls_refcoco_all.py`
- **Base configs** (inherited via `_base_`):
  - Dataset: `config/_base_/dataset/DEFAULT_TRAIN_DATASET.py`
  - Model: `config/_base_/model/shikra.py`
  - Training: `config/_base_/train/shikra_deepspeed_lora.py`

To modify training:
1. Update the config file in `config/training_configs/`
2. Key parameters:
   - `model_name_or_path`: Path to LLaVA checkpoint
   - `output_dir`: Where to save checkpoints (default: `./exp/perceptionGPT/`)
   - `lora_enable`, `lora_r`, `lora_alpha`, `lora_dropout`: LoRA settings
   - `freeze_autoencoder`, `pretrained_autoencoder`: Autoencoder settings
   - Loss weights: `lm_loss_weight`, `recon_loss_weight`, `box_loss_weight`, `l2_loss_weight`

### DeepSpeed Configuration

Three DeepSpeed configs available in `deepspeed/`:
- `ds_config_zero2.json`: ZeRO-2 (default)
- `ds_config_zero2_offload.json`: ZeRO-2 with offloading
- `ds_config_zero3.json`: ZeRO-3

Select by modifying `deepspeed` parameter in training config.

## Data Setup

### Directory Structure

Place annotation files (`.jsonl`) in the `data/` folder:
```
data/
  ├── blip_laion_cc_sbu_558k.jsonl
  ├── CAP_coco2014_train.jsonl
  ├── CWB_flickr30k_train.jsonl
  └── ...
```

Download annotations from: [Google Drive link](https://drive.google.com/file/d/1CNLu1zJKPtliQEYCZlZ8ykH00ppInnyN/view?usp=drive_link)

### Image Data

Images must be downloaded separately from official sources. Update `image_folder` paths in dataset configs at `config/_base_/dataset/DEFAULT_*_*.py`.

Example:
```python
flickr=dict(
    type='FlickrDataset',
    filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_train.jsonl',
    image_folder=r'path/to/flickr30k_images/on/your/computer',  # UPDATE THIS
    template_file=r'{{fileDirname}}/template/flickr30k.json',
),
```

## Code Architecture

### Directory Structure

```
mllm/                           # Main package
├── models/
│   ├── perceptionGPT/          # Core PerceptionGPT model implementation
│   │   ├── perceptionGPT.py    # Main model class (40KB)
│   │   ├── mask_decoder.py     # Mask decoding module
│   │   └── peft_for_shikra.py  # PEFT/LoRA integration
│   ├── autoencoder/            # Visual feature encoding/decoding
│   ├── vision_towers/          # Vision encoders (CLIP)
│   ├── enhancer/               # Visual enhancement modules (SAM, transformers)
│   └── builder/                # Model loading utilities
├── dataset/
│   ├── single_image_dataset/   # Dataset implementations (REC, RES, VQA, etc.)
│   ├── process_function/       # Data processing and formatting
│   └── utils/                  # Dataset utilities and transforms
├── engine/
│   ├── shikra.py              # ShikraTrainer (saves mm_projector separately)
│   ├── perception_trainer.py  # Custom trainer with perception losses
│   └── base_engine.py         # Base trainer class
├── conversation/              # Conversation templates and processing
├── pipeline/
│   └── finetune.py           # Main training script with DeepSpeed
└── demo/                     # Demo/inference scripts
```

### Model Type Hierarchy

The codebase uses a type-based model registry pattern:
- `type='shikra'`: Base Shikra model (in `_base_/model/shikra.py`)
- `type='perceptionGPT'`: PerceptionGPT model with autoencoder and mask prediction
- `type='medgemma'`: MedGemma model for medical imaging (Gemma 3 + SigLIP vision encoder)

**Architecture Differences**:
- **PerceptionGPT**: LLaMA/TinyLlama + CLIP ViT-L/14 (separate vision tower)
- **MedGemma**: Gemma 3 (4B) + SigLIP (integrated vision encoder) + FLARE25 medical LoRA adapters

### Dataset System

Datasets follow a plugin-based registration system:
- All datasets registered in `mllm/dataset/root.py` via `DATASETS`, `METRICS`, `TRANSFORMS`, `FUNCTIONS`
- Single-image datasets in `mllm/dataset/single_image_dataset/`
- Templates define conversation formats in `config/_base_/dataset/template/*.json`
- Process functions handle different formats (boxes, masks) in `mllm/dataset/process_function/`

Key dataset types:
- **REC**: Referring Expression Comprehension (bounding boxes)
- **RES**: Referring Expression Segmentation (masks)
- **VQA**: Visual Question Answering
- **GC**: Grounded Captioning
- **REG**: Referring Expression Generation
- **MedicalDetectionDataset**: Medical object detection (lesions, tumors, organs)
- **MedicalSegmentationDataset**: Medical image segmentation
- **DummyDataset**: Synthetic data for testing (10 samples)

### Training Flow

1. `mllm/pipeline/finetune.py` loads config and initializes DeepSpeed
2. `prepare_data()` builds datasets from config (supports `ConcatDataset` for mixing)
3. `load_pretrained()` initializes model from checkpoint
4. PEFT/LoRA applied if `lora_enable=True`
5. `prepare_trainer_collator()` creates trainer (ShikraTrainer or PerceptionTrainer)
6. Training loop with custom losses (LM loss + reconstruction + box prediction + L2)

### Key Model Components

- **Vision Tower**: CLIP ViT-L/14 (`openai/clip-vit-large-patch14`)
- **MM Projector**: Projects vision features to LLM space
- **Autoencoder**: Encodes/decodes visual features for mask prediction
- **Mask Decoder**: Generates segmentation masks from visual tokens
- **Target Processors**: Format outputs (boxes, masks) - `PlainBoxFormatter`, `UnifiedFormatter`

## Common Gotchas

1. **Config paths**: Use `{{fileDirname}}` for relative paths in configs (mmengine convention)
2. **Model checkpoint**: Must set `model_name_or_path` to valid LLaVA checkpoint (or MedGemma for medical tasks)
3. **DeepSpeed + LoRA**: When using LoRA, ShikraTrainer saves mm_projector weights separately
4. **Image token length**: Default is 256, controlled by `image_token_len` parameter (reduce to 128/64 for 4GB GPU)
5. **Conversation template**: Uses `vicuna_v1.1` by default (PerceptionGPT), `gemma` for MedGemma, defined in `conv_args`
6. **Multi-task training**: Use `ConcatDataset` type in data config to mix datasets
7. **Import errors**: `unwrap_model` moved in transformers 4.46+; use try/except fallback imports (already implemented)
8. **Vision tower initialization**: Check attribute exists before accessing via `hasattr(model, 'vision_tower')`
9. **4GB GPU limitations**: Even with optimizations (8-bit, LoRA, gradient checkpointing), may still OOM. Consider cloud GPU or CPU training
10. **MedGemma vs PerceptionGPT**: Use `type='medgemma'` for medical tasks, `type='shikra'` or `type='perceptionGPT'` for general visual grounding
11. **GPU Architecture Compatibility**:
    - **T4/Turing GPUs** (compute 7.5): Use `fp16=True, bf16=False, tf32=False` - Config: `medgemma_16gb_medical.py`
    - **RTX 3090/A100 (Ampere+)** (compute 8.0+): Can use `bf16=True, tf32=True` - Config: `medgemma_16gb_ampere.py`
    - **Error**: `ValueError: Your setup doesn't support bf16/gpu` means you're using BF16 on non-Ampere GPU
12. **Transformers Version for MedGemma**: Requires `transformers>=4.50.0` (Gemma 3 support). Older versions (4.46.3) will fail with tokenizer loading error: `Exception: data did not match any variant of untagged enum ModelWrapper`

## Medical Imaging Features

### Supported Modalities (MedGemma)
- **CT** (Computed Tomography): Tumor detection, organ segmentation
- **MRI** (Magnetic Resonance Imaging): Brain lesion detection, tissue classification
- **X-ray**: Abnormality detection, disease classification
- **Ultrasound**: Fetal monitoring, organ assessment
- **Fundus Photography**: Diabetic retinopathy, glaucoma detection
- **Pathology**: Cell counting, cancer detection
- **Endoscopy**: Polyp detection, lesion classification

### Medical Dataset Format
- **COCO JSON**: Standard detection/segmentation format with medical metadata
- **JSONL**: Line-by-line format with conversations, boxes, masks, modality
- See `DATASET_STRUCTURE.md` for detailed format specifications

### Model Download
```bash
# Download MedGemma base + FLARE25 LoRA adapters
bash scripts/download_medgemma.sh

# Download LLaVA checkpoint for PerceptionGPT
bash scripts/download_data.sh
```

## Memory Optimization for Low-Memory GPUs

### 4GB GPU Configuration
Key settings in `config/training_configs/perception_1gpu_4gb_lora.py` or `medgemma_4gb_test.py`:
```python
# Quantization
load_in_8bit=True          # 8-bit (or load_in_4bit for extreme cases)

# LoRA
lora_enable=True
lora_r=8                   # Reduced rank (16 for MedGemma)
lora_alpha=16

# Batch and sequence
per_device_train_batch_size=1
gradient_accumulation_steps=16
max_length=512             # Reduced from 1024+
image_token_len=128        # Reduced from 256

# Memory optimizations
gradient_checkpointing=True
fp16=True                  # or bf16 on Ampere+
```

### DeepSpeed ZeRO-3 with CPU Offloading
Config in `deepspeed/ds_config_zero3_offload_4gb.json`:
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  }
}
```

**Warning**: 4GB GPU training is extremely slow (5-10x slower) due to CPU offloading. Recommended minimum: 8GB GPU. Comfortable: 16GB+ GPU.

## Testing and Validation

### Environment Validation
```bash
# Comprehensive installation and setup test
python scripts/test_setup.py

# Test specific configuration loading
python scripts/test_config.py

# Quick 3-step training test (MedGemma)
bash scripts/test_medgemma_4gb.sh
```

### Model and Setup Testing
```bash
# Test PerceptionGPT model loading
python -c "
from mllm.models.perceptionGPT import PerceptionGPT
from mllm.config import prepare_args
print('✓ PerceptionGPT imports successful')
"

# Test MedGemma model loading (requires download)
python -c "
from mllm.models.medgemma import MedGemmaPerception
print('✓ MedGemma imports successful')
"

# Test dataset loading
python -c "
from mllm.dataset import prepare_data
from mllm.config import prepare_args
cfg = prepare_args(['config/training_configs/medgemma_4gb_test.py'])
data = prepare_data(cfg.model_args, cfg.data_args)
print(f'✓ Dataset loaded: {type(data).__name__}')
"
```

### Monitor Training
```bash
# GPU usage (real-time)
watch -n 0.5 nvidia-smi

# Training logs and progress
tail -f exp/*/logs/training.log

# TensorBoard for visualization
tensorboard --logdir exp/

# Check checkpoint saves
ls -la exp/*/checkpoints/
```

## Troubleshooting Common Issues

### 1. Model Type Not Recognized
**Error**: `NotImplementedError: shikra not implemented!`
**Fix**: Already fixed in `mllm/models/builder/builder.py` and `build_perceptionGPT.py`. Verify both `type='shikra'` and `type='perceptionGPT'` are supported.

### 2. Vision Tower Initialization
**Error**: `AttributeError: 'ShikraLlamaModel' object has no attribute 'vision_tower'`
**Fix**: Already fixed in `perceptionGPT.py:387-394`. Uses `hasattr()` checks before accessing.

### 3. Missing Trainer Type
**Error**: `KeyError: 'shikra'` in TYPE2TRAINER
**Fix**: Already fixed in `mllm/engine/builder.py:11`. Added `'shikra': PerceptionTrainer` mapping.

### 4. Import Errors (unwrap_model)
**Error**: `ImportError: cannot import name 'unwrap_model'`
**Fix**: Already fixed with try/except fallback in `base_engine.py`, `perception_trainer.py`, `shikra.py`.

### 5. Out of Memory
**Solutions**:
- Reduce `image_token_len` to 64
- Use 4-bit quantization (`load_in_4bit=True`)
- Freeze autoencoder (`freeze_autoencoder=True`)
- Use CPU training (very slow): `CUDA_VISIBLE_DEVICES="" python ...`
- Upgrade to cloud GPU (Google Colab Pro, Vast.ai, RunPod)

## Additional Documentation

- **MEDGEMMA.md**: Comprehensive MedGemma integration guide
- **SETUP_4GB.md**: Detailed 4GB GPU setup instructions (Vietnamese)
- **DATASET_STRUCTURE.md**: Medical dataset format specifications
- **QUICK_START.md**: Quick start guide (if exists)
- **TEST_README.md**: Testing documentation (if exists)
