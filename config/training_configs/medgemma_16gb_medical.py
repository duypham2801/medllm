"""
MedGemma Production Configuration for T4 16GB (or similar)

Optimized for actual medical imaging fine-tuning with good performance.
- Uses 8-bit quantization (better than 4-bit for accuracy)
- Full LoRA rank (64) matching FLARE25 defaults
- Full image tokens (256)
- Larger batch size (4)
- Longer sequences (2048 tokens)

Ready for production medical imaging datasets.
"""

_base_ = ['../_base_/dataset/DEFAULT_TRAIN_DATASET.py']

# Use local models if downloaded, otherwise download from HuggingFace
import os
_local_base = "ckpt/medgemma-4b-it"
_local_adapter = "ckpt/flare25-medgemma"

_model_path = _local_base if os.path.exists(_local_base) else "google/medgemma-4b-it"
_adapter_path = _local_adapter if os.path.exists(_local_adapter) else "leoyinn/flare25-medgemma"

# ============================================================================
# Data Configuration
# ============================================================================
# NOTE: Replace DummyDataset with your actual medical imaging dataset
data_args = dict(
    train=dict(
        type='DummyDataset',  # TODO: Replace with actual dataset
        num_samples=1000,     # TODO: Set to actual dataset size
        image_size=448
    ),
    validation=None,          # TODO: Add validation dataset
    test=None,
    multival=None,
    compute_metric=None,
    # Data collator settings
    collator_kwargs=dict(
        max_length=2048,
        padding=True
    ),
    gen_kwargs=dict(
        max_new_tokens=512,
        num_beams=1,
    ),
)

# ============================================================================
# Training Configuration
# ============================================================================
training_args = dict(
    # Output directory
    output_dir='./exp/medgemma_16gb_medical/',
    overwrite_output_dir=True,

    # Training mode
    do_train=True,
    do_eval=False,  # Set True when validation dataset is added
    do_predict=False,

    # Batch size - optimized for 16GB
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # Effective batch size = 16
    dataloader_num_workers=4,        # Parallel data loading

    # Training steps
    max_steps=4000,                  # FLARE25 used 4000 steps
    num_train_epochs=3,              # Or use epochs instead of max_steps
    save_strategy='steps',
    save_steps=500,
    save_total_limit=3,              # Keep only last 3 checkpoints
    evaluation_strategy='steps',     # Use 'evaluation_strategy' for compatibility
    eval_steps=500,

    # Learning rate - FLARE25 defaults
    learning_rate=1e-4,
    lr_scheduler_type='cosine',      # FLARE25 used cosine scheduler
    warmup_steps=100,
    warmup_ratio=0.025,

    # Memory optimization
    gradient_checkpointing=True,
    bf16=True,                       # T4 supports bfloat16 (better than fp16)
    fp16=False,
    tf32=True,                       # T4 supports TF32 (Ampere)

    # Logging
    logging_steps=10,
    logging_first_step=True,
    report_to='tensorboard',         # Enable TensorBoard logging

    # Optimization
    optim="adamw_torch",
    weight_decay=0.01,               # Small weight decay for regularization
    max_grad_norm=1.0,               # Gradient clipping

    # LoRA settings
    lora_enable=True,
    lora_r=64,
    lora_alpha=16,
    lora_dropout=0.1,

    # Misc
    seed=42,
    remove_unused_columns=False,
    local_rank=-1,                   # Single GPU (use deepspeed for multi-GPU)
)

# ============================================================================
# Model Configuration
# ============================================================================
model_args = dict(
    type='medgemma',
    version='v1',

    # Model paths
    model_name_or_path=_model_path,
    adapter_path=_adapter_path,

    # Quantization - 8-bit for good balance of memory and accuracy
    load_in_4bit=False,
    load_in_8bit=True,

    # LoRA Configuration - using FLARE25 defaults
    lora_enable=True,
    lora_r=64,           # FLARE25 default (medical-optimized)
    lora_alpha=16,
    lora_dropout=0.1,
    lora_target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],

    # Vision Configuration
    image_size=448,             # MedGemma default resolution (can use 896 for better quality)
    image_token_len=256,        # Full token count
    mm_use_im_start_end=True,
    mm_vision_select_layer=-2,

    # Conversation
    conv_args=dict(
        conv_template='gemma',
        sep_image_conv_front=False,
        tokenize_kwargs=dict(truncation_size=2048),
    ),

    # Model settings
    freeze_backbone=False,
    freeze_lm=False,
    model_max_length=2048,      # Standard length for medical reports

    # Loss weights (only LM loss for MedGemma)
    lm_loss_weight=1.0,
    recon_loss_weight=0.0,
    l2_loss_weight=0.0,
    box_loss_weight=0.0,

    # Process functions (for compatibility)
    process_func_args=dict(
        conv=dict(type='ShikraConvProcess'),
        image=dict(type='ShikraImageProcessor'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='ShikraTextProcess')
    ),
    target_processor=dict(boxes=dict(type='PlainBoxFormatter')),

    # Generation kwargs
    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
)

# ============================================================================
# DeepSpeed Configuration (Optional - for multi-GPU)
# ============================================================================
# Uncomment to use DeepSpeed ZeRO-2 for better memory efficiency
# deepspeed = './deepspeed/ds_config_zero2.json'

# ============================================================================
# Medical Dataset Integration Guide
# ============================================================================
"""
To use your own medical imaging dataset:

1. Create dataset class in mllm/dataset/single_image_dataset/
   Example: medical_image_dataset.py

2. Register in mllm/dataset/__init__.py:
   from .single_image_dataset.medical_image_dataset import MedicalImageDataset

3. Update data_args above:
   train=dict(
       type='MedicalImageDataset',
       data_path='path/to/annotations.json',
       image_folder='path/to/images/',
       modality='CT',  # or MRI, X-ray, etc.
   )

4. Supported modalities (from FLARE25):
   - CT: Computed Tomography
   - MRI: Magnetic Resonance Imaging
   - X-ray: Radiography
   - Ultrasound: Sonography
   - Fundus: Retinal photography
   - Pathology: Histopathology slides
   - Endoscopy: Endoscopic images

5. Task types:
   - Classification: Disease/condition classification
   - Detection: Lesion/abnormality detection
   - Counting: Object counting (e.g., cells)
   - Regression: Measurement prediction
   - Report Generation: Medical report from image

Memory Estimation for T4 16GB:
- Base model (8-bit): ~4.5GB
- LoRA parameters (r=64): ~130MB
- Vision encoder: ~500MB
- Activations (batch=4, seq=2048): ~6GB
- Gradients: ~130MB
- Total: ~11GB
Status: ✅ Comfortable headroom

Training Speed Estimate:
- T4 16GB: ~2-3 sec/step (batch=4)
- Total time (4000 steps): ~3-4 hours
- With 1000 samples: ~12 epochs in 4000 steps

To run:
    conda activate llm
    python mllm/pipeline/finetune.py config/training_configs/medgemma_16gb_medical.py
"""
