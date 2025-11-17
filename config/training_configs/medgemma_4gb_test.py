"""
MedGemma Test Configuration for GTX 1650 4GB GPU

⚠️ WARNING: This config is HEAVILY optimized for 4GB VRAM
- Uses 4-bit quantization (NF4)
- Minimal batch size (1)
- Reduced LoRA rank (16 instead of 64)
- Smaller image tokens (128 instead of 256)
- Short sequences (512 tokens max)
- Dummy dataset for quick testing

May still encounter OOM - if so, try:
1. Reduce image_token_len to 64
2. Reduce max_length to 256
3. Use CPU training (very slow)
"""

_base_ = ['../_base_/dataset/DEFAULT_TRAIN_DATASET.py']

# Use local models if downloaded, otherwise download from HuggingFace
import os

# Get project root - mmengine runs from project root, so just use cwd
_project_root = os.getcwd()
_local_base = os.path.join(_project_root, "ckpt/medgemma-4b-it")
_local_adapter = os.path.join(_project_root, "ckpt/flare25-medgemma")

print(f"[DEBUG] Project root: {_project_root}")
print(f"[DEBUG] Checking for local model at: {_local_base}")
print(f"[DEBUG] Exists: {os.path.exists(_local_base)}")

_model_path = _local_base if os.path.exists(_local_base) else "google/medgemma-4b-it"
_adapter_path = _local_adapter if os.path.exists(_local_adapter) else "leoyinn/flare25-medgemma"

print(f"[INFO] Using model: {_model_path}")
print(f"[INFO] Using adapter: {_adapter_path}")

# ============================================================================
# Data Configuration
# ============================================================================
data_args = dict(
    train=dict(
        type='DummyDataset',
        num_samples=10,  # Just 10 samples for quick test
        image_size=448
    ),
    validation=None,
    test=None,
    multival=None,
    compute_metric=None,
    collator_kwargs=dict(
        max_length=512,
        padding=True
    ),
    gen_kwargs=dict(
        max_new_tokens=128,
        num_beams=1,
    ),
)

# ============================================================================
# Training Configuration
# ============================================================================
training_args = dict(
    output_dir='./exp/medgemma_4gb_test/',
    overwrite_output_dir=True,
    report_to='none',

    # Training mode
    do_train=True,
    do_eval=False,
    do_predict=False,

    # Batch size - CRITICAL for 4GB
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    dataloader_num_workers=0,

    # Training steps
    max_steps=3,
    num_train_epochs=1,

    # Learning rate
    learning_rate=2e-4,
    lr_scheduler_type='constant',
    warmup_steps=0,

    # Checkpointing
    save_strategy='no',
    eval_strategy='no',           # transformers >= 4.57 uses 'eval_strategy'
    logging_steps=1,
    logging_first_step=True,

    # Memory optimization - CRITICAL
    gradient_checkpointing=True,
    fp16=True,
    bf16=False,
    tf32=False,

    # LoRA
    lora_enable=True,
    lora_r=16,
    lora_alpha=16,
    lora_dropout=0.1,

    # Optimization
    optim="adamw_torch",
    weight_decay=0.0,

    # Misc
    seed=42,
    remove_unused_columns=False,
    local_rank=-1,
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

    # Quantization - 4-bit for maximum memory savings
    load_in_4bit=True,
    load_in_8bit=False,

    # LoRA Configuration - reduced from FLARE25 defaults
    lora_enable=True,
    lora_r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    lora_target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],

    # Vision Configuration
    image_size=448,
    image_token_len=128,  # Reduced from 256
    mm_use_im_start_end=True,
    mm_vision_select_layer=-2,

    # Conversation
    conv_args=dict(
        conv_template='gemma',
        sep_image_conv_front=False,
        tokenize_kwargs=dict(truncation_size=512),
    ),

    # Model settings
    freeze_backbone=False,
    freeze_lm=False,
    model_max_length=512,

    # Loss weights
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

"""
Memory Estimation for GTX 1650 4GB:
- Base model (4-bit): ~2.5GB
- LoRA parameters (r=16): ~50MB
- Activations (batch=1, seq=512): ~800MB
- Gradients: ~50MB
- Total: ~3.4GB
Status: Should fit, but very close to limit

To run:
    conda activate llm
    python mllm/pipeline/finetune.py config/training_configs/medgemma_4gb_test.py
"""
