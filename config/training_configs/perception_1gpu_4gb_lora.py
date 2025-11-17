"""
Optimized training configuration for PerceptionGPT on 4GB GPU
This config uses aggressive memory optimization techniques:
- LoRA with small rank (r=8)
- DeepSpeed ZeRO-3 with CPU offloading
- Gradient checkpointing
- Small batch size with gradient accumulation
- 8-bit optimization with bitsandbytes
"""

_base_ = [
    '../_base_/dataset/DEFAULT_TRAIN_DATASET.py',
    '../_base_/dataset/DEFAULT_TEST_RES_VARIANT.py',
    '../_base_/model/shikra.py',
]

# ====================
# Data Configuration
# ====================
data_args = dict(
    train=dict(
        type='ConcatDataset',
        cfgs=[
            # Use only RefCOCO for initial testing/fine-tuning
            # This is much smaller than full dataset
            {{_base_.DEFAULT_TRAIN_DATASET.rec_mask_all}}
        ],
    ),
    validation=None,
    # Disable multival to save memory during training
    multival=None,
    test=None,

    # Compute metric
    compute_metric=None,

    # Padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=512,  # Reduced from 1024 to save memory
    ),

    # Generate config
    gen_kwargs=dict(
        max_new_tokens=256,  # Reduced from 1024
        num_beams=1,  # Greedy decoding only
    ),
)

# ====================
# Training Arguments - OPTIMIZED FOR 4GB GPU
# ====================
training_args = dict(
    # Output
    output_dir='./exp/perceptionGPT_4gb/',
    overwrite_output_dir=False,
    report_to='tensorboard',

    # Training loop
    do_train=True,
    do_eval=False,  # Disable eval to save memory
    do_predict=False,

    # Batch size - CRITICAL FOR 4GB GPU
    per_device_train_batch_size=1,  # Minimum possible
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch size = 16

    # Epochs and steps
    num_train_epochs=3,  # Reduced from 10 for faster iteration
    max_steps=-1,  # Use epochs instead

    # Learning rate
    learning_rate=2e-4,  # Slightly higher for LoRA
    lr_scheduler_type='cosine',
    warmup_ratio=0.03,
    weight_decay=0.0,

    # Optimization strategy
    evaluation_strategy='no',  # Disabled
    save_strategy='steps',
    save_steps=1000,  # Save less frequently
    save_total_limit=2,  # Keep only 2 checkpoints to save disk space

    # Logging
    logging_steps=10,
    logging_first_step=True,

    # Memory optimization - CRITICAL
    gradient_checkpointing=True,  # Trade compute for memory
    tf32=True,  # Use TF32 on Ampere GPUs
    bf16=False,  # Not supported on all GPUs
    fp16=True,  # Use FP16 to save memory

    # DeepSpeed configuration - CRITICAL FOR 4GB GPU
    deepspeed="deepspeed/ds_config_zero3_offload_4gb.json",

    # LoRA configuration - CRITICAL FOR 4GB GPU
    lora_enable=True,
    lora_r=8,  # Small rank to reduce memory
    lora_alpha=16,  # alpha = 2*r is common
    lora_dropout=0.05,

    # Data loading
    dataloader_num_workers=2,  # Reduced to save memory
    dataloader_pin_memory=True,
    remove_unused_columns=False,

    # Other
    seed=42,
)

# ====================
# Model Arguments - OPTIMIZED FOR 4GB GPU
# ====================
model_args = dict(
    type="perceptionGPT",
    version='v1',

    # Model paths - MUST BE UPDATED BY USER
    # Download from: https://huggingface.co/liuhaotian/llava-v1.5-7b
    model_name_or_path="ckpt/llava-v1.5-7b",  # UPDATE THIS PATH

    # Vision tower
    vision_tower=r'openai/clip-vit-large-patch14',
    mm_vision_select_layer=-2,

    # Image processing
    image_token_len=256,
    mm_use_im_start_end=True,
    sep_image_conv_front=False,

    # Model freezing - CRITICAL FOR 4GB GPU
    freeze_backbone=False,  # We use LoRA instead
    freeze_lm=False,
    tune_mm_mlp_adapter=False,
    freeze_mm_mlp_adapter=False,

    # LoRA settings - CRITICAL FOR 4GB GPU
    lora_enable=True,
    lora_r=8,  # Match training_args
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=['q_proj', 'v_proj'],  # Only attention, saves memory

    # 8-bit optimization - CRITICAL FOR 4GB GPU
    load_in_8bit=True,  # Load base model in 8-bit

    # Autoencoder settings
    init_peft_inside=False,
    freeze_autoencoder=False,
    pretrained_autoencoder=None,

    # Loss weights
    lm_loss_weight=1.0,
    recon_loss_weight=1.0,
    box_loss_weight=1.0,
    l2_loss_weight=0.0,

    # Target processor
    target_processor=dict(
        boxes=dict(type='UnifiedFormatter'),
    ),

    # Conversation settings
    conv_args=dict(
        conv_template='vicuna_v1.1',
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=512),  # Reduced from 4096
    ),

    # Process functions
    process_func_args=dict(
        conv=dict(type='ShikraConvProcess'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='ShikraTextProcess'),
        image=dict(type='ShikraImageProcessor'),
    ),

    # Generation settings
    model_max_length=512,  # Reduced from 2048
    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
)

# ====================
# Notes for 4GB GPU Users
# ====================
"""
IMPORTANT NOTES:
1. This config is highly optimized for 4GB GPU but may still encounter OOM
2. If you still get OOM errors, try:
   - Reduce image_token_len to 128 or 64
   - Use load_in_4bit instead of load_in_8bit (requires bitsandbytes)
   - Disable autoencoder training (freeze_autoencoder=True)
   - Use CPU training (much slower but won't OOM)

3. Expected training speed on 4GB GPU:
   - Very slow due to CPU offloading (5-10x slower than normal)
   - Approximately 5-10 seconds per step with batch_size=1
   - Full epoch may take several hours to days

4. To use this config:
   bash scripts/run_4gb.sh

5. Monitor GPU memory:
   watch -n 0.5 nvidia-smi
"""
