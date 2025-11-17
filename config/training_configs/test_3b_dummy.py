"""
Test configuration with 3B model and minimal dummy data
For quick testing to verify code runs
"""

_base_ = [
    '../_base_/model/shikra.py',
]

# ====================
# Dummy Data Configuration - NO REAL DATASET NEEDED
# ====================
data_args = dict(
    # Use a simple test dataset that we'll create
    train=dict(
        type='DummyDataset',  # Will create this
        num_samples=10,  # Just 10 samples for testing
    ),
    validation=None,
    multival=None,
    test=None,

    compute_metric=None,

    collator_kwargs=dict(
        padding=True,
        max_length=512,
    ),

    gen_kwargs=dict(
        max_new_tokens=128,
        num_beams=1,
    ),
)

# ====================
# Training Arguments - MINIMAL FOR TESTING
# ====================
training_args = dict(
    output_dir='./exp/test_3b/',
    overwrite_output_dir=True,
    report_to='none',  # Disable reporting for testing

    # Minimal training
    do_train=True,
    do_eval=False,
    do_predict=False,

    # Batch settings
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # No accumulation for testing

    # Just 3 steps to test
    max_steps=3,
    num_train_epochs=1,

    # Learning rate
    learning_rate=2e-4,
    lr_scheduler_type='constant',
    warmup_steps=0,

    # Checkpointing - disabled for quick test
    save_strategy='no',
    evaluation_strategy='no',  # Use 'evaluation_strategy' for older transformers versions
    logging_steps=1,
    logging_first_step=True,

    # Memory optimization
    gradient_checkpointing=True,
    tf32=False,  # GTX 1650 doesn't support TF32 (Ampere+ only)
    fp16=True,

    # Simple training - NO DeepSpeed for testing
    # deepspeed=None,

    # LoRA
    lora_enable=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    dataloader_num_workers=0,  # Single worker for testing
    remove_unused_columns=False,
    seed=42,
)

# ====================
# Model Arguments - 3B MODEL
# ====================
model_args = dict(
    type="shikra",  # Use simpler shikra instead of perceptionGPT for testing
    version='v1',

    # Model paths - OPTIONS (choose one):
    # Option 1: TinyLlama 1.1B - Fastest for testing (will auto-download ~2GB)
    model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",

    # Option 2: Phi-3.5-mini 3.8B - Better quality but larger (~8GB)
    # model_name_or_path="microsoft/Phi-3.5-mini-instruct",

    # Option 3: StableLM-3B (~6GB)
    # model_name_or_path="stabilityai/stablelm-3b-4e1t",

    # Option 4: Vicuna-7B (original, ~13GB)
    # model_name_or_path="lmsys/vicuna-7b-v1.5",

    # Option 5: If you have a local model
    # model_name_or_path="ckpt/your-model-path",

    # Vision tower (lightweight)
    vision_tower=r'openai/clip-vit-base-patch32',  # Smaller CLIP variant
    mm_vision_select_layer=-2,

    # Image processing - REDUCED
    image_token_len=64,  # Reduced from 256 to save memory
    mm_use_im_start_end=True,
    sep_image_conv_front=False,

    # Freezing
    freeze_backbone=False,
    tune_mm_mlp_adapter=False,

    # LoRA for 3.8GB GPU
    lora_enable=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=['q_proj', 'v_proj'],

    # 8-bit loading
    load_in_8bit=True,

    # Target processor - simplified
    target_processor=dict(
        boxes=dict(type='PlainBoxFormatter'),
    ),

    # Conversation
    conv_args=dict(
        conv_template='vicuna_v1.1',
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=512),
    ),

    # Process functions
    process_func_args=dict(
        conv=dict(type='ShikraConvProcess'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='ShikraTextProcess'),
        image=dict(type='ShikraImageProcessor'),
    ),

    model_max_length=512,
    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
)
