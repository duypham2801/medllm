# Simplified MedGemma Detection + Segmentation Configuration
# Avoids circular import issues

_base_ = [
    '../_base_/model/medgemma_4b_it.py',
    '../_base_/train/shikra_deepspeed_lora.py',
]

# Model Configuration
model_args = dict(
    type='medgemma',
    model_name_or_path='./ckpt/medgemma-4b-it',
    adapter_name_or_path='./ckpt/flare25-medgemma',

    # Tokenizer
    tokenizer_name_or_path='./ckpt/medgemma-4b-it',
    model_max_length=1024,

    # Vision
    vision_tower='openai/clip-vit-large-patch14',
    image_token_len=256,
    vision_select_layer=-2,

    # Training
    lora_enable=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,

    # Memory optimization
    gradient_checkpointing=True,
    use_cache=False,

    # Medical imaging settings
    freeze_vision_tower=False,
    freeze_mm_projector=False,
    use_flare25_adapters=True,
)

# Data Configuration - Direct JSONL loading
data_args = dict(
    # Train dataset - direct path
    train=dict(
        type='json',
        data_path='data_medgemma/medical_detection_segmentation_all.jsonl',
        image_folder='dataset',
        image_size=896,
        mask_size=448,

        # Simple conversation template
        conv_template='gemma',

        # Processing
        force_image_size=True,
        image_aspect_ratio='pad',
        use_im_start_end=True,
        replace_image_token=True,

        # Data loading
        lazy_load_data=True,
        num_workers=4,
        prefetch_factor=4,
    ),

    # Validation dataset - use same data but will be split later
    validation=dict(
        type='json',
        data_path='data_medgemma/medical_detection_segmentation_all.jsonl',
        image_folder='dataset',
        image_size=896,
        mask_size=448,

        conv_template='gemma',

        force_image_size=True,
        image_aspect_ratio='pad',
        use_im_start_end=True,
        replace_image_token=True,

        lazy_load_data=True,
        num_workers=4,
        prefetch_factor=4,
    ),

    # Split ratio
    validation_split=0.1,
)

# Training Configuration
training_args = dict(
    # Output
    output_dir='./exp/medgemma_detection_seg_simple_16gb',
    logging_dir='./exp/medgemma_detection_seg_simple_16gb/logs',

    # Training schedule
    num_train_epochs=3,
    max_steps=-1,

    # Batch size
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,

    # Learning rate
    learning_rate=1e-5,
    weight_decay=0.01,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,

    # Optimization
    optim='adamw_torch',
    max_grad_norm=1.0,

    # Precision
    fp16=False,
    bf16=True,
    tf32=True,

    # Memory optimization
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,

    # Evaluation
    evaluation_strategy='epoch',
    eval_steps=1,
    save_strategy='epoch',
    save_steps=1,

    # Logging
    logging_strategy='steps',
    logging_steps=20,
    report_to='tensorboard',

    # DeepSpeed
    deepspeed='deepspeed/ds_config_zero2.json',

    # Checkpointing
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,

    # Miscellaneous
    seed=42,
    dataloader_drop_last=False,
    run_name='medgemma_detection_seg_simple_16gb',
)