# MedGemma Detection + Segmentation Training Configuration
# Optimized for 16GB GPU (T4, RTX series)

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
    model_max_length=1024,  # Larger for 16GB GPU

    # Vision
    vision_tower='openai/clip-vit-large-patch14',
    image_token_len=256,    # Full resolution for 16GB GPU
    vision_select_layer=-2,

    # Training
    lora_enable=True,
    lora_r=16,              # Higher rank for better performance
    lora_alpha=32,
    lora_dropout=0.1,

    # Quantization
    load_in_8bit=False,     # No quantization needed for 16GB GPU
    load_in_4bit=False,

    # Memory optimization
    gradient_checkpointing=True,
    use_cache=False,

    # Medical imaging settings
    freeze_vision_tower=False,
    freeze_mm_projector=False,
    use_flare25_adapters=True,
)

# Data Configuration
data_args = dict(
    # Train dataset - all categories combined
    train=dict(
        type='MedicalDetectionSegmentationDataset',
        data_file='data_medgemma/medical_detection_segmentation_all.jsonl',
        image_root='dataset',
        image_size=896,          # Larger input for 16GB GPU
        mask_size=448,           # Larger masks
        max_num_instances=15,    # More instances per image

        # Conversation template
        conv_args=dict(
            conv_template='gemma',
            system_message='',
            roles=('human', 'gpt'),
            offset=0,
            sep_style=2,
            sep=' ',
            sep2='</s>',
        ),

        # Processing
        force_image_size=True,
        image_aspect_ratio='pad',
        use_im_start_end=True,
        replace_image_token=True,

        # Medical metadata
        modalities=None,         # All modalities
        categories=None,         # All categories
    ),

    # Validation dataset - split from master
    validation=dict(
        type='MedicalDetectionSegmentationDataset',
        data_file='data_medgemma/medical_detection_segmentation_all.jsonl',
        image_root='dataset',
        image_size=896,
        mask_size=448,
        max_num_instances=15,

        # Filter for validation split
        dataset_split='val',

        conv_args=dict(
            conv_template='gemma',
            system_message='',
            roles=('human', 'gpt'),
            offset=0,
            sep_style=2,
            sep=' ',
            sep2='</s>',
        ),

        force_image_size=True,
        image_aspect_ratio='pad',
        use_im_start_end=True,
        replace_image_token=True,
    ),

    # Data loading
    lazy_load_data=True,
    num_workers=4,          # More workers for 16GB GPU
    prefetch_factor=4,
)

# Training Configuration
training_args = dict(
    # Output
    output_dir='./exp/medgemma_detection_seg_16gb',
    logging_dir='./exp/medgemma_detection_seg_16gb/logs',

    # Training schedule
    num_train_epochs=10,
    max_steps=-1,

    # Batch size (reasonable for 16GB GPU)
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,   # Good balance

    # Learning rate
    learning_rate=1e-5,      # Lower LR for better convergence
    weight_decay=0.01,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,

    # Optimization
    optim='adamw_torch',
    max_grad_norm=1.0,

    # Precision
    fp16=False,            # Use full precision for 16GB GPU
    bf16=True,             # Better precision if supported
    tf32=True,             # Faster if supported

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
    deepspeed='deepspeed/ds_config_zero2.json',  # ZeRO-2 for 16GB GPU

    # Checkpointing
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,

    # Miscellaneous
    seed=42,
    dataloader_drop_last=False,
    run_name='medgemma_detection_seg_16gb',
)

# Loss Configuration
loss_args = dict(
    # Language modeling loss
    lm_loss_weight=1.0,

    # Detection loss
    box_loss_weight=2.0,
    classification_loss_weight=1.0,

    # Segmentation loss
    mask_loss_weight=3.0,
    dice_loss_weight=1.0,

    # Balance detection vs segmentation
    detection_weight=1.0,
    segmentation_weight=1.0,
)

# Evaluation Configuration
evaluation_args = dict(
    # Metrics for detection
    detection_metrics=['precision', 'recall', 'f1', 'map'],

    # Metrics for segmentation
    segmentation_metrics=['iou', 'dice', 'pixel_accuracy'],

    # IoU thresholds
    iou_thresholds=[0.5, 0.6, 0.7, 0.75, 0.8, 0.9],

    # Evaluation settings
    eval_num_samples=-1,    # Evaluate all samples
    confidence_threshold=0.5,
    nms_threshold=0.5,

    # Per-modality evaluation
    evaluate_by_modality=True,
    evaluate_by_category=True,
)

# Advanced Configuration
advanced_args = dict(
    # Data augmentation (more aggressive for 16GB GPU)
    enable_augmentation=True,
    augmentation_policies=[
        dict(type='horizontal_flip', p=0.5),
        dict(type='vertical_flip', p=0.2),
        dict(type='random_rotation', degrees=15, p=0.5),
        dict(type='random_brightness', brightness_factor=0.2, p=0.5),
        dict(type='random_contrast', contrast_factor=0.2, p=0.5),
        dict(type='random_saturation', saturation_factor=0.1, p=0.3),
        dict(type='gaussian_blur', kernel_size=3, p=0.2),
        dict(type='random_crop', scale=(0.8, 1.0), p=0.3),
    ],

    # Class handling
    ignore_label=-1,
    num_classes=None,  # Auto-detect from data

    # Post-processing
    apply_nms=True,
    min_box_size=32,
    min_mask_area=100,

    # Training tricks
    label_smoothing=0.1,
    focal_loss_alpha=0.25,
    focal_loss_gamma=2.0,

    # Advanced training strategies
    use_mixed_precision=True,
    gradient_clipping=True,
    ema_decay=0.9999,
    use_early_stopping=True,
    early_stopping_patience=5,

    # Learning rate scheduling
    use_cosine_annealing=True,
    warmup_tepochs=1,
    min_lr=1e-7,

    # Advanced augmentation
    mixup_alpha=0.2,
    cutmix_alpha=1.0,
    prob_mixup=0.3,
    prob_cutmix=0.3,
)