# MedGemma Detection + Segmentation Training Configuration
# Optimized for 4GB GPU (GTX 1650)

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
    model_max_length=512,  # Reduced for 4GB GPU

    # Vision
    vision_tower='openai/clip-vit-large-patch14',
    image_token_len=128,   # Reduced for 4GB GPU (was 256)
    vision_select_layer=-2,

    # Training
    lora_enable=True,
    lora_r=8,              # Reduced rank for 4GB GPU
    lora_alpha=16,
    lora_dropout=0.1,

    # Quantization
    load_in_8bit=True,     # 8-bit quantization for 4GB GPU
    load_in_4bit=False,    # Set to True if still OOM

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
        image_size=448,          # Smaller for 4GB GPU
        mask_size=224,           # Smaller masks
        max_num_instances=10,    # Max instances per image

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
        image_size=448,
        mask_size=224,
        max_num_instances=10,

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
    num_workers=1,          # Reduced for 4GB GPU
    prefetch_factor=2,
)

# Training Configuration
training_args = dict(
    # Output
    output_dir='./exp/medgemma_detection_seg_4gb',
    logging_dir='./exp/medgemma_detection_seg_4gb/logs',

    # Training schedule
    num_train_epochs=3,
    max_steps=-1,

    # Batch size (tiny for 4GB GPU)
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Large accumulation for effective batch

    # Learning rate
    learning_rate=2e-5,
    weight_decay=0.0,
    lr_scheduler_type='cosine',
    warmup_ratio=0.03,

    # Optimization
    optim='adamw_torch',
    max_grad_norm=1.0,

    # Precision
    fp16=True,             # Use fp16 for 4GB GPU
    bf16=False,            # Not supported on older GPUs
    tf32=False,            # Not supported on older GPUs

    # Memory optimization
    dataloader_pin_memory=False,
    dataloader_num_workers=1,
    remove_unused_columns=False,

    # Evaluation
    evaluation_strategy='epoch',
    eval_steps=1,
    save_strategy='epoch',
    save_steps=1,

    # Logging
    logging_strategy='steps',
    logging_steps=10,
    report_to='tensorboard',

    # DeepSpeed
    deepspeed='deepspeed/ds_config_zero3_offload_4gb.json',

    # Checkpointing
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,

    # Miscellaneous
    seed=42,
    dataloader_drop_last=False,
    run_name='medgemma_detection_seg_4gb',
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
    iou_thresholds=[0.5, 0.75],

    # Evaluation settings
    eval_num_samples=100,   # Limit for faster evaluation on 4GB GPU
    confidence_threshold=0.5,
    nms_threshold=0.5,
)

# Advanced Configuration
advanced_args = dict(
    # Data augmentation (light for medical images)
    enable_augmentation=True,
    augmentation_policies=[
        dict(type='horizontal_flip', p=0.5),
        dict(type='random_rotation', degrees=10, p=0.3),
        dict(type='random_brightness', brightness_factor=0.1, p=0.3),
        dict(type='random_contrast', contrast_factor=0.1, p=0.3),
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
)