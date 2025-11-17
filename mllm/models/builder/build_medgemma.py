"""
Builder for MedGemma-based Perception models.

Handles loading of:
1. Base google/medgemma-4b-it model
2. FLARE25 LoRA adapters from leoyinn/flare25-medgemma
3. Tokenizer and image processor
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from mllm.models.medgemma import MedGemmaPerception, MedGemmaConfig


def load_pretrained_medgemma(model_args, training_args):
    """
    Load MedGemma model with optional FLARE25 LoRA adapters.

    Args:
        model_args: Model configuration arguments with:
            - model_name_or_path: Base model path (e.g., "google/medgemma-4b-it")
            - adapter_path: Optional path to LoRA adapters (e.g., "leoyinn/flare25-medgemma")
            - load_in_8bit: Whether to use 8-bit quantization
            - load_in_4bit: Whether to use 4-bit quantization
            - lora_enable: Whether to enable LoRA training
            - lora_r, lora_alpha, lora_dropout: LoRA hyperparameters
        training_args: Training configuration

    Returns:
        tuple: (model, preprocessor_dict)
    """
    # Quantization config
    quantization_config = None
    if getattr(model_args, 'load_in_4bit', False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("Using 4-bit quantization (NF4)")
    elif getattr(model_args, 'load_in_8bit', False):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        print("Using 8-bit quantization")

    # Device map for multi-GPU or CPU offloading
    device_map = getattr(model_args, 'device_map', 'auto')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True
    )

    # Set padding token (Gemma doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load image processor
    try:
        image_processor = AutoImageProcessor.from_pretrained(
            model_args.model_name_or_path
        )
    except Exception as e:
        print(f"Warning: Could not load image processor: {e}")
        print("MedGemma may handle image processing internally.")
        image_processor = None

    # Check if we should load LoRA adapters
    adapter_path = getattr(model_args, 'adapter_path', None)

    if adapter_path:
        print(f"Loading base model from: {model_args.model_name_or_path}")
        print(f"Loading LoRA adapters from: {adapter_path}")

        # Load base model
        model = MedGemmaPerception.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16 if training_args.fp16 else (
                torch.bfloat16 if training_args.bf16 else torch.float32
            ),
            trust_remote_code=True
        )

        # Load LoRA adapters
        try:
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                is_trainable=training_args.do_train
            )
            print("✓ LoRA adapters loaded successfully")

            # Optionally merge adapters for inference
            if not training_args.do_train and getattr(model_args, 'merge_lora', False):
                print("Merging LoRA adapters into base model...")
                model = model.merge_and_unload()
                print("✓ LoRA adapters merged")

        except Exception as e:
            print(f"Warning: Could not load LoRA adapters from {adapter_path}: {e}")
            print("Continuing with base model only.")

    else:
        print(f"Loading base model: {model_args.model_name_or_path}")
        # Load model without adapters
        model = MedGemmaPerception.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16 if training_args.fp16 else (
                torch.bfloat16 if training_args.bf16 else torch.float32
            ),
            trust_remote_code=True
        )

    # Apply new LoRA if enabled for training
    if getattr(model_args, 'lora_enable', False) and training_args.do_train:
        print("\nApplying LoRA for training...")

        # If model already has PEFT adapters, we're fine-tuning on top
        # Otherwise, add new LoRA layers
        if not isinstance(model, PeftModel):
            lora_config = LoraConfig(
                r=getattr(model_args, 'lora_r', 64),
                lora_alpha=getattr(model_args, 'lora_alpha', 16),
                target_modules=getattr(
                    model_args,
                    'lora_target_modules',
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                ),
                lora_dropout=getattr(model_args, 'lora_dropout', 0.1),
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
            print("✓ New LoRA layers added")

        model.print_trainable_parameters()

    # Set loss weights
    model.set_loss_weights(model_args)

    # Initialize vision tokenizer
    model.config.mm_use_im_start_end = getattr(model_args, 'mm_use_im_start_end', True)
    model.config.image_size = getattr(model_args, 'image_size', 448)
    model.config.image_token_len = getattr(model_args, 'image_token_len', 256)

    # Add vision tokens to tokenizer
    model.initialize_vision_tokenizer(
        mm_use_im_start_end=model_args.mm_use_im_start_end,
        tokenizer=tokenizer,
        device=training_args.device,
        tune_mm_mlp_adapter=False,  # Not applicable for MedGemma
        pretrain_mm_mlp_adapter=None,
        vision_config=None  # Vision config is integrated in MedGemma
    )

    # Prepare preprocessor dict
    preprocessor = dict(
        image=image_processor,
        text=tokenizer,
        conv=dict(
            image_token_len=model.config.image_token_len,
            sep_image_conv_front=getattr(model_args, 'sep_image_conv_front', False),
            use_im_start_end=model.config.mm_use_im_start_end,
        )
    )

    return model, preprocessor


def print_trainable_params(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_params} || "
        f"trainable%: {100 * trainable_params / all_params:.4f}"
    )
