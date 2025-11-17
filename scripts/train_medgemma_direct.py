#!/usr/bin/env python3
"""
Direct training script for MedGemma detection+segmentation
Bypasses all complex dataset registration and import systems
"""

import json
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DirectMedicalDataset(Dataset):
    """Direct dataset that works without any PerceptionGPT dependencies"""

    def __init__(self, jsonl_file, image_root, tokenizer, image_size=448, max_length=512):
        self.jsonl_file = Path(jsonl_file)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length

        # Load data
        self.data = []
        print(f"Loading data from {jsonl_file}...")

        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        sample = json.loads(line.strip())
                        self.data.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"Error at line {line_num + 1}: {e}")

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load and process image
        image_path = self.image_root / sample['image']
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
            # For now, we'll just store the image path since we need proper vision processing
            image_info = str(sample['image'])
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image_info = "error_loading_image"

        # Process conversations into text
        conversations = sample['conversations']
        text = ""
        for conv in conversations:
            if conv['from'] == 'human':
                text += f"Human: {conv['value']}\n"
            else:
                text += f"Assistant: {conv['value']}\n"

        # Tokenize text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': inputs['input_ids'].squeeze().clone(),
            'image_path': image_info,
            'category': sample.get('category', 'Unknown'),
            'modality': sample.get('modality', 'Unknown')
        }

class MedGemmaTrainer:
    def __init__(self, model_path, adapter_path, data_file, image_root, use_existing_adapters=True):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.data_file = data_file
        self.image_root = image_root
        self.use_existing_adapters = use_existing_adapters

    def load_model(self):
        """Load MedGemma model with adapters"""
        print(f"Loading model from {self.model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side='right',
            truncation_side='right'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )

        # Handle LoRA adapters - either load existing or create new
        if (self.use_existing_adapters and self.adapter_path and
            Path(self.adapter_path).exists()):
            print(f"Attempting to load existing adapters from {self.adapter_path}")
            try:
                # Try to load adapters first
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
                print("✅ Existing adapters loaded successfully")

                # If we loaded existing adapters, we don't need to add new LoRA layers
                return self.model

            except Exception as e:
                print(f"⚠️  Failed to load existing adapters: {e}")
                print("Creating new LoRA adapters instead...")
        elif self.use_existing_adapters:
            print("⚠️  Existing adapters requested but not found, creating new adapters...")

        # Setup LoRA for training (new adapters)
        print("Creating new LoRA adapters for training...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
                'lm_head'
            ]
        )

        self.model = get_peft_model(self.model, lora_config)
        print("✅ New LoRA adapters created")
        self.model.print_trainable_parameters()

        return self.model

    def prepare_data(self):
        """Prepare dataset"""
        train_dataset = DirectMedicalDataset(
            self.data_file,
            self.image_root,
            self.tokenizer,
            image_size=448,
            max_length=512
        )

        # Split into train/validation
        total_size = len(train_dataset)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        return train_dataset, val_dataset

    def train(self):
        """Start training"""
        print("Starting training...")

        # Load model
        model = self.load_model()

        # Prepare data
        train_dataset, val_dataset = self.prepare_data()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./exp/medgemma_direct',
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_ratio=0.05,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy='steps',
            save_strategy='steps',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            fp16=True,
            dataloader_num_workers=4,
            report_to='tensorboard',
            run_name='medgemma_direct_training'
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Start training
        print(f"Training with {len(train_dataset)} samples...")
        trainer.train()

        # Save model
        trainer.save_model('./exp/medgemma_direct/final_model')
        self.tokenizer.save_pretrained('./exp/medgemma_direct/final_model')

        print("Training completed!")

def main():
    print("🚀 Direct MedGemma Training")
    print("=" * 50)

    # Configuration
    model_path = './ckpt/medgemma-4b-it'
    adapter_path = './ckpt/flare25-medgemma'
    data_file = 'data_medgemma/medical_detection_segmentation_all.jsonl'
    image_root = 'dataset'

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Direct MedGemma Training')
    parser.add_argument('--no-adapters', action='store_true',
                       help='Skip existing adapters and create new ones')
    args = parser.parse_args()

    use_existing_adapters = not args.no_adapters

    print(f"Model: {model_path}")
    print(f"Adapters: {adapter_path}")
    print(f"Use existing adapters: {use_existing_adapters}")
    print(f"Data: {data_file} ({len(list(Path(data_file).open())) if Path(data_file).exists() else 'Not found'} samples)")

    # Check files exist
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return 1

    if not Path(data_file).exists():
        print(f"❌ Data not found: {data_file}")
        return 1

    # Create trainer
    trainer = MedGemmaTrainer(model_path, adapter_path, data_file, image_root, use_existing_adapters)

    # Start training
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())