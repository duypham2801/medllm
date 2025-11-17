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
    """Direct dataset with proper conversation formatting"""

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

        # Check if coordinate tokens already exist, add if needed
        coordinate_tokens = ['<obj_vis_s>', '<obj_vis_e>']

        # Check if tokens already exist in vocabulary
        tokens_to_add = []
        for token in coordinate_tokens:
            if token not in self.tokenizer.get_vocab():
                tokens_to_add.append(token)
            else:
                print(f"Token '{token}' already exists in vocabulary")

        if tokens_to_add:
            print(f"Adding special tokens: {tokens_to_add}")
            self.tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
        else:
            print("All coordinate tokens already exist in vocabulary")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load image (we don't actually need image processing for text-only training)
        image_path = self.image_root / sample['image']
        image_info = str(sample['image'])

        # Format conversations properly for training
        conversations = sample['conversations']

        # Extract messages properly
        human_msg = ""
        assistant_msg = ""

        for conv in conversations:
            if conv['from'] == 'human':
                human_msg = conv['value']
            elif conv['from'] == 'gpt':
                assistant_msg = conv['value']

        # Format as chat template
        formatted_text = f"{human_msg}\n{assistant_msg}"

        # Add EOS token
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
            formatted_text += self.tokenizer.eos_token

        # Tokenize with proper padding
        inputs = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Create proper labels (shifted for causal LM)
        input_ids = inputs['input_ids'].squeeze()
        labels = input_ids.clone()

        # Create proper labels - only train on assistant response
        # Tokenize human and assistant parts separately
        human_tokens = self.tokenizer(human_msg, add_special_tokens=False)['input_ids']
        assistant_tokens = self.tokenizer(assistant_msg, add_special_tokens=False)['input_ids']

        # Validate vocabulary size and token bounds
        vocab_size = self.tokenizer.vocab_size

        # Ensure tokens are within vocabulary bounds
        human_tokens = [t if t < vocab_size else self.tokenizer.unk_token_id for t in human_tokens]
        assistant_tokens = [t if t < vocab_size else self.tokenizer.unk_token_id for t in assistant_tokens]

        # Combine with special tokens
        full_tokens = human_tokens + assistant_tokens
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
            eos_id = self.tokenizer.eos_token_id
            if eos_id and eos_id < vocab_size:
                full_tokens.append(eos_id)

        # Create attention mask and labels
        input_ids = torch.tensor(full_tokens[:self.max_length], dtype=torch.long)

        # Create labels - mask human part with -100
        labels = input_ids.clone()
        labels[:len(human_tokens)] = -100  # Only train on assistant response

        # Pad to max_length
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_length,), -100, dtype=torch.long)])
            attention_mask = torch.cat([torch.ones(len(input_ids) - pad_length, dtype=torch.long), torch.zeros(pad_length, dtype=torch.long)])
        else:
            attention_mask = torch.ones_like(input_ids)

        # Ensure we don't exceed max_length
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        attention_mask = attention_mask[:self.max_length]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
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
        # Create dataset first
        train_dataset = DirectMedicalDataset(
            self.data_file,
            self.image_root,
            self.tokenizer,
            image_size=448,
            max_length=512
        )

        # Resize model embeddings for new tokens (safer approach)
        if hasattr(self.model, 'resize_token_embeddings'):
            old_size = self.model.get_input_embeddings().weight.size(0)
            new_size = self.tokenizer.vocab_size
            if new_size > old_size:
                print(f"Resizing embeddings from {old_size} to {new_size}")
                self.model.resize_token_embeddings(new_size)
                print("✅ Embeddings resized successfully")
            elif new_size < old_size:
                print(f"⚠️  New vocab size ({new_size}) is smaller than old ({old_size})")
                print("   This might indicate tokenizer issues. Proceeding with caution.")
            else:
                print(f"✅ Embeddings size matches ({new_size})")
        else:
            print("⚠️  Model does not support embedding resizing")

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

        # Load model first
        model = self.load_model()

        # Then prepare data (to ensure tokenizer has been set up)
        train_dataset, val_dataset = self.prepare_data()

        # Custom data collator that handles our format
        def custom_data_collator(features):
            batch = {}

            # Stack tensors
            batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
            batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
            batch['labels'] = torch.stack([f['labels'] for f in features])

            return batch

        # Training arguments with version compatibility
        try:
            # Try with eval_strategy (newer transformers versions)
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
                eval_strategy='steps',  # Newer versions
                save_strategy='steps',
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss',
                greater_is_better=False,
                fp16=True,
                dataloader_num_workers=4,
                report_to='tensorboard',
                run_name='medgemma_direct_training'
            )
        except TypeError as e:
            if 'eval_strategy' in str(e):
                print("⚠️  Using older transformers version, switching to evaluation_strategy")
                # Fallback to evaluation_strategy (older transformers versions)
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
                    evaluation_strategy='steps',  # Older versions
                    save_strategy='steps',
                    load_best_model_at_end=True,
                    metric_for_best_model='eval_loss',
                    greater_is_better=False,
                    fp16=True,
                    dataloader_num_workers=4,
                    report_to='tensorboard',
                    run_name='medgemma_direct_training'
                )
            else:
                raise e

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=custom_data_collator,
            processing_class=self.tokenizer,
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