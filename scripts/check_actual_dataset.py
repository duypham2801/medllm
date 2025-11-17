#!/usr/bin/env python3
"""
Check what dataset is actually being used in training
"""

def check_actual_dataset():
    """Check what dataset training is actually using"""
    print("🔍 Checking Actual Dataset Being Used")
    print("=" * 50)

    try:
        import sys
        sys.path.append('.')
        from scripts.train_medgemma_direct import DirectMedicalDataset, MedGemmaTrainer
        from transformers import AutoTokenizer
        import sys

        # Exactly replicate what training script does
        model_path = './ckpt/medgemma-4b-it'
        adapter_path = './ckpt/flare25-medgemma'
        data_file = 'data_medgemma/medical_detection_segmentation_test.jsonl'
        image_root = 'dataset'
        use_existing_adapters = False  # --no-adapters

        print(f"Checking data_file: {data_file}")
        print(f"File exists: {Path(data_file).exists()}")

        # Create tokenizer exactly like in training
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Check and add coordinate tokens like in training
        coordinate_tokens = ['<obj_vis_s>', '<obj_vis_e>']
        existing_tokens = []
        tokens_to_add = []

        for token in coordinate_tokens:
            if token in tokenizer.get_vocab():
                existing_tokens.append(f"'{token}' (ID: {tokenizer.get_vocab()[token]})")
            else:
                tokens_to_add.append(token)
                print(f"⚠️  Token '{token}' not in Gemma vocabulary - will add")

        if existing_tokens:
            print(f"Using existing coordinate tokens: {existing_tokens}")

        # Add missing tokens
        if tokens_to_add:
            print(f"Adding special tokens: {tokens_to_add}")
            tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})

            # Verify tokens were added
            for token in tokens_to_add:
                if token in tokenizer.get_vocab():
                    token_id = tokenizer.get_vocab()[token]
                    print(f"✅ Token '{token}' added with ID: {token_id}")
                else:
                    print(f"❌ Failed to add token '{token}'")

        # Create dataset exactly like in training
        print(f"\nCreating dataset...")
        dataset = DirectMedicalDataset(data_file, image_root, tokenizer)

        print(f"Dataset length: {len(dataset)}")

        # Check first few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Input IDs shape: {sample['input_ids'].shape}")
            print(f"  Non-masked labels: {(sample['labels'] != -100).sum().item()}")
            print(f"  Max token ID: {sample['input_ids'].max().item()}")
            print(f"  Min token ID: {sample['input_ids'].min().item()}")

            # Check if there are any actual training tokens (not -100)
            valid_labels = sample['labels'][sample['labels'] != -100]
            if len(valid_labels) == 0:
                print(f"  ❌ NO VALID LABELS FOR TRAINING!")
            else:
                print(f"  ✅ {len(valid_labels)} tokens for training")

            # Check token bounds
            actual_vocab_size = len(tokenizer.get_vocab())
            if sample['input_ids'].max() >= actual_vocab_size:
                print(f"  ❌ TOKEN OUT OF BOUNDS!")
            else:
                print(f"  ✅ All tokens within bounds")

        return True

    except Exception as e:
        print(f"❌ Check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    from pathlib import Path
    check_actual_dataset()