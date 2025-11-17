#!/usr/bin/env python3
"""
Debug why loss = 0.0 and grad_norm = NaN
"""

def debug_loss_issue():
    """Debug the specific loss and gradient issues"""
    print("🔍 Debugging Loss = 0.0 and grad_norm = NaN Issues")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        import json
        from pathlib import Path

        # Check what dataset file is actually being used
        data_file = Path("data_medgemma/medical_detection_segmentation_test.jsonl")
        print(f"Dataset file: {data_file}")
        print(f"Dataset exists: {data_file.exists()}")

        if not data_file.exists():
            print("❌ Dataset file does not exist!")
            print("Available files:")
            data_dir = Path("data_medgemma")
            if data_dir.exists():
                for f in data_dir.glob("*"):
                    print(f"  {f}")
            return False

        # Load and examine the actual data
        samples = []
        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Just check first 3 samples
                    break
                sample = json.loads(line.strip())
                samples.append(sample)

        print(f"\nExamining {len(samples)} samples:")
        for i, sample in enumerate(samples):
            print(f"\nSample {i+1}:")
            print(f"  Image: {sample.get('image', 'N/A')}")
            print(f"  Category: {sample.get('category', 'N/A')}")
            print(f"  Modality: {sample.get('modality', 'N/A')}")
            print(f"  Conversations: {len(sample.get('conversations', []))}")

            for j, conv in enumerate(sample.get('conversations', [])):
                print(f"    {j+1}. {conv.get('from', 'unknown')}: {conv.get('value', '')[:50]}...")

        # Test data processing step by step
        print(f"\n🧪 Testing Data Processing:")

        from transformers import AutoTokenizer

        # Load tokenizer exactly like in training
        tokenizer = AutoTokenizer.from_pretrained('./ckpt/medgemma-4b-it')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Add coordinate tokens
        coordinate_tokens = ['<obj_vis_s>', '<obj_vis_e>']
        tokens_to_add = [token for token in coordinate_tokens if token not in tokenizer.get_vocab()]
        if tokens_to_add:
            tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
            print(f"✅ Added tokens: {tokens_to_add}")

        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"Actual vocab size: {len(tokenizer.get_vocab())}")

        # Process one sample exactly like in the dataset
        sample = samples[0]
        conversations = sample['conversations']

        # Extract messages
        human_msg = ""
        assistant_msg = ""
        for conv in conversations:
            if conv['from'] == 'human':
                human_msg = conv['value']
            elif conv['from'] == 'gpt':
                assistant_msg = conv['value']

        print(f"\nHuman message: {human_msg}")
        print(f"Assistant message: {assistant_msg}")

        # Tokenize like in our dataset
        human_tokens = tokenizer(human_msg, add_special_tokens=False)['input_ids']
        assistant_tokens = tokenizer(assistant_msg, add_special_tokens=False)['input_ids']

        print(f"\nHuman tokens: {len(human_tokens)}")
        print(f"Assistant tokens: {len(assistant_tokens)}")
        print(f"Human text preview: {tokenizer.decode(human_tokens[:5])}...")
        print(f"Assistant text preview: {tokenizer.decode(assistant_tokens[:5])}...")

        # Create full sequence and labels
        actual_vocab_size = len(tokenizer.get_vocab())
        full_tokens = human_tokens + assistant_tokens + [tokenizer.eos_token_id]

        # Create labels (mask human part)
        labels = [-100] * len(human_tokens) + assistant_tokens + [tokenizer.eos_token_id]

        print(f"\nFull tokens length: {len(full_tokens)}")
        print(f"Labels length: {len(labels)}")
        print(f"Non-masked labels: {sum(1 for l in labels if l != -100)}")

        # Check if all tokens are valid
        max_token_id = max(full_tokens)
        print(f"Max token ID: {max_token_id}")
        print(f"Actual vocab size: {actual_vocab_size}")

        if max_token_id >= actual_vocab_size:
            print("❌ Token out of bounds!")
            return False

        # Test loss calculation manually
        print(f"\n🧮 Testing Loss Calculation:")

        # Convert to tensors
        input_ids = torch.tensor([full_tokens[:512]], dtype=torch.long)  # Add batch dimension
        labels = torch.tensor([labels[:512]], dtype=torch.long)

        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Non-masked positions: {(labels != -100).sum().item()}")

        # Create dummy logits to test loss calculation
        batch_size, seq_len, vocab_size = 1, 512, actual_vocab_size

        # Create some realistic logits (not all zeros)
        logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1
        logits = logits.float()  # Ensure float32 for numerical stability

        # Calculate loss manually
        criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

        # Reshape for loss calculation
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        print(f"Logits shape: {logits_flat.shape}")
        print(f"Labels flat shape: {labels_flat.shape}")

        # Calculate loss
        loss = criterion(logits_flat, labels_flat)
        print(f"Manual loss: {loss.item():.6f}")

        # Check if loss is reasonable
        if loss.item() == 0.0:
            print("❌ Loss is exactly 0.0!")

            # Check if all labels are masked
            valid_labels = labels_flat[labels_flat != -100]
            print(f"Valid labels for training: {len(valid_labels)}")
            if len(valid_labels) == 0:
                print("❌ All labels are masked! This causes loss = 0")
                return False
        else:
            print("✅ Loss is non-zero and reasonable!")

        # Test with model (small test)
        print(f"\n🤖 Testing with Model (if possible):")
        try:
            from transformers import AutoModelForCausalLM

            # Load model with minimal resources
            model = AutoModelForCausalLM.from_pretrained(
                './ckpt/medgemma-4b-it',
                torch_dtype=torch.float16,
                device_map='cpu'  # Use CPU to avoid memory issues
            )

            # Resize embeddings
            model.resize_token_embeddings(actual_vocab_size)

            model.eval()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
                model_loss = outputs.loss
                print(f"Model loss: {model_loss.item():.6f}")

                if torch.isnan(model_loss):
                    print("❌ Model loss is NaN!")
                elif model_loss.item() == 0.0:
                    print("❌ Model loss is 0.0!")
                else:
                    print("✅ Model loss is reasonable!")

        except Exception as e:
            print(f"⚠️  Could not test with model: {e}")

        return True

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if debug_loss_issue():
        print("\n🎉 Debug completed - Issues identified!")
    else:
        print("\n❌ Debug failed!")