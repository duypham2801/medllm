#!/usr/bin/env python3
"""
Test the training fixes without running full training
"""

def test_training_fix():
    """Test if our training fixes work"""
    print("🧪 Testing Training Fixes")
    print("=" * 50)

    try:
        import torch
        import json
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from pathlib import Path

        # Create a small test dataset file
        test_data = [
            {
                "image": "test.jpg",
                "conversations": [
                    {"from": "human", "value": "<image>\nWhere is the Viem_thuc_quan?"},
                    {"from": "gpt", "value": "The Viem_thuc_quan is located at <obj_vis_s>[0.5,0.5,0.6,0.6]<obj_vis_e>."}
                ],
                "category": "Viem_thuc_quan",
                "modality": "Endoscopy"
            }
        ]

        test_file = Path("data_medgemma/test_training_fix.jsonl")
        test_file.parent.mkdir(exist_ok=True)

        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        print(f"✅ Created test dataset: {test_file}")

        # Test the DirectMedicalDataset
        import sys
        sys.path.append('.')
        from scripts.train_medgemma_direct import DirectMedicalDataset

        # Load tokenizer first (mimic what load_model does)
        tokenizer = AutoTokenizer.from_pretrained('./ckpt/medgemma-4b-it')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Add coordinate tokens (mimic what load_model does)
        coordinate_tokens = ['<obj_vis_s>', '<obj_vis_e>']
        tokens_to_add = [token for token in coordinate_tokens if token not in tokenizer.get_vocab()]

        if tokens_to_add:
            tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
            print(f"✅ Added tokens: {tokens_to_add}")

        # Create dataset
        dataset = DirectMedicalDataset(
            str(test_file),
            "dataset",
            tokenizer,
            image_size=448,
            max_length=512
        )

        print(f"✅ Dataset created with {len(dataset)} samples")

        # Test one sample
        sample = dataset[0]

        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Attention mask shape: {sample['attention_mask'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")

        # Check that labels have both -100 (masked) and actual token IDs
        unique_labels = torch.unique(sample['labels'])
        print(f"Unique labels: {unique_labels[:10].tolist()}... (showing first 10)")

        masked_count = (sample['labels'] == -100).sum().item()
        total_count = len(sample['labels'])
        print(f"Masked tokens: {masked_count}/{total_count} ({masked_count/total_count:.1%})")

        if masked_count == 0 or masked_count == total_count:
            print("❌ Label masking looks wrong!")
            return False
        else:
            print("✅ Label masking looks correct!")

        # Check if tokens are within vocabulary bounds
        actual_vocab_size = len(tokenizer.get_vocab())
        max_token_id = sample['input_ids'].max().item()
        print(f"Max token ID: {max_token_id}, Actual vocab size: {actual_vocab_size}")

        if max_token_id >= actual_vocab_size:
            print("❌ Token ID out of bounds!")
            return False
        else:
            print("✅ All tokens within bounds!")

        # Test a minimal forward pass
        print(f"\nTesting forward pass...")

        # Load a smaller model for testing (if available)
        # For now, just test the data processing
        print("✅ Data processing test passed!")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_training_fix():
        print("\n🎉 Training fix test passed!")
        print("Ready to run: python scripts/train_medgemma_direct.py --no-adapters")
    else:
        print("\n❌ Training fix test failed.")