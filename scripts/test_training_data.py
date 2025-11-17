#!/usr/bin/env python3
"""
Test training data formatting before actual training
"""

import json
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

def test_conversation_formatting():
    """Test if conversations are formatted correctly"""
    print("🧪 Testing Conversation Formatting")
    print("=" * 50)

    # Load first sample
    data_file = Path('data_medgemma/medical_detection_segmentation_all.jsonl')
    if not data_file.exists():
        print("❌ Dataset file not found")
        return False

    with open(data_file, 'r') as f:
        first_sample = json.loads(f.readline())

    print(f"Sample category: {first_sample['category']}")
    print(f"Sample modality: {first_sample['modality']}")
    print(f"Conversations:")
    for i, conv in enumerate(first_sample['conversations']):
        print(f"  {i+1}. {conv['from']}: {conv['value'][:80]}...")

    # Test tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('./ckpt/medgemma-4b-it')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"\n✅ Tokenizer loaded")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Pad token: {tokenizer.pad_token}")
        print(f"  EOS token: {tokenizer.eos_token}")

        # Test special tokens
        special_tokens = ['<obj_vis_s>', '<obj_vis_e>', '[', ']']
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        print(f"  Vocab size after adding special tokens: {tokenizer.vocab_size}")

        # Test conversation processing
        human_msg = ""
        assistant_msg = ""

        for conv in first_sample['conversations']:
            if conv['from'] == 'human':
                human_msg = conv['value']
            elif conv['from'] == 'gpt':
                assistant_msg = conv['value']

        print(f"\nHuman message: {human_msg}")
        print(f"Assistant message: {assistant_msg}")

        # Tokenize separately
        human_tokens = tokenizer(human_msg, add_special_tokens=False)['input_ids']
        assistant_tokens = tokenizer(assistant_msg, add_special_tokens=False)['input_ids']

        print(f"\nHuman tokens: {len(human_tokens)}")
        print(f"Assistant tokens: {len(assistant_tokens)}")
        print(f"Human text: {tokenizer.decode(human_tokens)}")
        print(f"Assistant text: {tokenizer.decode(assistant_tokens)}")

        # Test label creation
        full_tokens = human_tokens + assistant_tokens + [tokenizer.eos_token_id]
        labels = [-100] * len(human_tokens) + assistant_tokens + [tokenizer.eos_token_id]

        print(f"\nFull tokens length: {len(full_tokens)}")
        print(f"Labels length: {len(labels)}")
        print(f"Human part masked: {labels[:len(human_tokens)]}")
        print(f"Assistant part: {labels[len(human_tokens):len(human_tokens)+5]}...")  # First 5 tokens

        return True

    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False

def test_loss_computation():
    """Test if loss can be computed properly"""
    print("\n🧪 Testing Loss Computation")
    print("=" * 50)

    try:
        import torch
        import torch.nn as nn

        # Sample input
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  # 10 tokens
        labels = torch.tensor([[-100, -100, -100, 4, 5, 6, 7, 8, 9, 10]])  # Human part masked

        # Simple cross-entropy test
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        logits = torch.randn(1, 10, 1000)  # batch=1, seq=10, vocab=1000
        loss = criterion(logits.view(-1, 1000), labels.view(-1))

        print(f"✅ Loss computation successful")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Loss value: {loss.item():.4f}")

        # Check if loss is not zero
        if loss.item() > 0:
            print("✅ Loss is non-zero (good!)")
            return True
        else:
            print("❌ Loss is zero (problem!)")
            return False

    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False

def test_coordinate_format():
    """Test coordinate format in conversations"""
    print("\n🧪 Testing Coordinate Format")
    print("=" * 50)

    # Load a few samples to check coordinate format
    data_file = Path('data_medgemma/medical_detection_segmentation_all.jsonl')
    if not data_file.exists():
        print("❌ Dataset file not found")
        return False

    coordinate_samples = 0
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Check first 5 samples
                break
            sample = json.loads(line)

            for conv in sample['conversations']:
                if conv['from'] == 'gpt':
                    text = conv['value']
                    if '<obj_vis_s>' in text and '<obj_vis_e>' in text:
                        coordinate_samples += 1
                        print(f"Sample {i+1}: {text}")
                        break

    if coordinate_samples > 0:
        print(f"✅ Found {coordinate_samples} samples with proper coordinate format")
        return True
    else:
        print("❌ No samples with coordinate format found")
        return False

def main():
    print("🔬 Pre-Training Data Test")
    print("=" * 60)

    tests = [
        ("Conversation Formatting", test_conversation_formatting),
        ("Loss Computation", test_loss_computation),
        ("Coordinate Format", test_coordinate_format),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<25} {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n🎉 All tests passed! Ready for training.")
        print("\nNext step:")
        print("  python scripts/train_medgemma_direct.py --no-adapters")
    else:
        print("\n❌ Some tests failed. Please fix issues before training.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())