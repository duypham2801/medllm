#!/usr/bin/env python3
"""
Debug training issues step by step
"""

import json
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def debug_tokenizer_and_model():
    """Debug tokenizer and model compatibility"""
    print("🔍 Debugging Tokenizer and Model")
    print("=" * 50)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./ckpt/medgemma-4b-it')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Original tokenizer vocab_size: {tokenizer.vocab_size}")

    # Add coordinate tokens
    coordinate_tokens = ['<obj_vis_s>', '<obj_vis_e>']
    tokens_to_add = [token for token in coordinate_tokens if token not in tokenizer.get_vocab()]

    if tokens_to_add:
        print(f"Adding tokens: {tokens_to_add}")
        tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})

    print(f"Updated tokenizer vocab_size: {tokenizer.vocab_size}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        './ckpt/medgemma-4b-it',
        torch_dtype=torch.float16,
        device_map='auto'
    )

    # Check model embedding size
    old_embedding_size = model.get_input_embeddings().weight.size(0)
    print(f"Model embedding size before resize: {old_embedding_size}")

    # Only resize if needed
    if tokenizer.vocab_size != old_embedding_size:
        print(f"Resizing model embeddings to {tokenizer.vocab_size}")
        model.resize_token_embeddings(tokenizer.vocab_size)
        new_embedding_size = model.get_input_embeddings().weight.size(0)
        print(f"Model embedding size after resize: {new_embedding_size}")
    else:
        print("✅ Model embedding size already matches tokenizer")

    return tokenizer, model

def test_sample_processing(tokenizer):
    """Test sample processing step by step"""
    print("\n🔍 Debugging Sample Processing")
    print("=" * 50)

    # Load sample data
    data_file = Path('data_medgemma/medical_detection_segmentation_all.jsonl')
    with open(data_file, 'r') as f:
        sample = json.loads(f.readline())

    # Extract conversations
    human_msg = ""
    assistant_msg = ""
    for conv in sample['conversations']:
        if conv['from'] == 'human':
            human_msg = conv['value']
        elif conv['from'] == 'gpt':
            assistant_msg = conv['value']

    print(f"Human: {human_msg}")
    print(f"Assistant: {assistant_msg}")

    # Tokenize parts
    human_tokens = tokenizer(human_msg, add_special_tokens=False)['input_ids']
    assistant_tokens = tokenizer(assistant_msg, add_special_tokens=False)['input_ids']

    print(f"\nHuman tokens (first 10): {human_tokens[:10]}")
    print(f"Assistant tokens (first 10): {assistant_tokens[:10]}")
    print(f"Max human token: {max(human_tokens) if human_tokens else 0}")
    print(f"Max assistant token: {max(assistant_tokens) if assistant_tokens else 0}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Create labels
    full_tokens = human_tokens + assistant_tokens + [tokenizer.eos_token_id]
    labels = [-100] * len(human_tokens) + assistant_tokens + [tokenizer.eos_token_id]

    print(f"\nFull tokens length: {len(full_tokens)}")
    print(f"Labels length: {len(labels)}")
    print(f"Non-masked labels: {sum(1 for l in labels if l != -100)}")

    # Check for invalid tokens
    invalid_tokens = [t for t in full_tokens if t >= tokenizer.vocab_size]
    if invalid_tokens:
        print(f"❌ Invalid tokens found: {invalid_tokens}")
    else:
        print("✅ All tokens are valid")

    return full_tokens, labels

def test_loss_computation(model, input_ids, labels):
    """Test actual loss computation"""
    print("\n🔍 Debugging Loss Computation")
    print("=" * 50)

    model.eval()
    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
        loss = outputs.loss

        print(f"Loss value: {loss.item()}")
        print(f"Loss type: {type(loss)}")
        print(f"Loss is finite: {torch.isfinite(loss)}")

        if not torch.isfinite(loss):
            print("❌ Loss is not finite!")

            # Check logits
            logits = outputs.logits
            print(f"Logits shape: {logits.shape}")
            print(f"Logits contains nan: {torch.isnan(logits).any()}")
            print(f"Logits contains inf: {torch.isinf(logits).any()}")

            # Check input_ids
            print(f"Input IDs min/max: {input_ids.min()}/{input_ids.max()}")
            print(f"Labels min/max: {labels.min()}/{labels.max()}")

            return False
        else:
            print("✅ Loss computation looks good")
            return True

def main():
    try:
        # Test tokenizer and model
        tokenizer, model = debug_tokenizer_and_model()

        # Test sample processing
        full_tokens, labels = test_sample_processing(tokenizer)

        # Convert to tensors
        input_ids = torch.tensor(full_tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # Test loss computation
        if test_loss_computation(model, input_ids, labels):
            print("\n🎉 All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1

    except Exception as e:
        print(f"❌ Debug script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())