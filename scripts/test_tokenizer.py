#!/usr/bin/env python3
"""
Test tokenizer to avoid index errors
"""

import sys
from pathlib import Path

def test_tokenizer():
    """Test tokenizer without dataset complications"""
    print("🧪 Testing Tokenizer")
    print("=" * 50)

    try:
        from transformers import AutoTokenizer

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('./ckpt/medgemma-4b-it')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✅ Tokenizer loaded")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"  UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")

        # Test coordinate tokens
        coordinate_tokens = ['<obj_vis_s>', '<obj_vis_e>', '[', ']']
        tokens_to_add = []

        for token in coordinate_tokens:
            if token not in tokenizer.get_vocab():
                tokens_to_add.append(token)
                print(f"Token '{token}' not in vocab (ID: {tokenizer.get_vocab().get(token, 'N/A')})")
            else:
                print(f"Token '{token}' already in vocab (ID: {tokenizer.get_vocab()[token]})")

        if tokens_to_add:
            print(f"\nAdding special tokens: {tokens_to_add}")
            old_size = tokenizer.vocab_size
            tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
            new_size = tokenizer.vocab_size
            print(f"Vocab size changed: {old_size} → {new_size}")

        # Test tokenization with sample text
        sample_text = "<image>\nWhere is the Loet_hoanh_tao_trang in this endoscopy?\nThe Loet_hoanh_tao_trang is located at <obj_vis_s>[0.577,0.529]<obj_vis_e>."

        print(f"\nSample text: {sample_text}")
        tokens = tokenizer(sample_text, add_special_tokens=True)['input_ids']
        print(f"Token count: {len(tokens)}")
        print(f"Max token ID: {max(tokens)}")
        print(f"Vocab size: {tokenizer.vocab_size}")

        if max(tokens) >= tokenizer.vocab_size:
            print(f"❌ ERROR: Found token ID >= vocab_size!")
            print(f"   Problematic tokens: {[t for t in tokens if t >= tokenizer.vocab_size]}")
            return False
        else:
            print(f"✅ All tokens within vocabulary bounds")

        return True

    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if test_tokenizer():
        print("\n🎉 Tokenizer test passed!")
        print("Ready for training.")
        return 0
    else:
        print("\n❌ Tokenizer test failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())