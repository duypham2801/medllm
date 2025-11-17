#!/usr/bin/env python3
"""
Debug tokenizer vocabulary behavior
"""

def test_tokenizer_vocab_issue():
    """Debug the specific vocabulary size issue"""
    print("🔍 Debugging Tokenizer Vocabulary Issue")
    print("=" * 60)

    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./ckpt/medgemma-4b-it')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Initial vocab_size: {tokenizer.vocab_size}")
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    # Test sample text with special tokens
    sample_text = "<image>\nWhere is the Loet_hoanh_tao_trang in this endoscopy?\nThe Loet_hoanh_tao_trang is located at <obj_vis_s>[0.577,0.529]<obj_vis_e>."

    print(f"\nBefore adding special tokens:")
    print(f"Sample text: {sample_text}")

    # Tokenize without special tokens first
    tokens_no_special = tokenizer(sample_text, add_special_tokens=False)['input_ids']
    print(f"Tokens (no special): {len(tokens_no_special)}")
    print(f"Max token ID: {max(tokens_no_special) if tokens_no_special else 0}")

    # Now add special tokens
    coordinate_tokens = ['<obj_vis_s>', '<obj_vis_e>']
    tokens_to_add = [token for token in coordinate_tokens if token not in tokenizer.get_vocab()]

    print(f"\nAdding tokens: {tokens_to_add}")

    # Check vocab size before and after
    vocab_before = tokenizer.vocab_size
    print(f"Vocab size before: {vocab_before}")

    tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})

    vocab_after = tokenizer.vocab_size
    print(f"Vocab size after: {vocab_after}")

    # Tokenize again
    tokens_with_special = tokenizer(sample_text, add_special_tokens=True)['input_ids']
    print(f"Tokens (with special): {len(tokens_with_special)}")
    print(f"Max token ID: {max(tokens_with_special)}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Check specific tokens
    print(f"\nSpecial token mappings:")
    for token in coordinate_tokens:
        if token in tokenizer.get_vocab():
            token_id = tokenizer.get_vocab()[token]
            print(f"  '{token}' -> {token_id}")

            # Check if this ID is within bounds
            if token_id >= tokenizer.vocab_size:
                print(f"  ❌ ERROR: Token ID {token_id} >= vocab_size {tokenizer.vocab_size}")
            else:
                print(f"  ✅ OK: Token ID {token_id} < vocab_size {tokenizer.vocab_size}")
        else:
            print(f"  '{token}' -> NOT FOUND")

    # Try to understand what's happening
    print(f"\nInvestigating vocabulary structure:")
    print(f"  len(tokenizer.get_vocab()): {len(tokenizer.get_vocab())}")
    print(f"  tokenizer.vocab_size: {tokenizer.vocab_size}")

    # Check if vocab_size property is not being updated correctly
    actual_vocab_keys = list(tokenizer.get_vocab().keys())
    max_token_id_from_vocab = max(tokenizer.get_vocab().values())
    print(f"  Max token ID from vocab dict: {max_token_id_from_vocab}")

    # Try to resize model embeddings and see if it works
    print(f"\nTesting model compatibility:")
    try:
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            './ckpt/medgemma-4b-it',
            torch_dtype=torch.float16,
            device_map='auto'
        )

        model_embed_size = model.get_input_embeddings().weight.size(0)
        print(f"  Model embedding size: {model_embed_size}")

        if tokenizer.vocab_size != model_embed_size:
            print(f"  ⚠️  Size mismatch: Model {model_embed_size}, Tokenizer {tokenizer.vocab_size}")

            try:
                model.resize_token_embeddings(tokenizer.vocab_size)
                new_embed_size = model.get_input_embeddings().weight.size(0)
                print(f"  ✅ Model resized successfully: {new_embed_size}")
            except Exception as e:
                print(f"  ❌ Model resize failed: {e}")
        else:
            print(f"  ✅ Sizes match already")

    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")

if __name__ == "__main__":
    test_tokenizer_vocab_issue()