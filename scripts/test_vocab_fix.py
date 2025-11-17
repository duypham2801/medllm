#!/usr/bin/env python3
"""
Test the vocabulary fix without full training
"""

def test_vocab_fix():
    """Test if our vocabulary fix works"""
    print("🧪 Testing Vocabulary Fix")
    print("=" * 50)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained('./ckpt/medgemma-4b-it')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✅ Tokenizer loaded")
        print(f"  Original vocab_size: {tokenizer.vocab_size}")

        # Add coordinate tokens
        coordinate_tokens = ['<obj_vis_s>', '<obj_vis_e>']
        tokens_to_add = [token for token in coordinate_tokens if token not in tokenizer.get_vocab()]

        if tokens_to_add:
            print(f"Adding tokens: {tokens_to_add}")
            tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})

        print(f"  Reported vocab_size: {tokenizer.vocab_size}")
        print(f"  Actual vocab size: {len(tokenizer.get_vocab())}")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            './ckpt/medgemma-4b-it',
            torch_dtype=torch.float16,
            device_map='auto'
        )

        # Check model embedding size
        model_embed_size = model.get_input_embeddings().weight.size(0)
        print(f"  Model embedding size: {model_embed_size}")

        # Use actual vocabulary size for resizing (our fix)
        actual_vocab_size = len(tokenizer.get_vocab())
        print(f"  Using actual vocab size for resize: {actual_vocab_size}")

        if model_embed_size != actual_vocab_size:
            print(f"Resizing model embeddings...")
            model.resize_token_embeddings(actual_vocab_size)
            new_embed_size = model.get_input_embeddings().weight.size(0)
            print(f"  New model embedding size: {new_embed_size}")

        # Test tokenization and forward pass
        sample_text = "<image>\nWhere is the Loet_hoanh_tao_trang in this endoscopy?\nThe Loet_hoanh_tao_trang is located at <obj_vis_s>[0.5,0.5,0.6,0.6]<obj_vis_e>."

        print(f"\nTesting sample processing...")
        tokens = tokenizer(sample_text, return_tensors='pt', truncation=True, max_length=512)
        print(f"Token shape: {tokens['input_ids'].shape}")
        print(f"Max token ID: {tokens['input_ids'].max().item()}")
        print(f"Model vocab size: {model.get_input_embeddings().weight.size(0)}")

        if tokens['input_ids'].max() >= model.get_input_embeddings().weight.size(0):
            print("❌ Token ID out of bounds!")
            return False
        else:
            print("✅ All tokens within bounds!")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(**tokens, labels=tokens['input_ids'].clone())
            loss = outputs.loss
            print(f"✅ Forward pass successful!")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Loss is finite: {torch.isfinite(loss)}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_vocab_fix():
        print("\n🎉 Vocabulary fix test passed!")
        print("Ready for training.")
    else:
        print("\n❌ Vocabulary fix test failed.")