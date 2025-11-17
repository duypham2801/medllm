#!/usr/bin/env python3
"""
Test minimal training to isolate the loss = 0.0 issue
"""

def test_minimal_training():
    """Test training with minimal setup"""
    print("🧪 Testing Minimal Training")
    print("=" * 50)

    try:
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        import json
        from pathlib import Path

        # Create a very simple dataset with 2 samples
        simple_data = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [-100, -100, 3, 4, 5]  # Only train on last 3 tokens
            },
            {
                "input_ids": [6, 7, 8, 9, 10],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [-100, -100, 8, 9, 10]  # Only train on last 3 tokens
            }
        ]

        print(f"Created simple dataset with {len(simple_data)} samples")

        # Create a very small model for testing
        model_config = {
            'vocab_size': 1000,
            'hidden_size': 128,
            'num_hidden_layers': 2,
            'num_attention_heads': 2,
            'intermediate_size': 512
        }

        # Create a simple causal LM model
        class SimpleCausalLM(nn.Module):
            def __init__(self, vocab_size=1000, hidden_size=128):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

            def forward(self, input_ids=None, attention_mask=None, labels=None):
                embeddings = self.embedding(input_ids)
                logits = self.lm_head(embeddings)

                loss = None
                if labels is not None:
                    # Shift for causal LM
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    # Flatten
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)

                    # Calculate loss
                    criterion = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = criterion(shift_logits, shift_labels)

                return type('ModelOutput', (), {'loss': loss, 'logits': logits})()

        model = SimpleCausalLM()
        print("✅ Created simple model")

        # Convert data to tensors
        for item in simple_data:
            item['input_ids'] = torch.tensor(item['input_ids'], dtype=torch.long)
            item['attention_mask'] = torch.tensor(item['attention_mask'], dtype=torch.long)
            item['labels'] = torch.tensor(item['labels'], dtype=torch.long)

        # Test forward pass
        model.train()
        sample = simple_data[0]
        outputs = model(
            input_ids=sample['input_ids'].unsqueeze(0),
            labels=sample['labels'].unsqueeze(0)
        )

        print(f"Forward pass loss: {outputs.loss.item():.6f}")
        print(f"Loss is finite: {torch.isfinite(outputs.loss)}")
        print(f"Loss is zero: {outputs.loss.item() == 0.0}")

        # Test backward pass
        loss = outputs.loss
        loss.backward()

        # Check gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        print(f"Grad norm: {total_norm:.6f}")
        print(f"Grad norm is NaN: {torch.isnan(torch.tensor(total_norm))}")

        # Test with actual model if we can
        print(f"\n🤖 Testing with actual model setup...")
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained('./ckpt/medgemma-4b-it')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Add coordinate tokens
            coordinate_tokens = ['<obj_vis_s>', '<obj_vis_e>']
            tokens_to_add = [token for token in coordinate_tokens if token not in tokenizer.get_vocab()]
            if tokens_to_add:
                tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})

            # Create minimal dataset with actual tokenizer
            text = "<image>\nWhere is the test?\nThe test is at <obj_vis_s>[0.5,0.5]<obj_vis_e>."
            tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

            # Create labels manually
            human_part = "<image>\nWhere is the test?\n"
            assistant_part = "The test is at <obj_vis_s>[0.5,0.5]<obj_vis_e>."

            human_tokens = tokenizer(human_part, add_special_tokens=False)['input_ids']
            assistant_tokens = tokenizer(assistant_part, add_special_tokens=False)['input_ids']

            full_tokens = human_tokens + assistant_tokens + [tokenizer.eos_token_id]
            labels = [-100] * len(human_tokens) + assistant_tokens + [tokenizer.eos_token_id]

            # Pad to 512
            while len(full_tokens) < 512:
                full_tokens.append(tokenizer.pad_token_id)
                labels.append(-100)

            input_ids = torch.tensor([full_tokens[:512]], dtype=torch.long)
            labels = torch.tensor([labels[:512]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

            print(f"Actual tokenized input shape: {input_ids.shape}")
            print(f"Non-masked labels: {(labels != -100).sum().item()}")

            # Test with actual model (CPU only to avoid memory issues)
            actual_model = AutoModelForCausalLM.from_pretrained(
                './ckpt/medgemma-4b-it',
                torch_dtype=torch.float32,  # Use float32 for stability
                device_map='cpu'
            )

            # Resize embeddings
            actual_vocab_size = len(tokenizer.get_vocab())
            actual_model.resize_token_embeddings(actual_vocab_size)

            actual_model.train()
            outputs = actual_model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

            print(f"Actual model loss: {outputs.loss.item():.6f}")
            print(f"Loss is finite: {torch.isfinite(outputs.loss)}")
            print(f"Loss is zero: {outputs.loss.item() == 0.0}")

            if torch.isnan(outputs.loss):
                print("❌ Model loss is NaN!")
            elif outputs.loss.item() == 0.0:
                print("❌ Model loss is 0.0!")

                # Check if there's an issue with labels
                print(f"Label tensor stats:")
                print(f"  Min: {labels.min().item()}")
                print(f"  Max: {labels.max().item()}")
                print(f"  Unique non-masked: {labels[labels != -100].unique()[:5].tolist()}")
            else:
                print("✅ Model loss looks reasonable!")

        except Exception as e:
            print(f"⚠️  Could not test with actual model: {e}")
            import traceback
            traceback.print_exc()

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_minimal_training()