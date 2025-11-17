#!/usr/bin/env python3
"""
Test if training config loads properly
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("Testing Training Configuration")
print("=" * 60)

try:
    print("\n[1/4] Loading config...")
    from mllm.config import prepare_args

    # Load test config (pass as list, not string)
    config_path = "config/training_configs/test_3b_dummy.py"
    cfg, training_args = prepare_args([config_path])

    model_args = cfg.model_args
    data_args = cfg.data_args

    print(f"✓ Config loaded successfully")
    print(f"  Model type: {model_args['type']}")
    print(f"  Model path: {model_args['model_name_or_path']}")
    print(f"  Output dir: {training_args.output_dir}")
    print(f"  Max steps: {training_args.max_steps}")

    print("\n[2/4] Testing dataset...")
    from mllm.dataset import prepare_data
    from mmengine.config import Config

    # Convert dicts to Config objects for compatibility
    data_args_obj = Config(data_args)
    model_args_obj = Config(model_args)

    # Prepare data (with dummy dataset)
    datasets = prepare_data(data_args_obj, model_args_obj, training_args)
    train_dataset = datasets['train']

    print(f"✓ Dataset created")
    print(f"  Type: {type(train_dataset).__name__}")
    print(f"  Size: {len(train_dataset)} samples")

    # Get one sample
    sample = train_dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")

    print("\n[3/4] Testing model loading...")
    print("⚠  Skipping model load (would download ~2GB)")
    print("  To test model loading, run: bash scripts/test_training.sh")

    print("\n[4/4] Testing trainer setup...")
    from mllm.engine import prepare_trainer_collator
    print("✓ Trainer imports work")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYour setup is ready. To run actual training:")
    print("  bash scripts/test_training.sh")
    print("\nThis will:")
    print("  1. Auto-download TinyLlama-1.1B (~2GB)")
    print("  2. Run 3 training steps with dummy data")
    print("  3. Verify the full training pipeline works")

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
