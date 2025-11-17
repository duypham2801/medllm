#!/usr/bin/env python3
"""
Check library versions for compatibility
"""

import sys

def check_library_versions():
    """Check and print library versions"""
    print("🔍 Checking Library Versions")
    print("=" * 50)

    libraries = {
        'torch': lambda: __import__('torch').__version__,
        'transformers': lambda: __import__('transformers').__version__,
        'peft': lambda: __import__('peft').__version__,
        'accelerate': lambda: __import__('accelerate').__version__,
        'PIL': lambda: __import__('PIL').__version__,
        'numpy': lambda: __import__('numpy').__version__,
    }

    for lib_name, version_func in libraries.items():
        try:
            version = version_func()
            print(f"✅ {lib_name:<15} {version}")
        except ImportError:
            print(f"❌ {lib_name:<15} Not installed")
        except Exception as e:
            print(f"⚠️  {lib_name:<15} Error: {e}")

    print("\n" + "=" * 50)

    # Check transformers version compatibility
    try:
        import transformers
        version = transformers.__version__
        major, minor = map(int, version.split('.')[:2])

        if major >= 4:
            if minor >= 36:
                print("✅ Transformers version supports 'eval_strategy'")
            else:
                print("⚠️  Transformers version may need 'evaluation_strategy' instead")
        else:
            print("❌ Transformers version too old")
    except Exception as e:
        print(f"❌ Could not check transformers version: {e}")

def check_gpus():
    """Check GPU availability"""
    print("\n🖥️  GPU Information")
    print("=" * 50)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("❌ CUDA not available")
    except Exception as e:
        print(f"❌ GPU check failed: {e}")

def check_files():
    """Check required files"""
    print("\n📁 File Check")
    print("=" * 50)

    files_to_check = [
        ('MedGemma Model', './ckpt/medgemma-4b-it'),
        ('FLARE25 Adapters', './ckpt/flare25-medgemma'),
        ('Dataset', 'data_medgemma/medical_detection_segmentation_all.jsonl'),
        ('Images', 'dataset'),
    ]

    for name, path in files_to_check:
        from pathlib import Path
        if Path(path).exists():
            print(f"✅ {name:<20} {path}")
        else:
            print(f"❌ {name:<20} {path}")

def main():
    check_library_versions()
    check_gpus()
    check_files()

    print("\n" + "=" * 50)
    print("🚀 Ready for training!")
    print("Command: python scripts/train_medgemma_direct.py --no-adapters")

if __name__ == "__main__":
    main()