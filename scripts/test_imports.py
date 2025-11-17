#!/usr/bin/env python3
"""
Test imports to debug circular import issues
"""

import sys

def test_basic_imports():
    """Test basic imports without dataset system"""
    print("Testing basic imports...")

    try:
        import torch
        print("✅ torch")
    except ImportError as e:
        print(f"❌ torch: {e}")
        return False

    try:
        from PIL import Image
        print("✅ PIL")
    except ImportError as e:
        print(f"❌ PIL: {e}")
        return False

    return True

def test_mllm_structure():
    """Test mllm module structure without imports"""
    print("\nTesting mllm structure...")

    try:
        import mllm
        print("✅ mllm module")
    except ImportError as e:
        print(f"❌ mllm module: {e}")
        return False

    return True

def test_simple_dataset():
    """Test our simple dataset class"""
    print("\nTesting SimpleMedicalDataset...")

    try:
        # Import directly without going through __init__.py
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "simple_dataset",
            "scripts/quick_train_medgemma.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        SimpleMedicalDataset = module.SimpleMedicalDataset
        print("✅ SimpleMedicalDataset imported")

        # Test instantiation
        dataset = SimpleMedicalDataset('data_medgemma/medical_detection_segmentation_all.jsonl', 'dataset')
        print(f"✅ Dataset created: {len(dataset)} samples")

        return True

    except Exception as e:
        print(f"❌ SimpleMedicalDataset: {e}")
        return False

def main():
    print("🧪 Testing Imports for MedGemma")
    print("=" * 50)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("MLLM Module", test_mllm_structure),
        ("Simple Dataset", test_simple_dataset),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n🎉 Basic imports work!")
        print("Dataset conversion was successful!")
        print("For training, use a custom training script that doesn't rely on the complex dataset system.")
    else:
        print("\n❌ Some basic imports failed.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())