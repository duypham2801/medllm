#!/usr/bin/env python3
"""
Test and validation script for PerceptionGPT setup
Checks all dependencies, GPU availability, and basic functionality
"""

import sys
import os
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text):
    print(f"  {text}")

# ====================
# Test 1: Python packages
# ====================
def test_packages():
    print_header("Test 1: Checking Python Packages")

    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'transformers': 'HuggingFace Transformers',
        'datasets': 'HuggingFace Datasets',
        'accelerate': 'HuggingFace Accelerate',
        'peft': 'PEFT (LoRA)',
        'deepspeed': 'DeepSpeed',
        'mmengine': 'MMEngine',
        'bitsandbytes': 'BitsAndBytes (8-bit)',
        'einops': 'Einops',
        'sentencepiece': 'SentencePiece',
    }

    all_ok = True
    for package, name in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{name:30s} v{version}")
        except ImportError:
            print_error(f"{name:30s} NOT INSTALLED")
            all_ok = False

    return all_ok

# ====================
# Test 2: CUDA and GPU
# ====================
def test_cuda():
    print_header("Test 2: Checking CUDA and GPU")

    try:
        import torch

        # CUDA availability
        if torch.cuda.is_available():
            print_success(f"CUDA available: True")
        else:
            print_error("CUDA NOT available")
            return False

        # CUDA version
        cuda_version = torch.version.cuda
        print_info(f"CUDA version: {cuda_version}")

        # PyTorch version
        print_info(f"PyTorch version: {torch.__version__}")

        # GPU count
        gpu_count = torch.cuda.device_count()
        print_info(f"GPU count: {gpu_count}")

        # GPU details
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print_success(f"GPU {i}: {gpu_name} ({total_memory:.1f} GB)")

            # Warning for 4GB GPU
            if total_memory < 6:
                print_warning(f"GPU {i} has only {total_memory:.1f}GB VRAM - training will be very slow")

        # Test GPU allocation
        try:
            test_tensor = torch.zeros(100, 100).cuda()
            print_success("GPU memory allocation test passed")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print_error(f"GPU memory allocation failed: {e}")
            return False

        return True

    except Exception as e:
        print_error(f"CUDA test failed: {e}")
        return False

# ====================
# Test 3: Model imports
# ====================
def test_model_imports():
    print_header("Test 3: Checking MLLM Model Imports")

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    try:
        # Test basic imports
        from mllm.config import prepare_args
        print_success("mllm.config imports successfully")

        from mllm.models import load_pretrained
        print_success("mllm.models imports successfully")

        from mllm.dataset import prepare_data
        print_success("mllm.dataset imports successfully")

        from mllm.engine import prepare_trainer_collator
        print_success("mllm.engine imports successfully")

        return True

    except ImportError as e:
        print_error(f"Import failed: {e}")
        print_info("Make sure you installed all packages: bash scripts/install_packages.sh")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False

# ====================
# Test 4: Data and checkpoint paths
# ====================
def test_paths():
    print_header("Test 4: Checking Data and Checkpoint Paths")

    project_root = Path(__file__).parent.parent

    # Check directories
    data_dir = project_root / "data"
    ckpt_dir = project_root / "ckpt"

    if data_dir.exists():
        print_success(f"data/ directory exists")
        # Check for annotation files
        jsonl_files = list(data_dir.glob("*.jsonl"))
        if jsonl_files:
            print_info(f"Found {len(jsonl_files)} annotation files")
        else:
            print_warning("No .jsonl annotation files found in data/")
    else:
        print_warning("data/ directory does not exist")
        print_info("Run: bash scripts/download_data.sh")

    if ckpt_dir.exists():
        print_success(f"ckpt/ directory exists")
        # Check for model checkpoints
        llava_path = ckpt_dir / "llava-v1.5-7b"
        if llava_path.exists():
            print_success("LLaVA-v1.5-7b checkpoint found")
        else:
            print_warning("LLaVA checkpoint not found")
            print_info("Run: bash scripts/download_data.sh")
    else:
        print_warning("ckpt/ directory does not exist")
        print_info("Run: bash scripts/download_data.sh")

    return True

# ====================
# Test 5: Config file
# ====================
def test_config():
    print_header("Test 5: Checking Training Configuration")

    project_root = Path(__file__).parent.parent
    config_path = project_root / "config/training_configs/perception_1gpu_4gb_lora.py"

    if config_path.exists():
        print_success("Training config file exists")

        # Check if model path is updated
        with open(config_path, 'r') as f:
            content = f.read()
            if 'ckpt/llava-v1.5-7b' in content:
                print_info("Config uses default LLaVA path: ckpt/llava-v1.5-7b")
                print_warning("Update model_name_or_path if your checkpoint is elsewhere")
            else:
                print_info("Config has custom model path")

        return True
    else:
        print_error("Training config not found")
        return False

# ====================
# Test 6: DeepSpeed
# ====================
def test_deepspeed():
    print_header("Test 6: Checking DeepSpeed")

    try:
        import deepspeed
        print_success(f"DeepSpeed installed: v{deepspeed.__version__}")

        # Check config file
        project_root = Path(__file__).parent.parent
        ds_config_path = project_root / "deepspeed/ds_config_zero3_offload_4gb.json"

        if ds_config_path.exists():
            print_success("DeepSpeed config file exists")
        else:
            print_error("DeepSpeed config not found")
            return False

        return True

    except ImportError:
        print_error("DeepSpeed not installed")
        return False

# ====================
# Main test runner
# ====================
def main():
    print(f"\n{Colors.BOLD}PerceptionGPT Setup Validation{Colors.END}")
    print(f"Testing installation and configuration...\n")

    results = {
        'Packages': test_packages(),
        'CUDA/GPU': test_cuda(),
        'Model Imports': test_model_imports(),
        'Paths': test_paths(),
        'Config': test_config(),
        'DeepSpeed': test_deepspeed(),
    }

    # Summary
    print_header("Test Summary")

    all_passed = True
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name:20s} PASSED")
        else:
            print_error(f"{test_name:20s} FAILED")
            all_passed = False

    print()

    if all_passed:
        print_success("All tests passed! ✓")
        print()
        print("Next steps:")
        print("  1. Download data/models: bash scripts/download_data.sh")
        print("  2. Start training: bash scripts/run_4gb.sh")
        print()
        return 0
    else:
        print_error("Some tests failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("  - Install packages: bash scripts/install_packages.sh")
        print("  - Download data: bash scripts/download_data.sh")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
