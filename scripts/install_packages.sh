#!/bin/bash

# Installation script for PerceptionGPT on 4GB GPU
# This script installs all required packages into conda environment 'llm'

set -e  # Exit on error

echo "========================================="
echo "PerceptionGPT Package Installation"
echo "Target: conda environment 'llm'"
echo "========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Activate conda environment
echo ""
echo "[1/5] Activating conda environment 'llm'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm

# Verify we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "llm" ]]; then
    echo "ERROR: Failed to activate 'llm' environment"
    exit 1
fi
echo "✓ Environment 'llm' activated"

# Check Python version
echo ""
echo "[2/5] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
if [[ ! "$python_version" =~ ^3\.10 ]]; then
    echo "WARNING: Expected Python 3.10, got $python_version"
fi

# Install PyTorch with CUDA support
echo ""
echo "[3/5] Installing PyTorch with CUDA support..."
echo "This may take a while..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch CUDA availability
echo ""
echo "Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Install remaining packages from requirements
echo ""
echo "[4/5] Installing remaining packages from requirements_fixed.txt..."
pip install -r requirements_fixed.txt

# Install optional packages for better performance
echo ""
echo "[5/5] Installing optional optimization packages..."
echo "Installing xformers (memory-efficient attention)..."
pip install xformers --no-deps || echo "WARNING: xformers installation failed (optional package)"

# Verify critical packages
echo ""
echo "========================================="
echo "Verifying installation..."
echo "========================================="

python << 'EOF'
import sys

packages_to_check = [
    'torch',
    'torchvision',
    'transformers',
    'datasets',
    'accelerate',
    'peft',
    'deepspeed',
    'mmengine',
    'bitsandbytes',
    'einops',
    'sentencepiece',
]

print("\nPackage versions:")
print("-" * 50)

all_ok = True
for package in packages_to_check:
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package:20s} {version}")
    except ImportError as e:
        print(f"✗ {package:20s} NOT INSTALLED")
        all_ok = False

print("-" * 50)

if all_ok:
    print("\n✓ All critical packages installed successfully!")
    sys.exit(0)
else:
    print("\n✗ Some packages failed to install. Please check errors above.")
    sys.exit(1)
EOF

installation_status=$?

echo ""
echo "========================================="
if [ $installation_status -eq 0 ]; then
    echo "✓ Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run: bash scripts/download_data.sh"
    echo "2. Run: python scripts/test_setup.py"
    echo "3. Start training: bash scripts/run_4gb.sh"
else
    echo "✗ Installation completed with errors."
    echo "Please review the error messages above."
fi
echo "========================================="
