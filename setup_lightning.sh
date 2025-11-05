#!/bin/bash
# Setup script for Lightning AI Studio
# Run this once when you first start your Lightning AI Studio

set -e  # Exit on error

echo "================================"
echo "MRE-PINN Lightning AI Setup"
echo "================================"

# Check if CUDA is available
echo ""
echo "Checking CUDA availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Install dependencies
echo ""
echo "Installing dependencies from requirements-lightning.txt..."
pip install -r requirements-lightning.txt

# Set environment variable for DeepXDE backend
echo ""
echo "Setting DeepXDE backend to PyTorch..."
export DDEBACKEND=pytorch

# Install the mre_pinn package in development mode
echo ""
echo "Installing mre_pinn package..."
pip install -e .

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import mre_pinn; print('mre_pinn imported successfully')"
python -c "import deepxde; print(f'deepxde version: {deepxde.__version__}')"

echo ""
echo "================================"
echo "Setup complete! ✓"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Download data: python download_data.py (or run the notebook)"
echo "2. Train model: python train_lightning.py"
echo "3. Or open: MICCAI-2023/MICCAI-2023-simulation-training.ipynb"
echo ""
