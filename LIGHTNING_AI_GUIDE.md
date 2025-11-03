# Running MRE-PINN on Lightning AI

This guide explains how to train MRE-PINN models on Lightning AI's free GPU platform, which provides faster training times compared to CPU.

## Why Lightning AI?

- **Free GPU Credits**: Lightning AI offers free GPU credits for machine learning training
- **Fast Training**: ~2.5 hours for 100k iterations on GPU vs. much longer on CPU
- **Easy Setup**: Cloud-based environment with pre-installed PyTorch and CUDA
- **Jupyter Support**: Run notebooks or Python scripts

## Quick Start Guide

### 1. Create a Lightning AI Account

1. Go to [Lightning AI](https://lightning.ai/)
2. Sign up for a free account
3. Navigate to "Studios" to create a new workspace

### 2. Create a New Studio

1. Click "New Studio"
2. Select a GPU machine type (recommend: **GPU** for free tier)
3. Wait for the Studio to start

### 3. Clone the Repository

In the Lightning AI terminal, run:

```bash
git clone https://github.com/yeshu183/MRE_EigPIELM.git
cd MRE_EigPIELM
```

Or if you have a specific branch:

```bash
git clone -b claude/fix-cuda-memory-stats-011CUm7fwxr2FcWvMba2o9xd https://github.com/yeshu183/MRE_EigPIELM.git
cd MRE_EigPIELM
```

### 4. Run Setup Script

This will install all dependencies:

```bash
chmod +x setup_lightning.sh
./setup_lightning.sh
```

The setup script will:
- Verify CUDA/GPU availability
- Install all required Python packages
- Set up the mre_pinn package
- Verify the installation

### 5. Download Training Data

The BIOQIC simulation dataset needs to be downloaded. You can do this by running:

```bash
# Option 1: Run the data download cells from the notebook
jupyter notebook MICCAI-2023/MICCAI-2023-simulation-training.ipynb
# Then run cells 1-3 to download and preprocess data

# Option 2: Create a simple download script (if not already available)
```

Or run these Python commands:

```python
import os
os.environ['DDEBACKEND'] = 'pytorch'
import mre_pinn

# Download and preprocess BIOQIC data
bioqic = mre_pinn.data.BIOQICFEMBox('data/BIOQIC/downloads')
bioqic.download()
bioqic.load_mat()
bioqic.preprocess()
dataset = bioqic.to_dataset()
dataset.save_xarrays('data/BIOQIC/fem_box')
```

### 6. Train the Model

#### Option A: Using the Training Script (Recommended)

```bash
python train_lightning.py \
    --example_id 60 \
    --frequency 90 \
    --n_iters 100000 \
    --n_hidden 128 \
    --n_layers 5 \
    --learning_rate 1e-4 \
    --save_prefix lightning_run
```

This will:
- Automatically detect and use GPU
- Train for 100,000 iterations
- Save checkpoints every 10,000 iterations
- Display progress every 100 iterations
- Show GPU memory usage

#### Option B: Using the Jupyter Notebook

1. Open `MICCAI-2023/MICCAI-2023-simulation-training.ipynb`
2. Run all cells in sequence
3. The CUDA availability checks are already in place (thanks to our fix!)

## Training Parameters

Key parameters you can adjust in `train_lightning.py`:

### Data Parameters
- `--example_id`: Which example to train on (default: '60')
- `--frequency`: Training frequency in Hz (default: 'auto')
- `--xarray_dir`: Path to preprocessed data (default: 'data/BIOQIC/fem_box')

### Model Architecture
- `--n_layers`: Number of hidden layers (default: 5)
- `--n_hidden`: Hidden layer size (default: 128)
- `--activ_fn`: Activation function (default: 'ss' for sine)

### Training Hyperparameters
- `--n_iters`: Number of training iterations (default: 100000)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--n_points`: Number of training points per iteration (default: 1024)
- `--pde_loss_weight`: PDE loss weight (default: 1e-16)

### Output Control
- `--test_every`: Evaluate every N iterations (default: 1000)
- `--save_every`: Save checkpoint every N iterations (default: 10000)
- `--save_prefix`: Prefix for saved files (default: None)

## Example Training Commands

### Quick Test Run (1000 iterations)
```bash
python train_lightning.py --n_iters 1000 --save_prefix test
```

### Full Training (100k iterations)
```bash
python train_lightning.py --n_iters 100000 --save_prefix full_training
```

### Training Different Frequencies
```bash
# 60 Hz
python train_lightning.py --example_id 60 --frequency 60 --save_prefix freq_60

# 90 Hz
python train_lightning.py --example_id 90 --frequency 90 --save_prefix freq_90
```

### Larger Model
```bash
python train_lightning.py --n_hidden 256 --n_layers 7 --save_prefix large_model
```

## Monitoring Training

### GPU Memory Usage

The training script automatically displays GPU memory usage at the end of training:

```
Peak GPU memory usage: 2.45 GB
GPU 0 - Allocated: 2.30 GB, Reserved: 2.50 GB
```

### Training Progress

Progress is displayed every 100 iterations:
```
Iteration 1000/100000 - Loss: 0.0123 - Time: 12.5s
```

### Checkpoints

Models are automatically saved every 10,000 iterations (or as specified by `--save_every`).

## Expected Training Times

On Lightning AI free GPU (typically T4 or similar):
- **100k iterations**: ~2-3 hours
- **10k iterations**: ~15-20 minutes
- **1k iterations**: ~2-3 minutes

Compare to CPU (from your environment):
- **100k iterations**: ~49 hours!

## Troubleshooting

### "CUDA out of memory"

If you run out of GPU memory, try:
- Reduce `--n_points` (e.g., 512 instead of 1024)
- Reduce `--n_hidden` (e.g., 64 instead of 128)
- Reduce `--n_layers` (e.g., 3 instead of 5)

### "CUDA not available"

Make sure you:
1. Selected a GPU machine type when creating the Studio
2. Ran the setup script which verifies CUDA availability

### Import Errors

If you see import errors:
```bash
# Re-run the setup script
./setup_lightning.sh

# Or manually install
pip install -r requirements-lightning.txt
pip install -e .
```

## Saving and Downloading Results

### Save Results to Cloud Storage

Lightning AI Studios have persistent storage. Your results in the `data/` directory will be saved.

### Download Results

From Lightning AI Studio:
1. Use the file browser to locate your saved models
2. Right-click → Download
3. Or use `lightning` CLI to download files

### Export Trained Models

```bash
# Compress results
tar -czf results.tar.gz data/results/

# Download using Lightning CLI or web interface
```

## Cost and Free Tier

Lightning AI offers free GPU credits. Check current limits at:
https://lightning.ai/pricing

Typical free tier:
- **GPU hours**: Limited free hours per month
- **Storage**: Limited persistent storage
- **Compute**: Shared GPUs (T4 or similar)

## Tips for Efficient Use

1. **Test First**: Run a short training (1000 iterations) to verify everything works
2. **Use Checkpoints**: The `--save_every` parameter ensures you don't lose progress
3. **Monitor GPU**: Keep an eye on GPU memory usage
4. **Batch Experiments**: Train multiple models in sequence to maximize GPU time
5. **Download Results**: Don't forget to download your trained models!

## Next Steps

After training on Lightning AI:
1. Download the trained models
2. Use them for inference on your local machine
3. Run the evaluation notebooks to analyze results
4. Compare with FEM and AHI baselines

## Additional Resources

- [Lightning AI Documentation](https://lightning.ai/docs)
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
- [MRE-PINN Paper](https://arxiv.org/abs/...) <!-- Add actual paper link -->

## Support

If you encounter issues:
1. Check the Lightning AI [community forum](https://lightning.ai/forums)
2. Review the troubleshooting section above
3. Open an issue on the GitHub repository

---

Happy training! 🚀⚡
