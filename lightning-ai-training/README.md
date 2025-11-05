# MRE-PINN Training on Lightning AI

This folder contains a notebook configured to train MRE-PINN models on Lightning AI's free GPU instances.

## Why Use Lightning AI?

- **Free GPU Credits**: Lightning AI provides free GPU hours for development
- **Easy Setup**: No local GPU required - runs entirely in the cloud
- **Fast Training**: GPU acceleration significantly reduces training time
- **Automatic Environment**: Pre-configured with CUDA and PyTorch

## Setup Instructions

### 1. Create a Lightning AI Account

1. Go to [Lightning AI](https://lightning.ai/)
2. Sign up for a free account
3. Navigate to "Studios" in the dashboard

### 2. Create a New Studio

1. Click "New Studio"
2. Select a GPU runtime:
   - **T4** (recommended for this project) - Good balance of performance and availability
   - **A10G** - Faster but may have limited free hours
   - **L4** - Alternative option with good performance
3. Wait for the studio to initialize

### 3. Upload the Repository

**Option A: Upload Entire Folder (Recommended)**
1. Compress your local `MRE-PINN` folder to a ZIP file
2. Upload to Lightning AI via the file browser
3. Extract: `unzip MRE-PINN.zip`
4. Navigate: `cd MRE-PINN/lightning-ai-training`

**Option B: Clone from GitHub**
```bash
# In Lightning AI terminal
git clone https://github.com/<your-username>/MRE-PINN.git
cd MRE-PINN/lightning-ai-training
```

**Note:** Option A is recommended as it includes your conda environment and all dependencies.

### 4. Open the Notebook

1. Navigate to `lightning-ai-training/lightning-ai-simulation-training.ipynb`
2. Open the notebook in Lightning AI

### 5. Run the Training

No installation needed! The notebook uses the existing MRE-PINN environment with all dependencies already installed.

Execute the cells sequentially. The notebook will:
- Automatically detect and use the GPU
- Download the BIOQIC dataset
- Train the MRE-PINN model
- Save checkpoints and results

## Key Differences from Local Training

### GPU Configuration
The notebook automatically detects and uses GPU:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Optimizations Enabled
- `torch.backends.cudnn.benchmark = True` - Enables cuDNN auto-tuner
- Automatic memory management and cleanup
- Peak memory tracking

### Interactive Features Disabled
Interactive matplotlib widgets don't work well in cloud environments, so they're disabled:
```python
test_eval = mre_pinn.testing.TestEvaluator(
    interact=False  # Disabled for cloud compatibility
)
```

## Expected Training Time

With GPU acceleration (T4):
- **1,000 iterations**: ~30 seconds
- **10,000 iterations**: ~5 minutes
- **100,000 iterations**: ~50 minutes

Compare to CPU (local machine):
- **100,000 iterations**: ~46 hours

**Speed improvement: ~50x faster on GPU!**

## Memory Requirements

- **GPU Memory**: ~2-4 GB (well within T4's 16 GB)
- **System RAM**: ~4-8 GB
- **Storage**: ~500 MB for data and checkpoints

## Downloading Results

After training completes, download your results:

1. Run the final cell to create a zip file:
```python
!zip -r lightning_ai_results.zip checkpoints/ LIGHTNING_AI_*.png LIGHTNING_AI_*.pkl
```

2. Use the Lightning AI file browser to download `lightning_ai_results.zip`

3. Extract locally to view results and trained models

## Troubleshooting

### GPU Not Detected
If `torch.cuda.is_available()` returns `False`:
1. Check that you selected a GPU runtime (not CPU)
2. Restart the kernel
3. Check Lightning AI status for GPU availability

### Out of Memory Error
If you encounter CUDA OOM errors:
1. Reduce batch size: Change `n_points=1024` to `n_points=512`
2. Reduce model size: Change `n_hidden=128` to `n_hidden=64`
3. Clear GPU memory: `torch.cuda.empty_cache()`

### Module Import Issues
If you get import errors for `mre_pinn`:
1. Check that you've uploaded the entire MRE-PINN folder
2. Verify you're in the correct directory: `pwd`
3. Check the path setup in the first code cell
4. Manually add to path if needed: `sys.path.insert(0, '/path/to/MRE-PINN')`

### Data Download Fails
If BIOQIC download fails:
1. Check internet connection in Lightning AI
2. Try downloading with `force=True`: `bioqic.download(force=True)`
3. Upload data manually from local machine

## Cost Considerations

- Lightning AI provides **free GPU hours** for new accounts
- Monitor your usage in the Lightning AI dashboard
- Training this model (~1 hour on T4) should stay well within free tier limits
- Consider stopping the studio when not actively training

## File Structure

```
lightning-ai-training/
├── README.md                              # This file
├── lightning-ai-simulation-training.ipynb # Main training notebook
└── requirements.txt                       # Python dependencies (optional)
```

## Next Steps

After training:
1. Analyze the metrics in `test_eval.metrics`
2. Compare results with baseline methods (AHI, FEM)
3. Experiment with different hyperparameters
4. Try different frequencies or datasets

## Additional Resources

- [Lightning AI Documentation](https://lightning.ai/docs)
- [MRE-PINN Repository](https://github.com/<your-username>/MRE-PINN)
- [DeepXDE Documentation](https://deepxde.readthedocs.io/)
- [PyTorch CUDA Best Practices](https://pytorch.org/docs/stable/notes/cuda.html)

## Support

For issues specific to:
- **Lightning AI platform**: Contact Lightning AI support
- **MRE-PINN code**: Open an issue on the GitHub repository
- **GPU/CUDA issues**: Check PyTorch compatibility

---

**Happy Training! 🚀**
