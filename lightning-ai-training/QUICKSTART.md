# Quick Start Guide - Lightning AI Training

Get started training MRE-PINN on Lightning AI GPU in 5 minutes!

## 1. Prepare Your Files (Local Machine)

**Option A: Use the Automated Script (Recommended)**

```powershell
# Windows PowerShell - Navigate to lightning-ai-training folder
cd lightning-ai-training

# Run the automated zip creation script
.\create-zip.ps1

# This creates MRE-PINN-lightning.zip in the parent directory
# Automatically excludes data files (saves space and time!)
```

**Option B: Manual Compression**

First, close any Jupyter notebooks that might have locked files, then:

```powershell
# Windows PowerShell - Navigate to parent directory
cd ..

# Exclude data and cache files to avoid permission errors
$exclude = @("data", ".git", "__pycache__", ".ipynb_checkpoints")
Get-ChildItem -Path . -Exclude $exclude | Compress-Archive -DestinationPath MRE-PINN.zip

# Linux/Mac:
zip -r MRE-PINN.zip . -x "data/*" "*.git*" "*__pycache__*" "*.ipynb_checkpoints*"
```

**Why exclude data?** The data will be downloaded fresh on Lightning AI (only ~50MB), saving upload time!

## 2. Setup Lightning AI (5 minutes)

1. **Create Account**: Go to [lightning.ai](https://lightning.ai) → Sign up (free)

2. **Create Studio**:
   - Click "New Studio"
   - Select **T4 GPU** runtime
   - Wait for initialization (~2 minutes)

3. **Upload Files**:
   - Click upload icon in file browser
   - Upload `MRE-PINN.zip`
   - Open terminal: `unzip MRE-PINN.zip`

## 3. Run Training (1 hour)

1. **Open Notebook**:
   ```bash
   cd lightning-ai-training
   ```
   - Open `lightning-ai-simulation-training.ipynb`

2. **Run All Cells**:
   - Click "Run All" or execute cells one by one
   - Watch for GPU detection message

3. **Monitor Progress**:
   - Training displays every 100 iterations
   - Total time: ~50 minutes for 100k iterations

## 4. Download Results

```bash
# In the final notebook cell, this creates a zip file:
zip -r lightning_ai_results.zip checkpoints/ LIGHTNING_AI_*.png LIGHTNING_AI_*.pkl
```

Download `lightning_ai_results.zip` via the file browser.

## Expected Output

```
Using device: cuda
GPU: Tesla T4
Available GPU memory: 14.75 GiB

File already exists at data/BIOQIC/downloads/four_target_phantom.mat, skipping download

Starting training on cuda...
Iteration: 100, Loss: 0.0234, PDE Loss: 0.0012
Iteration: 200, Loss: 0.0198, PDE Loss: 0.0009
...
Iteration: 100000, Loss: 0.0023, PDE Loss: 0.0001

Peak GPU memory: 3.42 GiB
GPU utilization was efficient!

CPU times: user 45min 23s, sys: 2min 15s, total: 47min 38s
Wall time: 51min 42s
```

## Troubleshooting

### GPU Not Found?
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show T4 or A10G
```

If False:
- Verify you selected GPU runtime (not CPU)
- Restart kernel: Kernel → Restart Kernel
- Check Lightning AI status

### Out of Memory?
Reduce memory usage in the model configuration cell:
```python
# Change from:
n_points=1024, n_hidden=128

# To:
n_points=512, n_hidden=64
```

### Import Error: "No module named 'mre_pinn'"?
```bash
# Check current directory
pwd

# Should be: /path/to/MRE-PINN/lightning-ai-training
# If not, navigate there:
cd MRE-PINN/lightning-ai-training
```

Then restart kernel and run again.

## Tips for Success

✅ **Do:**
- Select T4 GPU runtime
- Upload entire MRE-PINN folder
- Let training complete uninterrupted
- Download results before closing studio

❌ **Don't:**
- Use CPU runtime (50x slower!)
- Upload only the notebook file
- Close browser during training
- Forget to download results

## Cost Estimate

**Free Tier:**
- 1 hour GPU training = ~1 hour of free credits
- T4 GPU is most cost-effective
- Total cost: **FREE** (within free tier limits)

**If you exceed free tier:**
- T4: ~$0.60/hour
- A10G: ~$1.00/hour
- This project: ~$0.50-1.00 total

## Next Steps

After training:
1. Analyze metrics in the notebook
2. Compare with baseline methods (AHI, FEM)
3. Try different hyperparameters
4. Experiment with different frequencies

## Need Help?

- Lightning AI Docs: https://lightning.ai/docs
- MRE-PINN Issues: https://github.com/<your-repo>/issues
- Check main [README.md](README.md) for detailed instructions

---

**Happy Training! 🚀⚡**
