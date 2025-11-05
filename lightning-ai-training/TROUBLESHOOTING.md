# Troubleshooting Guide - Lightning AI Setup

## Common Issues When Preparing Files

### Issue 1: "Permission denied" or "File is being used by another process"

**Symptom:**
```
The process cannot access the file 'C:\Users\...\data\BIOQIC\fem_box\90\mre_mask.nc'
because it is being used by another process.
```

**Cause:** Jupyter notebook or Python has the file open (NetCDF files remain locked).

**Solutions:**

1. **Close and restart Jupyter kernel:**
   - In Jupyter: `Kernel` → `Restart Kernel`
   - Close all notebooks that use the data
   - Wait 10 seconds, then try again

2. **Use the automated script (recommended):**
   ```powershell
   cd lightning-ai-training
   .\create-zip.ps1
   ```
   This script automatically excludes data files!

3. **Restart VS Code/Jupyter completely:**
   - Close VS Code or Jupyter
   - Wait 10 seconds
   - Reopen and try again

4. **Manual unlock (Windows):**
   ```powershell
   # Find what's locking the file
   Get-Process | Where-Object {$_.Modules.FileName -like "*netCDF4*"}

   # Kill Python processes
   Stop-Process -Name python -Force
   ```

### Issue 2: Zip file is too large (>500 MB)

**Cause:** Including data files that will be re-downloaded on Lightning AI anyway.

**Solution:**
Use the automated script that excludes data:
```powershell
cd lightning-ai-training
.\create-zip.ps1
```

Expected size: ~90 MB (without data) vs 500+ MB (with data)

### Issue 3: Lightning AI upload fails or times out

**Solutions:**
1. **Ensure you excluded data:** Use `create-zip.ps1` script
2. **Check your internet connection:** Large uploads need stable connection
3. **Try during off-peak hours:** Faster upload speeds
4. **Alternative:** Push to GitHub and clone on Lightning AI instead

### Issue 4: Module 'mre_pinn' not found on Lightning AI

**Symptoms:**
```python
ModuleNotFoundError: No module named 'mre_pinn'
```

**Solutions:**

1. **Check directory structure:**
   ```bash
   pwd  # Should show: /path/to/MRE-PINN/lightning-ai-training
   ls ..  # Should show mre_pinn folder
   ```

2. **Verify extraction:**
   ```bash
   unzip -l MRE-PINN-lightning.zip | grep mre_pinn
   # Should show mre_pinn/ directory
   ```

3. **Manual path fix:**
   ```python
   import sys
   sys.path.insert(0, '/home/zeus/studio/MRE-PINN')  # Adjust path as needed
   import mre_pinn
   ```

### Issue 5: GPU not detected on Lightning AI

**Symptoms:**
```python
torch.cuda.is_available()  # Returns False
```

**Solutions:**

1. **Verify GPU runtime:**
   - Go to Settings in Lightning AI Studio
   - Ensure "GPU" is selected (not "CPU")
   - Runtime should show T4, A10G, or L4

2. **Restart kernel:**
   ```python
   # Kernel → Restart Kernel
   import torch
   print(torch.cuda.is_available())
   ```

3. **Check Lightning AI status:**
   - GPU may be temporarily unavailable
   - Wait a few minutes and try again
   - Check Lightning AI status page

### Issue 6: Out of memory on GPU

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce batch size:**
   ```python
   # Change from:
   n_points=1024
   # To:
   n_points=512  # or even 256
   ```

2. **Reduce model size:**
   ```python
   # Change from:
   n_hidden=128
   # To:
   n_hidden=64
   ```

3. **Clear GPU cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. **Restart kernel and try again**

### Issue 7: Data download fails on Lightning AI

**Symptoms:**
```
URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
```

**Solutions:**

1. **Check internet on Lightning AI:**
   ```bash
   ping google.com
   ```

2. **Retry download:**
   ```python
   bioqic.download(force=True)
   ```

3. **Upload data manually:**
   - Download data on local machine first
   - Upload `data/BIOQIC/downloads/` folder to Lightning AI
   - Skip the download step in notebook

### Issue 8: Training is very slow

**Expected speeds:**
- GPU (T4): ~0.5 seconds/iteration → 100k iters in ~50 minutes
- CPU: ~30 seconds/iteration → 100k iters in ~46 hours

**Check:**

1. **Verify GPU is being used:**
   ```python
   import torch
   print(f"Device: {torch.cuda.get_device_name(0)}")  # Should show T4
   print(f"Using CUDA: {next(model.pinn.parameters()).is_cuda}")  # Should be True
   ```

2. **Enable cuDNN benchmarking:**
   ```python
   torch.backends.cudnn.benchmark = True  # Should already be in notebook
   ```

3. **Check GPU utilization:**
   ```bash
   # In Lightning AI terminal
   nvidia-smi
   # Should show GPU usage near 100%
   ```

## Need More Help?

- **Lightning AI Issues:** [Lightning AI Support](https://lightning.ai/support)
- **MRE-PINN Issues:** Open a GitHub issue
- **CUDA/PyTorch Issues:** [PyTorch Forums](https://discuss.pytorch.org/)

## Quick Checklist Before Starting

✅ Closed all Jupyter notebooks
✅ Ran `create-zip.ps1` script successfully
✅ Zip file is ~90 MB (not 500+ MB)
✅ Lightning AI account created
✅ GPU runtime selected (T4/A10G/L4)
✅ File uploaded and extracted
✅ In correct directory: `MRE-PINN/lightning-ai-training`

If all checked, you're ready to train! 🚀
