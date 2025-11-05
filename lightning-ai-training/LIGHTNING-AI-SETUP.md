# Lightning AI Complete Setup Guide

## The Easy Way: Use GitHub (Recommended!)

Since there's no obvious upload button in Lightning AI, **using GitHub is actually easier**:

### Step 1: Push to GitHub (Run this on your local machine)

```powershell
# Navigate to lightning-ai-training folder
cd "C:\Users\Yeshwanth Kesav\Desktop\MRE-PINN\lightning-ai-training"

# Run the automated GitHub setup script
.\push-to-github.ps1
```

This script will:
- ✅ Initialize Git repository
- ✅ Create proper `.gitignore` (excludes data files)
- ✅ Commit your code
- ✅ Push to GitHub
- ✅ Give you the exact commands for Lightning AI

### Step 2: Clone in Lightning AI (Easy!)

In Lightning AI Studio terminal:

```bash
# Clone your repository
git clone https://github.com/YourUsername/MRE-PINN.git

# Navigate to the training folder
cd MRE-PINN/lightning-ai-training

# Open the notebook
jupyter notebook lightning-ai-simulation-training.ipynb
```

**Done!** Now just run the notebook cells.

---

## Alternative: Manual File Upload Options

If you still want to upload files without Git:

### Option A: Look for File Manager

Lightning AI Studio should have a file manager. Look for:

1. **Left Sidebar** → File browser icon (usually looks like a folder 📁)
2. **Right-click in the file browser** → "Upload Files" option
3. **Or look for an upload icon** at the top (↑ or cloud icon)

### Option B: Drag and Drop

Try dragging `MRE-PINN-lightning.zip` directly into the Lightning AI file browser.

### Option C: Use a File Transfer Service

1. **Upload to Google Drive or Dropbox:**
   - Upload `MRE-PINN-lightning.zip` to Google Drive
   - Get shareable link
   - Set to "Anyone with link can view"

2. **Download in Lightning AI:**
   ```bash
   # For Google Drive
   pip install gdown
   gdown https://drive.google.com/uc?id=YOUR_FILE_ID
   unzip MRE-PINN-lightning.zip

   # For Dropbox (change ?dl=0 to ?dl=1 in the URL)
   wget -O MRE-PINN.zip "https://www.dropbox.com/s/YOUR_LINK?dl=1"
   unzip MRE-PINN.zip
   ```

### Option D: Create Files Manually (Last Resort)

If nothing else works, you can recreate the key files directly in Lightning AI:

1. **Create the notebook in Lightning AI:**
   - New Notebook → Copy paste from `lightning-ai-simulation-training.ipynb`

2. **Clone just the mre_pinn code:**
   ```bash
   # If your main code is already on GitHub
   git clone https://github.com/YourUsername/MRE-PINN.git
   ```

---

## Why Git Clone is Better

✅ **No file size limits**
✅ **No upload time**
✅ **Version controlled** - easy to update later
✅ **Works reliably** - no browser/network issues
✅ **Industry standard** - useful skill to learn

---

## Quick Decision Tree

```
Can you find an Upload button in Lightning AI?
│
├─ YES → Upload MRE-PINN-lightning.zip
│   └─ Extract and run notebook
│
└─ NO → Use GitHub instead
    ├─ Run: .\push-to-github.ps1
    ├─ In Lightning AI: git clone YOUR_URL
    └─ Run notebook
```

---

## Getting Help

**Can't find upload button?**
- Check Lightning AI documentation (they update UI frequently)
- Try their Discord/Community for quick help
- Use Git Clone method (always works!)

**GitHub not an option?**
- Try the file transfer services (Google Drive, Dropbox)
- Contact Lightning AI support for upload assistance

**Stuck?**
- Read [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Check [upload-to-lightning.md](upload-to-lightning.md) for more options

---

## Summary: Recommended Approach

### For Most Users:
```powershell
# Local machine
cd lightning-ai-training
.\push-to-github.ps1
```

```bash
# Lightning AI terminal
git clone https://github.com/YourUsername/MRE-PINN.git
cd MRE-PINN/lightning-ai-training
```

**That's it!** Now run the notebook and start training on GPU! 🚀
