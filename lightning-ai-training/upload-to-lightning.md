# How to Upload Files to Lightning AI Studio

## Option 1: Using the Web Interface (If Upload Button is Available)

### Look for these locations:

1. **File Browser Left Sidebar:**
   - Click on the folder icon on the left
   - Right-click in the file browser area
   - Look for "Upload Files" or "Upload Folder"

2. **Top Menu Bar:**
   - Look for an upload icon (↑ or cloud with arrow)
   - Usually near the file browser or toolbar

3. **Drag and Drop:**
   - Try dragging the zip file directly into the file browser
   - Some browsers support this

## Option 2: Git Clone (Recommended - No Upload Needed!)

### Step 1: Push your code to GitHub (if not already there)

On your local machine:

```bash
cd "C:\Users\Yeshwanth Kesav\Desktop\MRE-PINN"

# Initialize git if needed
git init

# Add files (excluding data)
git add .
git commit -m "Add Lightning AI training setup"

# Push to GitHub (replace with your repo URL)
git remote add origin https://github.com/YourUsername/MRE-PINN.git
git push -u origin main
```

### Step 2: Clone in Lightning AI Studio

In the Lightning AI terminal:

```bash
git clone https://github.com/YourUsername/MRE-PINN.git
cd MRE-PINN/lightning-ai-training
```

**Advantages:**
- No file size limits
- Version controlled
- Easy to update code later
- No upload time

## Option 3: Lightning AI Teamspaces (Alternative)

If you're using Lightning AI Teamspaces, you might need to:

1. Go to the Teamspace settings
2. Navigate to "Files" or "Storage"
3. Upload there instead of in the Studio

## Option 4: Transfer via URL

### If you can host the zip file temporarily:

**Using Google Drive:**
```bash
# In Lightning AI terminal
# First, get the shareable link from Google Drive, then:
gdown https://drive.google.com/uc?id=YOUR_FILE_ID
unzip MRE-PINN-lightning.zip
```

**Using Dropbox:**
```bash
wget -O MRE-PINN-lightning.zip "https://www.dropbox.com/s/YOUR_LINK?dl=1"
unzip MRE-PINN-lightning.zip
```

**Using file.io (temporary):**
```bash
# On local machine - upload and get URL
curl -F "file=@MRE-PINN-lightning.zip" https://file.io

# In Lightning AI - download using the returned URL
wget -O MRE-PINN-lightning.zip "https://file.io/XXXXX"
unzip MRE-PINN-lightning.zip
```

## Option 5: Use Individual Files Instead of Zip

If upload is difficult, you can create the files directly in Lightning AI:

### Step 1: Create the directory structure

```bash
mkdir -p MRE-PINN/lightning-ai-training
mkdir -p MRE-PINN/mre_pinn
```

### Step 2: Create files using the Lightning AI editor

1. Click "New File" in Lightning AI
2. Copy-paste the content from your local files
3. Save with the same names

**Files to create manually:**
- `lightning-ai-simulation-training.ipynb`
- Copy the `mre_pinn` module files

## Our Recommendation: Use Git Clone

Since you don't see an upload button, **Git Clone is the best option**:

### Quick Setup:

1. **On Local Machine:**
   ```bash
   cd "C:\Users\Yeshwanth Kesav\Desktop\MRE-PINN"

   # If not already a git repo
   git init
   git add .
   git commit -m "Initial commit for Lightning AI"

   # Push to GitHub (create repo first at github.com)
   git remote add origin https://github.com/YourUsername/MRE-PINN.git
   git push -u origin main
   ```

2. **In Lightning AI Studio Terminal:**
   ```bash
   git clone https://github.com/YourUsername/MRE-PINN.git
   cd MRE-PINN/lightning-ai-training
   jupyter notebook lightning-ai-simulation-training.ipynb
   ```

## Troubleshooting

### "No upload button visible"

**Possible reasons:**
1. Lightning AI interface updated (common with cloud platforms)
2. Browser compatibility issue
3. Using mobile/tablet (try desktop)
4. Permissions issue

**Try:**
- Refresh the browser
- Try a different browser (Chrome recommended)
- Check Lightning AI documentation for current upload method
- Use Git Clone instead (most reliable)

### "Upload fails or times out"

**Solutions:**
- File too large → Use Git Clone
- Network issue → Try file.io or similar
- Lightning AI issue → Check status page

## Need Help?

- **Lightning AI Docs:** https://lightning.ai/docs
- **Lightning AI Discord:** Join their community for quick help
- **Check their latest tutorials:** They may have updated the upload process

---

**Bottom Line:** If you can't find an upload button, **use Git Clone** - it's actually easier and more reliable!
