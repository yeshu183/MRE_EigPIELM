#!/usr/bin/env python
"""
Download and preprocess BIOQIC simulation dataset
This script downloads the BIOQIC FEM simulation data and converts it to xarray format.

Usage:
    python download_data.py
"""

import os
import sys

# Set DeepXDE backend before importing
os.environ['DDEBACKEND'] = 'pytorch'

print("="*60)
print("BIOQIC Data Download and Preprocessing")
print("="*60)
print()

print("Importing mre_pinn package...")
import mre_pinn

print("Initializing BIOQIC FEM Box dataset...")
bioqic = mre_pinn.data.BIOQICFEMBox('data/BIOQIC/downloads')

print("\n" + "="*60)
print("DOWNLOADING DATA")
print("="*60)
print("This will download the four_target_phantom.mat file (~10-20 MB)")
print("from https://bioqic-apps.charite.de/")
print()

try:
    bioqic.download()
    print("✓ Download complete!")
except Exception as e:
    print(f"✗ Download failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("LOADING MAT FILE")
print("="*60)

try:
    bioqic.load_mat()
    print("✓ MAT file loaded successfully!")
except Exception as e:
    print(f"✗ Loading failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)
print("Segmenting spatial regions and creating ground truth elastogram...")

try:
    bioqic.preprocess()
    print("✓ Preprocessing complete!")
except Exception as e:
    print(f"✗ Preprocessing failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("CONVERTING TO XARRAY FORMAT")
print("="*60)

try:
    dataset = bioqic.to_dataset()
    print("✓ Converted to xarray dataset!")
    print(f"\nDataset info:")
    print(dataset.wave)
except Exception as e:
    print(f"✗ Conversion failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("SAVING XARRAY FILES")
print("="*60)
print("Saving to data/BIOQIC/fem_box/")

try:
    dataset.save_xarrays('data/BIOQIC/fem_box')
    print("✓ All files saved successfully!")
except Exception as e:
    print(f"✗ Saving failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("VERIFICATION")
print("="*60)
print("Checking saved files...")

import os
frequencies = [50, 60, 70, 80, 90, 100]
all_exist = True

for freq in frequencies:
    freq_dir = f'data/BIOQIC/fem_box/{freq}'
    wave_file = f'{freq_dir}/wave.nc'
    mre_file = f'{freq_dir}/mre.nc'
    mask_file = f'{freq_dir}/mre_mask.nc'

    if os.path.exists(wave_file) and os.path.exists(mre_file) and os.path.exists(mask_file):
        print(f"✓ {freq} Hz: All files present")
    else:
        print(f"✗ {freq} Hz: Missing files")
        all_exist = False

if all_exist:
    print("\n" + "="*60)
    print("SUCCESS! Dataset ready for training")
    print("="*60)
    print("\nAvailable frequencies: 50, 60, 70, 80, 90, 100 Hz")
    print("\nYou can now run:")
    print("  python train_lightning.py --example_id 90 --frequency 90")
    print("\nOr open the Jupyter notebook:")
    print("  MICCAI-2023/MICCAI-2023-simulation-training.ipynb")
    print("="*60)
else:
    print("\n" + "="*60)
    print("WARNING: Some files are missing")
    print("="*60)
    sys.exit(1)
