# mre_pinn.data - Data Management Module

This module handles all data loading, preprocessing, and management for MRE-PINN training and evaluation.

---

## Module Structure

```
data/
├── __init__.py          # Module initialization
├── imaging.py           # Patient data preprocessing (⭐ CORE)
├── bioqic.py            # Simulation data loading
├── dataset.py           # Unified data interface
└── segment.py           # Liver segmentation (U-Net)
```

---

## Files Overview

| File | Purpose | Main Classes | Use Case |
|------|---------|--------------|----------|
| `imaging.py` | Patient data preprocessing | `ImagingPatient`, `ImagingCohort` | Load and process clinical MRI data |
| `bioqic.py` | Simulation data | `BIOQICFEMBox`, `BIOQICPhantom` | Load synthetic data for validation |
| `dataset.py` | Unified interface | `MREExample`, `MREDataset` | Common API for all data types |
| `segment.py` | Liver segmentation | `UNet3D` | Deep learning segmentation model |

---

## imaging.py - Patient Data Preprocessing

### Purpose

Process raw clinical MRI data into format suitable for PINN training.

### Key Classes

#### `ImagingPatient` - Single Patient

```python
class ImagingPatient:
    """
    Load and preprocess MRI/MRE images for a single patient.

    Attributes:
        patient_id: str - Patient identifier (e.g., '0006')
        images: dict - Dictionary of SimpleITK images
        sequences: list - Available MRI sequences

    Methods:
        load_images() - Load NIFTI files
        preprocess() - Full preprocessing pipeline
        segment_image() - Run liver segmentation
        register_images() - Spatial alignment
        resize_images() - Resample to target grid
    """
```

**Usage Example**:
```python
patient = mre_pinn.data.ImagingPatient('0006')
patient.load_images()
patient.preprocess(
    same_grid=True,
    mre_size=(256, 256, 4),
    anat_size=(256, 256, 4)
)
```

#### `ImagingCohort` - Multiple Patients

```python
class ImagingCohort:
    """
    Manage multiple patients as a cohort.

    Attributes:
        patient_ids: list - List of patient IDs
        patients: dict - Dictionary of ImagingPatient objects

    Methods:
        load_images() - Load all patient images
        preprocess() - Preprocess entire cohort
        to_dataset() - Convert to MREDataset
    """
```

**Usage Example**:
```python
cohort = mre_pinn.data.ImagingCohort(
    patient_ids=['0006', '0020', '0024'],
    nifti_dirs='/path/to/NIFTI/data'
)
cohort.load_images()
cohort.preprocess()  # Processes all patients
dataset = cohort.to_dataset()
```

### MRI Sequences

The following sequences are loaded for each patient:

| Sequence | Type | Purpose |
|----------|------|---------|
| `t1_pre_in` | Anatomical | T1-weighted in-phase |
| `t1_pre_out` | Anatomical | T1-weighted out-phase (main for segmentation) |
| `t1_pre_water` | Anatomical | T1 water image |
| `t1_pre_fat` | Anatomical | T1 fat image |
| `t2` | Anatomical | T2-weighted |
| `mre_raw` | MRE | Magnitude image (for registration) |
| `wave` | MRE | **Wave displacement field (INPUT)** |
| `mre` | MRE | **Elasticity map (GROUND TRUTH)** |

### Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Load NIFTI Files                                   │
│  - Read 8 MRI sequences from disk                           │
│  - Store as SimpleITK images                                │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Fix Metadata                                       │
│  - Correct spatial coordinates (origin, spacing)            │
│  - Restore wave field from RGB screenshots                  │
│  - Apply inpainting to remove text overlays                 │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Register Main Anatomical Image                     │
│  - Register t1_pre_out → mre_raw (rigid transform)          │
│  - 6 DOF: 3 rotations + 3 translations                      │
│  - Uses SimpleITK/Elastix                                   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Liver Segmentation                                 │
│  - Resize t1_pre_out → (256, 256, 32)                       │
│  - Run pre-trained U-Net model                              │
│  - Output: Binary liver mask                                │
│  - Model: Trained on CHAOS dataset                          │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Register Mask to MRE Grid                          │
│  - Apply same transform to mask                             │
│  - Create anat_mask and mre_mask (identical)                │
│  - Use nearest-neighbor interpolation (preserve labels)     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 6: Register All Other Sequences                       │
│  - Register each sequence to t1_pre_out                     │
│  - All images now in same coordinate system                 │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 7: Resize to Standard Grid                            │
│  - Anatomical: (256, 256, 4)                                │
│  - MRE: (256, 256, 4)                                       │
│  - Masks: (256, 256, 4)                                     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 8: Convert to XArray and Save                         │
│  Output:                                                     │
│  - wave.nc (complex displacement)                           │
│  - mre.nc (complex elasticity)                              │
│  - mre_mask.nc (liver segmentation)                         │
│  - anat.nc (5 MRI sequences stacked)                        │
│  - anat_mask.nc (liver segmentation)                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Functions

#### `load_nifti_file()`

```python
def load_nifti_file(nii_file, verbose=True):
    """
    Load NIFTI file as SimpleITK image.

    Args:
        nii_file: Path to .nii file
        verbose: Print loading message

    Returns:
        sitk.Image
    """
```

#### `register_image()`

```python
def register_image(moving_image, fixed_image, transform='rigid'):
    """
    Register (align) two 3D images.

    Uses Elastix registration framework.

    Args:
        moving_image: Image to transform
        fixed_image: Reference image
        transform: 'rigid' (6 DOF) or 'affine' (12 DOF)

    Returns:
        aligned_image: Transformed image
        transform_params: Transformation parameters
    """
```

**Transform Parameters**:
```python
# Example output
TransformParameters: [-0.055, 0.016, 0.014, 11.3, -5.8, -29.1]
                     └─── rotations (rad) ──┘ └─── translations (mm) ──┘
```

#### `segment_image()`

```python
def segment_image(image, model=None, verbose=True):
    """
    Segment liver using pre-trained U-Net.

    Args:
        image: SimpleITK image (256×256×32)
        model: Pre-trained UNet3D model
        verbose: Print segmentation message

    Returns:
        mask_image: Binary segmentation mask
    """
```

**Segmentation Process**:
1. Normalize intensity: percentile (0.5, 99.5) → [-1, 1]
2. Forward pass through U-Net
3. Sigmoid activation
4. Threshold at 0.5 → binary mask

#### `restore_wave_image()`

```python
def restore_wave_image(wave_image, vmax, verbose=True):
    """
    Convert wave field from RGB screenshot to grayscale values.

    The wave images are RGB screenshots with:
    - Red channel: Positive displacement
    - Blue channel: Negative displacement
    - Text overlays: Removed via inpainting

    Args:
        wave_image: RGB image (H×W×3)
        vmax: Maximum displacement value (m)

    Returns:
        restored_image: Grayscale displacement field
    """
```

**Restoration Algorithm**:
```python
# Decode from RGB
array = (array_r + array_gr)/512 - (array_b + array_gb)/512
array *= vmax

# Remove text with inpainting
array_txt = (R==255) & (G==255) & (B==255)  # White pixels
array_txt = binary_dilation(array_txt)      # Expand slightly
array = inpaint_biharmonic(array, array_txt)  # Interpolate
```

#### `resize_image()`

```python
def resize_image(image, out_size, verbose=True):
    """
    Resize image to target dimensions.

    Maintains spatial extent, adjusts spacing.

    Args:
        image: SimpleITK image
        out_size: Target size (e.g., (256, 256, 4))

    Returns:
        resized_image: Resampled image
    """
```

---

## bioqic.py - Simulation Data

### Purpose

Load and preprocess synthetic MRE data from finite element simulations.

### Key Classes

#### `BIOQICFEMBox` - Box Phantom

```python
class BIOQICFEMBox:
    """
    Load BIOQIC four-target box phantom simulation.

    Simulation of a box with 4 spherical inclusions at different stiffnesses.

    Ground truth elasticity:
    - Background: 3 kPa
    - Inclusions: 10 kPa

    Frequencies: 50, 60, 70, 80, 90, 100 Hz
    Grid: 80×100×10 voxels
    """
```

**Usage**:
```python
bioqic = mre_pinn.data.BIOQICFEMBox('../data/BIOQIC/downloads')
bioqic.download()           # Download .mat file
bioqic.load_mat()           # Load from MATLAB
bioqic.preprocess()         # Create elastogram
dataset = bioqic.to_dataset()
dataset.save_xarrays('../data/BIOQIC/fem_box')
```

#### `BIOQICPhantom` - Physical Phantom

```python
class BIOQICPhantom:
    """
    Load BIOQIC physical phantom MRI data.

    Real MRI scan of gelatin phantom with inclusions.

    Frequencies: 50, 60, 70, 80, 90, 100 Hz
    Grid: 80×100×25 voxels
    """
```

### Ground Truth Creation

For simulation data, ground truth is **analytical**:

```python
def create_elastogram(self):
    """
    Create ground truth elasticity map from spatial regions.

    Steps:
    1. Segment phantom into regions (background + inclusions)
    2. Assign known stiffness values to each region
    3. Apply Voigt model: μ = μ_real + iωη
    """

    # Ground truth stiffness (Pa)
    mu = np.array([
        0,      # Background (outside phantom)
        3e3,    # Phantom background
        10e3,   # Inclusion 1
        10e3,   # Inclusion 2
        10e3,   # Inclusion 3
        10e3    # Inclusion 4
    ])

    # Apply Voigt model (viscoelasticity)
    eta = 1  # Pa·s (viscosity)
    mu = mu + 1j * omega * eta
```

---

## dataset.py - Unified Data Interface

### Purpose

Provide common interface for all data types (simulation, patient).

### Key Classes

#### `MREExample` - Single Example

```python
class MREExample:
    """
    Single MRE dataset example (one patient or one simulation).

    Attributes:
        example_id: str or int
        wave: xarray.DataArray - Wave displacement field
        mre: xarray.DataArray - Elasticity (ground truth)
        mre_mask: xarray.DataArray - Segmentation mask
        anat: xarray.DataArray - Anatomical images (optional)

    Methods:
        view() - Visualize data
        save_xarrays() - Save to disk
        load_xarrays() - Load from disk
        describe() - Descriptive statistics
    """
```

**Data Structure**:
```python
example = MREExample(
    example_id='0006',
    wave=<xarray (256,256,4) complex>,
    mre=<xarray (256,256,4) complex>,
    mre_mask=<xarray (256,256,4) float>,
    anat=<xarray (256,256,4,5) float>  # 5 sequences
)
```

#### `MREDataset` - Multiple Examples

```python
class MREDataset:
    """
    Collection of MRE examples.

    Attributes:
        example_ids: list - List of example IDs
        examples: dict - Dictionary of MREExample objects

    Methods:
        __getitem__() - Access by index or ID
        __len__() - Number of examples
        split() - Train/val/test split
    """
```

### XArray File Format

Data is saved as NetCDF files (.nc):

```
data/NAFLD/v4/0006/
├── wave.nc         # Wave displacement (complex)
├── mre.nc          # Elasticity (complex)
├── mre_mask.nc     # Liver mask (float)
├── anat.nc         # Anatomical images (float)
└── anat_mask.nc    # Liver mask (float)
```

**XArray Structure**:
```python
<xarray.DataArray 'wave' (x: 256, y: 256, z: 4)>
array([...complex values...])
Coordinates:
  * x        (x) float64 -0.414 -0.411 ... 0.301 0.303  # meters
  * y        (y) float64 -0.392 -0.389 ... 0.323 0.325
  * z        (z) float64 -0.002 0.020 0.042 0.064
    region   (x, y, z) float32 ...                      # mask
```

### Key Methods

#### `load_xarrays()`

```python
@classmethod
def load_xarrays(cls, xarray_dir, example_id, anat=False, verbose=True):
    """
    Load preprocessed xarray files from disk.

    Args:
        xarray_dir: Base directory (e.g., '../data/NAFLD/v4')
        example_id: Patient/example ID
        anat: Load anatomical images
        verbose: Print loading messages

    Returns:
        MREExample object
    """
```

**Usage**:
```python
# Load patient data
example = mre_pinn.data.MREExample.load_xarrays(
    '../data/NAFLD/v4/',
    '0006',
    anat=True
)

# Load simulation data
example = mre_pinn.data.MREExample.load_xarrays(
    '../data/BIOQIC/fem_box/',
    frequency=90
)
```

#### `view()`

```python
def view(self, *variables, mask=False, **kwargs):
    """
    Interactive visualization of data.

    Args:
        *variables: Variable names to view ('wave', 'mre', 'anat')
        mask: Apply segmentation mask
        **kwargs: Visualization options (vmax, cmap, polar, etc.)
    """
```

**Usage**:
```python
# View wave field
example.view('wave', ax_height=3)

# View multiple variables
example.view('wave', 'mre', mask=True)

# Custom visualization
example.view('mre', polar=True, vmax=10e3, mask=True)
```

#### `describe()`

```python
def describe(self):
    """
    Get descriptive statistics for all variables.

    Returns:
        pandas.DataFrame with stats (mean, std, min, max, percentiles)
    """
```

**Example Output**:
```
                         dtype    count         mean          std
variable component
wave     scalar        float64  262144.0     0.000139     0.023588
mre      scalar          int16  262144.0  1505.871937  1335.735833
mre_mask scalar        float32  262144.0     0.171467     0.376916
```

---

## segment.py - Liver Segmentation

### Purpose

3D U-Net model for automatic liver segmentation from T1-weighted MRI.

### UNet3D Architecture

```python
class UNet3D(nn.Module):
    """
    3D U-Net for liver segmentation.

    Input: (1, 256, 256, 32) - T1-weighted MRI
    Output: (1, 256, 256, 32) - Binary liver mask

    Architecture:
    - Encoder: 4 down-sampling blocks (64→128→256→512 channels)
    - Decoder: 3 up-sampling blocks (512→256→128→64)
    - Skip connections between encoder/decoder
    - Batch normalization + ReLU activation
    """
```

**Network Structure**:
```
Input (256×256×32)
    │
    ├─► DownTransition (1→64)   ──┐
    ├─► DownTransition (64→128)  ─┤
    ├─► DownTransition (128→256) ─┤
    ├─► DownTransition (256→512) ─┘
    │                              │
    └─► UpTransition (512→256) ◄──┤
        UpTransition (256→128) ◄──┤
        UpTransition (128→64)  ◄──┘
        Conv3D (64→1) + Sigmoid
            │
        Output (256×256×32)
```

### Pre-trained Model

**Location**:
```
/ocean/projects/asc170022p/bpollack/mre_ai/data/CHAOS/
trained_models/001/model_2020-09-30_11-14-20.pkl
```

**Training Dataset**: CHAOS Challenge (Combined Healthy Abdominal Organ Segmentation)
- Task: Liver segmentation from CT and MRI
- Modality: T1-weighted MRI
- Date: September 2020

### Usage Example

```python
# Load pre-trained model
model = load_segment_model('cuda')

# Segment image
mask = segment_image(t1_image, model=model)
```

---

## Data Flow Diagrams

### Patient Data: Load → Preprocess → Save

```
NIFTI Files
    ├─ t1_pre_in.nii
    ├─ t1_pre_out.nii
    ├─ mre.nii
    └─ wave.nii
        │
        ▼
   ImagingPatient('0006')
        │
        ├─► load_images()
        │
        ├─► preprocess()
        │   ├─ fix_metadata()
        │   ├─ restore_wave()
        │   ├─ register()
        │   ├─ segment()
        │   └─ resize()
        │
        ├─► to_xarrays()
        │
        ▼
  XArray Files
    ├─ wave.nc
    ├─ mre.nc
    ├─ mre_mask.nc
    ├─ anat.nc
    └─ anat_mask.nc
```

### Simulation Data: Download → Load → Process

```
MATLAB File (.mat)
    └─ four_target_phantom.mat
        │
        ▼
   BIOQICFEMBox()
        │
        ├─► download()
        ├─► load_mat()
        ├─► preprocess()
        │   ├─ segment_regions()
        │   └─ create_elastogram()
        │
        ├─► to_dataset()
        │
        ▼
  XArray Files (per frequency)
    ├─ 50/wave.nc, mre.nc, mre_mask.nc
    ├─ 60/wave.nc, mre.nc, mre_mask.nc
    └─ ... (up to 100 Hz)
```

---

## Common Workflows

### 1. Load Patient Data

```python
import mre_pinn

# Single patient
example = mre_pinn.data.MREExample.load_xarrays(
    '../data/NAFLD/v4/',
    '0006',
    anat=True
)

# View data
example.view('wave', 'mre', mask=True)
print(example.describe())
```

### 2. Preprocess New Patient

```python
# Load raw data
patient = mre_pinn.data.ImagingPatient('0006')
patient.load_images()

# Preprocess
patient.preprocess(
    same_grid=True,
    mre_size=(256, 256, 4),
    anat_size=(256, 256, 4)
)

# Convert to MREExample
arrays = patient.convert_images()
example = mre_pinn.data.MREExample.from_patient(patient)

# Save
example.save_xarrays('../data/processed/')
```

### 3. Load Simulation Data

```python
# Download and preprocess
bioqic = mre_pinn.data.BIOQICFEMBox('../data/BIOQIC/downloads')
bioqic.download()
bioqic.load_mat()
bioqic.preprocess()

# Convert to dataset
dataset = bioqic.to_dataset()
dataset.save_xarrays('../data/BIOQIC/fem_box')

# Load specific frequency
example = mre_pinn.data.MREExample.load_xarrays(
    '../data/BIOQIC/fem_box/',
    frequency=90
)
```

### 4. Process Entire Cohort

```python
# Load cohort
with open('patient_list.txt') as f:
    patient_ids = [line.strip() for line in f]

cohort = mre_pinn.data.ImagingCohort(patient_ids)

# Preprocess all
cohort.load_images()   # Takes ~1 min per patient
cohort.preprocess()     # Takes ~3 min per patient (includes segmentation)

# Save
dataset = cohort.to_dataset()
dataset.save_xarrays('../data/processed/')
```

---

## Coordinate Systems and Units

### Spatial Coordinates

- **Units**: Meters (m)
- **Convention**: Right-handed Cartesian (x, y, z)
- **Origin**: Varies by scanner/patient

**Conversion from mm to m**:
```python
coords[dim] = (origin[i] + np.arange(size[i]) * spacing[i]) * 2e-3
```

### Image Metadata

Each image has associated metadata:
```python
image.GetSize()      # (width, height, depth) in voxels
image.GetSpacing()   # (dx, dy, dz) in mm/voxel
image.GetOrigin()    # (x0, y0, z0) in mm
image.GetDirection() # 3×3 rotation matrix
```

---

## Troubleshooting

### Common Issues

**1. Missing sequences**
```python
# Error: Patient missing 'wave' sequence
# Solution: Check NIFTI directory structure
patient = ImagingPatient('0006')
print(patient.find_sequences())  # See what's available
```

**2. Registration failure**
```python
# Error: Registration produces large residuals
# Solution: Check if images are in similar coordinate systems
print(patient.metadata)  # Check origin/spacing
```

**3. Segmentation quality**
```python
# Low quality segmentation
# Solution: Adjust intensity normalization
a_min, a_max = np.percentile(array, (1.0, 99.0))  # Try different percentiles
```

**4. Memory errors**
```python
# Error: Out of memory during cohort processing
# Solution: Process patients individually
for pid in patient_ids:
    patient = ImagingPatient(pid)
    patient.load_images()
    patient.preprocess()
    # ... save and clear
    del patient
```

---

## Performance Considerations

### Memory Usage

- **Single patient** (all sequences): ~500 MB RAM
- **Cohort** (100 patients): ~50 GB RAM (process sequentially!)
- **Segmentation model**: ~200 MB GPU memory

### Processing Time

| Operation | Time | Hardware |
|-----------|------|----------|
| Load patient images | ~30 sec | SSD |
| Register images | ~2 min | CPU |
| Segment liver | ~5 sec | GPU |
| Full preprocessing | ~3 min | CPU+GPU |
| Cohort (100 patients) | ~5 hours | CPU+GPU |

---

## See Also

- [../PACKAGE_OVERVIEW.md](../PACKAGE_OVERVIEW.md) - Package overview
- [../model/MODEL_ARCHITECTURES.md](../model/MODEL_ARCHITECTURES.md) - Neural network models
- [../training/TRAINING_MODULE.md](../training/TRAINING_MODULE.md) - Training procedures
- [../../MICCAI-2023/EXPERIMENTS_GUIDE.md](../../MICCAI-2023/EXPERIMENTS_GUIDE.md) - Experiment notebooks
