# Data Formats Guide for MRE-PINN

This document explains all the data formats used in the MRE-PINN project, including medical imaging formats, array formats, and why complex values are used.

---

## Table of Contents

1. [Overview](#overview)
2. [Medical Imaging Formats](#medical-imaging-formats)
3. [Scientific Data Formats](#scientific-data-formats)
4. [Complex Numbers in MRE](#complex-numbers-in-mre)
5. [Data Types Used](#data-types-used)
6. [Data Flow Pipeline](#data-flow-pipeline)

---

## Overview

The MRE-PINN project processes **Magnetic Resonance Elastography (MRE)** data using multiple data formats. The pipeline converts medical imaging data into formats suitable for Physics-Informed Neural Networks (PINNs).

**Key Data Transformations:**
```
NIfTI (.nii) → SimpleITK Image → NumPy Array → xarray DataArray → NetCDF (.nc)
MATLAB (.mat) → NumPy Array → xarray DataArray → NetCDF (.nc)
```

---

## Medical Imaging Formats

### 1. NIfTI (Neuroimaging Informatics Technology Initiative)

**File Extension:** `.nii` or `.nii.gz` (compressed)

**What is NIfTI?**
- NIfTI is a medical imaging file format designed for storing 3D/4D volumetric data
- Commonly used for MRI, fMRI, PET scans, and other medical imaging modalities
- Successor to the older ANALYZE format
- Contains both image data and spatial metadata (orientation, spacing, origin)

**Structure:**
- **Header:** Contains metadata about the image (dimensions, data type, spatial information)
- **Image Data:** Multi-dimensional array of voxel intensities

**Key Metadata in NIfTI:**
- `size`: Number of voxels in each dimension (e.g., 512 × 512 × 88)
- `spacing`: Physical distance between voxels in millimeters (e.g., 0.703 mm)
- `origin`: Physical coordinates of the first voxel in space
- `direction`: Orientation matrix defining anatomical axes

**In This Project:**
NIfTI files store patient MRI and MRE sequences:
- Anatomical sequences: `t1_pre_in`, `t1_pre_out`, `t1_pre_water`, `t1_pre_fat`, `t2`
- MRE sequences: `mre_raw`, `wave`, `mre`

**Example from code** ([imaging.py:206](mre_pinn/data/imaging.py#L206)):
```python
nii_file = self.nifti_dir / self.patient_id / (seq + '.nii')
image = sitk.ReadImage(str(nii_file))  # Load NIfTI using SimpleITK
```

**Why NIfTI?**
- Industry standard for medical imaging research
- Preserves spatial metadata critical for accurate analysis
- Compatible with medical imaging software (ITK-SNAP, 3D Slicer, FSL)

---

### 2. MATLAB Files (.mat)

**File Extension:** `.mat`

**What are MATLAB files?**
- Binary files created by MATLAB for saving workspace variables
- Can store arrays, structures, and other MATLAB data types
- Two versions: v7 (older, uses scipy) and v7.3 (HDF5-based, uses h5py)

**In This Project:**
Used for BIOQIC simulation datasets downloaded from https://bioqic-apps.charite.de/

**Example Data Structure** ([bioqic.py:462-488](mre_pinn/data/bioqic.py#L462-L488)):
```python
data = scipy.io.loadmat('four_target_phantom.mat')
# Contains:
#   'u_ft': (100, 80, 10, 3, 6) complex128 array
#   '__header__': File metadata
#   '__version__': MATLAB version
```

**Key Variables:**
- `u_ft`: Fourier-transformed displacement field (wave data)
- `phase_unwrapped`: Unwrapped phase images
- `magnitude`: Anatomical magnitude images

---

## Scientific Data Formats

### 3. xarray DataArray

**What is xarray?**
- Python library for working with labeled multi-dimensional arrays
- Think of it as "pandas for N-dimensional data"
- Provides named dimensions and coordinate labels

**Why xarray?**
- **Self-describing:** Data comes with labels and coordinates
- **Alignment:** Automatic alignment of data along shared dimensions
- **Metadata:** Preserves units, names, and spatial information
- **NetCDF integration:** Native support for saving/loading

**Structure in This Project:**

**Wave Field DataArray:**
```python
<xarray.DataArray 'wave' (x: 256, y: 256, z: 4, component: 3)>
Coordinates:
  * x          (x) float64 -0.4138 ... 0.3034  # meters
  * y          (y) float64 -0.3917 ... 0.3255  # meters
  * z          (z) float64 -0.002113 ... 0.06389  # meters
  * component  (component) <U1 'x' 'y' 'z'  # displacement direction
    region     (x, y, z) int32  # tissue segmentation mask
```

**Dimensions Explained:**
- `x, y, z`: Spatial coordinates in meters
- `component`: Vector components of displacement (x, y, z directions)
- `frequency`: Vibration frequency in Hz (for multi-frequency data)

**Elastogram DataArray:**
```python
<xarray.DataArray 'mre' (x: 256, y: 256, z: 4)>
dtype: complex128  # Complex shear modulus
```

**Custom Field Accessor:**
The project extends xarray with a custom `.field` accessor ([fields.py](mre_pinn/fields.py)) to provide:
- `array.field.spatial_dims`: Returns `['x', 'y', 'z']`
- `array.field.spatial_shape`: Returns `(256, 256, 4)`
- `array.field.spatial_resolution`: Returns voxel spacing
- `array.field.values()`: Flattens for neural network input

---

### 4. NetCDF (Network Common Data Form)

**File Extension:** `.nc`

**What is NetCDF?**
- Self-describing, machine-independent data format
- Standard for scientific data (climate, oceanography, medical imaging)
- Stores multi-dimensional arrays with metadata
- Created by Unidata (University Corporation for Atmospheric Research)

**Why NetCDF in This Project?**
- Efficient storage of multi-dimensional MRE data
- Preserves all spatial metadata automatically
- Platform-independent (works on Windows, Linux, Mac)
- Compression support for large datasets

**Example from code** ([dataset.py:262-267](mre_pinn/data/dataset.py#L262-L267)):
```python
def save_xarray_file(nc_file, array, verbose=True):
    if np.iscomplexobj(array):
        # Split complex into real and imaginary parts
        new_dim = xr.DataArray(['real', 'imag'], dims=['part'])
        array = xr.concat([array.real, array.imag], dim=new_dim)
    array.to_netcdf(nc_file)  # Save to NetCDF format
```

**Directory Structure:**
```
data/BIOQIC/fem_box/
├── 50/
│   ├── wave.nc       # Displacement field at 50 Hz
│   ├── mre.nc        # Elasticity map at 50 Hz
│   └── mre_mask.nc   # Tissue segmentation mask
├── 60/
│   ├── wave.nc
│   ├── mre.nc
│   └── mre_mask.nc
...
```

---

## Complex Numbers in MRE

### Why Complex Values?

MRE data uses **complex numbers** extensively. Here's why:

#### 1. Wave Propagation is Harmonic

MRE uses **mechanical vibrations** (shear waves) propagating through tissue:

```
Displacement: u(x, t) = A × exp(i(k·x - ωt))
```

Where:
- `A`: Wave amplitude
- `k`: Wave vector (spatial frequency)
- `ω`: Angular frequency (temporal frequency)
- `i`: Imaginary unit (√-1)

**In Fourier Domain:**
After Fourier transform, the time-varying wave becomes a complex field:
- **Real part:** Represents in-phase component
- **Imaginary part:** Represents 90° out-of-phase component

**Example from notebook:**
```python
# Wave field shape: (x, y, z, component)
# dtype: complex128
wave[i, j, k, 'z'] = 0.00428 + 0.00195j  # Complex displacement in z-direction
```

#### 2. Viscoelastic Materials Have Complex Modulus

Biological tissues are **viscoelastic** (both elastic and viscous):

```
μ* = μ' + iμ''  # Complex shear modulus
```

- **μ' (real part):** Storage modulus (elasticity) - energy stored
- **μ'' (imaginary part):** Loss modulus (viscosity) - energy dissipated

**Physical Interpretation:**
- Pure elastic material: `μ'' = 0` (perfectly bouncy, no energy loss)
- Pure viscous material: `μ' = 0` (like honey, no bounce)
- Real tissue: Both components exist

**Example from code** ([bioqic.py:156-158](mre_pinn/data/bioqic.py#L156-L158)):
```python
# Voigt viscoelastic model
eta = 1  # Pa·s (viscosity)
mu = mu_real + 1j * omega * eta  # Complex modulus
```

#### 3. Mathematical Convenience

Complex representation simplifies wave equations:

**Without complex numbers:**
```python
u_x(x,t) = A cos(kx - ωt)
u_y(x,t) = A sin(kx - ωt)
```

**With complex numbers:**
```python
u(x,t) = A exp(i(kx - ωt))  # Single equation!
```

#### 4. Phase Information

Complex values naturally encode **both amplitude and phase**:

```python
z = r × exp(iθ)  # Polar form
  = r cos(θ) + i × r sin(θ)  # Cartesian form

r = |z| = sqrt(real² + imag²)  # Magnitude
θ = angle(z) = arctan(imag/real)  # Phase
```

**Example from code:**
```python
# Viewing data in polar form
example.view('mre', polar=True, vmax=20e3)
# Shows magnitude and phase separately
```

---

### Complex Data Storage

Since NetCDF doesn't natively support complex numbers, they're split:

**Saving** ([dataset.py:264-266](mre_pinn/data/dataset.py#L264-L266)):
```python
if np.iscomplexobj(array):
    array = xr.concat([array.real, array.imag],
                      dim=xr.DataArray(['real', 'imag'], dims=['part']))
```

**Loading** ([dataset.py:273-278](mre_pinn/data/dataset.py#L273-L278)):
```python
if 'part' in array.dims:
    real = array.sel(part='real')
    imag = array.sel(part='imag')
    return real + 1j * imag
```

---

## Data Types Used

### NumPy Data Types

| Type | Bytes | Description | Usage |
|------|-------|-------------|-------|
| `float32` | 4 | Single precision float | Anatomical MRI images |
| `float64` | 8 | Double precision float | Coordinates, spatial metadata |
| `complex64` | 8 | Single precision complex | Not used (insufficient precision) |
| `complex128` | 16 | Double precision complex | **Wave fields, elastograms** |
| `int32` | 4 | 32-bit integer | Segmentation masks, region labels |

**Why complex128?**
- MRE involves phase-sensitive measurements
- Small phase errors accumulate in wave inversion
- Double precision minimizes numerical errors

**Example from notebook:**
```python
example.describe()
#                     dtype    count                      mean
# wave/y          complex128  80000.0    -0.000001-0.000005j
# wave/x          complex128  80000.0     0.000079-0.000073j
# wave/z          complex128  80000.0    -0.000287-0.000249j
# mre/scalar      complex128  80000.0  3382.375000+565.487j
```

---

## Data Flow Pipeline

### Simulation Data (BIOQIC)

```
1. Download:
   MATLAB .mat file
   ↓
2. Load and Parse:
   scipy.io.loadmat() or h5py
   ↓
3. Add Metadata:
   numpy array → xarray.DataArray
   + spatial coordinates (x, y, z)
   + frequency coordinates
   + component labels
   ↓
4. Preprocessing:
   - Segment spatial regions (tissue types)
   - Create ground truth elastograms
   - Apply physics-based models (Voigt, Springpot)
   ↓
5. Save to NetCDF:
   wave.nc, mre.nc, mre_mask.nc
   (complex values split into real/imag)
```

### Patient Data (Clinical MRI/MRE)

```
1. Medical Scanner:
   DICOM files (from MRI scanner)
   ↓
2. Conversion:
   DICOM → NIfTI (.nii files)
   ↓
3. Load with SimpleITK:
   NIfTI → sitk.Image object
   (preserves spatial metadata)
   ↓
4. Preprocessing:
   a) Metadata correction
   b) Wave image restoration (RGB → grayscale, inpainting)
   c) Image registration (align sequences)
   d) Image segmentation (liver detection with UNet3D)
   e) Resizing/resampling
   ↓
5. Convert to xarray:
   sitk.Image → numpy array → xarray.DataArray
   + coordinates from image metadata
   ↓
6. Save to NetCDF:
   Patient folders with wave.nc, mre.nc, anat.nc, masks
```

### Neural Network Training

```
1. Load NetCDF:
   xr.open_dataarray('wave.nc')
   ↓
2. Extract Values:
   array.field.values()  # (N_points, N_components)
   ↓
3. Normalization:
   (values - mean) / std
   ↓
4. PyTorch Tensors:
   torch.tensor(values, dtype=torch.float32)
   ↓
5. PINN Training:
   Input: (x, y, z) coordinates
   Output: Complex wave field u(x,y,z), elasticity μ(x,y,z)
```

---

## Detailed Format Specifications

### Wave Field Format

**Dimensions:** `(frequency, x, y, z, component)` or `(x, y, z, component)`

**Coordinates:**
- `frequency`: [50, 60, 70, 80, 90, 100] Hz (BIOQIC) or single value (patient)
- `x, y, z`: Spatial coordinates in meters
- `component`: ['x', 'y', 'z'] for displacement vector components

**Data Type:** `complex128`

**Physical Units:**
- Displacement in **meters** (typically microns: 10⁻⁶ m)
- Complex representation of harmonic motion

**Example Values:**
```
wave[40, 50, 2, 'z'] = -0.000287 - 0.000249j  # meters
|wave| = sqrt(0.000287² + 0.000249²) = 0.00038 m = 380 μm
phase = arctan(-0.000249 / -0.000287) = 0.71 rad = 41°
```

### Elastogram (MRE) Format

**Dimensions:** `(frequency, x, y, z)` or `(x, y, z)`

**Data Type:** `complex128`

**Physical Units:** Pascals (Pa)

**Components:**
- **Real part (μ'):** Shear storage modulus (elasticity)
- **Imaginary part (μ''):** Shear loss modulus (viscosity)

**Typical Values:**
- Healthy liver: μ' ≈ 2000-2500 Pa
- Fibrotic liver: μ' ≈ 3000-5000 Pa
- Cirrhotic liver: μ' ≈ 5000-10000+ Pa
- Viscosity: μ'' ≈ 500-1000 Pa

**Example from simulation:**
```python
mu = np.array([0, 3e3, 10e3, 10e3, 10e3, 10e3])  # Pa
# Region 0: Background (0 Pa)
# Region 1: Soft matrix (3000 Pa)
# Regions 2-5: Stiff inclusions (10000 Pa)
```

### Anatomical Image Format

**Dimensions:** `(x, y, z, component)` where component = MRI sequences

**Sequences (component dimension):**
- `t1_pre_in`: T1-weighted in-phase
- `t1_pre_out`: T1-weighted out-of-phase
- `t1_pre_water`: T1-weighted water image
- `t1_pre_fat`: T1-weighted fat image
- `t2`: T2-weighted image

**Data Type:** `float32` (pixel intensities, arbitrary units)

**Purpose:** Anatomical context for elasticity maps

### Mask Format

**Dimensions:** `(x, y, z)`

**Data Type:** `int32`

**Values:**
- `0`: Background (outside tissue)
- `1`: Tissue region 1 (e.g., liver parenchyma)
- `2-N`: Additional tissue regions or inclusions

**Purpose:** Spatial segmentation for analysis

---

## Summary

| Format | Extension | Library | Purpose | Preserves Metadata |
|--------|-----------|---------|---------|-------------------|
| **NIfTI** | `.nii` | SimpleITK | Medical image storage | ✓ (spatial info) |
| **MATLAB** | `.mat` | scipy.io / h5py | Simulation data input | ✗ (must add) |
| **xarray** | In-memory | xarray | Labeled arrays with coords | ✓ (dimensions, coords) |
| **NetCDF** | `.nc` | xarray/netCDF4 | Efficient scientific data storage | ✓ (all metadata) |
| **PyTorch** | In-memory | torch | Neural network training | ✗ (numerical only) |

**Why These Formats?**

1. **NIfTI:** Standard medical imaging format, maintains spatial information from MRI scanner
2. **xarray:** Provides labeled dimensions, automatic alignment, and readable code
3. **NetCDF:** Industry-standard scientific data format with compression and metadata
4. **Complex numbers:** Natural representation of wave physics and viscoelastic properties

**Key Insight:**
The data flow preserves **physical meaning** at every step. A voxel at coordinate (0.02, 0.03, 0.005) meters always refers to the same physical location in space, whether in NIfTI, xarray, or NetCDF format.

---

## Additional Resources

- **xarray documentation:** http://xarray.pydata.org/
- **NetCDF User Guide:** https://www.unidata.ucar.edu/software/netcdf/
- **NIfTI format specification:** https://nifti.nimh.nih.gov/
- **SimpleITK:** https://simpleitk.org/
- **MRE physics:** Muthupillai et al. (1995) "Magnetic resonance elastography"

---

*This guide was created for the MRE-PINN project to help understand the various data formats and transformations involved in processing Magnetic Resonance Elastography data for Physics-Informed Neural Networks.*
