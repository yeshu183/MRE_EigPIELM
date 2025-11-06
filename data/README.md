# data/ - Dataset Storage

Storage directory for MRE datasets (simulation and patient data).

---

## Directory Structure

```
data/
├── BIOQIC/              # Simulation data
│   ├── downloads/       # Original .mat files
│   ├── fem_box/         # Preprocessed box phantom
│   │   ├── 50/         # 50 Hz data
│   │   ├── 60/         # 60 Hz data
│   │   └── ...         # Up to 100 Hz
│   └── phantom/        # Physical phantom data
│
└── NAFLD/              # Patient data (not included)
    ├── v3/             # Version 3 preprocessing
    └── v4/             # Version 4 preprocessing (current)
        ├── 0006/       # Patient 0006
        │   ├── wave.nc
        │   ├── mre.nc
        │   ├── mre_mask.nc
        │   ├── anat.nc
        │   └── anat_mask.nc
        └── ...         # 155 patients total
```

---

## Datasets

### BIOQIC (Simulation)

**Description**: Synthetic MRE data from finite element simulations

**Source**: BioQIC Quality Control phantom

**Contents**:
- `fem_box`: Four-target box phantom
- `phantom`: Physical phantom MRI data

**Download**:
```python
python download_data.py
# Or run MICCAI-2023-simulation-training.ipynb
```

**Size**: ~50 MB (compressed), ~200 MB (preprocessed)

**Frequencies**: 50, 60, 70, 80, 90, 100 Hz

---

### NAFLD (Patient Data)

**Description**: Clinical MRE data from NAFLD (liver disease) patients

**Source**: Clinical MRI scans (not publicly available)

**Contents**: 155 patients with full MRI sequences

**Note**: Patient data not included in repository (protected health information)

**Access**: Contact authors for collaboration

---

## File Formats

### XArray NetCDF (.nc)

All preprocessed data saved as NetCDF files containing xarray DataArrays.

**Example**:
```python
import xarray as xr

# Load wave field
wave = xr.open_dataarray('data/BIOQIC/fem_box/90/wave.nc')

print(wave)
# <xarray.DataArray 'wave' (x: 80, y: 100, z: 10)>
# Coordinates:
#   * x: float64 (meters)
#   * y: float64 (meters)
#   * z: float64 (meters)
# Data: complex128
```

---

## Data Files Per Example

| File | Description | Type | Size |
|------|-------------|------|------|
| `wave.nc` | Wave displacement field (INPUT) | complex | ~1 MB |
| `mre.nc` | Elasticity map (GROUND TRUTH) | complex | ~1 MB |
| `mre_mask.nc` | Segmentation mask | float | ~1 MB |
| `anat.nc` | Anatomical MRI (5 sequences) | float | ~5 MB |
| `anat_mask.nc` | Anatomical segmentation | float | ~1 MB |

**Total per patient**: ~9 MB

---

## Usage

### Load Data

```python
import mre_pinn

# Simulation data
example = mre_pinn.data.MREExample.load_xarrays(
    'data/BIOQIC/fem_box/',
    frequency=90
)

# Patient data (if available)
example = mre_pinn.data.MREExample.load_xarrays(
    'data/NAFLD/v4/',
    '0006',
    anat=True
)
```

### View Data

```python
# Interactive visualization
example.view('wave', 'mre', mask=True)

# Statistics
print(example.describe())
```

---

## Download Instructions

### Automated Download

```bash
python download_data.py
```

Downloads and preprocesses BIOQIC simulation data.

### Manual Download

1. Download from: [BioQIC website]
2. Place .mat files in `data/BIOQIC/downloads/`
3. Run preprocessing notebook

---

## Storage Requirements

| Dataset | Compressed | Preprocessed | Notes |
|---------|-----------|--------------|-------|
| BIOQIC | 50 MB | 200 MB | 6 frequencies |
| NAFLD (1 patient) | N/A | 9 MB | All sequences |
| NAFLD (155 patients) | N/A | 1.4 GB | Full cohort |

---

## Data Versioning

### NAFLD Versions

- **v1**: Initial preprocessing (deprecated)
- **v2**: Fixed metadata issues (deprecated)
- **v3**: Added anatomical sequences
- **v4**: Current version (improved segmentation)

**Recommendation**: Use v4 for all new experiments

---

## gitignore

Large data files excluded from git:
```
data/NAFLD/        # Patient data (protected)
data/BIOQIC/*.mat  # Original MATLAB files
```

Only directory structure and README included in repository.

---

## See Also

- [../mre_pinn/data/README.md](../mre_pinn/data/README.md) - Data loading code
- [../MICCAI-2023/README.md](../MICCAI-2023/README.md) - Experiment notebooks
- [../download_data.py](../download_data.py) - Download script
