# MRE-PINN Architecture Overview

This document provides a comprehensive overview of the MRE-PINN (Magnetic Resonance Elastography - Physics-Informed Neural Network) repository structure, data flow, and implementation details.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Data Flow Overview](#data-flow-overview)
3. [Key Concepts](#key-concepts)
4. [Module Descriptions](#module-descriptions)
5. [Workflow Diagrams](#workflow-diagrams)

---

## Repository Structure

```
MRE-PINN/
├── mre_pinn/                    # Main Python package
│   ├── data/                    # Data loading and preprocessing
│   ├── model/                   # PINN architecture
│   ├── training/                # Training loop and losses
│   ├── testing/                 # Evaluation and metrics
│   ├── baseline/                # Baseline methods (AHI, FEM)
│   ├── pde.py                   # Physics equations
│   ├── fields.py                # Spatial field operations
│   ├── utils.py                 # Utility functions
│   └── visual.py                # Visualization tools
├── MICCAI-2023/                 # Experiment notebooks
├── data/                        # Dataset storage
└── train.py                     # Training script
```

---

## Data Flow Overview

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. DATA ACQUISITION                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ├─ Simulation: BIOQIC dataset (FEM)
                              └─ Patient: NAFLD cohort (Clinical MRI)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2. DATA PREPROCESSING                        │
│  (mre_pinn/data/imaging.py, bioqic.py)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼────────────────────────┐
    │                         │                        │
    ▼                         ▼                        ▼
┌──────────┐          ┌──────────────┐         ┌──────────┐
│  Wave    │          │  Elasticity  │         │  Masks   │
│  Field   │          │  (Ground     │         │ (Liver   │
│ (Input)  │          │   Truth)     │         │  Segm.)  │
└──────────┘          └──────────────┘         └──────────┘
    │                         │                        │
    └─────────────────────────┼────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. PINN TRAINING                             │
│  (mre_pinn/training/pinn_training.py)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼────────────────────────┐
    │                         │                        │
    ▼                         ▼                        ▼
┌──────────┐          ┌──────────────┐         ┌──────────┐
│  Wave    │          │   Physics    │         │ Anatomic │
│  Loss    │          │   Loss (PDE) │         │   Loss   │
└──────────┘          └──────────────┘         └──────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    4. PREDICTION                                │
│  Neural Network learns to:                                      │
│  - Reconstruct wave field from spatial coordinates             │
│  - Predict elasticity (stiffness) map                          │
│  - Satisfy wave equation (physics constraint)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    5. EVALUATION                                │
│  (mre_pinn/testing/generic.py)                                 │
│  - Compute metrics (MSAE, correlation, MAD)                    │
│  - Compare with baselines (AHI, FEM)                           │
│  - Visualize results                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### 1. What is MRE (Magnetic Resonance Elastography)?

**Medical imaging technique** to measure tissue stiffness:
- Apply mechanical vibrations to tissue (40-100 Hz)
- MRI measures resulting wave propagation
- Reconstruct **elasticity map** (tissue stiffness)
- Clinical use: Diagnose liver disease (fibrosis, cirrhosis)

### 2. The Problem

```
INPUT:   Wave field u(x,y,z) - tissue displacement (measured)
OUTPUT:  Elasticity μ(x,y,z) - tissue stiffness (unknown)

Traditional Methods (AHI, FEM):
❌ Sensitive to noise
❌ Low spatial resolution
❌ Make simplifying assumptions

PINN Approach:
✅ Physics-informed (wave equation constraint)
✅ Robust to noise (learned from data)
✅ Can use anatomical priors
✅ Higher accuracy
```

### 3. Physics-Informed Neural Networks (PINNs)

**Key Idea**: Train neural network with two objectives:
1. **Data fitting**: Match measured wave field
2. **Physics**: Satisfy wave equation PDE

**Wave Equation** (governs tissue motion):
```
∇·[μ(∇u + ∇u^T)] + ρω²u = 0

Where:
- u = displacement field (complex-valued)
- μ = shear modulus (tissue stiffness) - what we want!
- ρ = tissue density (≈1000 kg/m³)
- ω = angular frequency (2π × 40-100 Hz)
```

**PINN Architecture**:
```
Input: (x, y, z) spatial coordinates
       ↓
   [Neural Network]
       ↓
Output: u(x,y,z) - wave field
        μ(x,y,z) - elasticity
```

### 4. Ground Truth vs Predictions

| Data Type | Wave Field (Input) | Elasticity (Output) |
|-----------|-------------------|---------------------|
| **Simulation** | FEM-computed | Analytical (known values) |
| **Patient** | MRI measurement | Clinical scanner reconstruction |

**Important**: For patient data, "ground truth" elasticity is from clinical MRI scanner (not perfect, but best available).

---

## Module Descriptions

### mre_pinn/data/ - Data Management

**Purpose**: Load, preprocess, and manage MRE datasets

| File | Purpose | Key Functions |
|------|---------|---------------|
| `imaging.py` | Patient data preprocessing | `ImagingPatient`, `ImagingCohort` |
| `bioqic.py` | Simulation data loading | `BIOQICFEMBox`, `BIOQICPhantom` |
| `dataset.py` | Unified data interface | `MREExample`, `MREDataset` |
| `segment.py` | Liver segmentation (U-Net) | `UNet3D` |

**Data Preprocessing Pipeline** (Patient Data):
1. Load NIFTI files (T1, T2, MRE sequences)
2. Register images (align to same coordinate system)
3. Segment liver (U-Net deep learning)
4. Resize to standard grid (256×256×4)
5. Convert to xarray format
6. Save preprocessed data

### mre_pinn/model/ - Neural Network Architecture

**Purpose**: Define PINN neural network models

| File | Purpose | Key Classes |
|------|---------|-------------|
| `pinn.py` | Main PINN architecture | `MREPINN` |
| `generic.py` | Base network components | `PINN` (base class) |

**MREPINN Architecture**:
```python
MREPINN(
    u_pinn: PINN,    # Wave field network
    mu_pinn: PINN    # Elasticity network
)

Each PINN:
    Input: (x, y, z) → [3 or 6 features with polar encoding]
    Hidden: [n_layers × n_hidden units] with skip connections
    Activation: sin, tanh, or swish
    Output: Complex-valued predictions
```

### mre_pinn/training/ - Training Loop

**Purpose**: Train PINN models with physics constraints

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `pinn_training.py` | Main training class | `MREPINNModel`, `MREPINNData` |
| `losses.py` | Loss functions | `msae_loss` (Mean Squared Absolute Error) |
| `callbacks.py` | Training callbacks | `Resampler` |

**Training Process**:
1. Sample training points from masked region (liver only)
2. Forward pass through network
3. Compute losses:
   - Wave loss: Match measured displacement
   - Elasticity loss: Match ground truth (optional)
   - PDE loss: Satisfy wave equation
   - Anatomical loss: Incorporate MRI features
4. Backpropagate and update weights
5. Periodically resample training points

**Loss Weights Schedule**:
```
Iterations 0-10k:     PDE weight = 0 (warmup phase)
Iterations 10k+:      PDE weight increases progressively
```

### mre_pinn/baseline/ - Comparison Methods

**Purpose**: Implement traditional MRE reconstruction methods

| File | Purpose | Methods |
|------|---------|---------|
| `direct.py` | Algebraic Helmholtz Inversion | AHI baseline |
| `fem.py` | Finite Element Method | FEM baseline |
| `filters.py` | Signal processing | Savitzky-Golay, curl |

**Baseline Methods**:
- **AHI (Algebraic Helmholtz Inversion)**: Fast, algebraic solution
- **FEM (Finite Element Method)**: Numerical PDE solver (accurate but slow)

### mre_pinn/testing/ - Evaluation

**Purpose**: Evaluate models and compute metrics

| File | Purpose | Key Classes |
|------|---------|-------------|
| `generic.py` | Test evaluator | `TestEvaluator` |

**Metrics Computed**:
- **MSAE**: Mean Squared Absolute Error
- **MAD**: Mean Absolute Deviation
- **R**: Pearson correlation
- **PSD**: Power Spectral Density

### mre_pinn/pde.py - Physics Equations

**Purpose**: Define wave equation PDEs

**Wave Equation Variants**:
1. **Helmholtz**: Assumes homogeneous medium
2. **Heterogeneous**: Allows spatially-varying elasticity (more accurate)

### mre_pinn/fields.py - Spatial Operations

**Purpose**: Spatial field operations (gradients, smoothing)

**Key Operations**:
- Compute spatial derivatives
- Smooth fields (Gaussian, Savitzky-Golay)
- Extract regions of interest

### mre_pinn/visual.py - Visualization

**Purpose**: Interactive visualization tools

**Features**:
- Interactive viewers for 3D data
- Colormaps for wave/elasticity
- Subplot grids for comparisons

---

## Workflow Diagrams

### Training Workflow

```
┌─────────────────────────────────────────────────────────┐
│  START: Load preprocessed data (wave, mre, masks)      │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Initialize PINN model                                  │
│  - u_pinn: wave network                                 │
│  - mu_pinn: elasticity network                          │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Training Loop (100k iterations)                        │
└─────────────────────────────────────────────────────────┘
         │
         ├─► Sample n_points from liver region (e.g., 1024)
         │
         ├─► Forward pass: (x,y,z) → u_pred, μ_pred
         │
         ├─► Compute losses:
         │   ├─ L_wave = MSAE(u_true, u_pred)
         │   ├─ L_elasticity = MSAE(μ_true, μ_pred)
         │   ├─ L_PDE = wave equation residual
         │   └─ L_anat = anatomical consistency
         │
         ├─► Total loss = w₁·L_wave + w₂·L_elast + w₃·L_PDE + w₄·L_anat
         │
         ├─► Backpropagate and update weights
         │
         ├─► Every 100 iters: Evaluate on full grid
         │
         └─► Every 1000 iters: Resample training points
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Save trained model and predictions                     │
└─────────────────────────────────────────────────────────┘
```

### Data Preprocessing Workflow (Patient Data)

```
┌─────────────────────────────────────────────────────────┐
│  START: Raw DICOM/NIFTI files from MRI scanner          │
│  - T1-weighted (in-phase, out-phase, water, fat)        │
│  - T2-weighted                                           │
│  - MRE wave field (RGB screenshots!)                    │
│  - MRE elasticity (clinical reconstruction)             │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Fix metadata                                   │
│  - Correct spatial coordinates                          │
│  - Restore wave field from RGB screenshots              │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: Image registration                             │
│  - Align T1 image to MRE coordinate system              │
│  - Use rigid transform (6 DOF: 3 rotations + 3 shifts) │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: Liver segmentation                             │
│  - Resize T1 to (256, 256, 32)                          │
│  - Run pre-trained U-Net model                          │
│  - Output: Binary liver mask                            │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4: Register mask to MRE grid                      │
│  - Apply same transform to mask                         │
│  - Create anat_mask and mre_mask                        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 5: Register all other images                      │
│  - Align remaining sequences to reference               │
│  - Resize all to (256, 256, 4)                          │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 6: Convert to xarray and save                     │
│  Output files:                                           │
│  - wave.nc (complex displacement field)                 │
│  - mre.nc (complex elasticity - ground truth)           │
│  - mre_mask.nc (liver segmentation)                     │
│  - anat.nc (5 MRI sequences)                            │
│  - anat_mask.nc (liver segmentation)                    │
└─────────────────────────────────────────────────────────┘
```

---

## Important Implementation Details

### 1. Coordinate Systems

All spatial coordinates are in **meters** (converted from mm):
```python
coords[dim] = (origin[i] + np.arange(size[i]) * spacing[i]) * 2e-3  # mm → m
```

### 2. Complex-Valued Data

Wave fields and elasticity are **complex-valued**:
```python
u = u_real + 1j * u_imag  # Displacement
μ = μ_real + 1j * μ_imag  # Elasticity (storage + loss modulus)
```

### 3. Masking Strategy

**Purpose**: Train only on liver tissue, ignore background

```python
# During training
mu_mask = example.mre_mask.values.reshape(-1)  # Boolean mask
x, u, mu = x[mu_mask], u[mu_mask], mu[mu_mask]  # Filter points
sample = torch.randperm(n_masked)[:n_points]    # Random subsample
```

### 4. Polar Coordinate Input

**Option**: Use polar coordinates for better performance:
```python
if polar_input:
    r = sqrt(x² + y²)
    sin_theta = y / r
    cos_theta = x / r
    input = [x, y, z, r, sin_theta, cos_theta]  # 6 features
else:
    input = [x, y, z]  # 3 features
```

### 5. Skip Connections

Networks use **dense skip connections** (similar to DenseNet):
```python
class PINN(nn.Module):
    def forward(self, x):
        h = x
        for layer in self.hidden_layers:
            h_new = layer(h)
            h = torch.cat([h, h_new], dim=1)  # Concatenate
        return self.output(h)
```

---

## File Naming Conventions

### Notebook Files
```
MICCAI-2023-{dataset}-{purpose}.ipynb

Examples:
- MICCAI-2023-simulation-training.ipynb
- MICCAI-2023-patient-experiments.ipynb
```

### Data Files
```
{dataset}/{example_id}/{variable}.nc

Examples:
- BIOQIC/fem_box/90/wave.nc
- NAFLD/v4/0006/mre.nc
```

### Model Checkpoints
```
{experiment_name}/{job_name}_model.pt

Examples:
- train_patient_0006_30_hetero_1e-02_model.pt
```

---

## Next Steps

For detailed information about specific modules:
- See `mre_pinn/README.md` for package overview
- See `mre_pinn/data/README.md` for data pipeline details
- See `mre_pinn/training/README.md` for training details
- See `MICCAI-2023/README.md` for experiment descriptions

---

## Quick Reference

### Common Abbreviations
- **MRE**: Magnetic Resonance Elastography
- **PINN**: Physics-Informed Neural Network
- **AHI**: Algebraic Helmholtz Inversion
- **FEM**: Finite Element Method
- **MSAE**: Mean Squared Absolute Error
- **PDE**: Partial Differential Equation
- **NAFLD**: Non-Alcoholic Fatty Liver Disease
- **BIOQIC**: BioQIC Quality Control phantom dataset

### Typical Values
- **Frequency**: 40-100 Hz (mechanical vibration)
- **Liver elasticity**: 1-15 kPa (healthy to cirrhotic)
- **Grid size**: 256×256×4 voxels
- **Training iterations**: 100k
- **Training time**: 2-3 hours (GPU)
- **Batch size**: 1024 spatial points
