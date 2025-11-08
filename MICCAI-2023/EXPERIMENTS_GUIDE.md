# MICCAI-2023 Experiment Notebooks

Jupyter notebooks for the MICCAI 2023 paper: *Physics-informed neural networks for tissue elasticity reconstruction in magnetic resonance elastography*.

---

## Notebooks Overview

| Notebook | Purpose | Dataset | Runtime |
|----------|---------|---------|---------|
| `simulation-training.ipynb` | Train PINN on synthetic data | BIOQIC | ~2.5h (GPU) |
| `simulation-experiments.ipynb` | Simulation experiments & analysis | BIOQIC | N/A (analysis) |
| `patient-preprocessing.ipynb` | Preprocess patient MRI data | NAFLD | ~5h (155 patients) |
| `patient-training.ipynb` | Train PINN on patient data | NAFLD | ~3h (GPU) |
| `patient-experiments.ipynb` | Patient experiments & analysis | NAFLD | N/A (analysis) |
| `FEM-baseline.ipynb` | Compute FEM baselines | Both | ~30min |
| `FEM-development.ipynb` | FEM development/testing | Both | N/A (dev) |

---

## Detailed Descriptions

### 1. MICCAI-2023-simulation-training.ipynb

**Purpose**: Full training workflow on simulation data

**Workflow**:
1. Download BIOQIC dataset
2. Load and visualize data
3. Create PINN model
4. Train for 100k iterations
5. Compare with AHI baseline
6. Visualize results

**Key Outputs**:
- Trained model
- Predicted elasticity maps
- Performance metrics

**Usage**:
```bash
jupyter notebook MICCAI-2023-simulation-training.ipynb
```

---

### 2. MICCAI-2023-simulation-experiments.ipynb

**Purpose**: Hyperparameter studies and experiments on simulation data

**Contents**:
- Grid search over model architectures
- Loss weight optimization
- Frequency sensitivity analysis
- Noise robustness tests

---

### 3. MICCAI-2023-patient-preprocessing.ipynb

**Purpose**: Preprocess raw patient DICOM/NIFTI data

**Workflow**:
1. Load raw NIFTI files
2. Image registration (alignment)
3. Liver segmentation (U-Net)
4. Resize to standard grid
5. Save as xarray .nc files

**Input**: Raw NIFTI files from MRI scanner
**Output**: Preprocessed .nc files

**Processes 155 patients** from NAFLD cohort.

---

### 4. MICCAI-2023-patient-training.ipynb

**Purpose**: Train PINN on single patient example

**Features**:
- Loads preprocessed patient data
- Includes anatomical MRI sequences
- Larger network (5 layers, 128 units)
- Demonstrates anatomical loss

**Example patient**: Subject 0006

---

### 5. MICCAI-2023-patient-experiments.ipynb

**Purpose**: Large-scale experiments on 155 patients

**Experiments**:
- Compare methods: AHI, FEM-HH, FEM-het, PINN-HH, PINN-het
- Test anatomical loss weights: [0, 1e-4, 1e-2, 1]
- Compute statistics across cohort
- Generate publication figures

**Key Results**:
- PINN-het achieves R=0.84 (best method)
- Anatomical loss improves to R=0.86
- Significant improvement over clinical standard (AHI: R=0.57)

---

### 6. MICCAI-2023-FEM-baseline.ipynb

**Purpose**: Compute FEM baseline reconstructions

**Methods**:
- AHI (Algebraic Helmholtz Inversion)
- FEM-Helmholtz
- FEM-Heterogeneous

**Note**: Requires Linux + FEniCS/dolfinx

---

### 7. MICCAI-2023-FEM-development.ipynb

**Purpose**: Development and testing of FEM implementations

**Contents**:
- FEM solver development
- Mesh generation tests
- Element type comparisons

---

## Data Flow

```
Raw Data
    │
    ├─► simulation-training.ipynb
    │   └─► Trained PINN model
    │
    └─► patient-preprocessing.ipynb
        └─► Preprocessed .nc files
            │
            ├─► patient-training.ipynb
            │   └─► Trained PINN model
            │
            └─► patient-experiments.ipynb
                └─► Results & figures
```

---

## Running Experiments

### Quick Test (Simulation)

```bash
# Start Jupyter
jupyter notebook

# Open: MICCAI-2023-simulation-training.ipynb
# Run all cells

# Expected runtime: 2.5 hours (GPU)
# Expected output: Elasticity maps, metrics
```

### Full Patient Study

```bash
# 1. Preprocess data (one-time, ~5 hours)
# Open: MICCAI-2023-patient-preprocessing.ipynb
# Run all cells

# 2. Train models (run on cluster)
# See: patient-experiments.ipynb for batch job submission

# 3. Analyze results
# Open: MICCAI-2023-patient-experiments.ipynb
```

---

## Generated Figures

### Main Figures

1. **patient_image_grid.png**: Visual comparison of methods
2. **patient_plots.png**: Box plots of correlation metrics
3. **patient_method_R_bar_plot.png**: Method comparison
4. **simulation_results.png**: Simulation study results

### Figure Generation

Figures automatically generated in `images/` directory when running experiment notebooks.

---

## Hyperparameters Used

### Simulation Data

```python
n_layers = 2
n_hidden = 64
activ_fn = 'sin'
omega = 60  # Hz (network frequency)
frequency = 90  # Hz (data frequency)
n_iters = 100000
lr = 1e-4
loss_weights = [1, 0, 0, 1e-8]  # [wave, elast, pde, anat]
```

### Patient Data

```python
n_layers = 5
n_hidden = 128
activ_fn = 'sin'
polar_input = True
omega = 30  # Hz
frequency = 40  # Hz
n_iters = 100000
lr = 1e-4
loss_weights = [1, 0, 1, 1e-4]  # Include anatomical loss
```

---

## Troubleshooting

**Kernel dies during training**:
- Reduce `n_points` (batch size)
- Use CPU instead of GPU

**FEM baseline fails**:
- Check dolfinx installation (Linux only)
- Windows: FEM not supported

**Out of memory**:
- Restart kernel
- Process patients sequentially (not all at once)

---

## Citation

```bibtex
@inproceedings{mre-pinn-2023,
  title={Physics-informed neural networks for tissue elasticity reconstruction in magnetic resonance elastography},
  booktitle={MICCAI 2023},
  year={2023}
}
```

---

## See Also

- [../README.md](../README.md) - Repository overview
- [../ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [../mre_pinn/PACKAGE_OVERVIEW.md](../mre_pinn/PACKAGE_OVERVIEW.md) - Package documentation
