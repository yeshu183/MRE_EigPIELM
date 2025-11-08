# MRE-PINN

> **Note**: This repository is being modified with the following objectives:
> 1. Resolve various implementation issues to ensure smooth execution
> 2. Integrate EIG-PIELM methodology into the existing framework
> 3. Comprehensive Testing and Comparison:
>    - Test the EIG-PIELM implementation on simulation datasets
>    - Validate on real-world datasets available in the repository
>    - Compare performance against:
>      - Original MRE-PINN implementation
>      - Existing FEM-based approaches (using available code in this repository)

This repository contains code for the paper *Physics-informed neural networks for tissue elasticity reconstruction in magnetic resonance elastography* which is to be presented at MICCAI 2023.

![MRE-PINN examples](MICCAI-2023/images/patient_image_grid.png)

## Installation

### Requirements

- **Python**: 3.8+
- **OS**: Linux (recommended), Windows (limited), macOS
- **GPU**: NVIDIA GPU with CUDA support (recommended) or CPU

### Environment Setup

#### GPU Environment (Recommended)

For training with GPU acceleration:

```bash
mamba env create --file=environment.yml
mamba activate MRE-PINN
python -m ipykernel install --user --name=MRE-PINN
```

#### CPU-Only Environment

If you don't have a GPU or want CPU-only installation:

```bash
mamba env create --file=env-cpu.yml
mamba activate MRE-PINN-CPU
python -m ipykernel install --user --name=MRE-PINN-CPU
```

**Note**: CPU training is significantly slower (~10-50x) but works for testing and small experiments.

### Platform-Specific Notes

#### Windows Users ⚠️

**FEM Baseline Limitation**: The FEM (Finite Element Method) baseline using FEniCS/dolfinx is **not available on Windows**.

- **Issue**: FEniCS/dolfinx requires Linux due to PETSc dependencies
- **Impact**: `mre_pinn.baseline.eval_fem_baseline()` will fail on Windows
- **Workaround**:
  - FEM baselines are optional - PINN training works fine without them
  - Use WSL2 (Windows Subsystem for Linux) for full FEM support
  - Or use Linux/macOS for complete functionality

**What still works on Windows**:
- ✅ PINN training and evaluation
- ✅ AHI (Algebraic Helmholtz Inversion) baseline
- ✅ Data preprocessing and visualization
- ✅ All core functionality except FEM

#### Linux Users

Full functionality available, including FEM baselines with dolfinx.

#### macOS Users

FEniCS/dolfinx may have installation issues. Follow Linux instructions and refer to FEniCS documentation for macOS-specific setup.

## Usage

### Jupyter Notebook (Recommended)

This [notebook](MICCAI-2023/MICCAI-2023-simulation-training.ipynb) downloads the BIOQIC simulation data set and trains PINNs to reconstruct a map of shear elasticity from the displacement field.

The notebook takes roughly 2.5 h to train for 100,000 iterations on an RTX 5000 and uses 2.5 GiB of GPU memory.

### Python Script

For training via command line:

```bash
# Download data first
python download_data.py

# Train model using the provided script
python train.py
```

See the [MICCAI-2023 notebooks](MICCAI-2023/) for detailed examples and usage.

---

## Troubleshooting

### Common Issues

#### 1. FEniCS/dolfinx Import Error (Windows)

**Error**:
```
ModuleNotFoundError: No module named 'dolfinx'
ImportError: cannot import name 'dolfinx' from 'mre_pinn.baseline'
```

**Solution**: FEniCS/dolfinx is not available on Windows. This is expected behavior.
- FEM baselines will be skipped automatically
- PINN training continues normally
- For FEM support, use WSL2 or Linux

#### 2. CUDA Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Use CPU environment: `mamba activate MRE-PINN-CPU`
2. Reduce batch size: Edit `train.py` and set `n_points=512` (instead of 1024)
3. Use smaller network: Set `n_hidden=64` and `n_layers=2`

#### 3. Slow Training on CPU

**Issue**: Training takes many hours on CPU

**Solutions**:
1. Reduce iterations: `n_iters=10000` for testing
2. Use smaller network: `n_layers=2, n_hidden=32`
3. Consider cloud GPU (Google Colab, AWS, etc.)

#### 4. DeepXDE Backend Warning

**Warning**:
```
Using backend: pytorch
```

**Solution**: This is informational, not an error. The code uses PyTorch backend by default.

---

## Documentation

Comprehensive documentation is available:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and data flow
- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete documentation index
- **[mre_pinn/PACKAGE_OVERVIEW.md](mre_pinn/PACKAGE_OVERVIEW.md)** - Package overview
- **[MICCAI-2023/EXPERIMENTS_GUIDE.md](MICCAI-2023/EXPERIMENTS_GUIDE.md)** - Experiment notebooks guide

---

## File Structure

```
MRE-PINN/
├── download_data.py              # Download BIOQIC dataset
├── train.py                      # Command-line training script
├── mre_pinn/                     # Main Python package
│   ├── data/                     # Data loading and preprocessing
│   ├── model/                    # PINN architecture
│   ├── training/                 # Training procedures
│   ├── baseline/                 # Baseline methods (AHI, FEM)
│   └── testing/                  # Evaluation and metrics
├── MICCAI-2023/                  # Experiment notebooks
│   ├── simulation-training.ipynb # Main training notebook
│   ├── patient-training.ipynb    # Patient data notebook
│   └── *-experiments.ipynb       # Experiment analysis
└── data/                         # Dataset storage
    └── BIOQIC/                   # Simulation data
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{mre-pinn-2023,
  title={Physics-informed neural networks for tissue elasticity reconstruction in magnetic resonance elastography},
  booktitle={MICCAI 2023},
  year={2023}
}
```
