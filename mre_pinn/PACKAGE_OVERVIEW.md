# mre_pinn Package

Main Python package containing all core functionality for Physics-Informed Neural Networks applied to Magnetic Resonance Elastography.

---

## Package Structure

```
mre_pinn/
├── __init__.py              # Package initialization
├── data/                    # Data loading and preprocessing → See data/DATASETS_GUIDE.md
├── model/                   # Neural network architectures → See model/README.md
├── training/                # Training loops and losses → See training/README.md
├── testing/                 # Evaluation and metrics → See testing/README.md
├── baseline/                # Baseline comparison methods → See baseline/README.md
├── pde.py                   # Wave equation PDEs
├── fields.py                # Spatial field operations
├── utils.py                 # Utility functions
└── visual.py                # Visualization tools
```

---

## Module Overview

### Core Modules

| Module | Purpose | Key Components |
|--------|---------|----------------|
| `data/` | Data management | Loading, preprocessing, segmentation |
| `model/` | Neural networks | PINN architecture definition |
| `training/` | Training | Loss functions, optimization |
| `testing/` | Evaluation | Metrics, visualization |
| `baseline/` | Comparisons | AHI, FEM baselines |

### Utility Modules

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `pde.py` | Physics equations | `WaveEquation`, `HelmholtzEquation` |
| `fields.py` | Field operations | `Field` (gradients, smoothing) |
| `utils.py` | Utilities | File I/O, conversions |
| `visual.py` | Visualization | `XArrayViewer`, color maps |

---

## Quick Start

### Basic Usage Example

```python
import mre_pinn

# 1. Load data
example = mre_pinn.data.MREExample.load_xarrays(
    '../data/BIOQIC/fem_box/',
    frequency=90
)

# 2. Define PDE
pde = mre_pinn.pde.WaveEquation.from_name('hetero', omega=90)

# 3. Create PINN model
pinn = mre_pinn.model.MREPINN(
    example,
    omega=60,
    n_layers=2,
    n_hidden=64,
    activ_fn='sin'
)

# 4. Setup training
model = mre_pinn.training.MREPINNModel(
    example, pinn, pde,
    loss_weights=[1, 0, 0, 1e-8],
    n_points=1024
)
model.compile(optimizer='adam', lr=1e-4)

# 5. Train
model.train(100000)

# 6. Evaluate
dataset, arrays = model.test()
```

---

## Detailed Module Documentation

### pde.py - Wave Equations

Defines the partial differential equations that govern wave propagation in tissue.

#### Classes

**`WaveEquation`**
- Base class for wave equations
- Implements PDE residual computation
- Supports automatic differentiation

**Variants**:
```python
# Helmholtz equation (assumes homogeneous medium)
pde = WaveEquation.from_name('helmholtz', omega=40)

# Heterogeneous equation (spatially-varying elasticity)
pde = WaveEquation.from_name('hetero', omega=40)
```

#### Mathematical Formulation

**Heterogeneous Wave Equation**:
```
∇·[μ(∇u + ∇u^T)] + ρω²u = 0

Where:
  u(x,y,z) = displacement field (complex)
  μ(x,y,z) = shear modulus (what we're solving for)
  ρ = 1000 kg/m³ (tissue density)
  ω = 2πf (angular frequency)
```

#### Key Methods

```python
def pde_residual(self, x, u, mu):
    """
    Compute PDE residual (should be ~0 if physics satisfied)

    Args:
        x: Spatial coordinates (N, 3)
        u: Wave field (N, 3) complex
        mu: Elasticity (N, 1) complex

    Returns:
        residual: PDE violation (N, 3) complex
    """
```

---

### fields.py - Spatial Field Operations

Provides tools for working with 3D spatial fields (gradients, smoothing, etc).

#### Main Class: `Field`

Wraps xarray DataArrays with convenient methods for spatial operations.

#### Key Methods

```python
class Field:
    def points(self, reshape=True):
        """Get spatial coordinate grid"""

    def values(self, reshape=True):
        """Get field values as numpy array"""

    def gradient(self, savgol=True, order=2, kernel_size=3):
        """Compute spatial derivatives"""

    def smooth(self, method='savgol', order=2, kernel_size=3):
        """Apply spatial smoothing"""

    def laplacian(self):
        """Compute Laplacian (∇²)"""
```

#### Usage Example

```python
import mre_pinn

# Load wave field
wave = example.wave  # xarray DataArray

# Convert to Field
field = wave.field

# Get coordinates
x = field.points()  # Shape: (N, 3)

# Compute gradient
grad_u = field.gradient()  # Shape: (N, 3, 3)

# Smooth field
smoothed = field.smooth(kernel_size=5)
```

---

### utils.py - Utility Functions

Common utilities for file I/O, data conversion, and formatting.

#### Key Functions

```python
def exists(path):
    """Check if file/directory exists"""

def as_xarray(array, like=None):
    """Convert numpy array to xarray with metadata"""

def print_if(verbose, *args):
    """Conditional printing"""

def progress(iterable, desc=None):
    """Progress bar wrapper"""

def braced_glob(pattern):
    """Glob with brace expansion: '{a,b}' → [a, b]"""
```

---

### visual.py - Visualization

Interactive visualization tools for 3D medical imaging data.

#### Main Class: `XArrayViewer`

Interactive Jupyter widget for viewing 3D xarray data.

```python
class XArrayViewer:
    """
    Interactive viewer with sliders for:
    - Spatial slicing (x, y, z)
    - Component selection
    - Real/imaginary parts
    """
```

#### Usage Example

```python
import mre_pinn

# Load data
example = mre_pinn.data.MREExample.load_xarrays(...)

# View wave field
example.view('wave', ax_height=3)

# View elasticity
example.view('mre', polar=True, vmax=10e3)

# View with mask
example.view('wave', 'mre', mask=True)
```

#### Color Maps

```python
def wave_color_map(n_colors=255, symmetric=True):
    """Blue-white-red colormap for wave fields"""

def mre_color_map(n_colors=255, symmetric=False):
    """Rainbow colormap for elasticity"""

def region_color_map(n_colors=255, has_background=False):
    """Colormap for segmentation regions"""
```

---

## Data Types and Conventions

### Coordinate System

- **Units**: Meters (m)
- **Origin**: Varies by dataset
- **Dimensions**: (x, y, z) in Cartesian coordinates

### Complex Numbers

Wave fields and elasticity are complex-valued:
```python
u = u_real + 1j * u_imag  # Displacement
μ = μ_real + 1j * μ_imag  # Elasticity

# Storage modulus: μ_real (elastic response)
# Loss modulus: μ_imag (viscous response)
```

### XArray Structure

Standard xarray structure for MRE data:

```python
<xarray.DataArray 'wave' (x: 256, y: 256, z: 4)>
Coordinates:
  * x        (x) float64 - Spatial coordinates (meters)
  * y        (y) float64
  * z        (z) float64
    region   (x, y, z) float32 - Segmentation mask
```

---

## Physics Background

### Wave Propagation in Tissue

When tissue is vibrated:
1. Mechanical waves propagate through tissue
2. Wave speed depends on tissue stiffness
3. Stiffer tissue → faster waves → shorter wavelength

**Relationship**:
```
λ = 2π/k = 2π√(μ/ρω²)

Where:
  λ = wavelength
  k = wave number
  μ = shear modulus (stiffness)
  ρ = density
  ω = frequency
```

### Why PINNs Work

**Traditional methods**:
- Solve inverse problem algebraically
- Sensitive to noise
- Make simplifying assumptions

**PINN approach**:
- Learn from noisy data
- Enforce physics (wave equation)
- Regularized by neural network
- More robust and accurate

---

## Common Parameters

### Model Architecture

```python
n_layers: int = 2-5           # Number of hidden layers
n_hidden: int = 64-128        # Units per layer
activ_fn: str = 'sin'         # Activation: 'sin', 'tanh', 'swish'
polar_input: bool = False     # Use polar coordinates
```

### Training

```python
n_iters: int = 100000         # Training iterations
n_points: int = 1024          # Batch size (spatial points)
lr: float = 1e-4              # Learning rate
loss_weights: list            # [wave, elast, pde, anat]
```

### Physics

```python
omega: float                  # Network frequency (Hz)
frequency: float              # Data frequency (Hz)
rho: float = 1000            # Tissue density (kg/m³)
```

---

## Error Handling

### Common Issues

**1. CUDA out of memory**
```python
# Solution: Reduce batch size
model = MREPINNModel(..., n_points=512)  # Instead of 1024
```

**2. NaN losses**
```python
# Solution: Reduce learning rate
model.compile(lr=1e-5)  # Instead of 1e-4
```

**3. PDE loss doesn't decrease**
```python
# Solution: Adjust warmup period
model = MREPINNModel(
    ...,
    pde_warmup_iters=20000,  # Train longer before PDE
    pde_init_weight=1e-10    # Start with smaller weight
)
```

---

## Performance Tips

### GPU Acceleration

```python
# Specify device
model = MREPINNModel(..., device='cuda')

# Monitor GPU memory
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
```

### Faster Training

1. **Use larger batch size** (if GPU memory allows)
2. **Enable CUDNN benchmarking**:
   ```python
   torch.backends.cudnn.benchmark = True
   ```
3. **Use mixed precision** (PyTorch 1.6+):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   ```

### Memory Optimization

```python
# Resample training points less frequently
callbacks = [Resampler(period=10000)]  # Every 10k iters

# Use gradient checkpointing for very deep networks
```

---

## Testing and Validation

### Unit Tests

```python
# Test field operations
from mre_pinn import fields
field = wave.field
assert field.points().shape == (N, 3)

# Test PDE computation
from mre_pinn import pde
residual = pde.pde_residual(x, u, mu)
assert residual.shape == u.shape
```

### Integration Tests

See `MICCAI-2023/` notebooks for full workflow examples.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{mre-pinn-2023,
  title={Physics-informed neural networks for tissue elasticity reconstruction in magnetic resonance elastography},
  author={...},
  booktitle={MICCAI 2023},
  year={2023}
}
```

---

## See Also

- [data/DATASETS_GUIDE.md](data/DATASETS_GUIDE.md) - Data loading and preprocessing details
- [model/README.md](model/README.md) - Neural network architecture details
- [training/README.md](training/README.md) - Training loop implementation
- [../ARCHITECTURE.md](../ARCHITECTURE.md) - Overall system architecture
