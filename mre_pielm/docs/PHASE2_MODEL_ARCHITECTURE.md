# Phase 2: Model Architecture

## Overview

Phase 2 implements the **MREPIELM** model class and supporting utilities for MRE forward and inverse problems. This phase builds on the Bernstein basis functions from Phase 1 and provides a high-level interface for solving MRE problems using Physics-Informed Extreme Learning Machines.

## Components Implemented

### 1. MREPIELM Model Class (`model.py`)

The core model class that manages dual Bernstein bases for displacement (u) and elasticity (mu).

#### Architecture

**Dual Basis Representation:**
```
u(x,y,z) = sum_{i,j,k} w_u[i,j,k] * Phi_{i,j,k}(x,y,z)   [complex-valued]
mu(x,y,z) = sum_{i,j,k} w_mu[i,j,k] * Psi_{i,j,k}(x,y,z)  [real-valued]
```

where:
- Phi: Bernstein basis for displacement (typically higher degree)
- Psi: Bernstein basis for elasticity (typically lower degree)
- w_u, w_mu: Weights to be solved via linear system

**Key Design Decisions:**

1. **Separate Bases**: u and mu use independent Bernstein bases
   - Allows different polynomial degrees for each field
   - u typically needs higher resolution (wave field)
   - mu can use lower resolution (smoothly varying elasticity)

2. **Complex-Valued u**: Displacement is complex to represent wave phase
   - Store 2 weight vectors per component (real + imaginary)
   - Forward pass combines real and imaginary predictions

3. **Normalization**: Matches MREPINN normalization strategy
   - Inputs: centered and scaled by extent, then scaled by omega
   - Outputs: standardized by mean and std from training data

#### Initialization

```python
from mre_pielm import MREPIELM

# Manual initialization
model = MREPIELM(
    u_degrees=(10, 12, 8),      # u basis: (nx, ny, nz)
    mu_degrees=(6, 8, 6),        # mu basis: (nx, ny, nz)
    domain=((0, 0.08), (0, 0.1), (0, 0.01)),  # Physical bounds (m)
    omega=2*np.pi*60,            # Angular frequency (rad/s)
    rho=1000.0,                  # Density (kg/m^3)
    device='cpu'
)

# From MREExample
model = MREPIELM.from_example(
    example=mre_example,
    u_degrees=(10, 12, 8),
    mu_degrees=(6, 8, 6),
    frequency=60  # Hz
)
```

**Feature Counts:**
```python
print(f"u basis features: {model.n_u_features}")  # (11 x 13 x 9) = 1287
print(f"mu basis features: {model.n_mu_features}") # (7 x 9 x 7) = 441
print(f"Total parameters: {model.n_parameters}")   # Depends on components
```

#### Forward Pass

**Basic prediction:**
```python
import torch

# Predict at arbitrary points
x = torch.rand(1000, 3) * torch.tensor([0.08, 0.1, 0.01])  # Physical coords

outputs = model.forward(x, compute_derivatives=False)
u_pred = outputs['u']   # (1000, n_u_components) complex
mu_pred = outputs['mu']  # (1000, n_mu_components) real
```

**With derivatives (for PDE residuals):**
```python
outputs = model.forward(x, compute_derivatives=True)

u = outputs['u']          # (N, n_components) complex
mu = outputs['mu']        # (N, n_mu_components) real
grad_u = outputs['grad_u']  # (N, n_components, 3) complex
lap_u = outputs['lap_u']    # (N, n_components) complex
grad_mu = outputs['grad_mu'] # (N, n_mu_components, 3) real
```

#### Normalization

**Input Normalization:**
```python
x_norm = (x - input_loc) / input_scale * omega
```
- `input_loc`: Center of spatial domain
- `input_scale`: Extent of spatial domain
- `omega`: Angular frequency (scales with wave speed)

**Output Normalization:**
```python
# u (complex)
u_real = u_pred_real * u_scale + u_loc
u_imag = u_pred_imag * u_scale + u_loc
u = torch.complex(u_real, u_imag)

# mu (real)
mu = mu_pred * mu_scale + mu_loc
```

Set from example:
```python
model.set_normalization_from_example(example)
```

### 2. Utility Functions (`utils.py`)

Helper functions for data handling and coordinate manipulation.

#### Domain Extraction

```python
from mre_pielm.utils import extract_domain_bounds

# Extract physical bounds from xarray
domain = extract_domain_bounds(example.wave)
# Returns: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
```

#### Data Conversion

```python
from mre_pielm.utils import xarray_to_points_and_values

# Convert xarray to tensors
coords, values = xarray_to_points_and_values(
    example.wave,
    mask=example.mre_mask,  # Optional: only extract masked region
    flatten=True,
    device='cpu'
)
# coords: (N, 3) tensor
# values: (N, n_components) tensor (complex if input is complex)
```

#### Point Sampling

**Random sampling:**
```python
from mre_pielm.utils import sample_random_points

x_random = sample_random_points(
    domain=((0, 0.08), (0, 0.1), (0, 0.01)),
    n_points=10000,
    device='cpu'
)  # (10000, 3)
```

**Grid sampling:**
```python
from mre_pielm.utils import sample_grid_points

x_grid = sample_grid_points(
    domain=((0, 0.08), (0, 0.1), (0, 0.01)),
    n_points_per_dim=(20, 25, 10),  # nx, ny, nz
    device='cpu'
)  # (20*25*10, 3)
```

**Collocation points (for PDE):**
```python
from mre_pielm.utils import create_collocation_points

x_colloc = create_collocation_points(
    domain=domain,
    n_collocation=50000,
    sampling='random',  # or 'grid'
    device='cpu'
)
```

#### Error Metrics

```python
from mre_pielm.utils import compute_relative_error

error = compute_relative_error(pred, target)
# Returns: ||pred - target|| / ||target||
```

## Model Capabilities

### Parameter Estimation

**Example configuration:**
```python
u_degrees = (10, 12, 8)  # 1287 features
mu_degrees = (6, 8, 6)   # 441 features
n_u_components = 2       # real and imag
n_mu_components = 1      # scalar elasticity

# Weights to solve:
# u: 2 * n_u_components * n_u_features = 2 * 2 * 1287 = 5148
# mu: n_mu_components * n_mu_features = 1 * 441 = 441
# Total: 5589 parameters
```

### Forward vs Inverse Problems

**Forward Problem (Phase 3):**
- Given: mu (elasticity field)
- Solve for: u (displacement field)
- Approach: PDE constraints + boundary conditions

**Inverse Problem (Phase 4):**
- Given: u (measured displacement)
- Solve for: mu (elasticity field)
- Approach: Data fitting + PDE constraints

**Joint Solve:**
- Solve for both u and mu simultaneously
- Use data fitting + PDE constraints
- Typical for MRE reconstruction

## Comparison with MREPINN

| Aspect | MREPINN | MREPIELM |
|--------|---------|----------|
| **Learning** | Iterative (gradient descent) | One-shot (linear solve) |
| **Speed** | Slow (~1000s iterations) | Fast (~1s solve) |
| **Basis** | Neural network (tanh/sin) | Bernstein polynomials |
| **Parameters** | ~10k-50k (network weights) | ~1k-10k (basis weights) |
| **Derivatives** | Autograd | Analytical formulas |
| **Conditioning** | Can be unstable | Well-conditioned |
| **Normalization** | Center + extent | Same strategy |
| **Complex handling** | Polar or Cartesian | Cartesian (real + imag) |

## Implementation Details

### Weight Storage

```python
# u weights (list of tensors)
model.u_weights = [
    w_u_real_comp0,  # (n_u_features,)
    w_u_imag_comp0,  # (n_u_features,)
    w_u_real_comp1,  # (n_u_features,)
    w_u_imag_comp1,  # (n_u_features,)
    ...
]

# mu weights (tensor)
model.mu_weights = torch.tensor(...)  # (n_mu_components, n_mu_features)
```

### Derivative Computation

Uses `einsum` for efficient tensor contraction:

```python
# Gradient: (N, n_features, 3) @ (n_features,) -> (N, 3)
grad_u = torch.einsum('nfi,f->ni', grad_phi_u, w_u)

# Multiple components: (N, n_features, 3) @ (C, n_features) -> (N, C, 3)
grad_mu = torch.einsum('nfi,cf->nci', grad_phi_mu, w_mu)
```

### Device Support

```python
# CPU
model = MREPIELM(..., device='cpu')

# GPU (if available)
model = MREPIELM(..., device='cuda')
```

All tensors (bases, weights, outputs) automatically placed on specified device.

## Testing

### Test Coverage

```bash
python tests/test_phase2_model.py
```

**Tests verify:**
1. Model initialization
2. Dual basis creation
3. Random and grid point sampling
4. Coordinate normalization
5. Forward pass with mock weights
6. Derivative computation (gradient and Laplacian)
7. Collocation point creation
8. Utility functions
9. Error metrics
10. Model representation

All tests pass successfully.

## Usage Example

```python
import torch
import numpy as np
from mre_pielm import MREPIELM
from mre_pielm.utils import sample_random_points

# Create model
domain = ((0, 0.08), (0, 0.1), (0, 0.01))
model = MREPIELM(
    u_degrees=(8, 10, 6),
    mu_degrees=(5, 6, 4),
    domain=domain,
    omega=2*np.pi*60,
    device='cpu'
)

# Set normalization (normally from example)
model.input_loc = torch.tensor([0.04, 0.05, 0.005])
model.input_scale = torch.tensor([0.08, 0.1, 0.01])
model.u_loc = torch.zeros(2)
model.u_scale = torch.ones(2)
model.mu_loc = torch.tensor([5000.0])
model.mu_scale = torch.tensor([2000.0])

# Initialize weights (normally from solver)
n_u_feat = model.n_u_features
n_mu_feat = model.n_mu_features

model.u_weights = [
    torch.randn(n_u_feat),  # u_real_x
    torch.randn(n_u_feat),  # u_imag_x
    torch.randn(n_u_feat),  # u_real_y
    torch.randn(n_u_feat),  # u_imag_y
]
model.mu_weights = torch.randn(1, n_mu_feat)

# Predict at random points
x = sample_random_points(domain, 100)
outputs = model.forward(x, compute_derivatives=True)

print(f"u: {outputs['u'].shape}")       # (100, 2) complex
print(f"mu: {outputs['mu'].shape}")      # (100, 1) real
print(f"grad_u: {outputs['grad_u'].shape}")  # (100, 2, 3) complex
print(f"lap_u: {outputs['lap_u'].shape}")    # (100, 2) complex
```

## Next Steps (Phase 3)

- Implement `HelmholtzForwardSolver` for forward MRE problem
- Assemble linear system: H @ W = K
  - H: Feature matrix from PDE + boundary conditions
  - K: Right-hand side from known data
- Solve for u weights given mu
- Test on synthetic data
- Validate against analytical solutions

## File Structure

```
mre_pielm/
├── __init__.py           # Package exports
├── model.py              # MREPIELM class (440 lines)
├── utils.py              # Utility functions (360 lines)
├── core/                 # From Phase 1
│   ├── bernstein.py
│   ├── solver.py
│   └── derivatives.py
└── docs/
    ├── PHASE1_BERNSTEIN_BASIS.md
    └── PHASE2_MODEL_ARCHITECTURE.md  # This file
```

## API Quick Reference

```python
# Main model class
from mre_pielm import MREPIELM

# Model methods
model.forward(x, compute_derivatives=False)  # Predict u and mu
model.predict_u(x)                           # Predict u only
model.predict_mu(x)                          # Predict mu only
model.normalize_input(x)                     # Apply input normalization
model.set_normalization_from_example(ex)     # Extract normalization params

# Utility functions
from mre_pielm.utils import (
    extract_domain_bounds,
    extract_normalization_stats,
    xarray_to_points_and_values,
    sample_random_points,
    sample_grid_points,
    create_collocation_points,
    compute_relative_error,
    split_complex_components,
    merge_complex_components,
)
```

---

**Phase 2 Status**: COMPLETE

All model architecture components implemented, tested, and ready for Phase 3 (Forward Solver Implementation).
