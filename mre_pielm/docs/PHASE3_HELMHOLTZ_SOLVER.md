# Phase 3: Helmholtz Forward Solver

## Overview

Phase 3 implements the **HelmholtzForwardSolver** for solving the MRE forward problem using the Helmholtz approximation:

```
μ∇²u + ρω²u = 0
```

Given elasticity μ, solve for displacement u by assembling and solving a linear system combining PDE constraints and optional data fitting.

## Helmholtz Equation

**Physical Meaning:**
- Simplified wave equation assuming homogeneous elasticity (∇μ ≈ 0)
- Valid when elasticity variations are smooth
- Faster to solve than full heterogeneous equation

**PDE Form:**
```
μ * Laplacian(u) + ρ * ω² * u = 0
```

where:
- μ: Shear modulus (Pa)
- u: Displacement (complex, meters)
- ρ: Density (kg/m³)
- ω: Angular frequency (rad/s)

## Implementation

### HelmholtzForwardSolver Class

**Location:** `mre_pielm/forward/helmholtz.py`

**Key Methods:**

1. **assemble_pde_system()**: Build PDE constraint matrix
2. **assemble_data_system()**: Build data fitting matrix
3. **solve_for_component()**: Solve for real & imaginary weights
4. **solve()**: Complete forward solution

### Linear System Formulation

**For basis representation:**
```
u(x,y,z) = sum_i w_i * phi_i(x,y,z)
```

**PDE becomes:**
```
μ * sum_i w_i * Laplacian(phi_i) + ρω² * sum_i w_i * phi_i = 0
```

**Matrix form:**
```
(μ * Lap_phi + ρω² * phi) @ W = 0
```

**With data:**
```
[H_pde  ]     [0    ]
[H_data ] @ W = [u_data]
```

### Usage Example

```python
from mre_pielm import MREPIELM
from mre_pielm.forward import HelmholtzForwardSolver
import torch

# Create model
model = MREPIELM(
    u_degrees=(8, 10, 6),
    mu_degrees=(5, 6, 4),
    domain=((0, 0.08), (0, 0.1), (0, 0.01)),
    omega=2*np.pi*60,
    device='cpu'
)

# Create solver
solver = HelmholtzForwardSolver(
    model=model,
    n_collocation=10000,
    pde_weight=1.0,
    data_weight=1.0,
    ridge=1e-8
)

# Solve with known mu
mu_colloc = torch.ones(10000) * 5000.0  # Constant 5kPa
results = solver.solve(mu=mu_colloc, n_components=2)

# Access results
u_pred = results['u_pred']         # Predicted displacement
pde_residual = results['pde_residual']  # PDE error
```

### Solving with Data

```python
# With measured displacement data
x_data = ...  # (n_data, 3) coordinates
u_data = ...  # (n_data, n_components) complex displacement

results = solver.solve(
    mu=mu_colloc,
    x_data=x_data,
    u_data=u_data,
    n_components=2,
    solver='ridge'
)

print(f"Data error: {results['data_error']:.4e}")
print(f"PDE residual: {results['pde_residual']:.4e}")
```

## Test Results

All Phase 3 tests pass successfully:

```
✓ HelmholtzForwardSolver initialization
✓ PDE system assembly (Helmholtz equation)
✓ Data system assembly
✓ Solve for individual components
✓ Full solve (PDE only)
✓ Solve with data fitting
✓ Prediction at arbitrary points
✓ Derivative computation and PDE residual
✓ Multiple solvers (ridge and lstsq)
```

**Performance:**
- Collocation points: 1000
- u basis features: 210 (degrees 5×6×4)
- PDE residual: ~1e-6 to 1e-7
- Solve time: < 1 second

## Model Updates

Modified `MREPIELM.forward()` to handle cases where mu_weights is not set (forward problem):
- Only requires u_weights for prediction
- mu_weights optional (not needed for forward solve)
- grad_mu only computed if mu_weights available

## API Reference

### HelmholtzForwardSolver

```python
solver = HelmholtzForwardSolver(
    model,               # MREPIELM instance
    n_collocation=10000, # Number of PDE points
    pde_weight=1.0,      # PDE loss weight
    data_weight=1.0,     # Data loss weight
    ridge=1e-10,         # Regularization
    verbose=True         # Print diagnostics
)
```

### solve()

```python
results = solver.solve(
    mu,              # (n_colloc,) elasticity values
    x_data=None,     # Optional data coordinates
    u_data=None,     # Optional displacement data
    n_components=2,  # Number of displacement components
    solver='ridge'   # 'ridge' or 'lstsq'
)
```

**Returns:**
- `u_weights`: List of weight tensors
- `u_pred`: Predicted displacement at collocation points
- `pde_residual`: Mean PDE residual norm
- `data_error`: Relative data error (if data provided)

## Next Steps (Phase 4)

Phase 4 will implement the **heterogeneous forward solver**:
```
μ∇²u + ∇μ·∇u + ρω²u = 0
```

This includes the gradient term ∇μ·∇u for spatially varying elasticity.

## File Structure

```
mre_pielm/
├── forward/
│   ├── __init__.py           # Exports HelmholtzForwardSolver
│   └── helmholtz.py          # Helmholtz solver (490 lines)
├── model.py                  # Updated for optional mu_weights
└── docs/
    └── PHASE3_HELMHOLTZ_SOLVER.md  # This file
```

---

**Phase 3 Status**: ✅ COMPLETE

Helmholtz forward solver implemented, tested, and ready for use.
