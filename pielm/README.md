# PIELM for MRE

Physics-Informed Extreme Learning Machine implementation for Magnetic Resonance Elastography.

## Overview

PIELM is an alternative to PINN that uses random features instead of trainable neural networks. Key advantages:

- ✅ **One-shot training**: No iterations, just solve linear system once
- ✅ **Analytical derivatives**: Computed via PyTorch autograd on random features
- ✅ **Fast inference**: Linear prediction φ(x) @ W
- ✅ **Same interface**: Compatible with existing MRE-PINN workflow

## Module Structure

```
pielm/
├── __init__.py       # Package exports
├── features.py       # RandomFeatures class with autograd derivatives
├── solver.py         # Linear system solver (ridge regression)
├── model.py          # MREPIELM class (analog of MREPINN)
├── training.py       # MREPIELMModel class (one-shot solver) [TODO]
└── equations.py      # PDE constraint matrices [TODO]
```

## Implementation Status

### ✅ Completed (Phase 1)

1. **`features.py`**: Random Fourier features with PyTorch autograd
   - `RandomFeatures` class
   - Computes φ(x), ∇φ(x), ∇²φ(x) using autograd
   - Supports both cos and [cos, sin] features

2. **`solver.py`**: Linear system solver
   - Ridge regression: W = (A^T A + λI)^{-1} A^T b
   - Multiple methods: ridge, lstsq, pinv
   - Helper functions for condition number and residual

3. **`model.py`**: MREPIELM model class
   - Dual random feature architecture (u_features, mu_features)
   - Same normalization as MREPINN
   - Compatible forward() interface

### 🚧 In Progress (Phase 2)

4. **`training.py`**: MREPIELMModel solver class
   - Data loading and sampling (from MREPINNData pattern)
   - solve() method (instead of train())
   - test() method (compatible with TestEvaluator)

### 📋 TODO (Phase 3-4)

5. **`equations.py`**: PDE constraint matrix construction
   - Helmholtz equation: μ∇²u + ρω²u = 0
   - Hetero equation: μ∇²u + ∇μ·∇u + ρω²u = 0

6. **Integration**: Connect all pieces
   - PDE constraints in solve()
   - Full testing on BIOQIC data
   - Comparison notebook (PIELM vs PINN)

## Usage Example (Planned)

```python
import mre_pinn
import pielm

# 1. Load data (SAME as PINN)
example = mre_pinn.data.MREExample.load_xarrays('data/BIOQIC/fem_box', frequency=60)

# 2. Define PDE (SAME as PINN)
pde = mre_pinn.pde.WaveEquation.from_name('hetero', omega=60)

# 3. Create PIELM model (DIFFERENT: random features)
model = pielm.MREPIELM(example, omega=60, n_features=2000)

# 4. Create solver (DIFFERENT: one-shot)
solver = pielm.MREPIELMModel(
    example, model, pde,
    loss_weights=[1, 0, 0, 1e-8],  # Same as PINN
    n_points=4096
)

# 5. Solve (DIFFERENT: one-shot instead of iterative)
solver.solve()

# 6. Evaluate (SAME as PINN)
test_eval = mre_pinn.testing.TestEvaluator()
test_eval.model = solver
test_eval.test()
```

## Two MRE Equations

### Equation 1: Helmholtz (Homogeneous)
```
μ∇²u + ρω²u = 0
```
- Assumes constant elasticity (∇μ = 0)
- Simpler to implement
- Good baseline

### Equation 2: Hetero (Heterogeneous)
```
μ∇²u + ∇μ·∇u + ρω²u = 0
```
- Allows spatially varying elasticity
- Matches PINN experiments
- More accurate for real tissue

## Technical Notes

### Random Features
We use random Fourier features:
```
φ(x) = [cos(Wx + b), sin(Wx + b)]
```
where W ~ N(0, σ²) and b ~ Uniform(0, 2π).

Derivatives are computed via PyTorch autograd:
- First: ∂φ/∂x via torch.autograd.grad()
- Second: ∇²φ via double autograd

### Linear System
PIELM reduces MRE to solving:
```
[√w_data * Φ_data  ]     [√w_data * u_data]
[√w_pde  * Φ_PDE   ] W = [√w_pde  * 0     ]
```
where Φ_PDE enforces PDE constraints.

## Next Steps

1. Implement `training.py` (data fitting only first)
2. Test data fitting on BIOQIC example
3. Implement `equations.py` (Helmholtz then Hetero)
4. Add PDE constraints to solver
5. Full testing and comparison with PINN
