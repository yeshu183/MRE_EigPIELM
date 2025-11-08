# mre_pinn.baseline - Baseline Methods

Traditional MRE reconstruction methods for comparison.

---

## Files

| File | Purpose | Methods |
|------|---------|---------|
| `direct.py` | Algebraic methods | AHI (Algebraic Helmholtz Inversion) |
| `fem.py` | Numerical PDE solver | FEM (Finite Element Method) |
| `filters.py` | Signal processing | Savitzky-Golay, curl, divergence |

---

## AHI - Algebraic Helmholtz Inversion

### Purpose
Fast, algebraic solution to Helmholtz equation. Similar to clinical MRE scanners.

### Method
```python
mre_pinn.baseline.eval_ahi_baseline(example, frequency=40)
```

### Algorithm
```
1. Compute Laplacian: ∇²u
2. Solve algebraically: μ = -ρω²u / ∇²u
3. Apply spatial filtering (optional)
```

**Advantages**:
- Very fast (~1 second)
- No iterative optimization
- Works on clinical scanners

**Disadvantages**:
- Assumes homogeneous Helmholtz equation
- Sensitive to noise
- Lower spatial resolution

---

## FEM - Finite Element Method

### Purpose
Numerical solution to heterogeneous wave equation using FEniCS/dolfinx.

### Method
```python
mre_pinn.baseline.eval_fem_baseline(
    example,
    frequency=40,
    hetero=True,           # Use heterogeneous equation
    u_elem_type='CG-3',    # Cubic elements for displacement
    mu_elem_type='DG-1'    # Linear discontinuous for elasticity
)
```

### Algorithm
```
1. Create finite element mesh from data grid
2. Interpolate wave field onto FEM basis
3. Solve variational problem:
   Find μ such that:
   ∫ μ(∇u:∇v) dx = ∫ ρω²u·v dx
4. Extract elasticity at grid points
```

**Advantages**:
- Physically accurate
- Handles heterogeneity
- Well-established method

**Disadvantages**:
- Slow (~30 seconds per slice)
- Requires FEniCS (Linux only)
- Memory intensive

**Note**: FEM baseline disabled on Windows (FEniCS not available).

---

## Filters

### Savitzky-Golay Filtering
```python
from mre_pinn.baseline.filters import savgol_filter_3d

filtered = savgol_filter_3d(
    data,
    window_size=5,
    order=2
)
```

**Purpose**: Smooth data while preserving derivatives.

### Curl Computation
```python
from mre_pinn.baseline.filters import curl

curl_u = curl(u, dx, dy, dz)
```

**Purpose**: Compute rotational component of vector field.

---

## Performance Comparison

| Method | Speed | Accuracy | Noise Robustness |
|--------|-------|----------|------------------|
| **AHI** | ⚡⚡⚡ (~1s) | ⭐⭐ | ⭐ (low) |
| **FEM** | ⚡ (~30s) | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **PINN** | ⚡⚡ (~2h train, instant inference) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Usage Examples

### Evaluate AHI
```python
import mre_pinn

example = mre_pinn.data.MREExample.load_xarrays(...)

# Evaluate AHI baseline
mre_pinn.baseline.eval_ahi_baseline(example, frequency=40)

# Results stored in example
direct_pred = example.direct  # AHI prediction
```

### Evaluate FEM (Linux only)
```python
# Requires dolfinx installation
mre_pinn.baseline.eval_fem_baseline(
    example,
    frequency=40,
    hetero=True
)

fem_pred = example.fem  # FEM prediction
```

---

## Results from MICCAI-2023

### Patient Data (155 patients)

| Method | Correlation (R) | Description |
|--------|----------------|-------------|
| AHI | 0.57 | Clinical standard |
| FEM-Helmholtz | 0.68 | Numerical baseline |
| FEM-Heterogeneous | 0.68 | Numerical baseline |
| PINN-Helmholtz | 0.75 | PINN without anatomy |
| **PINN-Heterogeneous** | **0.84** | **Best method** |

---

## See Also

- [../training/TRAINING_MODULE.md](../training/TRAINING_MODULE.md) - PINN training
- [../../ARCHITECTURE.md](../../ARCHITECTURE.md) - System overview
