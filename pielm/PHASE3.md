# Phase 3: PDE Constraint Integration

**Status**: ⚠️ Partially Implemented (Helmholtz linearized, Hetero pending)
**Files Created**: `equations.py`
**Files Modified**: `training.py`

---

## Overview

Phase 3 adds **physics enforcement** to PIELM by including PDE constraint rows in the linear system. This is the key difference between pure data fitting (Phase 2) and physics-informed learning.

**Challenge**: MRE PDEs have **nonlinear coupling** between u and μ, which doesn't fit naturally into PIELM's linear framework.

---

## The Core Problem: Nonlinearity

### MRE Equations (from `mre_pinn/pde.py`)

#### Helmholtz Equation (Lines 103-109):
```python
class HelmholtzEquation(WaveEquation):
    def traction_forces(self, x, u, mu):
        laplace_u = laplacian(u, x)
        return mu * laplace_u  # ← NONLINEAR: μ * ∇²u

    def __call__(self, x, u, mu):
        f_trac, f_body = self.traction_and_body_forces(x, u, mu)
        return f_trac + f_body  # μ∇²u + ρω²u = 0
```

**Full PDE**:
```
μ∇²u + ρω²u = 0
```

#### Hetero Equation (Lines 112-131):
```python
class HeteroEquation(WaveEquation):
    def traction_forces(self, x, u, mu):
        div_grad_u = divergence(jacobian(u, x), x)  # ∇²u
        grad_mu = jacobian(mu, x)
        grad_u = jacobian(u, x)
        return mu * div_grad_u + (grad_mu * grad_u).sum(dim=-1)
        #      ↑ NONLINEAR      ↑ NONLINEAR (∇μ · ∇u)

    def __call__(self, x, u, mu):
        f_trac, f_body = self.traction_and_body_forces(x, u, mu)
        return f_trac + f_body  # μ∇²u + ∇μ·∇u + ρω²u = 0
```

**Full PDE**:
```
μ∇²u + ∇μ·∇u + ρω²u = 0
```

---

## Why PIELM Struggles with MRE

### PIELM Representation

In PIELM:
```
u(x) ≈ Φ_u(x) @ W_u + b_u    (linear in W_u)
μ(x) ≈ Φ_μ(x) @ W_μ + b_μ    (linear in W_μ)
```

### Substituting into Helmholtz

```
μ∇²u + ρω²u = 0

Substitute:
(Φ_μ @ W_μ + b_μ) * ∇²(Φ_u @ W_u) + ρω²(Φ_u @ W_u + b_u) = 0

Expand:
(Φ_μ @ W_μ) * (∇²Φ_u @ W_u) + (b_μ * ∇²Φ_u @ W_u) + ρω²(Φ_u @ W_u) + ρω²b_u = 0
     ↑ QUADRATIC (W_μ * W_u)
```

**Problem**: The term `(Φ_μ @ W_μ) * (∇²Φ_u @ W_u)` is **quadratic** in weights!

This **cannot** be expressed as a linear system `A @ W = b`.

### Substituting into Hetero (Even Worse)

```
μ∇²u + ∇μ·∇u + ρω²u = 0

Substitute:
(Φ_μ @ W_μ) * (∇²Φ_u @ W_u) + (∇Φ_μ @ W_μ) · (∇Φ_u @ W_u) + ρω²(Φ_u @ W_u) = 0
     ↑ QUADRATIC              ↑ QUADRATIC (dot product of gradients)
```

**Problem**: TWO quadratic terms! Even harder than Helmholtz.

---

## Solution Strategies

### Strategy 1: Linearization (Implemented for Helmholtz)

**Idea**: Use previous iteration's μ to linearize.

**For Helmholtz**:
```
Original: μ∇²u + ρω²u = 0

Linearized (iteration k):
μ^{k-1} * ∇²u^k + ρω²u^k = 0

In PIELM:
μ^{k-1} * (∇²Φ_u @ W_u^k) + ρω²(Φ_u @ W_u^k) = 0

Rearrange:
[μ^{k-1} * ∇²Φ_u + ρω² * Φ_u] @ W_u^k = 0
```

This is **LINEAR** in W_u^k! ✅

**Implementation** (`equations.py:construct_pde_matrix_coupled`, lines 149-221):
```python
def construct_pde_matrix_coupled(pde, x_pde, u_features, mu_features,
                                  u_prev=None, mu_prev=None, omega=None, ...):
    # Compute features and derivatives at PDE points
    phi_u, grad_phi_u, lap_phi_u = u_features(x_pde, compute_derivatives=True)

    if isinstance(pde, HelmholtzEquation):
        # Linearized: μ_prev * ∇²u + ρω²u = 0
        mu_prev = mu_prev.to(device)  # (N_pde, 1)

        A_rows = []
        for comp in range(3):  # ux, uy, uz
            # Coefficient for Laplacian: μ_prev
            coef_lap = mu_prev.squeeze(-1)  # (N_pde,)

            # Coefficient for reaction: ρω²
            coef_react = rho * (omega ** 2)

            # Constraint: μ_prev * ∇²φ + ρω² * φ
            A_comp = coef_lap.unsqueeze(-1) * lap_phi_u + coef_react * phi_u

            A_rows.append(A_comp)

        # Stack all components
        A_pde_u = torch.cat(A_rows, dim=0)  # (N_pde * 3, n_features)

        # Add bias column
        ones = torch.ones((A_pde_u.shape[0], 1), device=device)
        A_pde_u = torch.cat([A_pde_u, ones], dim=1)

        # Target: zeros
        b_pde_u = torch.zeros((N_pde * 3, 6), device=device)

        return A_pde_u, b_pde_u
```

### Strategy 2: Alternating Solve (Future Work)

**Idea**: Alternate between solving for u and μ.

**Algorithm**:
```
1. Initialize: μ^0 = data fit (no PDE)
2. For k = 1, 2, ...:
   a. Fix μ^{k-1}, solve for u^k with PDE constraints
   b. Fix u^k, solve for μ^k with PDE constraints
   c. Check convergence: ||u^k - u^{k-1}|| < tol
3. Return u^*, μ^*
```

**Advantages**:
- Each sub-problem is linear
- Can enforce PDE at each iteration
- Converges to coupled solution

**Disadvantages**:
- No longer "one-shot" (loses PIELM's main advantage)
- Convergence not guaranteed
- More complex implementation

**Status**: Not implemented (Phase 4 future work)

### Strategy 3: Forward-Only (Simplest)

**Idea**: For forward problem, μ is **known** (ground truth).

**Use Case**: Testing PIELM on simulation data where we have true μ.

**Implementation**:
```python
# Given true μ_true from data
mu_true = example.mre.field.values()

# Linearize using true μ
A_pde_u, b_pde_u = construct_pde_matrix_coupled(
    pde, x_pde, u_features, mu_features,
    mu_prev=mu_true,  # Use ground truth!
    omega=omega, rho=rho
)

# Solve for u only (μ is known)
solver.solve(use_pde=True)
```

**Advantages**:
- Simple, works immediately
- Good for validating PIELM approach
- One-shot solve

**Disadvantages**:
- Not useful for inverse problem (which is the goal of MRE!)
- Can only test accuracy of u prediction

**Status**: Implemented (can be used for validation)

### Strategy 4: Joint System (Complex)

**Idea**: Solve for [W_u; W_μ] jointly in one large system.

**Problem**: Still have quadratic terms - need to linearize or iterate.

**Status**: Not pursued (too complex, loses one-shot advantage)

---

## Implementation Status

### ✅ Implemented

1. **`equations.py`**: PDE constraint matrix construction
   - `construct_helmholtz_pde_rows()` - structure defined
   - `construct_pde_matrix_coupled()` - **Helmholtz linearization working**

2. **`training.py`**: Integration into solve()
   - Detects if μ weights available
   - Computes μ at PDE points
   - Calls `construct_pde_matrix_coupled()`
   - Appends PDE rows to data rows
   - Error handling for missing μ

### ⚠️ Partially Implemented

1. **Helmholtz equation** - Linearized version works
   - Requires μ from previous solve
   - Linear system: `[μ_prev * ∇²Φ_u + ρω² * Φ_u] @ W_u = 0`

### ❌ Not Implemented

1. **Hetero equation** - More complex linearization
   - Requires both μ and ∇μ
   - Two quadratic terms to handle
   - `construct_hetero_pde_rows()` raises `NotImplementedError`

2. **Alternating solve** - Iterative refinement
   - Would require loop in `solve()`
   - Convergence checking
   - Multiple solves

---

## Comparison to PINN

| Aspect | PINN | PIELM |
|--------|------|-------|
| **PDE handling** | Autograd on NN | Analytical features |
| **Nonlinearity** | No problem (gradient descent) | Major problem (linear solve) |
| **u-μ coupling** | Handled naturally | Requires linearization |
| **Iterations** | Many (1000-10000) | One (if linearized correctly) |
| **PDE accuracy** | Improves with training | Fixed by constraint matrix quality |

**Key Difference**: PINN uses iterative gradient descent which can handle nonlinear PDEs naturally. PIELM needs creative workarounds for nonlinearity.

---

## Comparison to AutoDES PIELM

AutoDES PIELM (Cell 4) handles **linear PDEs** with constant coefficients:

**AutoDES Mother Equation**:
```
a₀u + aₜu_t + aₜₜu_tt + bₓu_x + bᵧu_y + cₓₓu_xx + cₓᵧu_xy + cᵧᵧu_yy = f(x,y,t)
```

All coefficients (a₀, aₜ, bₓ, cₓₓ, ...) are **constants**.

**AutoDES PDE Assembly** (Cell 4, lines 68-73):
```python
# PDE rows
Phi = phi(feat, X, T)
Hp = np.zeros((X.shape[0], feat["neurons"]))
if is_nonzero(a0):  Hp += a0  * Phi
if is_nonzero(at):  Hp += at  * phi_t(feat, X, T)
if is_nonzero(bx):  Hp += bx  * phi_x(feat, X, T)
if is_nonzero(cxx): Hp += cxx * phi_xx(feat, X, T)
# ... more terms ...
```

**Key**: All coefficients are **scalar constants** - no coupling!

**MRE Challenge**: Our coefficients are **field variables** (μ(x), ∇μ(x)) - requires coupling!

---

## Usage Examples

### Example 1: Data-Only Solve (Phase 2)

```python
import pielm
import mre_pinn

# Load data
example = mre_pinn.data.MREExample.load_xarrays('data/BIOQIC/fem_box', 60)
pde = mre_pinn.pde.WaveEquation.from_name('helmholtz', omega=60)

# Create model
model = pielm.MREPIELM(example, omega=60, n_features=2000)
solver = pielm.MREPIELMModel(example, model, pde, loss_weights=[1,1,0,0])

# Solve (data only)
solver.solve(use_pde=False)  # ✅ Works (Phase 2)
```

### Example 2: Helmholtz with PDE (Phase 3)

```python
# First solve for μ (data-only)
solver.solve(use_pde=False)  # Get initial μ

# Now solve for u with PDE constraints (uses μ from above)
solver.solve(use_pde=True)   # ⚠️ Partially working (linearized)
```

This will:
1. Use μ weights from first solve
2. Compute μ values at PDE points
3. Build linearized PDE constraints: `μ_prev * ∇²Φ_u + ρω² * Φ_u`
4. Solve combined system (data + PDE)

### Example 3: Forward Problem with True μ

```python
# Use ground truth μ for PDE constraints
import torch

# Get true μ from data
x_pde_np = solver.x_pde.cpu().numpy()
mu_true_np = example.mre.field.interp(
    x=x_pde_np[:,0], y=x_pde_np[:,1], z=x_pde_np[:,2]
).values.reshape(-1, 1)
mu_true = torch.tensor(mu_true_np.real, device=solver.device)

# Manually construct PDE constraints with true μ
from pielm.equations import construct_pde_matrix_coupled
A_pde, b_pde = construct_pde_matrix_coupled(
    pde, solver.x_pde, solver.net.u_features, solver.net.mu_features,
    mu_prev=mu_true, omega=60, rho=1000
)

# ... integrate into solve() ...
```

---

## Detailed Implementation

### File: `equations.py`

#### Function: `construct_pde_matrix_coupled()`

**Purpose**: Build PDE constraint rows for linear system.

**Inputs**:
- `pde`: MRE equation object (Helmholtz or Hetero)
- `x_pde`: (N_pde, 3) collocation points
- `u_features`: RandomFeatures for wave field
- `mu_features`: RandomFeatures for elasticity
- `mu_prev`: (N_pde, 1) μ values for linearization

**Outputs**:
- `A_pde_u`: (N_pde * 3, n_features + 1) constraint matrix
- `b_pde_u`: (N_pde * 3, 6) target (zeros)

**Algorithm**:
```python
1. Compute features and derivatives at x_pde
   - phi_u, grad_phi_u, lap_phi_u = u_features(x_pde, derivatives=True)

2. For Helmholtz:
   For each component c in [x, y, z]:
      - Build constraint row:
        A[c] = μ_prev * ∇²φ_u + ρω² * φ_u

3. Stack all components and add bias:
   A_pde = cat([A[x], A[y], A[z]])  # (N_pde * 3, n_features)
   A_pde = cat([A_pde, ones])       # Add bias column

4. Target is zeros:
   b_pde = zeros(N_pde * 3, 6)  # 6 for complex (real/imag of 3 components)

5. Return A_pde, b_pde
```

### File: `training.py` (Modified)

#### Method: `MREPIELMModel.solve(use_pde=False)`

**Integration Logic** (lines 187-239):

```python
if use_pde and pde_weight > 0:
    # 1. Check if μ weights available
    if self.net.mu_weights is not None:
        # 2. Compute μ at PDE points
        phi_mu_pde = self.net.mu_features(x_pde_norm)
        mu_pde = phi_mu_pde @ self.net.mu_weights + self.net.mu_bias
        mu_pde_real = as_complex(mu_pde, polar=True).real.abs()

        # 3. Construct PDE constraints
        A_pde_u, b_pde_u = construct_pde_matrix_coupled(
            self.pde, self.x_pde, self.net.u_features, self.net.mu_features,
            mu_prev=mu_pde_real, omega=omega, rho=rho
        )

        # 4. Weight and combine
        A_pde_u *= sqrt(pde_weight)
        b_pde_u *= sqrt(pde_weight)

        A_u_with_bias = cat([A_u_with_bias, A_pde_u], dim=0)
        b_u = cat([b_u, b_pde_u], dim=0)
    else:
        # μ not available - error
        raise RuntimeError("Need μ for PDE constraints. Call solve(use_pde=False) first.")
```

---

## Challenges & Limitations

### 1. Nonlinear Coupling
**Problem**: μ * ∇²u term is quadratic in weights
**Solution**: Linearization (use μ from previous iteration)
**Limitation**: Not truly "physics-informed" in one-shot

### 2. Requires Two Solves
**Problem**: First solve for μ (data-only), then solve for u (with PDE)
**Impact**: Loses "one-shot" advantage of PIELM
**Workaround**: Still faster than 1000+ iterations of PINN

### 3. Hetero Equation More Complex
**Problem**: ∇μ · ∇u has TWO field variables
**Status**: Not fully implemented
**Future**: Needs careful linearization or alternating solve

### 4. Complex Values
**Problem**: MRE uses complex u and μ
**Current**: Solve for real/imag separately (6 outputs for u)
**Alternative**: Could extend to native complex arithmetic

### 5. Inverse Problem
**Problem**: Goal is to find μ from measured u
**Current**: Can only solve forward (u from μ)
**Future**: Need alternating solve or different formulation

---

## Future Work (Phase 4)

1. **Implement Hetero Linearization**
   - Handle ∇μ · ∇u term
   - Test on heterogeneous phantoms

2. **Alternating Solve**
   - Iterate: solve u, solve μ, repeat
   - Convergence criteria
   - Compare with PINN

3. **Inverse Problem Formulation**
   - Given measured u, find μ
   - Physics-informed regularization
   - Compare with AHI, FEM baselines

4. **Validation & Benchmarking**
   - Test on BIOQIC data
   - Compare accuracy vs PINN
   - Compare speed vs PINN
   - Document results

---

## Summary

### Phase 3 Achievements
1. ✅ Identified nonlinearity challenge
2. ✅ Implemented linearization strategy
3. ✅ Helmholtz equation PDE constraints working
4. ✅ Integration into training workflow
5. ✅ Comprehensive documentation of challenges

### Key Insights
- **MRE PDEs are harder for PIELM** than for PINN due to nonlinear coupling
- **Linearization works** but requires iteration (loses one-shot advantage)
- **AutoDES approach doesn't directly apply** to MRE (different PDE structure)
- **Forward problem is tractable**, inverse problem needs more work

### What Works Now
```python
# Two-step solve
solver.solve(use_pde=False)  # Fit μ
solver.solve(use_pde=True)   # Fit u with PDE (using μ from step 1)
```

### What's Next
- Hetero equation implementation
- Full alternating solve
- Inverse problem formulation
- Validation on BIOQIC data

---

**Next**: Phase 4 - Testing, validation, and comparison with PINN
