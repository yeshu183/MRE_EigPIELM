# Solver Implementation Comparison

## Question: Is the PIELM solver using the same implementation as AutoDES?

**Short Answer**: ✅ **Yes, mathematically identical ridge regression**, but adapted for PyTorch and MRE-specific needs.

---

## Ridge Regression Formula (Both Use Same)

### AutoDES PIELM_solver_v2.ipynb (Cell 5, lines 8-15):
```python
# Add bias column
ones = np.ones((H.shape[0], 1))
Hs = np.hstack([H, ones])

# Ridge regression
if ridge and ridge > 0:
    A = Hs.T @ Hs + ridge * np.eye(Hs.shape[1])
    b = Hs.T @ K
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
else:
    c = np.linalg.pinv(Hs) @ K
```

### Our PIELM `solver.py` + `training.py`:
```python
# solver.py: Ridge regression core
def solve_linear_system(A, b, regularization=1e-6, method='ridge'):
    if method == 'ridge':
        ATA = A.T @ A  # (N, N)
        ATb = A.T @ b  # (N, K)
        ATA_reg = ATA + regularization * torch.eye(N, device=device)
        W = torch.linalg.solve(ATA_reg, ATb)
        return W

# training.py: Add bias column (like AutoDES)
ones_u = torch.ones((A_u.shape[0], 1), device=self.device)
A_u_with_bias = torch.cat([A_u, ones_u], dim=1)

# Solve
weights_with_bias = solve_linear_system(A_u_with_bias, b_u, ...)

# Split weights and bias
self.net.u_weights = weights_with_bias[:-1]  # Features
self.net.u_bias = weights_with_bias[-1:]     # Bias
```

### Mathematical Equivalence

Both solve:
```
W = (A^T A + λI)^{-1} A^T b
```

where:
- `A = [Φ | 1]` is the feature matrix with bias column
- `b` is the target vector
- `λ` is the ridge regularization parameter

---

## Key Similarities

| Aspect | AutoDES | Our PIELM | Match? |
|--------|---------|-----------|--------|
| **Ridge formula** | `(Hs^T Hs + λI)^{-1} Hs^T K` | `(A^T A + λI)^{-1} A^T b` | ✅ 100% |
| **Bias column** | Adds `ones` column to `H` | Adds `ones` column to `A_u` | ✅ 100% |
| **Fallback** | Uses `lstsq` if `solve` fails | Try/except in `solve_linear_system` | ✅ Same |
| **Pseudoinverse** | Option if `ridge=0` | Option with `method='pinv'` | ✅ Same |
| **Weight splitting** | `c_feat, c_bias = c[:-1], c[-1]` | `u_weights, u_bias = [:-1], [-1:]` | ✅ Same |

---

## Key Differences

### 1. Framework
- **AutoDES**: NumPy (CPU only)
- **Our PIELM**: PyTorch (CPU/GPU)

**Why**: Integration with MRE-PINN which uses PyTorch

### 2. System Assembly
- **AutoDES**: Single system for generic PDEs
  ```python
  H, K = assemble_system_soft(problem, feat)
  # H has PDE + BC + IC rows all at once
  ```

- **Our PIELM**: Separate systems for u and μ
  ```python
  # Solve for u (wave field)
  A_u = phi_u_data * sqrt(u_weight)
  u_weights = solve(A_u_with_bias, u_data)

  # Solve for μ (elasticity field)
  A_mu = phi_mu_data * sqrt(mu_weight)
  mu_weights = solve(A_mu_with_bias, mu_data)
  ```

**Why**: MRE has different output dimensions (u=6, μ=2)

### 3. Weighting Strategy
- **AutoDES**: Multiply rows by weights during assembly
  ```python
  H_list = [Hp * w["pde"]]
  K_list = [Kp * w["pde"]]
  H_list.append(Hb * w["bc"])
  # ... stack all
  ```

- **Our PIELM**: Multiply by sqrt of weights
  ```python
  A_u = phi_u_data * np.sqrt(u_weight)
  b_u = self.u_data * np.sqrt(u_weight)
  ```

**Why**: Mathematically equivalent (sqrt weights give same weighted least squares)

### 4. PDE Constraints
- **AutoDES**: Assembles PDE rows based on mother equation coefficients
  ```python
  Hp = a0*Phi + at*phi_t + bx*phi_x + cxx*phi_xx + ...
  ```

- **Our PIELM**: Phase 2 only does data fitting
  ```python
  # TODO Phase 3: Add PDE rows
  ```

**Why**: Incremental implementation (PDE constraints in Phase 3)

---

## Detailed Code Correspondence

### AutoDES Cell 5:
```python
def solve_problem(problem):
    # Line 1-4: Validate and get config
    validate_problem(problem)
    dim = problem["dim"]
    model = problem["model"]
    ridge = float(model.get("ridge", 0.0))

    # Line 5-6: Initialize features and assemble system
    feat = init_features(model["neurons"], dim, time_dep, seed=model["seed"])
    H, K = assemble_system_soft(problem, feat)  # PDE + BC + IC rows

    # Line 8-9: Add bias column
    ones = np.ones((H.shape[0], 1))
    Hs = np.hstack([H, ones])

    # Line 10-15: Ridge regression solve
    if ridge and ridge > 0:
        A = Hs.T @ Hs + ridge * np.eye(Hs.shape[1])
        b = Hs.T @ K
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
    else:
        c = np.linalg.pinv(Hs) @ K

    # Line 16-17: Split weights and bias
    c = c.reshape(-1,1)
    c_feat, c_bias = c[:-1], c[-1]

    # Line 19-23: Create predictor
    def predict_u(x, y=None, t=None):
        # ... compute X, T ...
        return (phi(feat, X, T) @ c_feat).ravel() + float(c_bias)

    return predict_u, info
```

### Our PIELM (training.py:solve()):
```python
def solve(self, use_pde=False):
    # Get weights
    u_weight, mu_weight, a_weight, pde_weight = self.loss_weights

    # === Solve for u (wave field) ===
    # Compute features
    phi_u_data = self.net.u_features(
        self.net.normalize_input(self.x_data),
        compute_derivatives=False
    )

    # Build weighted system
    A_u = phi_u_data * np.sqrt(u_weight)
    b_u = self.u_data * np.sqrt(u_weight)

    # Add bias column (SAME AS AUTODES)
    ones_u = torch.ones((A_u.shape[0], 1), device=self.device)
    A_u_with_bias = torch.cat([A_u, ones_u], dim=1)

    # Solve ridge regression (SAME FORMULA AS AUTODES)
    weights_with_bias = solve_linear_system(
        A_u_with_bias, b_u,
        regularization=self.regularization,
        method=self.solver_method
    )

    # Split weights and bias (SAME AS AUTODES)
    self.net.u_weights = weights_with_bias[:-1]
    self.net.u_bias = weights_with_bias[-1:]

    # === Solve for μ (elasticity field) - SIMILAR ===
    # ... same pattern ...
```

### Our PIELM (model.py:forward()):
```python
def forward(self, inputs):
    # Compute features
    phi_u = self.u_features(x, compute_derivatives=False)
    phi_mu = self.mu_features(x, compute_derivatives=False)

    # Linear prediction with bias (SAME AS AUTODES)
    u_pred = phi_u @ self.u_weights + self.u_bias
    mu_pred = phi_mu @ self.mu_weights + self.mu_bias

    # Denormalize and return
    # ...
```

---

## Summary

### ✅ What's the Same (Core Algorithm)
1. **Ridge regression formula**: `W = (A^T A + λI)^{-1} A^T b`
2. **Bias column**: Append ones to feature matrix
3. **Weight splitting**: Separate feature weights from bias
4. **Fallback solver**: lstsq if direct solve fails
5. **Prediction**: `φ(x) @ weights + bias`

### 🔄 What's Different (Implementation Details)
1. **Framework**: NumPy → PyTorch (for GPU + integration)
2. **System structure**: Single solve → Dual solve (u and μ separate)
3. **Weighting**: Row multiplication → sqrt weighting (mathematically equivalent)
4. **PDE constraints**: Cell 4 assembly → Phase 3 (TODO)

### 🎯 Bottom Line

**The solver IS using the same AutoDES PIELM algorithm**, with these adaptations:

1. ✅ PyTorch instead of NumPy (for GPU support)
2. ✅ Dual systems (u and μ) instead of single system
3. ✅ Integrated with MRE-PINN workflow
4. ⏳ PDE constraints coming in Phase 3

The **mathematical core is identical** - we're solving the same ridge regression problem with the same formula, just adapted for MRE-specific requirements and PyTorch framework.

---

## Verification

To verify the solver is working correctly, we can:

1. **Test on simple problem** (1D Poisson like AutoDES TC-2)
2. **Compare numerical results** with AutoDES on same problem
3. **Check residuals** `||Aw - b||` should be similar

This will be done in validation testing (Phase 4).
