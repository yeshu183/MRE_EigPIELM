# Phase 4: Two-Stage Inverse Problem Solver

## Overview

This phase implements a **physics-based inverse solver** that reconstructs the elasticity field μ(x) from measured wave displacement u(x) **without requiring ground truth μ values**. The approach uses a two-stage strategy where:

1. **Stage 1**: Fit random features to noisy wave data → smooth wave field reconstruction
2. **Stage 2**: Find elasticity field (as random features) that satisfies the PDE physics → inverse elastography

**Key Innovation**: No direct algebraic inversion (which amplifies noise). Instead, we solve a **regularized linear system where the PDE physics acts as the constraint**.

---

## Mathematical Formulation

### Random Features - Different Basis for Each Neuron

Each random feature has its own frequency vector ω and phase shift b:

**Wave field features:**
```
Φ_u(x) = [cos(ω₁ᵘ·x + b₁ᵘ), sin(ω₁ᵘ·x + b₁ᵘ), ..., cos(ωₙᵘ·x + bₙᵘ), sin(ωₙᵘ·x + bₙᵘ)]ᵀ
```

where ωᵢᵘ ~ N(0, σᵤ²I) and bᵢᵘ ~ Uniform(0, 2π) are randomly sampled.

**Elasticity features:**
```
Φ_μ(x) = [cos(ω₁ᵐ·x + b₁ᵐ), sin(ω₁ᵐ·x + b₁ᵐ), ..., cos(ωₘᵐ·x + bₘᵐ), sin(ωₘᵐ·x + bₘᵐ)]ᵀ
```

with **different** random ω and b than the wave features.

---

## Stage 1: Wave Field Reconstruction

### Problem
Given noisy measurements ũ_data at points {xᵢ}, find smooth representation.

### Solution
Represent wave field as:
```
u(x) = Φ_u(x)ᵀ W_u + b_u
```

Solve ridge regression:
```
minimize: Σᵢ ||Φ_u(xᵢ)ᵀ W_u + b_u - ũ_data(xᵢ)||² + λ_u ||W_u||²
```

**Matrix form:**
```
┌─────────────────┐ ┌────┐   ┌──────────────┐
│ Φ_u(x₁)ᵀ    1  │ │ W_u│   │ ũ_data(x₁)   │
│ Φ_u(x₂)ᵀ    1  │ │ b_u│ = │ ũ_data(x₂)   │
│     ⋮       ⋮  │ └────┘   │      ⋮       │
│ Φ_u(xₙ)ᵀ    1  │          │ ũ_data(xₙ)   │
└─────────────────┘          └──────────────┘
```

**No ground truth μ needed!** This is pure wave data fitting.

**Output:** Weights W_u that define smooth u(x) everywhere via u(x) = Φ_u(x)ᵀW_u + b_u

---

## Stage 2: Physics-Based Elasticity Inversion

### Problem
Given reconstructed u(x) from Stage 1, find μ(x) that satisfies the **Helmholtz PDE**:
```
μ(x)∇²u(x) + ρω²u(x) = 0
```

### Why This is NOT Direct Inversion

**Direct inversion (AHI) - what we avoid:**
```
μ(x) = -ρω²u(x) / ∇²u(x)  ❌
```
Problems: Division by small ∇²u → noise amplification

**Our approach - regularized linear system:**

Represent elasticity as:
```
μ(x) = Φ_μ(x)ᵀ W_μ + b_μ
```

Substitute into PDE:
```
[Φ_μ(x)ᵀ W_μ + b_μ] · ∇²u(x) + ρω²u(x) = 0
```

Rearrange:
```
Φ_μ(x)ᵀ W_μ · ∇²u(x) = -ρω²u(x) - b_μ · ∇²u(x)
```

### For Complex 3-Component Wave Field

Wave field: u = [u_x, u_y, u_z]ᵀ where each component is complex.

For each component k ∈ {x, y, z} at collocation point xⱼ:
```
Φ_μ(xⱼ)ᵀ W_μ · ∇²u_k(xⱼ) = -ρω²u_k(xⱼ) - b_μ · ∇²u_k(xⱼ)
```

### Linear System Assembly

Stack equations for all points and all components:

```
┌──────────────────────────────────┐ ┌────┐   ┌────────────────┐
│ Φ_μ(x₁)ᵀ · ∇²u_x(x₁)        1   │ │ W_μ│   │ -ρω²u_x(x₁)    │
│ Φ_μ(x₁)ᵀ · ∇²u_y(x₁)        1   │ │ b_μ│ = │ -ρω²u_y(x₁)    │
│ Φ_μ(x₁)ᵀ · ∇²u_z(x₁)        1   │ └────┘   │ -ρω²u_z(x₁)    │
│ Φ_μ(x₂)ᵀ · ∇²u_x(x₂)        1   │          │ -ρω²u_x(x₂)    │
│          ⋮                   ⋮   │          │       ⋮        │
│ Φ_μ(xₙ)ᵀ · ∇²u_z(xₙ)        1   │          │ -ρω²u_z(xₙ)    │
└──────────────────────────────────┘          └────────────────┘
```

**This is a linear system in W_μ!**

Solve with ridge regression:
```
minimize: Σⱼ Σₖ ||Φ_μ(xⱼ)ᵀW_μ · ∇²u_k(xⱼ) + ρω²u_k(xⱼ) + b_μ·∇²u_k(xⱼ)||² + λ_μ||W_μ||²
```

---

## Why This Works Without Ground Truth μ

### The Physics Provides the Supervision!

**Stage 1 supervision:** Wave measurements ũ_data
- Input: spatial coordinates x
- Output: wave displacement u(x)
- Constraint: Match measured data

**Stage 2 supervision:** PDE residual = 0 (**not** ground truth μ!)
- Input: spatial coordinates x
- Output: elasticity μ(x)
- Constraint: **Physics equation must balance**

The PDE equation:
```
μ∇²u + ρω²u = 0
```

tells us that at every point, the combination of:
- Local stiffness (μ)
- Wave curvature (∇²u)
- Inertial force (ρω²u)

must balance to zero. This **constraint alone** is enough to determine μ!

### Intuitive Explanation

Think of it like this:
1. You measure how the wave moves (u_data)
2. Stage 1 smooths out the noise in the wave measurement
3. Stage 2 asks: "What material stiffness would create this wave pattern?"
4. The PDE physics provides the answer: the μ that makes the equation balance

**No need to know μ in advance** - the physics tells us what it must be!

---

## Computing Derivatives

### For Stage 2, we need ∇²u(xⱼ)

Since u(x) = Φ_u(x)ᵀW_u + b_u, we compute:
```
∇²u(x) = [∇²Φ_u(x)]ᵀ W_u
```

Where ∇²Φ_u is computed using **PyTorch autograd** (already implemented):

For each feature:
```
∇²cos(ω·x + b) = -||ω||² cos(ω·x + b)
∇²sin(ω·x + b) = -||ω||² sin(ω·x + b)
```

PyTorch autograd handles this automatically via double differentiation.

---

## Comparison: Direct vs Physics-Based Inversion

| Aspect | Direct (AHI) | Our Approach |
|--------|--------------|--------------|
| **Formula** | μ = -ρω²u / ∇²u | Φ_μᵀW_μ·∇²u = -ρω²u |
| **Regularization** | None | Random features + ridge |
| **Noise handling** | Amplifies noise | Smooths via basis |
| **Computation** | Point-wise division | Global linear solve |
| **Stability** | Poor (division) | Good (regularized) |
| **Speed** | ~1s | ~1s (two solves) |

---

## Heterogeneous Equation (with ∇μ term)

### PDE: μ∇²u + ∇μ·∇u + ρω²u = 0

**Challenge:** The ∇μ·∇u term creates additional coupling.

**Solution:** Iterative linearization

```
Iteration k:
1. Compute ∇μₖ = [∇Φ_μ]ᵀ W_μ^(k-1) using previous weights
2. Rearrange PDE: μ∇²u = -∇μₖ·∇u - ρω²u
3. Solve: Φ_μᵀW_μ^(k) · ∇²u = -∇μₖ·∇u - ρω²u
4. Repeat until ||W_μ^(k) - W_μ^(k-1)|| < tol
```

Each iteration is still a **linear solve**, just repeated until convergence.

---

## Implementation Overview

### File Structure

```
pielm/
├── inverse.py                 # NEW - Inverse solver functions
│   ├── compute_wave_derivatives()
│   ├── solve_inverse_helmholtz()
│   └── solve_inverse_hetero_iterative()
│
├── training.py                # MODIFIED - Add inverse_mode
│   └── MREPIELMModel.solve(inverse_mode=True)
│
└── test_inverse.py           # NEW - Validation test
```

### Workflow

```python
# Initialize model
model = MREPIELMModel(example, net, pde, ...)

# Two-stage inverse solver
model.solve(inverse_mode=True)

# Stage 1 happens internally:
#   - Fit Φ_u to wave data
#   - Get smooth u(x) = Φ_uᵀW_u + b_u

# Stage 2 happens internally:
#   - Compute ∇²u using W_u from Stage 1
#   - Solve Φ_μᵀW_μ·∇²u = -ρω²u for W_μ
#   - Get μ(x) = Φ_μᵀW_μ + b_μ

# Predict on new points
u_pred, mu_pred, _ = model.predict(x_new)
```

---

## Comparison with Other Methods

### Speed

| Method | Training | Inference | Total |
|--------|----------|-----------|-------|
| AHI (Direct) | 0s | ~1s | ~1s |
| FEM | 0s | ~30s/slice | ~30s |
| PINN | ~2 hours | instant | ~2 hours |
| **PIELM (Ours)** | **~1s** | **instant** | **~1s** |

### Accuracy (R correlation with biopsy on NAFLD patients)

| Method | Correlation |
|--------|-------------|
| AHI | 0.57 |
| FEM-Helmholtz | 0.68 |
| FEM-Hetero | 0.68 |
| PINN-Helmholtz | 0.75 |
| PINN-Hetero | 0.84 |
| **PIELM (Ours)** | **TBD (expect 0.70-0.80)** |

### Noise Robustness

| Method | Robustness | Notes |
|--------|------------|-------|
| AHI | ⭐ | Laplacian amplifies noise |
| FEM | ⭐⭐⭐ | Needs preprocessing |
| PINN | ⭐⭐⭐⭐⭐ | Implicit denoising via network |
| **PIELM** | **⭐⭐⭐⭐** | **Random features regularization** |

---

## Advantages of Two-Stage PIELM

✅ **No ground truth μ needed** - Pure physics-based inversion
✅ **Fast** - Two linear solves (~1 second total) vs PINN's 100k gradient steps
✅ **Regularized** - Random features provide built-in smoothness
✅ **Interpretable** - Clear separation: Stage 1 denoises, Stage 2 inverts
✅ **Flexible** - Easy to add constraints, change regularization per stage
✅ **Stable** - No direct division, uses ridge regression

---

## Challenges and Limitations

⚠️ **Error propagation** - Stage 2 depends on Stage 1 accuracy
⚠️ **Requires good wave data** - If u is too noisy, Stage 1 may struggle
⚠️ **Laplacian sensitivity** - ∇²u computation can amplify noise if features insufficient
⚠️ **Hetero needs iteration** - Not pure one-shot for ∇μ term
⚠️ **No joint optimization** - Can't balance u and μ like PINN does

---

## Future Enhancements

1. **Hybrid approach** - Use PIELM for u, small PINN for μ
2. **Iterative refinement** - Alternate between Stage 1 and Stage 2 multiple times
3. **Adaptive regularization** - Different λ for different spatial regions
4. **Multi-frequency** - Solve for multiple ω simultaneously
5. **Uncertainty quantification** - Bayesian extension with feature sampling

---

## Summary

**Phase 4 implements a two-stage inverse solver that:**

1. Reconstructs smooth wave field from noisy data (Stage 1)
2. Inverts for elasticity using PDE physics as constraint (Stage 2)
3. Achieves this **without needing ground truth μ values**
4. Maintains PIELM's speed advantage over PINN
5. Provides regularization through random feature basis

**The key insight:** The PDE physics equation itself provides enough information to constrain the inverse problem. We don't need to know the answer (μ) in advance - we just need to enforce that the physics balances!

---

## References

- AutoDES PIELM: Ridge regression with random features
- MRE-PINN: Physics-informed neural networks for MRE inverse problem
- FEM baseline: Direct variational formulation
- AHI: Algebraic Helmholtz inversion (clinical standard)

See [PHASE1.md](PHASE1.md), [PHASE2.md](PHASE2.md), [PHASE3.md](PHASE3.md) for previous implementation phases.
