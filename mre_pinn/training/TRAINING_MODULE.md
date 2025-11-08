# mre_pinn.training - Training Module

Training loops, loss functions, and optimization for MRE-PINN models.

---

## Files

| File | Purpose |
|------|---------|
| `pinn_training.py` | Main training class and data loader |
| `losses.py` | Loss function definitions |
| `callbacks.py` | Training callbacks (resampling, etc.) |

---

## Key Classes

### MREPINNModel - Main Training Class

```python
class MREPINNModel(deepxde.Model):
    """
    Training wrapper for MREPINN.

    Handles:
    - Training loop
    - Loss computation (wave + elasticity + PDE + anatomical)
    - Evaluation and visualization
    - Model checkpointing
    """
```

**Usage**:
```python
model = mre_pinn.training.MREPINNModel(
    example,                    # MREExample
    pinn,                       # MREPINN network
    pde,                        # WaveEquation
    loss_weights=[1, 0, 0, 1e-8],  # [wave, elast, pde, anat]
    pde_warmup_iters=10000,     # PDE warmup
    n_points=1024,              # Batch size
    device='cuda'
)

model.compile(optimizer='adam', lr=1e-4, loss=msae_loss)
model.train(100000, display_every=1000)
```

### MREPINNData - Data Loader

```python
class MREPINNData:
    """
    Data management for training.

    Features:
    - Point sampling from masked regions
    - Automatic resampling
    - Train/test splits
    """
```

---

## Loss Functions

### Total Loss

```python
L_total = w₁·L_wave + w₂·L_elast + w₃·L_PDE + w₄·L_anat

where:
  w₁, w₂, w₃, w₄ = loss_weights
```

### Component Losses

**1. Wave Loss** - Match measured displacement:
```python
L_wave = MSAE(u_true, u_pred)
```

**2. Elasticity Loss** - Match ground truth elasticity:
```python
L_elast = MSAE(μ_true, μ_pred)
```

**3. PDE Loss** - Satisfy wave equation:
```python
residual = ∇·[μ(∇u + ∇u^T)] + ρω²u
L_PDE = MSAE(residual, 0)
```

**4. Anatomical Loss** - Use MRI features:
```python
L_anat = MSAE(anat_features, 0)
```

### MSAE Loss (Mean Squared Absolute Error)

```python
def msae_loss(y_true, y_pred):
    """
    Loss for complex-valued predictions.

    MSAE = mean(|y_true - y_pred|²)
         = mean((Δreal)² + (Δimag)²)
    """
    return torch.mean(torch.abs(y_true - y_pred)**2)
```

---

## Training Schedule

### PDE Warmup

```python
Iterations 0-10k:      PDE weight = 0 (learn data first)
Iterations 10k-15k:    PDE weight increases gradually
Iterations 15k+:       PDE weight = final value
```

### Progressive Loss Weighting

```python
def get_pde_weight(iteration):
    if iteration < pde_warmup_iters:
        return 0
    else:
        step = (iteration - pde_warmup_iters) // pde_step_iters
        return pde_init_weight * (pde_step_factor ** step)
```

---

## Training Loop

```python
for iteration in range(n_iters):
    # 1. Sample training points
    x, u, μ = data.sample(n_points)  # From masked liver region

    # 2. Forward pass
    u_pred, μ_pred = model(x)

    # 3. Compute losses
    L_wave = msae_loss(u, u_pred)
    L_elast = msae_loss(μ, μ_pred)
    L_PDE = pde.residual(x, u_pred, μ_pred)
    L_total = w₁·L_wave + w₂·L_elast + w₃·L_PDE + w₄·L_anat

    # 4. Backpropagation
    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()

    # 5. Logging
    if iteration % display_every == 0:
        print(f"Iter {iteration}: Loss = {L_total:.6f}")

    # 6. Evaluation
    if iteration % test_every == 0:
        evaluate_on_full_grid()
```

---

## Callbacks

### Resampler

```python
class Resampler(Callback):
    """
    Periodically resample training points.

    Prevents overfitting to specific spatial locations.
    """

    def __init__(self, period=1000):
        self.period = period
```

**Usage**:
```python
model.train(100000, callbacks=[Resampler(period=1000)])
```

---

## Optimization

### Optimizers

**Adam** (default):
```python
model.compile(optimizer='adam', lr=1e-4)
```

**L-BFGS** (for fine-tuning):
```python
model.compile(optimizer='L-BFGS', lr=1e-3)
```

### Learning Rate Scheduling

```python
# Manual scheduling
for phase in [(1e-4, 50000), (1e-5, 50000)]:
    lr, iters = phase
    model.compile(optimizer='adam', lr=lr)
    model.train(iters)
```

---

## Monitoring

### Metrics Logged

- Total loss
- Component losses (wave, elasticity, PDE, anatomical)
- Gradient norms
- Training time per iteration

### Benchmarking

```python
model.benchmark(n_iters=100)
# Output:
# Data time/iter:  0.030s (8%)
# Model time/iter: 0.242s (66%)
# Loss time/iter:  0.094s (26%)
# Total time/iter: 0.366s
```

---

## Model Evaluation

```python
# Evaluate on full grid
dataset, arrays = model.test()

# arrays contains:
# - u: wave field [u_pred, u_diff, u_true]
# - mu: elasticity [μ_pred, μ_diff, μ_true]
# - pde: PDE residual
# - direct: AHI baseline (if available)
# - fem: FEM baseline (if available)
```

---

## Checkpointing

```python
# Save model
model.save('checkpoints/model_iter_100k.pt')

# Load model
model.restore('checkpoints/model_iter_100k.pt')
```

---

## Common Training Configurations

### Quick Test (Simulation)
```python
model = MREPINNModel(
    example, pinn, pde,
    loss_weights=[1, 0, 0, 1e-8],
    n_points=1024
)
model.train(10000)  # 10k iterations
```

### Full Training (Patient)
```python
model = MREPINNModel(
    example, pinn, pde,
    loss_weights=[1, 0, 1, 1e-4],  # Include elasticity + anatomical loss
    pde_warmup_iters=10000,
    pde_step_iters=5000,
    n_points=1024
)
model.train(100000)  # 100k iterations
```

---

## Troubleshooting

**NaN losses**:
- Reduce learning rate
- Increase PDE warmup period
- Check data normalization

**PDE loss doesn't decrease**:
- Increase PDE weight gradually
- Check PDE implementation
- Ensure gradients enabled

**Slow training**:
- Increase batch size (`n_points`)
- Enable CUDNN benchmark
- Use fewer hidden layers

---

## See Also

- [../model/MODEL_ARCHITECTURES.md](../model/MODEL_ARCHITECTURES.md) - Model architecture
- [../pde.py](../pde.py) - Physics equations
- [../../ARCHITECTURE.md](../../ARCHITECTURE.md) - System overview
