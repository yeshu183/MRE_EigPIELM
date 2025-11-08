# mre_pinn.testing - Evaluation Module

Model evaluation, metrics computation, and visualization.

---

## Files

| File | Purpose |
|------|---------|
| `generic.py` | Test evaluator and metrics |

---

## TestEvaluator

```python
class TestEvaluator(Callback):
    """
    Periodic evaluation during training.

    Features:
    - Evaluate on full spatial grid
    - Compute metrics (MSAE, MAD, R, PSD)
    - Update visualizations
    - Save predictions
    """
```

### Usage

```python
test_eval = mre_pinn.testing.TestEvaluator(
    test_every=100,      # Evaluate every 100 iterations
    save_every=1000,     # Save every 1000 iterations
    save_prefix='exp1',  # Prefix for saved files
    interact=True        # Interactive plots in Jupyter
)

model.train(100000, callbacks=[test_eval])
```

---

## Metrics

### MSAE - Mean Squared Absolute Error
```python
MSAE = mean(|y_true - y_pred|²)
```

### MAD - Mean Absolute Deviation
```python
MAD = mean(|y_true - y_pred|)
```

### R - Pearson Correlation
```python
R = corr(|y_true|, |y_pred|)
```

### PSD - Power Spectral Density
Computed in spatial frequency domain.

---

## Metrics by Region

Metrics computed for different spatial regions:
- `all`: Entire masked domain
- `1.0`: High-confidence region (mask > threshold)
- Frequency bins: Low, mid, high spatial frequencies

---

## Saved Outputs

### Files Saved

```
{save_prefix}_model.pt          # Model checkpoint
{save_prefix}_elastogram.nc     # Predicted elasticity
{save_prefix}_wave.nc           # Predicted wave field
{save_prefix}_metrics.csv       # Metrics log
```

---

## See Also

- [../training/TRAINING_MODULE.md](../training/TRAINING_MODULE.md) - Training procedures
- [../../ARCHITECTURE.md](../../ARCHITECTURE.md) - System overview
