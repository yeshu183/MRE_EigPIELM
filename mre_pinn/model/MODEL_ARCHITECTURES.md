# mre_pinn.model - Neural Network Architectures

This module defines the Physics-Informed Neural Network (PINN) architectures for MRE reconstruction.

---

## Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `pinn.py` | Main PINN architecture | `MREPINN` |
| `generic.py` | Base network components | `PINN` (base class) |

---

## MREPINN - Main Model

```python
class MREPINN(nn.Module):
    """
    Dual-network PINN for MRE reconstruction.

    Two separate networks:
    1. u_pinn: Predicts wave field u(x,y,z)
    2. mu_pinn: Predicts elasticity μ(x,y,z)

    Both networks share input (spatial coordinates) but have
    independent parameters.
    """
```

### Architecture

```
Input: (x, y, z) coordinates [batch_size, 3]
       │
       ├──► u_pinn  ──► Wave field predictions [batch_size, 6]
       │                (real + imag for 3 components)
       │
       └──► mu_pinn ──► Elasticity predictions [batch_size, 2]
                        (real + imag)
```

### Network Configuration

```python
pinn = mre_pinn.model.MREPINN(
    example,                 # MREExample with data
    omega=60,               # Characteristic frequency (Hz)
    n_layers=2,             # Hidden layers per network
    n_hidden=64,            # Units per hidden layer
    activ_fn='sin',         # Activation: 'sin', 'tanh', 'swish'
    polar_input=False       # Use polar coordinates
)
```

### Key Parameters

| Parameter | Options | Purpose |
|-----------|---------|---------|
| `omega` | 30-100 | Network characteristic frequency |
| `n_layers` | 2-5 | Depth of network |
| `n_hidden` | 32-128 | Width of network |
| `activ_fn` | `'sin'`, `'tanh'`, `'swish'` | Non-linearity |
| `polar_input` | `True`, `False` | Use (r,θ) encoding |

### Activation Functions

**Sine Activation** (recommended for periodic problems):
```python
h = sin(Wx + b)
# Good for: Wave fields, periodic boundaries
```

**Tanh Activation** (standard):
```python
h = tanh(Wx + b)
# Good for: General purpose
```

**Swish Activation** (smooth):
```python
h = x * sigmoid(Wx + b)
# Good for: Deep networks
```

---

## PINN - Base Network

```python
class PINN(nn.Module):
    """
    Base fully-connected network with skip connections.

    Features:
    - Dense (DenseNet-style) skip connections
    - Flexible activation functions
    - Normalization via location/scale parameters
    """
```

### Skip Connection Pattern

```
Input (3 or 6 features)
    │
    ├──► Hidden Layer 1 (n_hidden) ──┐
    │         │                       │
    └─────────┴──► concat ──┐         │
              │              │         │
    ┌─────────┴──────────────┘         │
    │                                  │
    ├──► Hidden Layer 2 (n_hidden) ───┤
    │         │                        │
    └─────────┴──► concat ──┐          │
              │              │          │
    ┌─────────┴──────────────┴──────────┘
    │
    ├──► Output Layer ──► Predictions
```

### Input Encoding

**Cartesian** (`polar_input=False`):
```python
input = [x, y, z]  # 3 features
```

**Polar** (`polar_input=True`):
```python
r = sqrt(x² + y²)
sin_theta = y / r
cos_theta = x / r
input = [x, y, z, r, sin_theta, cos_theta]  # 6 features
```

### Normalization

Networks use location-scale normalization:
```python
# Input normalization
x_norm = (x - x_loc) / x_scale

# Output denormalization
u = u_pred * u_scale + u_loc
μ = μ_pred * μ_scale + μ_loc
```

Statistics (`loc`, `scale`) computed from training data.

---

## Forward Pass

```python
# Create model
pinn = MREPINN(example, omega=60, n_layers=2, n_hidden=64)

# Forward pass
x = torch.tensor([[0.1, 0.2, 0.05]])  # (batch_size, 3)
u_pred, mu_pred = pinn(x)

# u_pred: (batch_size, 6) - wave field (3 components × 2 real/imag)
# mu_pred: (batch_size, 2) - elasticity (1 value × 2 real/imag)
```

---

## Network Initialization

### Weight Initialization

```python
# Xavier/Glorot initialization (default)
nn.init.xavier_normal_(layer.weight)
nn.init.zeros_(layer.bias)
```

### Frequency Scaling

For sine activation with frequency encoding:
```python
# Scale weights by frequency
layer.weight.data *= omega / omega_data
```

---

## Model Size

### Parameter Count

Example configuration (`n_layers=2`, `n_hidden=64`):

**Wave Network (u_pinn)**:
```
Input→Hidden1:  3 × 64  = 192 + 64 (bias)  = 256
Hidden1→Hidden2: 67 × 64 = 4,288 + 64      = 4,352
Hidden2→Output:  131 × 6 = 786 + 6         = 792
Total: ~5,400 parameters
```

**Elasticity Network (mu_pinn)**:
```
Input→Hidden1:  3 × 64  = 192 + 64 (bias)  = 256
Hidden1→Hidden2: 67 × 64 = 4,288 + 64      = 4,352
Hidden2→Output:  131 × 2 = 262 + 2         = 264
Total: ~4,900 parameters
```

**Total MREPINN**: ~10,000 parameters (very small!)

---

## Usage Examples

### Basic Training Setup

```python
import mre_pinn

# Load data
example = mre_pinn.data.MREExample.load_xarrays('data/', '0006')

# Create model
pinn = mre_pinn.model.MREPINN(
    example,
    omega=40,
    n_layers=3,
    n_hidden=128,
    activ_fn='sin',
    polar_input=True
)

# Print architecture
print(pinn)
# Output:
# MREPINN(
#   (u_pinn): PINN(...)
#   (mu_pinn): PINN(...)
# )

# Count parameters
n_params = sum(p.numel() for p in pinn.parameters())
print(f"Total parameters: {n_params:,}")
```

### Custom Network

```python
from mre_pinn.model import PINN

# Create custom u_pinn
u_pinn = PINN(
    input_size=6,        # Polar coordinates
    output_size=6,       # 3 components × 2 (real/imag)
    n_layers=4,
    n_hidden=256,
    activ_fn='swish'
)
```

---

## Advanced Topics

### Gradient Computation

PINNs require gradients for physics loss:
```python
x.requires_grad = True  # Enable gradient tracking

u, mu = pinn(x)

# Compute gradients
grad_u = torch.autograd.grad(
    u, x,
    grad_outputs=torch.ones_like(u),
    create_graph=True  # For higher-order derivatives
)[0]
```

### Multi-GPU Training

```python
# Wrap model for DataParallel
pinn = nn.DataParallel(pinn)

# Forward pass unchanged
u, mu = pinn(x)
```

---

## Comparison with Traditional NNs

| Aspect | Traditional NN | PINN |
|--------|---------------|------|
| Input | Features (e.g., images) | Spatial coordinates |
| Output | Class labels, values | Physical fields |
| Loss | Data fitting only | Data + Physics (PDE) |
| Regularization | Dropout, L2 | Physics equations |
| Interpretability | Black box | Physics-constrained |

---

## See Also

- [../PACKAGE_OVERVIEW.md](../PACKAGE_OVERVIEW.md) - Package overview
- [../training/TRAINING_MODULE.md](../training/TRAINING_MODULE.md) - Training procedures
- [../pde.py](../pde.py) - Physics equations
- [../../ARCHITECTURE.md](../../ARCHITECTURE.md) - System architecture
