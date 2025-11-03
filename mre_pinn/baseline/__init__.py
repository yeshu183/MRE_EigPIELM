from . import filters, direct

# Optional FEM support (only available on Linux with dolfinx/FEniCS)
try:
    from . import fem
    from .fem import MREFEM, eval_fem_baseline
    HAS_FEM = True
except ImportError:
    HAS_FEM = False
    print("Warning: FEM baseline not available (dolfinx not installed - this is expected on Windows)")
    print("PINN training will work normally. Only FEM baseline comparisons are disabled.")

from .direct import helmholtz_inversion, eval_ahi_baseline
