"""
Core components for MRE-PIELM

- bernstein: Bernstein polynomial basis functions
- solver: Linear system solvers (ridge regression, etc.)
- derivatives: Derivative computation utilities
"""

from .bernstein import BernsteinBasis3D
from .solver import (
    solve_ridge,
    solve_lstsq,
    solve_pinv,
    condition_number,
    compute_residual,
    diagnose_system
)
from .derivatives import (
    split_complex_tensor,
    merge_complex_tensor,
    verify_derivatives_finite_diff,
    compute_pde_residual_helmholtz,
    batch_interpolate
)

__all__ = [
    'BernsteinBasis3D',
    'solve_ridge',
    'solve_lstsq',
    'solve_pinv',
    'condition_number',
    'compute_residual',
    'diagnose_system',
    'split_complex_tensor',
    'merge_complex_tensor',
    'verify_derivatives_finite_diff',
    'compute_pde_residual_helmholtz',
    'batch_interpolate',
]
