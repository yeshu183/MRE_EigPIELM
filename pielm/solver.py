"""
Linear system solver for PIELM.

PIELM reduces to solving a linear least squares problem:
    min ||AW - b||²  + λ||W||²

where:
- A: Feature matrix (data + PDE constraints)
- W: Weight matrix (to be solved)
- b: Target values (data) or zeros (PDE)
- λ: Regularization parameter
"""

import torch
import numpy as np


def solve_linear_system(A, b, regularization=1e-6, method='ridge'):
    """
    Solve linear system AW = b for W.

    Args:
        A: (M, N) feature matrix
           M = number of constraints (data points + PDE points)
           N = number of features
        b: (M, K) target matrix
           K = output dimension (6 for complex u with 3 components, 2 for complex μ)
        regularization: Ridge regularization parameter λ
        method: Solver method
            'ridge': Ridge regression W = (A^T A + λI)^{-1} A^T b
            'lstsq': PyTorch least squares (uses QR or SVD)
            'pinv': Pseudoinverse W = A^+ b

    Returns:
        W: (N, K) weight matrix
    """
    device = A.device
    M, N = A.shape
    K = b.shape[1]

    if method == 'ridge':
        # Ridge regression: W = (A^T A + λI)^{-1} A^T b
        # More stable for ill-conditioned problems
        ATA = A.T @ A  # (N, N)
        ATb = A.T @ b  # (N, K)

        # Add regularization
        ATA_reg = ATA + regularization * torch.eye(N, device=device)

        # Solve
        try:
            W = torch.linalg.solve(ATA_reg, ATb)
        except RuntimeError as e:
            print(f"Warning: torch.linalg.solve failed ({e}), falling back to lstsq")
            W, _ = torch.lstsq(ATb, ATA_reg)
            W = W[:N]

        return W

    elif method == 'lstsq':
        # PyTorch least squares (uses LAPACK gelsd - SVD-based)
        # Automatically handles regularization via rcond parameter
        rcond = regularization if regularization > 0 else None
        solution = torch.linalg.lstsq(A, b, rcond=rcond)
        W = solution.solution
        return W

    elif method == 'pinv':
        # Pseudoinverse: W = (A^T A)^{-1} A^T b = A^+ b
        # Least stable but simple
        A_pinv = torch.linalg.pinv(A, rcond=regularization)
        W = A_pinv @ b
        return W

    else:
        raise ValueError(f"Unknown solver method: {method}. "
                         f"Choose from: 'ridge', 'lstsq', 'pinv'")


def solve_with_constraints(A_data, b_data, A_pde, b_pde,
                            weights=(1.0, 1.0), regularization=1e-6, method='ridge'):
    """
    Solve linear system with separate data and PDE constraints.

    Constructs weighted system:
        [√w_data * A_data]   [√w_data * b_data]
        [√w_pde  * A_pde ] W = [√w_pde  * b_pde ]

    Args:
        A_data: (M_data, N) data feature matrix
        b_data: (M_data, K) data targets
        A_pde: (M_pde, N) PDE feature matrix
        b_pde: (M_pde, K) PDE targets (usually zeros)
        weights: (w_data, w_pde) constraint weights
        regularization: Ridge parameter λ
        method: Solver method

    Returns:
        W: (N, K) weight matrix
    """
    w_data, w_pde = weights

    # Weight and stack constraints
    A = torch.cat([
        A_data * np.sqrt(w_data),
        A_pde * np.sqrt(w_pde)
    ], dim=0)

    b = torch.cat([
        b_data * np.sqrt(w_data),
        b_pde * np.sqrt(w_pde)
    ], dim=0)

    # Solve
    return solve_linear_system(A, b, regularization=regularization, method=method)


def compute_condition_number(A):
    """
    Compute condition number of matrix A.

    Useful for diagnosing ill-conditioned systems.

    Args:
        A: (M, N) matrix

    Returns:
        cond: Condition number κ(A) = σ_max / σ_min
    """
    s = torch.linalg.svdvals(A)
    cond = s[0] / s[-1]
    return cond.item()


def compute_residual(A, W, b):
    """
    Compute residual ||AW - b||.

    Useful for checking solution quality.

    Args:
        A: (M, N) feature matrix
        W: (N, K) weight matrix
        b: (M, K) target matrix

    Returns:
        residual: Scalar residual norm
    """
    pred = A @ W
    residual = torch.norm(pred - b)
    return residual.item()
