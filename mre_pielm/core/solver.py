"""
Linear System Solvers for PIELM

Implements ridge regression and other linear solvers for the PIELM framework.
The key equation is:
    H @ W = K

where:
- H: Feature matrix (N_rows, n_features)
- W: Weight vector (n_features,)
- K: Target vector (N_rows,)

For physics-informed learning, H includes both data fitting rows and PDE constraint rows.
"""

import torch
import numpy as np
from typing import Optional, Tuple
import warnings


def solve_ridge(H: torch.Tensor,
                K: torch.Tensor,
                ridge: float = 1e-10,
                verbose: bool = False) -> torch.Tensor:
    """
    Solve linear system using ridge regression.

    Solves: (H^T H + λI) W = H^T K

    Ridge regression adds a small diagonal term for numerical stability.
    For Bernstein polynomials, smaller ridge values (1e-10 to 1e-12) are
    typically sufficient due to good conditioning.

    Parameters
    ----------
    H : torch.Tensor, shape (N_rows, n_features)
        Feature matrix (may include data + PDE rows)
    K : torch.Tensor, shape (N_rows,) or (N_rows, 1)
        Target vector
    ridge : float, optional
        Ridge regularization parameter (default: 1e-10)
    verbose : bool, optional
        Print solver diagnostics (default: False)

    Returns
    -------
    W : torch.Tensor, shape (n_features,)
        Solution weights

    Notes
    -----
    The normal equation (H^T H) can become ill-conditioned for:
    - Large number of features
    - Nearly linearly dependent basis functions
    - High polynomial degrees

    Ridge regularization improves stability at the cost of slight bias.

    Examples
    --------
    >>> H = torch.randn(1000, 100)
    >>> K = torch.randn(1000)
    >>> W = solve_ridge(H, K, ridge=1e-10)
    >>> residual = torch.norm(H @ W - K)
    """
    # Ensure K is 1D
    if K.ndim == 2 and K.shape[1] == 1:
        K = K.squeeze(1)

    N_rows, n_features = H.shape

    if verbose:
        print(f"Ridge regression:")
        print(f"  System size: {N_rows} rows × {n_features} features")
        print(f"  Ridge parameter: λ = {ridge:.2e}")

    # Compute normal equation matrices
    # A = H^T H + λI
    # b = H^T K
    HtH = H.T @ H  # (n_features, n_features)
    HtK = H.T @ K  # (n_features,)

    # Add ridge regularization
    I = torch.eye(n_features, dtype=H.dtype, device=H.device)
    A = HtH + ridge * I

    if verbose:
        # Compute condition number for diagnostics
        try:
            cond = torch.linalg.cond(A).item()
            print(f"  Condition number: {cond:.2e}")
            if cond > 1e10:
                warnings.warn(f"Large condition number ({cond:.2e}). Consider increasing ridge.")
        except:
            pass  # Condition number computation may fail

    # Solve using Cholesky decomposition (most efficient for symmetric positive definite)
    try:
        # Try Cholesky first (fastest)
        L = torch.linalg.cholesky(A)
        W = torch.cholesky_solve(HtK.unsqueeze(1), L).squeeze(1)

        if verbose:
            print("  Solver: Cholesky decomposition")

    except RuntimeError:
        # Fall back to general linear solve if Cholesky fails
        warnings.warn("Cholesky failed, using general linear solve")
        W = torch.linalg.solve(A, HtK)

        if verbose:
            print("  Solver: General linear solve")

    if verbose:
        # Compute residual
        residual = torch.norm(H @ W - K).item()
        rel_residual = residual / torch.norm(K).item()
        print(f"  Residual: {residual:.4e} (relative: {rel_residual:.4e})")

    return W


def solve_lstsq(H: torch.Tensor,
                K: torch.Tensor,
                rcond: Optional[float] = None,
                verbose: bool = False) -> torch.Tensor:
    """
    Solve linear system using least squares (QR or SVD).

    Directly minimizes ||HW - K||² without forming normal equations.
    More numerically stable than ridge for ill-conditioned problems.

    Parameters
    ----------
    H : torch.Tensor, shape (N_rows, n_features)
        Feature matrix
    K : torch.Tensor, shape (N_rows,) or (N_rows, 1)
        Target vector
    rcond : float, optional
        Cutoff for small singular values (default: None = machine precision)
    verbose : bool, optional
        Print diagnostics

    Returns
    -------
    W : torch.Tensor, shape (n_features,)
        Solution weights

    Examples
    --------
    >>> H = torch.randn(1000, 100)
    >>> K = torch.randn(1000)
    >>> W = solve_lstsq(H, K)
    """
    if K.ndim == 2 and K.shape[1] == 1:
        K = K.squeeze(1)

    if verbose:
        print(f"Least squares (lstsq):")
        print(f"  System size: {H.shape[0]} rows × {H.shape[1]} features")

    # PyTorch lstsq
    result = torch.linalg.lstsq(H, K, rcond=rcond)
    W = result.solution

    if verbose:
        residual = torch.norm(H @ W - K).item()
        rel_residual = residual / torch.norm(K).item()
        print(f"  Residual: {residual:.4e} (relative: {rel_residual:.4e})")

    return W


def solve_pinv(H: torch.Tensor,
               K: torch.Tensor,
               rcond: float = 1e-8,
               verbose: bool = False) -> torch.Tensor:
    """
    Solve using Moore-Penrose pseudoinverse.

    W = pinv(H) @ K

    where pinv(H) = V @ Σ^{-1} @ U^T from SVD H = U Σ V^T.

    Parameters
    ----------
    H : torch.Tensor, shape (N_rows, n_features)
        Feature matrix
    K : torch.Tensor, shape (N_rows,)
        Target vector
    rcond : float, optional
        Cutoff for small singular values (default: 1e-8)
    verbose : bool, optional
        Print diagnostics

    Returns
    -------
    W : torch.Tensor, shape (n_features,)
        Solution weights

    Notes
    -----
    Pseudoinverse automatically handles rank deficiency by zeroing
    small singular values. Slower than ridge regression but more robust.

    Examples
    --------
    >>> H = torch.randn(1000, 100)
    >>> K = torch.randn(1000)
    >>> W = solve_pinv(H, K, rcond=1e-8)
    """
    if K.ndim == 2 and K.shape[1] == 1:
        K = K.squeeze(1)

    if verbose:
        print(f"Pseudoinverse:")
        print(f"  System size: {H.shape[0]} rows × {H.shape[1]} features")
        print(f"  rcond: {rcond:.2e}")

    # Compute pseudoinverse
    H_pinv = torch.linalg.pinv(H, rcond=rcond)
    W = H_pinv @ K

    if verbose:
        residual = torch.norm(H @ W - K).item()
        rel_residual = residual / torch.norm(K).item()
        print(f"  Residual: {residual:.4e} (relative: {rel_residual:.4e})")

    return W


def condition_number(H: torch.Tensor) -> float:
    """
    Compute condition number of matrix H.

    Condition number κ(H) = σ_max / σ_min where σ are singular values.
    Large condition numbers (>10^6) indicate numerical instability.

    Parameters
    ----------
    H : torch.Tensor, shape (N, M)
        Matrix to analyze

    Returns
    -------
    cond : float
        Condition number

    Examples
    --------
    >>> H = torch.randn(1000, 100)
    >>> cond = condition_number(H)
    >>> print(f"Condition number: {cond:.2e}")
    """
    cond = torch.linalg.cond(H).item()
    return cond


def compute_residual(H: torch.Tensor,
                    W: torch.Tensor,
                    K: torch.Tensor) -> Tuple[float, float]:
    """
    Compute solution residual ||HW - K||.

    Parameters
    ----------
    H : torch.Tensor, shape (N_rows, n_features)
        Feature matrix
    W : torch.Tensor, shape (n_features,)
        Solution weights
    K : torch.Tensor, shape (N_rows,)
        Target vector

    Returns
    -------
    residual : float
        Absolute residual ||HW - K||
    rel_residual : float
        Relative residual ||HW - K|| / ||K||

    Examples
    --------
    >>> H = torch.randn(1000, 100)
    >>> K = torch.randn(1000)
    >>> W = solve_ridge(H, K)
    >>> res, rel_res = compute_residual(H, W, K)
    >>> print(f"Residual: {res:.4e}, Relative: {rel_res:.4e}")
    """
    if K.ndim == 2 and K.shape[1] == 1:
        K = K.squeeze(1)

    prediction = H @ W
    residual = torch.norm(prediction - K).item()
    rel_residual = residual / torch.norm(K).item() if torch.norm(K).item() > 0 else 0.0

    return residual, rel_residual


def diagnose_system(H: torch.Tensor,
                    K: torch.Tensor,
                    ridge: float = 1e-10) -> dict:
    """
    Comprehensive diagnostics for linear system.

    Parameters
    ----------
    H : torch.Tensor
        Feature matrix
    K : torch.Tensor
        Target vector
    ridge : float, optional
        Ridge parameter to test

    Returns
    -------
    diagnostics : dict
        Dictionary with diagnostic information:
        - 'shape': System dimensions
        - 'condition_number': κ(H)
        - 'ridge_condition': κ(H^T H + λI)
        - 'rank': Numerical rank
        - 'smallest_sv': Smallest singular value
        - 'largest_sv': Largest singular value

    Examples
    --------
    >>> H = torch.randn(1000, 100)
    >>> K = torch.randn(1000)
    >>> diag = diagnose_system(H, K)
    >>> print(diag)
    """
    N_rows, n_features = H.shape

    # Compute SVD
    try:
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)

        diagnostics = {
            'shape': (N_rows, n_features),
            'condition_number': (S[0] / S[-1]).item() if S[-1] > 0 else float('inf'),
            'rank': (S > 1e-10).sum().item(),
            'smallest_sv': S[-1].item(),
            'largest_sv': S[0].item(),
        }

        # Condition number with ridge
        HtH = H.T @ H
        I = torch.eye(n_features, dtype=H.dtype, device=H.device)
        A = HtH + ridge * I
        diagnostics['ridge_condition'] = torch.linalg.cond(A).item()

    except Exception as e:
        diagnostics = {
            'shape': (N_rows, n_features),
            'error': str(e)
        }

    return diagnostics
