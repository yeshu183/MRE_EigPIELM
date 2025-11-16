"""
Derivative Computation Utilities for PIELM

Helper functions for computing derivatives in batch mode and handling
complex-valued fields common in MRE applications.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def split_complex_tensor(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split complex tensor into real and imaginary parts.

    Parameters
    ----------
    z : torch.Tensor (complex)
        Complex-valued tensor

    Returns
    -------
    real : torch.Tensor (float)
        Real part
    imag : torch.Tensor (float)
        Imaginary part

    Examples
    --------
    >>> z = torch.tensor([1+2j, 3+4j])
    >>> real, imag = split_complex_tensor(z)
    >>> print(real)  # tensor([1., 3.])
    >>> print(imag)  # tensor([2., 4.])
    """
    if torch.is_complex(z):
        return z.real, z.imag
    else:
        # Assume interleaved real/imag if not complex dtype
        if z.shape[-1] % 2 == 0:
            real = z[..., 0::2]
            imag = z[..., 1::2]
            return real, imag
        else:
            raise ValueError("Cannot split non-complex tensor with odd last dimension")


def merge_complex_tensor(real: torch.Tensor,
                         imag: torch.Tensor,
                         as_complex_dtype: bool = True) -> torch.Tensor:
    """
    Merge real and imaginary parts into complex tensor.

    Parameters
    ----------
    real : torch.Tensor
        Real part
    imag : torch.Tensor
        Imaginary part
    as_complex_dtype : bool, optional
        If True, return complex dtype tensor. If False, interleave real/imag.

    Returns
    -------
    z : torch.Tensor
        Complex tensor

    Examples
    --------
    >>> real = torch.tensor([1., 3.])
    >>> imag = torch.tensor([2., 4.])
    >>> z = merge_complex_tensor(real, imag)
    >>> print(z)  # tensor([1.+2.j, 3.+4.j])
    """
    if as_complex_dtype:
        return torch.complex(real, imag)
    else:
        # Interleave real and imag
        shape = list(real.shape)
        shape[-1] *= 2
        z = torch.zeros(shape, dtype=real.dtype, device=real.device)
        z[..., 0::2] = real
        z[..., 1::2] = imag
        return z


def verify_derivatives_finite_diff(basis,
                                   x: torch.Tensor,
                                   h: float = 1e-5,
                                   component: int = 0) -> dict:
    """
    Verify analytical derivatives against finite differences.

    Useful for testing Bernstein polynomial derivatives.

    Parameters
    ----------
    basis : BernsteinBasis3D
        Basis object with gradient() and laplacian() methods
    x : torch.Tensor, shape (N, 3)
        Test points
    h : float, optional
        Finite difference step size (default: 1e-5)
    component : int, optional
        Spatial component to test (0=x, 1=y, 2=z)

    Returns
    -------
    results : dict
        - 'gradient_error': Max error in gradient
        - 'laplacian_error': Max error in Laplacian
        - 'gradient_rel_error': Relative error
        - 'laplacian_rel_error': Relative error

    Examples
    --------
    >>> from mre_pielm.core.bernstein import BernsteinBasis3D
    >>> basis = BernsteinBasis3D((5,5,5), ((0,1),(0,1),(0,1)))
    >>> x = torch.rand(10, 3)
    >>> results = verify_derivatives_finite_diff(basis, x)
    >>> print(f"Gradient error: {results['gradient_error']:.2e}")
    """
    N = x.shape[0]

    # Analytical derivatives
    grad_analytical = basis.gradient(x)  # (N, n_feat, 3)
    lap_analytical = basis.laplacian(x)  # (N, n_feat)

    # Finite difference for gradient
    grad_fd = torch.zeros_like(grad_analytical)

    for dim in range(3):
        # Perturb in dimension dim
        x_plus = x.clone()
        x_minus = x.clone()
        x_plus[:, dim] += h
        x_minus[:, dim] -= h

        # Central difference
        phi_plus = basis(x_plus)
        phi_minus = basis(x_minus)
        grad_fd[:, :, dim] = (phi_plus - phi_minus) / (2 * h)

    # Finite difference for Laplacian
    lap_fd = torch.zeros_like(lap_analytical)

    for dim in range(3):
        x_plus = x.clone()
        x_minus = x.clone()
        x_plus[:, dim] += h
        x_minus[:, dim] -= h

        phi_center = basis(x)
        phi_plus = basis(x_plus)
        phi_minus = basis(x_minus)

        # Second derivative: (f(x+h) - 2f(x) + f(x-h)) / h²
        lap_fd += (phi_plus - 2 * phi_center + phi_minus) / h**2

    # Compute errors
    grad_error = torch.abs(grad_analytical - grad_fd).max().item()
    lap_error = torch.abs(lap_analytical - lap_fd).max().item()

    grad_rel_error = grad_error / torch.abs(grad_analytical).max().item() if torch.abs(grad_analytical).max() > 0 else 0
    lap_rel_error = lap_error / torch.abs(lap_analytical).max().item() if torch.abs(lap_analytical).max() > 0 else 0

    return {
        'gradient_error': grad_error,
        'laplacian_error': lap_error,
        'gradient_rel_error': grad_rel_error,
        'laplacian_rel_error': lap_rel_error,
    }


def compute_pde_residual_helmholtz(u: torch.Tensor,
                                  lap_u: torch.Tensor,
                                  mu: torch.Tensor,
                                  rho: float,
                                  omega: float) -> torch.Tensor:
    """
    Compute Helmholtz PDE residual: μ∇²u + ρω²u.

    Parameters
    ----------
    u : torch.Tensor, shape (N,) or (N, n_components)
        Wave field
    lap_u : torch.Tensor, shape (N,) or (N, n_components)
        Laplacian of wave field
    mu : torch.Tensor, shape (N,)
        Elasticity values
    rho : float
        Density (kg/m³)
    omega : float
        Angular frequency (rad/s)

    Returns
    -------
    residual : torch.Tensor
        PDE residual at each point

    Examples
    --------
    >>> u = torch.rand(100)
    >>> lap_u = torch.rand(100)
    >>> mu = torch.rand(100) * 5000  # Pa
    >>> residual = compute_pde_residual_helmholtz(u, lap_u, mu, 1000, 2*np.pi*60)
    """
    if mu.ndim == 1 and u.ndim > 1:
        mu = mu.unsqueeze(-1)  # Broadcast for multiple components

    residual = mu * lap_u + rho * omega**2 * u

    return residual


def batch_interpolate(values: np.ndarray,
                     coords_src: np.ndarray,
                     coords_dst: torch.Tensor,
                     method: str = 'linear') -> torch.Tensor:
    """
    Interpolate values from source grid to destination coordinates.

    Useful for interpolating elasticity from xarray grid to arbitrary points.

    Parameters
    ----------
    values : np.ndarray, shape (Nx, Ny, Nz) or (Nx, Ny, Nz, ...)
        Values on regular grid
    coords_src : np.ndarray, shape (3, ) tuples
        Source grid coordinates [(x0, x1, ...), (y0, y1, ...), (z0, z1, ...)]
    coords_dst : torch.Tensor, shape (N, 3)
        Destination coordinates
    method : str, optional
        Interpolation method: 'linear' or 'nearest' (default: 'linear')

    Returns
    -------
    values_interp : torch.Tensor, shape (N,) or (N, ...)
        Interpolated values

    Notes
    -----
    Uses scipy's RegularGridInterpolator internally.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Convert torch to numpy for interpolation
    coords_np = coords_dst.cpu().numpy()

    # Create interpolator
    interp = RegularGridInterpolator(
        coords_src,
        values,
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )

    # Interpolate
    values_interp = interp(coords_np)

    # Convert back to torch
    result = torch.tensor(values_interp, dtype=coords_dst.dtype, device=coords_dst.device)

    return result
