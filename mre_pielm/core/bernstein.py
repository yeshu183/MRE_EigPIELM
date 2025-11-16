"""
Bernstein Polynomial Basis Functions for 3D MRE

Implements 3D tensor product Bernstein polynomials with analytical derivatives
for use in Physics-Informed Extreme Learning Machines (PIELM).

Based on approach from: arxiv:2508.15343
"Physics-informed extreme learning machines for forward and inverse PDE problems"

Mathematical Foundation:
-----------------------
Bernstein polynomial of degree n:
    B_{i,n}(t) = C(n,i) * t^i * (1-t)^{n-i}  for t ∈ [0,1]

where C(n,i) = n! / (i!(n-i)!) is the binomial coefficient.

Properties:
- Non-negative: B_{i,n}(t) ≥ 0
- Partition of unity: Σ_{i=0}^n B_{i,n}(t) = 1
- Endpoint interpolation: B_{0,n}(0) = B_{n,n}(1) = 1
- Well-conditioned basis (better than monomials)

Derivatives:
    dB_{i,n}/dt = n * (B_{i-1,n-1}(t) - B_{i,n-1}(t))
    d²B_{i,n}/dt² = n(n-1) * (B_{i-2,n-2}(t) - 2*B_{i-1,n-2}(t) + B_{i,n-2}(t))

3D Tensor Product:
    Φ_{i,j,k}(x,y,z) = B_{i,nx}(x) * B_{j,ny}(y) * B_{k,nz}(z)
"""

import torch
import numpy as np
from typing import Tuple, Optional
import warnings


class BernsteinBasis3D:
    """
    3D Tensor product Bernstein polynomial basis for PIELM.

    This class implements Bernstein polynomials on a 3D domain using tensor products.
    It provides efficient evaluation of basis functions and their derivatives using
    PyTorch for GPU acceleration.

    Parameters
    ----------
    degrees : tuple of int
        Polynomial degrees (nx, ny, nz) for each spatial dimension.
        Total basis functions = (nx+1) * (ny+1) * (nz+1)
    domain : tuple of tuple of float
        Physical domain bounds ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    device : str, optional
        'cpu' or 'cuda' for computation device (default: 'cpu')
    cache_size : int, optional
        Maximum number of point sets to cache (default: 10)

    Attributes
    ----------
    n_features : int
        Total number of basis functions
    degrees : tuple
        Polynomial degrees per dimension
    domain : tuple
        Physical domain bounds

    Examples
    --------
    >>> basis = BernsteinBasis3D(degrees=(10, 12, 8),
    ...                          domain=((0, 0.08), (0, 0.1), (0, 0.01)))
    >>> x = torch.rand(100, 3) * torch.tensor([0.08, 0.1, 0.01])
    >>> phi = basis(x)  # Evaluate basis
    >>> print(phi.shape)  # (100, 11*13*9) = (100, 1287)
    >>> grad_phi = basis.gradient(x)  # First derivatives
    >>> lap_phi = basis.laplacian(x)  # Laplacian
    """

    def __init__(self,
                 degrees: Tuple[int, int, int],
                 domain: Tuple[Tuple[float, float],
                              Tuple[float, float],
                              Tuple[float, float]],
                 device: str = 'cpu',
                 cache_size: int = 10):

        self.nx, self.ny, self.nz = degrees
        self.degrees = degrees
        self.n_features = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        self.domain = domain
        self.device = device
        self.cache_size = cache_size

        # Extract domain bounds
        (self.x_min, self.x_max), \
        (self.y_min, self.y_max), \
        (self.z_min, self.z_max) = domain

        # Precompute binomial coefficients for stability
        self._precompute_binomials()

        # Cache for basis evaluations
        self._cache = {}
        self._cache_keys = []

        print(f"BernsteinBasis3D initialized:")
        print(f"  Degrees: nx={self.nx}, ny={self.ny}, nz={self.nz}")
        print(f"  Total basis functions: {self.n_features}")
        print(f"  Domain: x in [{self.x_min:.4f}, {self.x_max:.4f}], "
              f"y in [{self.y_min:.4f}, {self.y_max:.4f}], "
              f"z in [{self.z_min:.4f}, {self.z_max:.4f}]")
        print(f"  Device: {self.device}")

    def _precompute_binomials(self):
        """
        Precompute binomial coefficients C(n,i) using log-space for numerical stability.

        For large n, direct computation of n! can overflow. We use:
            log C(n,i) = log(n!) - log(i!) - log((n-i)!)
        then exponentiate to get C(n,i).
        """
        max_deg = max(self.nx, self.ny, self.nz)

        # Precompute for each degree
        self.binomial_x = self._compute_binomial_row(self.nx)
        self.binomial_y = self._compute_binomial_row(self.ny)
        self.binomial_z = self._compute_binomial_row(self.nz)

        # Move to device
        self.binomial_x = self.binomial_x.to(self.device)
        self.binomial_y = self.binomial_y.to(self.device)
        self.binomial_z = self.binomial_z.to(self.device)

    def _compute_binomial_row(self, n: int) -> torch.Tensor:
        """Compute all binomial coefficients C(n, i) for i=0..n."""
        indices = torch.arange(n + 1, dtype=torch.float64)

        # Use log-gamma for stability: log C(n,i) = lgamma(n+1) - lgamma(i+1) - lgamma(n-i+1)
        log_binom = (torch.lgamma(torch.tensor(n + 1, dtype=torch.float64)) -
                     torch.lgamma(indices + 1) -
                     torch.lgamma(torch.tensor(n + 1, dtype=torch.float64) - indices))

        binom = torch.exp(log_binom)
        return binom.float()  # Convert back to float32 for efficiency

    def _normalize_coords(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map physical coordinates to [0,1]³ parameter space.

        Parameters
        ----------
        x : torch.Tensor, shape (N, 3)
            Physical coordinates

        Returns
        -------
        t : torch.Tensor, shape (N, 3)
            Normalized coordinates in [0,1]³
        """
        domain_min = torch.tensor([self.x_min, self.y_min, self.z_min],
                                  dtype=x.dtype, device=x.device)
        domain_max = torch.tensor([self.x_max, self.y_max, self.z_max],
                                  dtype=x.dtype, device=x.device)

        t = (x - domain_min) / (domain_max - domain_min)

        # Clamp to [0, 1] to handle numerical errors at boundaries
        t = torch.clamp(t, 0.0, 1.0)

        return t

    def _bernstein_1d(self, t: torch.Tensor, i: int, n: int,
                     binom_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate single 1D Bernstein polynomial B_{i,n}(t).

        B_{i,n}(t) = C(n,i) * t^i * (1-t)^{n-i}

        Parameters
        ----------
        t : torch.Tensor, shape (N,)
            Parameter values in [0, 1]
        i : int
            Basis index (0 <= i <= n)
        n : int
            Polynomial degree
        binom_coeffs : torch.Tensor
            Precomputed binomial coefficients

        Returns
        -------
        B : torch.Tensor, shape (N,)
            Bernstein polynomial values
        """
        # Handle edge cases for numerical stability
        if i == 0:
            # B_{0,n}(t) = (1-t)^n
            return torch.pow(1.0 - t, n)
        elif i == n:
            # B_{n,n}(t) = t^n
            return torch.pow(t, n)
        else:
            # General case: C(n,i) * t^i * (1-t)^{n-i}
            c = binom_coeffs[i]
            ti = torch.pow(t, i)
            ti_comp = torch.pow(1.0 - t, n - i)
            return c * ti * ti_comp

    def _bernstein_1d_deriv(self, t: torch.Tensor, i: int, n: int,
                           binom_coeffs: torch.Tensor,
                           order: int = 1) -> torch.Tensor:
        """
        Evaluate derivative of 1D Bernstein polynomial.

        First derivative:
            dB_{i,n}/dt = n * (B_{i-1,n-1}(t) - B_{i,n-1}(t))

        Second derivative:
            d²B_{i,n}/dt² = n(n-1) * (B_{i-2,n-2}(t) - 2*B_{i-1,n-2}(t) + B_{i,n-2}(t))

        Parameters
        ----------
        t : torch.Tensor
            Parameter values
        i : int
            Basis index
        n : int
            Polynomial degree
        binom_coeffs : torch.Tensor
            Binomial coefficients
        order : int
            Derivative order (1 or 2)

        Returns
        -------
        dB : torch.Tensor
            Derivative values
        """
        if order == 1:
            # First derivative
            if n == 0:
                return torch.zeros_like(t)

            term1 = torch.zeros_like(t)
            term2 = torch.zeros_like(t)

            if i > 0:
                term1 = self._bernstein_1d(t, i-1, n-1, binom_coeffs[:n])
            if i < n:
                term2 = self._bernstein_1d(t, i, n-1, binom_coeffs[:n])

            return n * (term1 - term2)

        elif order == 2:
            # Second derivative
            if n <= 1:
                return torch.zeros_like(t)

            term1 = torch.zeros_like(t)
            term2 = torch.zeros_like(t)
            term3 = torch.zeros_like(t)

            if i >= 2:
                term1 = self._bernstein_1d(t, i-2, n-2, binom_coeffs[:n-1])
            if i >= 1 and i <= n-1:
                term2 = self._bernstein_1d(t, i-1, n-2, binom_coeffs[:n-1])
            if i <= n-2:
                term3 = self._bernstein_1d(t, i, n-2, binom_coeffs[:n-1])

            return n * (n - 1) * (term1 - 2 * term2 + term3)

        else:
            raise ValueError(f"Derivative order {order} not supported. Use 1 or 2.")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all basis functions at given points.

        Computes the tensor product:
            Φ_{i,j,k}(x,y,z) = B_{i,nx}(x) * B_{j,ny}(y) * B_{k,nz}(z)

        Parameters
        ----------
        x : torch.Tensor, shape (N, 3)
            Physical coordinates (x, y, z)

        Returns
        -------
        phi : torch.Tensor, shape (N, n_features)
            Basis function evaluations
            where n_features = (nx+1) * (ny+1) * (nz+1)
        """
        # Ensure input is on correct device
        x = x.to(self.device)

        # Normalize to [0,1]³
        t = self._normalize_coords(x)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        N = x.shape[0]

        # Evaluate all 1D Bernstein polynomials for each dimension
        # B_x[n, i] = B_{i,nx}(tx[n])
        B_x = torch.zeros((N, self.nx + 1), dtype=x.dtype, device=self.device)
        B_y = torch.zeros((N, self.ny + 1), dtype=x.dtype, device=self.device)
        B_z = torch.zeros((N, self.nz + 1), dtype=x.dtype, device=self.device)

        for i in range(self.nx + 1):
            B_x[:, i] = self._bernstein_1d(tx, i, self.nx, self.binomial_x)

        for j in range(self.ny + 1):
            B_y[:, j] = self._bernstein_1d(ty, j, self.ny, self.binomial_y)

        for k in range(self.nz + 1):
            B_z[:, k] = self._bernstein_1d(tz, k, self.nz, self.binomial_z)

        # Compute tensor product: Φ_{i,j,k} = B_x[i] * B_y[j] * B_z[k]
        # Reshape for efficient tensor product
        phi = torch.zeros((N, self.n_features), dtype=x.dtype, device=self.device)

        idx = 0
        for k in range(self.nz + 1):
            for j in range(self.ny + 1):
                for i in range(self.nx + 1):
                    phi[:, idx] = B_x[:, i] * B_y[:, j] * B_z[:, k]
                    idx += 1

        return phi

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of all basis functions.

        Returns ∇Φ = [∂Φ/∂x, ∂Φ/∂y, ∂Φ/∂z] for each basis function.

        For tensor product:
            ∂Φ_{i,j,k}/∂x = B'_{i,nx}(x) * B_{j,ny}(y) * B_{k,nz}(z)

        Parameters
        ----------
        x : torch.Tensor, shape (N, 3)
            Physical coordinates

        Returns
        -------
        grad_phi : torch.Tensor, shape (N, n_features, 3)
            Gradient of each basis function
            grad_phi[n, idx, 0] = ∂Φ_idx/∂x at point n
        """
        x = x.to(self.device)
        t = self._normalize_coords(x)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        N = x.shape[0]

        # Scale factors for chain rule: d/dx = d/dt * dt/dx
        scale_x = 1.0 / (self.x_max - self.x_min)
        scale_y = 1.0 / (self.y_max - self.y_min)
        scale_z = 1.0 / (self.z_max - self.z_min)

        # Evaluate basis and derivatives in each dimension
        B_x = torch.zeros((N, self.nx + 1), dtype=x.dtype, device=self.device)
        B_y = torch.zeros((N, self.ny + 1), dtype=x.dtype, device=self.device)
        B_z = torch.zeros((N, self.nz + 1), dtype=x.dtype, device=self.device)

        dB_x = torch.zeros((N, self.nx + 1), dtype=x.dtype, device=self.device)
        dB_y = torch.zeros((N, self.ny + 1), dtype=x.dtype, device=self.device)
        dB_z = torch.zeros((N, self.nz + 1), dtype=x.dtype, device=self.device)

        for i in range(self.nx + 1):
            B_x[:, i] = self._bernstein_1d(tx, i, self.nx, self.binomial_x)
            dB_x[:, i] = self._bernstein_1d_deriv(tx, i, self.nx, self.binomial_x, order=1) * scale_x

        for j in range(self.ny + 1):
            B_y[:, j] = self._bernstein_1d(ty, j, self.ny, self.binomial_y)
            dB_y[:, j] = self._bernstein_1d_deriv(ty, j, self.ny, self.binomial_y, order=1) * scale_y

        for k in range(self.nz + 1):
            B_z[:, k] = self._bernstein_1d(tz, k, self.nz, self.binomial_z)
            dB_z[:, k] = self._bernstein_1d_deriv(tz, k, self.nz, self.binomial_z, order=1) * scale_z

        # Compute gradients via product rule
        grad_phi = torch.zeros((N, self.n_features, 3), dtype=x.dtype, device=self.device)

        idx = 0
        for k in range(self.nz + 1):
            for j in range(self.ny + 1):
                for i in range(self.nx + 1):
                    # ∂Φ/∂x = B'_x * B_y * B_z
                    grad_phi[:, idx, 0] = dB_x[:, i] * B_y[:, j] * B_z[:, k]
                    # ∂Φ/∂y = B_x * B'_y * B_z
                    grad_phi[:, idx, 1] = B_x[:, i] * dB_y[:, j] * B_z[:, k]
                    # ∂Φ/∂z = B_x * B_y * B'_z
                    grad_phi[:, idx, 2] = B_x[:, i] * B_y[:, j] * dB_z[:, k]
                    idx += 1

        return grad_phi

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian of all basis functions.

        For tensor product:
            ∇²Φ_{i,j,k} = B''_{i,nx}(x)*B_{j,ny}(y)*B_{k,nz}(z) +
                          B_{i,nx}(x)*B''_{j,ny}(y)*B_{k,nz}(z) +
                          B_{i,nx}(x)*B_{j,ny}(y)*B''_{k,nz}(z)

        Parameters
        ----------
        x : torch.Tensor, shape (N, 3)
            Physical coordinates

        Returns
        -------
        lap_phi : torch.Tensor, shape (N, n_features)
            Laplacian of each basis function
        """
        x = x.to(self.device)
        t = self._normalize_coords(x)
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        N = x.shape[0]

        # Scale factors for second derivatives
        scale_xx = 1.0 / (self.x_max - self.x_min)**2
        scale_yy = 1.0 / (self.y_max - self.y_min)**2
        scale_zz = 1.0 / (self.z_max - self.z_min)**2

        # Evaluate basis and second derivatives
        B_x = torch.zeros((N, self.nx + 1), dtype=x.dtype, device=self.device)
        B_y = torch.zeros((N, self.ny + 1), dtype=x.dtype, device=self.device)
        B_z = torch.zeros((N, self.nz + 1), dtype=x.dtype, device=self.device)

        d2B_x = torch.zeros((N, self.nx + 1), dtype=x.dtype, device=self.device)
        d2B_y = torch.zeros((N, self.ny + 1), dtype=x.dtype, device=self.device)
        d2B_z = torch.zeros((N, self.nz + 1), dtype=x.dtype, device=self.device)

        for i in range(self.nx + 1):
            B_x[:, i] = self._bernstein_1d(tx, i, self.nx, self.binomial_x)
            d2B_x[:, i] = self._bernstein_1d_deriv(tx, i, self.nx, self.binomial_x, order=2) * scale_xx

        for j in range(self.ny + 1):
            B_y[:, j] = self._bernstein_1d(ty, j, self.ny, self.binomial_y)
            d2B_y[:, j] = self._bernstein_1d_deriv(ty, j, self.ny, self.binomial_y, order=2) * scale_yy

        for k in range(self.nz + 1):
            B_z[:, k] = self._bernstein_1d(tz, k, self.nz, self.binomial_z)
            d2B_z[:, k] = self._bernstein_1d_deriv(tz, k, self.nz, self.binomial_z, order=2) * scale_zz

        # Compute Laplacian
        lap_phi = torch.zeros((N, self.n_features), dtype=x.dtype, device=self.device)

        idx = 0
        for k in range(self.nz + 1):
            for j in range(self.ny + 1):
                for i in range(self.nx + 1):
                    # ∇²Φ = ∂²Φ/∂x² + ∂²Φ/∂y² + ∂²Φ/∂z²
                    lap_phi[:, idx] = (d2B_x[:, i] * B_y[:, j] * B_z[:, k] +
                                      B_x[:, i] * d2B_y[:, j] * B_z[:, k] +
                                      B_x[:, i] * B_y[:, j] * d2B_z[:, k])
                    idx += 1

        return lap_phi

    def verify_partition_of_unity(self, n_test: int = 100) -> bool:
        """
        Verify that basis functions sum to 1 (partition of unity property).

        Parameters
        ----------
        n_test : int
            Number of random test points

        Returns
        -------
        bool
            True if partition of unity holds within tolerance
        """
        # Generate random test points
        x_test = torch.rand(n_test, 3, device=self.device)
        x_test[:, 0] = x_test[:, 0] * (self.x_max - self.x_min) + self.x_min
        x_test[:, 1] = x_test[:, 1] * (self.y_max - self.y_min) + self.y_min
        x_test[:, 2] = x_test[:, 2] * (self.z_max - self.z_min) + self.z_min

        # Evaluate basis
        phi = self(x_test)

        # Sum across basis functions
        sums = phi.sum(dim=1)

        # Check if all sums are close to 1
        max_error = torch.abs(sums - 1.0).max().item()

        if max_error > 1e-8:
            warnings.warn(f"Partition of unity violated! Max error: {max_error:.2e}")
            return False

        return True
