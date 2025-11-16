"""
MREPIELM Model Architecture

This module implements the core MREPIELM class for MRE forward and inverse problems.
Uses dual Bernstein polynomial bases for displacement (u) and elasticity (mu).
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
import xarray as xr

from .core import BernsteinBasis3D
from .utils import (
    extract_domain_bounds,
    extract_normalization_stats,
    standardize_coords,
    unstandardize_output
)


class MREPIELM(torch.nn.Module):
    """
    MRE Physics-Informed Extreme Learning Machine.

    This model uses dual Bernstein polynomial bases to represent:
    - u: Displacement field (complex-valued wave field)
    - mu: Elasticity field (shear modulus)

    The model solves linear systems of the form H @ W = K where:
    - H: Feature matrix from basis evaluations
    - W: Weight vector to be solved
    - K: Target vector from data + PDE constraints

    Parameters
    ----------
    u_degrees : tuple of int
        Polynomial degrees for u basis (nx, ny, nz)
    mu_degrees : tuple of int
        Polynomial degrees for mu basis (nx, ny, nz)
    domain : tuple of tuples
        Physical domain bounds ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    omega : float
        Angular frequency (rad/s)
    rho : float, optional
        Material density (kg/m^3, default: 1000)
    device : str, optional
        Device for computation ('cpu' or 'cuda')
    normalize_inputs : bool, optional
        Whether to normalize spatial coordinates (default: True)
    normalize_outputs : bool, optional
        Whether to normalize u and mu outputs (default: True)

    Attributes
    ----------
    u_basis : BernsteinBasis3D
        Basis functions for displacement field
    mu_basis : BernsteinBasis3D
        Basis functions for elasticity field
    u_weights : torch.Tensor or None
        Learned weights for u basis (real and imag components)
    mu_weights : torch.Tensor or None
        Learned weights for mu basis
    """

    def __init__(
        self,
        u_degrees: Tuple[int, int, int],
        mu_degrees: Tuple[int, int, int],
        domain: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        omega: float,
        rho: float = 1000.0,
        device: str = 'cpu',
        normalize_inputs: bool = True,
        normalize_outputs: bool = True
    ):
        super().__init__()

        self.domain = domain
        self.omega = torch.tensor(omega, dtype=torch.float32, device=device)
        self.rho = torch.tensor(rho, dtype=torch.float32, device=device)
        self.device = device
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs

        # Create Bernstein bases
        print("\n" + "="*70)
        print("Initializing MREPIELM Model")
        print("="*70)

        print("\nDisplacement (u) basis:")
        self.u_basis = BernsteinBasis3D(
            degrees=u_degrees,
            domain=domain,
            device=device
        )

        print("\nElasticity (mu) basis:")
        self.mu_basis = BernsteinBasis3D(
            degrees=mu_degrees,
            domain=domain,
            device=device
        )

        # Normalization parameters (to be set from data)
        self.input_loc = None    # Center of coordinates
        self.input_scale = None  # Extent of coordinates
        self.u_loc = None        # Mean of u (real and imag)
        self.u_scale = None      # Std of u
        self.mu_loc = None       # Mean of mu
        self.mu_scale = None     # Std of mu

        # Learned weights (to be solved for)
        self.u_weights = None  # (2 * n_u_features,) for real and imag
        self.mu_weights = None # (n_mu_features,)

        print(f"\nModel configuration:")
        print(f"  Domain: {domain}")
        print(f"  Frequency: {omega / (2*np.pi):.1f} Hz (omega = {omega:.2f} rad/s)")
        print(f"  Density: {rho} kg/m^3")
        print(f"  Device: {device}")
        print(f"  Normalize inputs: {normalize_inputs}")
        print(f"  Normalize outputs: {normalize_outputs}")
        print("="*70 + "\n")

    @classmethod
    def from_example(
        cls,
        example: Any,
        u_degrees: Tuple[int, int, int],
        mu_degrees: Tuple[int, int, int],
        frequency: Optional[float] = None,
        device: str = 'cpu'
    ):
        """
        Create MREPIELM from MREExample.

        Parameters
        ----------
        example : MREExample
            MRE data example
        u_degrees : tuple of int
            Polynomial degrees for u basis
        mu_degrees : tuple of int
            Polynomial degrees for mu basis
        frequency : float, optional
            Frequency in Hz (if None, extracted from example)
        device : str, optional
            Device for computation

        Returns
        -------
        model : MREPIELM
            Initialized model with normalization parameters set
        """
        # Extract domain bounds from wave field
        domain = extract_domain_bounds(example.wave)

        # Get frequency
        if frequency is None:
            # Try to extract from example
            if 'frequency' in example.wave.coords:
                frequency = float(example.wave.frequency.item())
            else:
                # Parse from example_id if it's a frequency string
                try:
                    frequency = float(example.example_id)
                except:
                    raise ValueError("Could not determine frequency from example")

        omega = 2 * np.pi * frequency

        # Create model
        model = cls(
            u_degrees=u_degrees,
            mu_degrees=mu_degrees,
            domain=domain,
            omega=omega,
            device=device
        )

        # Set normalization parameters from example
        model.set_normalization_from_example(example)

        return model

    def set_normalization_from_example(self, example: Any):
        """
        Extract and set normalization parameters from MREExample.

        Matches normalization strategy from MREPINN:
        - Inputs normalized by center and extent
        - Outputs normalized by mean and std

        Parameters
        ----------
        example : MREExample
            MRE data example
        """
        # Extract metadata
        metadata = example.metadata

        # Spatial normalization (from wave field)
        center_vals = np.array([v.values for v in metadata['center'].wave.values])
        extent_vals = np.array([v.values for v in metadata['extent'].wave.values])

        self.input_loc = torch.as_tensor(center_vals, dtype=torch.float32, device=self.device)
        self.input_scale = torch.as_tensor(extent_vals, dtype=torch.float32, device=self.device)

        # Output normalization (from statistics)
        stats = example.describe()

        # Wave field (u) - handle complex components
        u_mean = []
        u_std = []
        for comp in example.wave.component.values:
            u_mean.append(stats['mean'].loc['wave', comp])
            u_std.append(stats['std'].loc['wave', comp])
        self.u_loc = torch.tensor(u_mean, dtype=torch.float32, device=self.device)
        self.u_scale = torch.tensor(u_std, dtype=torch.float32, device=self.device)

        # Elasticity (mu) - handle components
        mu_mean = []
        mu_std = []
        for comp in example.mre.component.values:
            mu_mean.append(stats['mean'].loc['mre', comp])
            mu_std.append(stats['std'].loc['mre', comp])
        self.mu_loc = torch.tensor(mu_mean, dtype=torch.float32, device=self.device)
        self.mu_scale = torch.tensor(mu_std, dtype=torch.float32, device=self.device)

        print("Normalization parameters set from example:")
        print(f"  Input center: {self.input_loc.cpu().numpy()}")
        print(f"  Input scale: {self.input_scale.cpu().numpy()}")
        print(f"  u mean: {self.u_loc.cpu().numpy()}")
        print(f"  u std: {self.u_scale.cpu().numpy()}")
        print(f"  mu mean: {self.mu_loc.cpu().numpy()}")
        print(f"  mu std: {self.mu_scale.cpu().numpy()}")

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize spatial coordinates.

        Normalization: x_norm = (x - center) / extent * omega

        Parameters
        ----------
        x : torch.Tensor, shape (N, 3)
            Physical coordinates

        Returns
        -------
        x_norm : torch.Tensor, shape (N, 3)
            Normalized coordinates
        """
        if not self.normalize_inputs or self.input_loc is None:
            return x

        x_norm = (x - self.input_loc) / self.input_scale
        x_norm = x_norm * self.omega
        return x_norm

    def unnormalize_u(self, u_norm: torch.Tensor) -> torch.Tensor:
        """Unnormalize displacement output."""
        if not self.normalize_outputs or self.u_loc is None:
            return u_norm
        return u_norm * self.u_scale + self.u_loc

    def unnormalize_mu(self, mu_norm: torch.Tensor) -> torch.Tensor:
        """Unnormalize elasticity output."""
        if not self.normalize_outputs or self.mu_loc is None:
            return mu_norm
        return mu_norm * self.mu_scale + self.mu_loc

    def forward(
        self,
        x: torch.Tensor,
        compute_derivatives: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: evaluate u and mu at spatial coordinates.

        Parameters
        ----------
        x : torch.Tensor, shape (N, 3)
            Physical spatial coordinates
        compute_derivatives : bool, optional
            If True, also compute gradients and Laplacians

        Returns
        -------
        outputs : dict
            Dictionary containing:
            - 'u': Displacement (complex, shape (N, n_components))
            - 'mu': Elasticity (real, shape (N, n_components))
            - 'grad_u': Gradient of u (if compute_derivatives=True)
            - 'lap_u': Laplacian of u (if compute_derivatives=True)
            - 'grad_mu': Gradient of mu (if compute_derivatives=True)
        """
        if self.u_weights is None or self.mu_weights is None:
            raise RuntimeError("Model weights not initialized. Call fit() first.")

        # Normalize inputs if enabled
        if self.normalize_inputs:
            x_norm = self.normalize_input(x)
        else:
            x_norm = x

        # Evaluate bases
        phi_u = self.u_basis(x)  # (N, n_u_features)
        phi_mu = self.mu_basis(x)  # (N, n_mu_features)

        # Predict u (complex-valued)
        n_components = len(self.u_loc)
        u_pred = torch.zeros(x.shape[0], n_components, dtype=torch.complex64, device=self.device)

        for i in range(n_components):
            # Real and imaginary parts
            w_real = self.u_weights[2*i]
            w_imag = self.u_weights[2*i + 1]

            u_real = phi_u @ w_real
            u_imag = phi_u @ w_imag

            u_pred[:, i] = torch.complex(u_real, u_imag)

        # Predict mu (real-valued)
        mu_pred = phi_mu @ self.mu_weights.T  # (N, n_mu_components)

        # Unnormalize outputs
        if self.normalize_outputs:
            # For complex u, normalize real and imag separately
            u_real_norm = u_pred.real * self.u_scale + self.u_loc
            u_imag_norm = u_pred.imag * self.u_scale + self.u_loc
            u_pred = torch.complex(u_real_norm, u_imag_norm)

            mu_pred = self.unnormalize_mu(mu_pred)

        outputs = {
            'u': u_pred,
            'mu': mu_pred
        }

        # Compute derivatives if requested
        if compute_derivatives:
            grad_phi_u = self.u_basis.gradient(x)  # (N, n_u_features, 3)
            lap_phi_u = self.u_basis.laplacian(x)  # (N, n_u_features)
            grad_phi_mu = self.mu_basis.gradient(x)  # (N, n_mu_features, 3)

            # Gradient of u (complex)
            grad_u = torch.zeros(x.shape[0], n_components, 3, dtype=torch.complex64, device=self.device)
            lap_u = torch.zeros(x.shape[0], n_components, dtype=torch.complex64, device=self.device)

            for i in range(n_components):
                w_real = self.u_weights[2*i]
                w_imag = self.u_weights[2*i + 1]

                # Gradient: (N, n_features, 3) @ (n_features,) -> (N, 3)
                # Need to contract over feature dimension
                grad_u_real = torch.einsum('nfi,f->ni', grad_phi_u, w_real)  # (N, 3)
                grad_u_imag = torch.einsum('nfi,f->ni', grad_phi_u, w_imag)  # (N, 3)
                grad_u[:, i, :] = torch.complex(grad_u_real, grad_u_imag)

                # Laplacian: (N, n_features) @ (n_features,) -> (N,)
                lap_u_real = lap_phi_u @ w_real  # (N,)
                lap_u_imag = lap_phi_u @ w_imag  # (N,)
                lap_u[:, i] = torch.complex(lap_u_real, lap_u_imag)

            # Gradient of mu (real): (N, n_features, 3) @ (n_components, n_features) -> (N, n_components, 3)
            # Contract over feature dimension for each component
            grad_mu = torch.einsum('nfi,cf->nci', grad_phi_mu, self.mu_weights)  # (N, n_mu_components, 3)

            outputs['grad_u'] = grad_u
            outputs['lap_u'] = lap_u
            outputs['grad_mu'] = grad_mu

        return outputs

    def predict_u(self, x: torch.Tensor) -> torch.Tensor:
        """Predict displacement at coordinates."""
        return self.forward(x, compute_derivatives=False)['u']

    def predict_mu(self, x: torch.Tensor) -> torch.Tensor:
        """Predict elasticity at coordinates."""
        return self.forward(x, compute_derivatives=False)['mu']

    @property
    def n_u_features(self) -> int:
        """Number of basis functions for u."""
        return self.u_basis.n_features

    @property
    def n_mu_features(self) -> int:
        """Number of basis functions for mu."""
        return self.mu_basis.n_features

    @property
    def n_parameters(self) -> int:
        """Total number of parameters to solve for."""
        # u has 2 weights per component (real + imag) per basis function
        # mu has 1 weight per component per basis function
        n_u_components = len(self.u_loc) if self.u_loc is not None else 1
        n_mu_components = len(self.mu_loc) if self.mu_loc is not None else 1

        n_u_params = 2 * n_u_components * self.n_u_features
        n_mu_params = n_mu_components * self.n_mu_features

        return n_u_params + n_mu_params

    def __repr__(self) -> str:
        return (
            f"MREPIELM(\n"
            f"  u_basis: {self.n_u_features} features (degrees {self.u_basis.degrees}),\n"
            f"  mu_basis: {self.n_mu_features} features (degrees {self.mu_basis.degrees}),\n"
            f"  omega: {self.omega.item():.2f} rad/s ({self.omega.item()/(2*np.pi):.1f} Hz),\n"
            f"  rho: {self.rho.item():.0f} kg/m^3,\n"
            f"  total_parameters: {self.n_parameters},\n"
            f"  device: {self.device}\n"
            f")"
        )
