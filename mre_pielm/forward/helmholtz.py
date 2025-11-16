"""
Helmholtz Equation Forward Solver

Solves the forward MRE problem using the Helmholtz approximation:
    μ∇²u + ρω²u = 0

where:
- u: Displacement field (complex-valued)
- μ: Shear modulus (elasticity)
- ρ: Density
- ω: Angular frequency

The solver assembles a linear system H @ W = K where:
- H: Feature matrix from PDE constraints + data fitting
- W: Weights for Bernstein basis
- K: Right-hand side from known data
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any

from ..model import MREPIELM
from ..core import solve_ridge, solve_lstsq
from ..utils import (
    xarray_to_points_and_values,
    create_collocation_points,
    compute_relative_error
)


class HelmholtzForwardSolver:
    """
    Forward solver for Helmholtz equation in MRE.

    Given elasticity field μ, solves for displacement field u such that:
        μ∇²u + ρω²u = 0

    The solver constructs a linear system combining:
    1. PDE residual constraints at collocation points
    2. Data fitting constraints (if available)
    3. Boundary conditions (optional)

    Parameters
    ----------
    model : MREPIELM
        Model with initialized bases
    n_collocation : int, optional
        Number of collocation points for PDE (default: 10000)
    pde_weight : float, optional
        Weight for PDE residual loss (default: 1.0)
    data_weight : float, optional
        Weight for data fitting loss (default: 1.0)
    ridge : float, optional
        Ridge regularization parameter (default: 1e-10)
    verbose : bool, optional
        Print solver diagnostics (default: True)

    Attributes
    ----------
    model : MREPIELM
        The underlying model
    x_colloc : torch.Tensor
        Collocation points for PDE
    """

    def __init__(
        self,
        model: MREPIELM,
        n_collocation: int = 10000,
        pde_weight: float = 1.0,
        data_weight: float = 1.0,
        ridge: float = 1e-10,
        verbose: bool = True
    ):
        self.model = model
        self.n_collocation = n_collocation
        self.pde_weight = pde_weight
        self.data_weight = data_weight
        self.ridge = ridge
        self.verbose = verbose

        # Generate collocation points
        self.x_colloc = create_collocation_points(
            domain=model.domain,
            n_collocation=n_collocation,
            sampling='random',
            device=model.device
        )

        if verbose:
            print(f"\nHelmholtzForwardSolver initialized:")
            print(f"  Collocation points: {n_collocation}")
            print(f"  PDE weight: {pde_weight}")
            print(f"  Data weight: {data_weight}")
            print(f"  Ridge parameter: {ridge}")

    def assemble_pde_system(
        self,
        mu: torch.Tensor,
        component_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble linear system for PDE constraints.

        For Helmholtz equation: μ∇²u + ρω²u = 0

        We want: μ * (Laplacian basis @ weights) + ρω² * (basis @ weights) = 0
        Rearranging: (μ * Laplacian basis + ρω² * basis) @ weights = 0

        Parameters
        ----------
        mu : torch.Tensor, shape (n_colloc,)
            Elasticity values at collocation points
        component_idx : int, optional
            Which component of u to solve for (default: 0)

        Returns
        -------
        H_pde : torch.Tensor, shape (n_colloc, n_u_features)
            Feature matrix for PDE constraints
        K_pde : torch.Tensor, shape (n_colloc,)
            Right-hand side (zeros for homogeneous PDE)
        """
        n_colloc = self.x_colloc.shape[0]
        n_u_features = self.model.n_u_features

        # Evaluate basis functions and Laplacian at collocation points
        phi_u = self.model.u_basis(self.x_colloc)  # (n_colloc, n_u_features)
        lap_phi_u = self.model.u_basis.laplacian(self.x_colloc)  # (n_colloc, n_u_features)

        # PDE: μ∇²u + ρω²u = 0
        # H @ W = μ * (lap_phi @ W) + ρω² * (phi @ W) = 0
        # H = μ * lap_phi + ρω² * phi

        mu_expanded = mu.unsqueeze(1)  # (n_colloc, 1)
        omega = self.model.omega
        rho = self.model.rho

        H_pde = mu_expanded * lap_phi_u + rho * omega**2 * phi_u  # (n_colloc, n_u_features)

        # Right-hand side is zero (homogeneous PDE)
        K_pde = torch.zeros(n_colloc, dtype=torch.float32, device=self.model.device)

        return H_pde, K_pde

    def assemble_data_system(
        self,
        x_data: torch.Tensor,
        u_data: torch.Tensor,
        component_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble linear system for data fitting constraints.

        We want: basis @ weights = u_data

        Parameters
        ----------
        x_data : torch.Tensor, shape (n_data, 3)
            Spatial coordinates of data points
        u_data : torch.Tensor, shape (n_data,)
            Displacement values at data points (real or imag part)
        component_idx : int, optional
            Which component of u (default: 0)

        Returns
        -------
        H_data : torch.Tensor, shape (n_data, n_u_features)
            Feature matrix for data constraints
        K_data : torch.Tensor, shape (n_data,)
            Right-hand side (target data values)
        """
        # Evaluate basis at data points
        phi_u_data = self.model.u_basis(x_data)  # (n_data, n_u_features)

        H_data = phi_u_data
        K_data = u_data

        return H_data, K_data

    def solve_for_component(
        self,
        mu: torch.Tensor,
        component_idx: int = 0,
        x_data: Optional[torch.Tensor] = None,
        u_real_data: Optional[torch.Tensor] = None,
        u_imag_data: Optional[torch.Tensor] = None,
        solver: str = 'ridge'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve for real and imaginary weights of one u component.

        Parameters
        ----------
        mu : torch.Tensor, shape (n_colloc,)
            Elasticity at collocation points
        component_idx : int
            Component index
        x_data : torch.Tensor, optional
            Data point coordinates
        u_real_data : torch.Tensor, optional
            Real part of u at data points
        u_imag_data : torch.Tensor, optional
            Imaginary part of u at data points
        solver : str, optional
            Solver to use: 'ridge' or 'lstsq' (default: 'ridge')

        Returns
        -------
        w_real : torch.Tensor, shape (n_u_features,)
            Weights for real part
        w_imag : torch.Tensor, shape (n_u_features,)
            Weights for imaginary part
        """
        if self.verbose:
            print(f"\nSolving for component {component_idx}...")

        # Assemble PDE system
        H_pde, K_pde = self.assemble_pde_system(mu, component_idx)

        # Initialize full system with PDE constraints
        H_list = [H_pde * self.pde_weight]
        K_real_list = [K_pde * self.pde_weight]
        K_imag_list = [K_pde * self.pde_weight]

        # Add data constraints if provided
        if x_data is not None and u_real_data is not None:
            H_data, K_real_data = self.assemble_data_system(x_data, u_real_data, component_idx)
            H_list.append(H_data * self.data_weight)
            K_real_list.append(K_real_data * self.data_weight)

            if self.verbose:
                print(f"  Data points (real): {x_data.shape[0]}")

        if x_data is not None and u_imag_data is not None:
            H_data, K_imag_data = self.assemble_data_system(x_data, u_imag_data, component_idx)
            if len(H_list) == 1:  # Only PDE so far
                H_list.append(H_data * self.data_weight)
            K_imag_list.append(K_imag_data * self.data_weight)

            if self.verbose:
                print(f"  Data points (imag): {x_data.shape[0]}")

        # Stack all constraints
        H = torch.cat(H_list, dim=0)  # (n_total_constraints, n_u_features)
        K_real = torch.cat(K_real_list, dim=0)
        K_imag = torch.cat(K_imag_list, dim=0)

        if self.verbose:
            print(f"  System size: {H.shape[0]} constraints × {H.shape[1]} features")

        # Solve for real part
        if solver == 'ridge':
            w_real = solve_ridge(H, K_real, ridge=self.ridge, verbose=self.verbose)
        elif solver == 'lstsq':
            w_real = solve_lstsq(H, K_real, verbose=self.verbose)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Solve for imaginary part
        if solver == 'ridge':
            w_imag = solve_ridge(H, K_imag, ridge=self.ridge, verbose=False)
        elif solver == 'lstsq':
            w_imag = solve_lstsq(H, K_imag, verbose=False)

        if self.verbose:
            print(f"  Weights solved successfully")

        return w_real, w_imag

    def solve(
        self,
        mu: torch.Tensor,
        x_data: Optional[torch.Tensor] = None,
        u_data: Optional[torch.Tensor] = None,
        n_components: int = 2,
        solver: str = 'ridge'
    ) -> Dict[str, Any]:
        """
        Solve complete forward problem.

        Parameters
        ----------
        mu : torch.Tensor, shape (n_colloc,) or (n_colloc, n_mu_components)
            Elasticity field at collocation points
        x_data : torch.Tensor, optional
            Data point coordinates (n_data, 3)
        u_data : torch.Tensor, optional
            Complex displacement data (n_data, n_components)
        n_components : int, optional
            Number of displacement components (default: 2 for 2D)
        solver : str, optional
            Linear solver: 'ridge' or 'lstsq' (default: 'ridge')

        Returns
        -------
        results : dict
            - 'u_weights': List of weight tensors
            - 'u_pred': Predicted displacement at collocation points
            - 'pde_residual': PDE residual norm
            - 'data_error': Data fitting error (if data provided)
        """
        if self.verbose:
            print("\n" + "="*70)
            print("Helmholtz Forward Solver - Solving for u")
            print("="*70)

        # Handle mu shape
        if mu.ndim == 1:
            mu_colloc = mu
        else:
            mu_colloc = mu[:, 0]  # Use first component

        # Initialize weights list
        u_weights = []

        # Solve for each component
        for comp_idx in range(n_components):
            # Extract data for this component if provided
            u_real_data = None
            u_imag_data = None

            if u_data is not None:
                u_real_data = u_data[:, comp_idx].real
                u_imag_data = u_data[:, comp_idx].imag

            # Solve
            w_real, w_imag = self.solve_for_component(
                mu=mu_colloc,
                component_idx=comp_idx,
                x_data=x_data,
                u_real_data=u_real_data,
                u_imag_data=u_imag_data,
                solver=solver
            )

            u_weights.append(w_real)
            u_weights.append(w_imag)

        # Store weights in model
        self.model.u_weights = u_weights

        # Predict at collocation points
        outputs = self.model.forward(self.x_colloc, compute_derivatives=True)
        u_pred = outputs['u']
        lap_u = outputs['lap_u']

        # Compute PDE residual
        pde_residual = mu_colloc.unsqueeze(1) * lap_u + self.model.rho * self.model.omega**2 * u_pred
        pde_residual_norm = torch.abs(pde_residual).mean().item()

        results = {
            'u_weights': u_weights,
            'u_pred': u_pred,
            'pde_residual': pde_residual_norm,
        }

        # Compute data error if data provided
        if x_data is not None and u_data is not None:
            u_pred_data = self.model.predict_u(x_data)
            data_error = compute_relative_error(u_pred_data, u_data)
            results['data_error'] = data_error

            if self.verbose:
                print(f"\nData fitting error: {data_error:.4e}")

        if self.verbose:
            print(f"PDE residual: {pde_residual_norm:.4e}")
            print("="*70 + "\n")

        return results

    def solve_from_example(
        self,
        example: Any,
        use_data: bool = True,
        n_data_samples: Optional[int] = None,
        solver: str = 'ridge'
    ) -> Dict[str, Any]:
        """
        Solve using data from MREExample.

        Parameters
        ----------
        example : MREExample
            MRE data example
        use_data : bool, optional
            Whether to use wave data for fitting (default: True)
        n_data_samples : int, optional
            Number of data points to use (default: all)
        solver : str, optional
            Linear solver (default: 'ridge')

        Returns
        -------
        results : dict
            Solution results
        """
        # Extract mu at collocation points
        from ..utils import xarray_to_points_and_values

        # Get mu values
        x_mu, mu_values = xarray_to_points_and_values(
            example.mre,
            mask=example.mre_mask,
            device=self.model.device
        )

        # Interpolate mu to collocation points
        # For now, use a simple nearest neighbor approach
        # TODO: Implement proper interpolation
        mu_colloc = mu_values[0].repeat(self.n_collocation)  # Placeholder

        # Extract data if requested
        x_data = None
        u_data = None

        if use_data:
            x_data, u_data = xarray_to_points_and_values(
                example.wave,
                mask=example.mre_mask,
                device=self.model.device
            )

            # Subsample if requested
            if n_data_samples is not None and n_data_samples < x_data.shape[0]:
                indices = torch.randperm(x_data.shape[0])[:n_data_samples]
                x_data = x_data[indices]
                u_data = u_data[indices]

        # Determine number of components
        n_components = u_data.shape[1] if u_data is not None else 2

        # Solve
        results = self.solve(
            mu=mu_colloc,
            x_data=x_data,
            u_data=u_data,
            n_components=n_components,
            solver=solver
        )

        return results
