"""
PDE constraint matrix construction for PIELM.

This module builds the constraint rows that enforce MRE physics equations
in the PIELM linear system. Unlike PINN which uses autograd on neural networks,
PIELM uses analytical derivatives of random features.

Two main equations:
1. Helmholtz (homogeneous): μ∇²u + ρω²u = 0
2. Hetero (heterogeneous): μ∇²u + ∇μ·∇u + ρω²u = 0
"""

import torch
import numpy as np
from mre_pinn.pde import HelmholtzEquation, HeteroEquation


def construct_helmholtz_pde_rows(
    phi_u, grad_phi_u, lap_phi_u,
    phi_mu,
    omega, rho=1000,
    device='cpu'
):
    """
    Construct PDE constraint rows for Helmholtz equation.

    Equation (from mre_pinn/pde.py:103-109):
        Traction forces: f_trac = μ∇²u
        Body forces:     f_body = ρω²u
        PDE residual:    μ∇²u + ρω²u = 0

    For PIELM, we need to express this in terms of random features:
        u ≈ Φ_u @ W_u + b_u
        μ ≈ Φ_μ @ W_μ + b_μ

    Substituting:
        (Φ_μ @ W_μ + b_μ) * (∇²Φ_u @ W_u) + ρω² * (Φ_u @ W_u + b_u) = 0

    This is NONLINEAR in weights (μ * ∇²u term), which makes it difficult
    for PIELM's linear system.

    **Simplification Strategy**:
    For forward problem (given true μ, solve for u), we can treat μ as known:
        μ_true * (∇²Φ_u @ W_u) + ρω² * (Φ_u @ W_u) = 0

    Rearranging:
        [μ_true * ∇²Φ_u + ρω² * Φ_u] @ W_u = 0

    Args:
        phi_u: (N_pde, n_features) - wave field features φ_u(x)
        grad_phi_u: (N_pde, n_features, 3) - gradient ∇φ_u(x)
        lap_phi_u: (N_pde, n_features) - Laplacian ∇²φ_u(x)
        phi_mu: (N_pde, n_features) - elasticity features φ_μ(x)
        omega: Angular frequency (rad/s)
        rho: Material density (kg/m³)
        device: 'cpu' or 'cuda'

    Returns:
        A_pde_u: (N_pde * 3, n_features) - PDE constraint matrix for u
                 Each component (ux, uy, uz) gets separate constraints
        b_pde_u: (N_pde * 3, 6) - Target (zeros) for real/imag parts
    """
    N_pde = phi_u.shape[0]
    n_features = phi_u.shape[1]

    # For forward problem, we need μ values at PDE points
    # This will be provided by MREPIELMModel during solve
    # For now, return structure for integration

    raise NotImplementedError(
        "Helmholtz PDE constraints require μ values at collocation points. "
        "This needs coupling between u and μ systems, which is complex for PIELM. "
        "See PHASE3.md for discussion of implementation strategies."
    )


def construct_hetero_pde_rows(
    phi_u, grad_phi_u, lap_phi_u,
    phi_mu, grad_phi_mu,
    omega, rho=1000,
    device='cpu'
):
    """
    Construct PDE constraint rows for Hetero equation.

    Equation (from mre_pinn/pde.py:112-131):
        Traction forces: f_trac = μ∇²u + ∇μ·∇u
        Body forces:     f_body = ρω²u
        PDE residual:    μ∇²u + ∇μ·∇u + ρω²u = 0

    This is even more complex than Helmholtz due to the ∇μ·∇u coupling term.

    For PIELM:
        u ≈ Φ_u @ W_u + b_u
        μ ≈ Φ_μ @ W_μ + b_μ
        ∇u ≈ ∇Φ_u @ W_u
        ∇μ ≈ ∇Φ_μ @ W_μ

    Substituting:
        (Φ_μ @ W_μ) * (∇²Φ_u @ W_u) + (∇Φ_μ @ W_μ) · (∇Φ_u @ W_u) + ρω² * (Φ_u @ W_u) = 0

    This has QUADRATIC terms: W_μ * W_u, which cannot be expressed as a linear system.

    **Possible Strategies**:
    1. Alternating solve: Fix μ, solve for u; fix u, solve for μ; iterate
    2. Linearization: Use previous iteration's μ/u to linearize
    3. Forward only: Given true μ, solve for u (same as Helmholtz)

    Args:
        phi_u: (N_pde, n_features) - wave field features
        grad_phi_u: (N_pde, n_features, 3) - gradient of features
        lap_phi_u: (N_pde, n_features) - Laplacian of features
        phi_mu: (N_pde, n_features) - elasticity features
        grad_phi_mu: (N_pde, n_features, 3) - gradient of mu features
        omega: Angular frequency
        rho: Material density
        device: Device

    Returns:
        A_pde_u: PDE constraint matrix
        b_pde_u: Target vector
    """
    raise NotImplementedError(
        "Hetero PDE constraints have quadratic coupling between u and μ. "
        "PIELM's linear system cannot directly handle this. "
        "Requires iterative or linearization approach. "
        "See PHASE3.md for detailed discussion."
    )


def construct_pde_matrix_coupled(
    pde,
    x_pde,
    u_features, mu_features,
    u_prev=None, mu_prev=None,
    omega=None,
    rho=1000,
    device='cpu'
):
    """
    Construct PDE constraint matrices with coupling.

    This is a more sophisticated approach that handles the nonlinear coupling
    between u and μ in MRE equations.

    **Strategy**: Use previous iteration values to linearize.

    For Helmholtz: μ∇²u + ρω²u = 0
        Linearize: μ_prev * ∇²u + ρω²u = 0
        This becomes linear in u: [μ_prev * ∇²Φ_u + ρω² * Φ_u] @ W_u = 0

    For Hetero: μ∇²u + ∇μ·∇u + ρω²u = 0
        Linearize: μ_prev * ∇²u + ∇μ_prev · ∇u + ρω²u = 0
        This becomes linear in u: [μ_prev * ∇²Φ_u + ∇μ_prev · ∇Φ_u + ρω² * Φ_u] @ W_u = 0

    Args:
        pde: MRE PDE equation object (HelmholtzEquation or HeteroEquation)
        x_pde: (N_pde, 3) PDE collocation points
        u_features: RandomFeatures object for wave field
        mu_features: RandomFeatures object for elasticity
        u_prev: (N_pde, 3) Previous iteration u values (for linearization)
        mu_prev: (N_pde, 1) Previous iteration μ values (for linearization)
        omega: Angular frequency
        rho: Material density
        device: Device

    Returns:
        A_pde_u: (N_pde * 3, n_features + 1) PDE constraints for u (with bias)
        b_pde_u: (N_pde * 3, 6) Target vector (zeros)
    """
    # Require previous values for linearization
    if mu_prev is None:
        raise ValueError(
            "PDE constraints require μ values for linearization. "
            "For first iteration, use data-fitting only (use_pde=False), "
            "then enable PDE constraints in subsequent refinement iterations."
        )

    # Compute features and derivatives at PDE points
    x_pde.requires_grad = True

    # Compute u features and derivatives
    phi_u, grad_phi_u, lap_phi_u = u_features(x_pde, compute_derivatives=True)
    # phi_u: (N_pde, n_features)
    # grad_phi_u: (N_pde, n_features, 3)
    # lap_phi_u: (N_pde, n_features)

    N_pde = x_pde.shape[0]
    n_features = phi_u.shape[1]

    # Get omega from PDE if not provided
    if omega is None:
        omega = pde.omega
    omega = torch.tensor(omega, device=device) if not isinstance(omega, torch.Tensor) else omega

    # Handle complex output (u has 3 components, real and imag)
    # For complex u = u_real + i*u_imag, we need to enforce PDE on both parts
    # Output dimension: 6 (3 components * 2 for real/imag)

    if isinstance(pde, HelmholtzEquation):
        # Helmholtz: μ∇²u + ρω²u = 0
        # Linearized: μ_prev * ∇²u + ρω²u = 0

        # Ensure μ_prev is on correct device and has correct shape
        mu_prev = mu_prev.to(device)  # (N_pde, 1)

        # For each component (ux, uy, uz), build constraint
        A_rows = []
        for comp in range(3):  # 3 spatial components
            # Coefficient for Laplacian term: μ_prev
            coef_lap = mu_prev.squeeze(-1)  # (N_pde,)

            # Coefficient for reaction term: ρω²
            coef_react = rho * (omega ** 2)

            # Constraint row: μ_prev * ∇²φ + ρω² * φ
            # lap_phi_u: (N_pde, n_features)
            # phi_u: (N_pde, n_features)
            A_comp = coef_lap.unsqueeze(-1) * lap_phi_u + coef_react * phi_u  # (N_pde, n_features)

            A_rows.append(A_comp)

        # Stack all components
        A_pde_u = torch.cat(A_rows, dim=0)  # (N_pde * 3, n_features)

        # Add bias column
        ones = torch.ones((A_pde_u.shape[0], 1), device=device)
        A_pde_u = torch.cat([A_pde_u, ones], dim=1)  # (N_pde * 3, n_features + 1)

        # Target: zeros (PDE residual should be zero)
        # Need to match output dimension: 6 (real and imag for 3 components)
        b_pde_u = torch.zeros((N_pde * 3, 6), device=device)

        return A_pde_u, b_pde_u

    elif isinstance(pde, HeteroEquation):
        # Hetero: μ∇²u + ∇μ·∇u + ρω²u = 0
        # Linearized: μ_prev * ∇²u + ∇μ_prev · ∇u + ρω²u = 0

        # Compute μ features and derivatives
        phi_mu, grad_phi_mu, _ = mu_features(x_pde, compute_derivatives=True)
        # grad_phi_mu: (N_pde, n_features_mu, 3)

        # Predict ∇μ using current mu_features weights (if available)
        # For now, use provided ∇μ_prev or compute from mu_prev
        # This is complex - need to handle properly

        raise NotImplementedError(
            "Hetero equation linearization requires careful handling of ∇μ. "
            "Implementation requires access to mu_weights to compute ∇μ = ∇Φ_μ @ W_μ. "
            "This creates dependency: need mu_weights to build u constraints. "
            "Requires alternating solve or joint system. See PHASE3.md."
        )

    else:
        raise ValueError(f"Unsupported PDE type: {type(pde)}")
