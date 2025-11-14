"""
Phase 4: Two-Stage Inverse Problem Solver

This module implements physics-based inverse elastography that reconstructs
the elasticity field μ(x) from measured wave displacement u(x) WITHOUT
requiring ground truth μ values.

Two-stage approach:
1. Stage 1: Fit random features to wave data → smooth wave reconstruction
2. Stage 2: Solve for elasticity using PDE physics constraint

Key: The PDE provides supervision for Stage 2, not ground truth μ!
"""

import torch
import numpy as np
from mre_pinn.utils import as_complex
from .solver import solve_linear_system


def compute_wave_derivatives(net, x_pde, device='cpu'):
    """
    Compute wave field u and its derivatives at PDE collocation points.

    Uses Stage 1 weights W_u to compute:
    - u(x) = Φ_u(x)^T W_u + b_u
    - ∇u(x) = [∇Φ_u(x)]^T W_u
    - ∇²u(x) = [∇²Φ_u(x)]^T W_u

    All derivatives are computed using PyTorch autograd.

    Args:
        net: MREPIELM model with trained u_weights from Stage 1
        x_pde: (N_pde, 3) collocation points for PDE
        device: 'cpu' or 'cuda'

    Returns:
        u: (N_pde, n_components) complex wave displacement
        grad_u: (N_pde, n_components, 3) complex gradient ∇u
        laplace_u: (N_pde, n_components) complex Laplacian ∇²u
    """
    if net.u_weights is None:
        raise ValueError(
            "Stage 1 not complete. Call solve(inverse_mode=False) first "
            "to fit wave field before computing derivatives."
        )

    # Ensure x_pde requires gradients for autograd
    x_pde = x_pde.clone().detach().requires_grad_(True).to(device)

    # Normalize inputs
    x_norm = net.normalize_input(x_pde)

    # Compute features and derivatives using autograd
    # This gives us Φ_u, ∇Φ_u, ∇²Φ_u at PDE points
    phi_u, grad_phi_u, laplace_phi_u = net.u_features(
        x_norm,
        compute_derivatives=True
    )
    # phi_u: (N_pde, n_features)
    # grad_phi_u: (N_pde, n_features, 3)
    # laplace_phi_u: (N_pde, n_features)

    # Compute u = Φ_u^T W_u + b_u
    u_pred = phi_u @ net.u_weights + net.u_bias  # (N_pde, 6)

    # Convert to complex (N_pde, 3)
    u_complex = as_complex(u_pred, polar=False)

    # Compute ∇u = [∇Φ_u]^T W_u
    # grad_phi_u is (N_pde, n_features, 3)
    # u_weights is (n_features, 6) - need to handle real/imag separately

    n_pde = x_pde.shape[0]
    n_features = phi_u.shape[1]

    # Split weights into real/imag components (each is n_features x 3)
    W_u_real = net.u_weights[:, :3]  # (n_features, 3)
    W_u_imag = net.u_weights[:, 3:]  # (n_features, 3)

    # Compute gradient for each component
    grad_u_real = torch.zeros(n_pde, 3, 3, device=device)  # (N, component, spatial_dim)
    grad_u_imag = torch.zeros(n_pde, 3, 3, device=device)

    for c in range(3):  # For each component (x, y, z)
        # grad_u_c = ∇Φ_u^T @ W_u[:, c]
        # grad_phi_u: (N_pde, n_features, 3)
        # Need: (N_pde, 3) = sum over features
        grad_u_real[:, c, :] = (grad_phi_u * W_u_real[:, c].unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        grad_u_imag[:, c, :] = (grad_phi_u * W_u_imag[:, c].unsqueeze(0).unsqueeze(-1)).sum(dim=1)

    grad_u_complex = grad_u_real + 1j * grad_u_imag  # (N_pde, 3, 3)

    # Compute ∇²u = [∇²Φ_u]^T W_u
    laplace_u_real = laplace_phi_u @ W_u_real  # (N_pde, 3)
    laplace_u_imag = laplace_phi_u @ W_u_imag  # (N_pde, 3)
    laplace_u_complex = laplace_u_real + 1j * laplace_u_imag

    return u_complex, grad_u_complex, laplace_u_complex


def solve_inverse_helmholtz(net, pde, x_pde, regularization=1e-6, device='cpu'):
    """
    Stage 2: Solve for elasticity μ using Helmholtz PDE constraint.

    PDE: μ(x)∇²u(x) + ρω²u(x) = 0

    Represent: μ(x) = Φ_μ(x)^T W_μ + b_μ

    Substitute into PDE:
        [Φ_μ(x)^T W_μ + b_μ] · ∇²u(x) + ρω²u(x) = 0

    Rearrange:
        Φ_μ(x)^T W_μ · ∇²u(x) = -ρω²u(x) - b_μ · ∇²u(x)

    For complex 3-component wave field, this gives 3*N_pde equations:
        [Φ_μ(x_j)^T · ∇²u_k(x_j)] W_μ = -ρω²u_k(x_j) - b_μ · ∇²u_k(x_j)

    This is a LINEAR system in W_μ!

    Args:
        net: MREPIELM model with trained u_weights from Stage 1
        pde: HelmholtzEquation with omega and rho
        x_pde: (N_pde, 3) collocation points
        regularization: Ridge regularization parameter
        device: 'cpu' or 'cuda'

    Returns:
        W_μ: (n_features_mu, 2) weights for elasticity (magnitude, phase)
        b_μ: (2,) bias for elasticity
    """
    print("  [Stage 2] Solving inverse Helmholtz equation...")

    # Step 1: Compute wave derivatives from Stage 1
    u, grad_u, laplace_u = compute_wave_derivatives(net, x_pde, device)
    # u: (N_pde, 3) complex
    # laplace_u: (N_pde, 3) complex

    n_pde = x_pde.shape[0]

    # Step 2: Compute μ features at PDE points
    x_norm = net.normalize_input(x_pde)
    phi_mu = net.mu_features(x_norm, compute_derivatives=False)
    # phi_mu: (N_pde, n_features_mu)

    # Step 3: Build linear system for each component
    # For each point j and component k:
    #   [Φ_μ(x_j)^T · ∇²u_k(x_j)] W_μ = -ρω²u_k(x_j) - b_μ · ∇²u_k(x_j)

    # Stack all equations: 3*N_pde rows
    n_eq = 3 * n_pde
    n_features = phi_mu.shape[1]

    # For complex fields, we need to split into real and imaginary parts
    # Real equations: Re([Φ_μ^T · ∇²u] W_μ) = Re(-ρω²u)
    # Imag equations: Im([Φ_μ^T · ∇²u] W_μ) = Im(-ρω²u)

    omega = pde.omega
    rho = pde.rho

    # Build system row by row (can be optimized later)
    A_rows = []
    b_rows = []

    for j in range(n_pde):
        for k in range(3):  # x, y, z components
            # LHS: Φ_μ(x_j)^T · ∇²u_k(x_j)
            # Φ_μ is real, ∇²u_k is complex
            phi_j = phi_mu[j, :]  # (n_features,)
            lap_uk_j = laplace_u[j, k]  # complex scalar

            # Row of A: phi_j * lap_uk_j (broadcast)
            # This is complex, so we need 2 rows (real and imag)
            a_row_complex = phi_j * lap_uk_j

            # RHS: -ρω²u_k(x_j)
            u_k_j = u[j, k]  # complex scalar
            b_val_complex = -rho * omega**2 * u_k_j

            # Split into real and imaginary equations
            A_rows.append(a_row_complex.real)
            b_rows.append(b_val_complex.real)

            A_rows.append(a_row_complex.imag)
            b_rows.append(b_val_complex.imag)

    # Stack into matrices
    A_mu = torch.stack(A_rows, dim=0)  # (6*N_pde, n_features)
    b_mu = torch.stack(b_rows, dim=0)  # (6*N_pde,)

    # Add bias column
    ones_mu = torch.ones((A_mu.shape[0], 1), device=device)
    A_mu_with_bias = torch.cat([A_mu, ones_mu], dim=1)

    # Step 4: Solve ridge regression
    print(f"    System size: A={A_mu_with_bias.shape}, b={b_mu.shape}")
    weights_with_bias = solve_linear_system(
        A_mu_with_bias, b_mu.unsqueeze(-1),
        regularization=regularization,
        method='ridge'
    )

    # Split weights and bias
    W_mu = weights_with_bias[:-1]  # (n_features, 1)
    b_mu_val = weights_with_bias[-1:]  # (1,)

    # Store in network
    # Note: For Helmholtz, μ is real and positive
    # We need to match output format: (magnitude, phase)
    # Since μ is real, phase = 0
    # Output is (n_features, 2) where col 0 is magnitude, col 1 is phase

    # Duplicate W_mu to create 2-column output
    W_mu_out = torch.cat([W_mu, torch.zeros_like(W_mu)], dim=1)  # (n_features, 2)
    b_mu_out = torch.cat([b_mu_val, torch.zeros_like(b_mu_val)], dim=1).squeeze(0)  # (2,)

    net.mu_weights = W_mu_out
    net.mu_bias = b_mu_out

    print(f"    [OK] Elasticity field inverted (Helmholtz)")

    return W_mu_out, b_mu_out


def solve_inverse_hetero_iterative(net, pde, x_pde, max_iter=10, tol=1e-4,
                                   regularization=1e-6, device='cpu'):
    """
    Stage 2: Solve for elasticity μ using heterogeneous PDE with iteration.

    PDE: μ∇²u + ∇μ·∇u + ρω²u = 0

    The ∇μ·∇u term creates coupling, so we use iterative linearization:

    Iteration k:
        1. Compute ∇μ_k using W_μ^(k-1) from previous iteration
        2. Linearize: μ∇²u = -∇μ_k·∇u - ρω²u
        3. Solve: Φ_μ^T W_μ^(k) · ∇²u = -∇μ_k·∇u - ρω²u
        4. Repeat until ||W_μ^(k) - W_μ^(k-1)|| < tol

    Args:
        net: MREPIELM model with trained u_weights
        pde: HeteroEquation or similar with omega, rho
        x_pde: (N_pde, 3) collocation points
        max_iter: Maximum iterations for convergence
        tol: Convergence tolerance on weight change
        regularization: Ridge regularization
        device: 'cpu' or 'cuda'

    Returns:
        W_μ: (n_features_mu, 2) final weights
        b_μ: (2,) final bias
    """
    print("  [Stage 2] Solving inverse heterogeneous equation (iterative)...")

    # Step 1: Compute wave derivatives from Stage 1
    u, grad_u, laplace_u = compute_wave_derivatives(net, x_pde, device)
    # u: (N_pde, 3) complex
    # grad_u: (N_pde, 3, 3) complex
    # laplace_u: (N_pde, 3) complex

    n_pde = x_pde.shape[0]
    omega = pde.omega
    rho = pde.rho

    # Step 2: Initialize μ with Helmholtz solution (ignoring ∇μ term)
    print(f"    Iteration 0: Helmholtz initialization")
    W_mu_prev, b_mu_prev = solve_inverse_helmholtz(net, pde, x_pde, regularization, device)

    # Step 3: Iterative refinement
    for iter_idx in range(max_iter):
        print(f"    Iteration {iter_idx + 1}/{max_iter}")

        # Compute ∇μ using current weights
        x_norm = net.normalize_input(x_pde)
        phi_mu, grad_phi_mu, _ = net.mu_features(x_norm, compute_derivatives=True)
        # phi_mu: (N_pde, n_features)
        # grad_phi_mu: (N_pde, n_features, 3)

        # μ = Φ_μ^T W_μ + b_μ
        mu_current = phi_mu @ net.mu_weights + net.mu_bias  # (N_pde, 1)

        # ∇μ = [∇Φ_μ]^T W_μ
        # grad_phi_mu: (N_pde, n_features, 3)
        # W_μ: (n_features, 1)
        grad_mu = torch.zeros(n_pde, 3, device=device)
        for d in range(3):
            grad_mu[:, d] = (grad_phi_mu[:, :, d] @ net.mu_weights).squeeze()
        # grad_mu: (N_pde, 3) real

        # Build linear system with ∇μ·∇u term
        # For each point j and component k:
        #   Φ_μ(x_j)^T W_μ · ∇²u_k(x_j) = -∇μ_j·∇u_k(x_j) - ρω²u_k(x_j)

        A_rows = []
        b_rows = []

        for j in range(n_pde):
            for k in range(3):  # x, y, z components
                # LHS: Φ_μ(x_j)^T · ∇²u_k(x_j)
                phi_j = phi_mu[j, :]
                lap_uk_j = laplace_u[j, k]
                a_row_complex = phi_j * lap_uk_j

                # RHS: -∇μ_j·∇u_k(x_j) - ρω²u_k(x_j)
                # ∇μ is real, ∇u_k is complex
                grad_mu_j = grad_mu[j, :]  # (3,) real
                grad_uk_j = grad_u[j, k, :]  # (3,) complex

                grad_term = -(grad_mu_j * grad_uk_j).sum()  # complex scalar
                inertia_term = -rho * omega**2 * u[j, k]
                b_val_complex = grad_term + inertia_term

                # Split into real and imaginary
                A_rows.append(a_row_complex.real)
                b_rows.append(b_val_complex.real)

                A_rows.append(a_row_complex.imag)
                b_rows.append(b_val_complex.imag)

        # Stack into matrices
        A_mu = torch.stack(A_rows, dim=0)
        b_mu = torch.stack(b_rows, dim=0)

        # Add bias
        ones_mu = torch.ones((A_mu.shape[0], 1), device=device)
        A_mu_with_bias = torch.cat([A_mu, ones_mu], dim=1)

        # Solve
        weights_with_bias = solve_linear_system(
            A_mu_with_bias, b_mu.unsqueeze(-1),
            regularization=regularization,
            method='ridge'
        )

        W_mu_new = weights_with_bias[:-1]
        b_mu_new = weights_with_bias[-1:]

        # Check convergence
        weight_change = torch.norm(W_mu_new - W_mu_prev).item()
        print(f"      Weight change: {weight_change:.6e}")

        if weight_change < tol:
            print(f"      [OK] Converged in {iter_idx + 1} iterations")
            net.mu_weights = W_mu_new
            net.mu_bias = b_mu_new
            return W_mu_new, b_mu_new

        # Update for next iteration
        W_mu_prev = W_mu_new
        b_mu_prev = b_mu_new
        net.mu_weights = W_mu_new
        net.mu_bias = b_mu_new

    print(f"      [WARN] Max iterations reached without full convergence")
    return W_mu_new, b_mu_new
