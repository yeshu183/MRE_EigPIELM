"""
Random feature generation for PIELM.

Uses random Fourier features with PyTorch autograd for computing derivatives.
This is the core difference from PINN - instead of trainable neural networks,
we use random projections with analytical (autograd) derivatives.
"""

import torch
import numpy as np


class RandomFeatures:
    """
    Random Fourier Features for PIELM.

    Features: φ(x) = cos(Wx + b) or [cos(Wx + b), sin(Wx + b)]

    Uses PyTorch autograd to compute:
    - φ(x): Random features
    - ∇φ(x): Gradient (first derivatives)
    - ∇²φ(x): Laplacian (second derivatives)

    This is similar to how PINN computes derivatives of neural networks,
    but applied to random features instead.
    """

    def __init__(self, n_input, n_features, frequency_scale=1.0,
                 use_sin_cos=True, seed=None, device='cpu'):
        """
        Initialize random features.

        Args:
            n_input: Input dimension (3 for spatial coordinates x,y,z)
            n_features: Number of random features (hidden neurons)
            frequency_scale: Scale of random frequencies (σ in exp(-σ||x||²))
            use_sin_cos: If True, use [cos, sin]; if False, use only cos
            seed: Random seed for reproducibility
            device: 'cpu' or 'cuda'
        """
        self.n_input = n_input
        self.n_features = n_features
        self.frequency_scale = frequency_scale
        self.use_sin_cos = use_sin_cos
        self.device = device

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Random projection matrix W ~ N(0, frequency_scale²)
        self.W = torch.randn(n_input, n_features, device=device) * frequency_scale

        # Random bias b ~ Uniform(0, 2π)
        self.b = torch.rand(n_features, device=device) * 2 * np.pi

        # Effective feature count
        self.n_effective = n_features * 2 if use_sin_cos else n_features

    def __call__(self, x, compute_derivatives=False):
        """
        Compute random features and optionally their derivatives.

        Args:
            x: (N, n_input) input coordinates
                Must have requires_grad=True if compute_derivatives=True
            compute_derivatives: If True, compute ∇φ and ∇²φ using autograd

        Returns:
            If compute_derivatives=False:
                features: (N, n_effective) random features

            If compute_derivatives=True:
                features: (N, n_effective)
                grad_features: (N, n_effective, n_input) - ∂φ/∂x
                laplace_features: (N, n_effective) - ∇²φ
        """
        # Ensure x is on correct device
        if x.device != self.device:
            x = x.to(self.device)

        # Linear transformation: z = Wx + b
        z = x @ self.W + self.b  # (N, n_features)

        # Apply activation
        if self.use_sin_cos:
            features = torch.cat([torch.cos(z), torch.sin(z)], dim=-1)  # (N, 2*n_features)
        else:
            features = torch.cos(z)  # (N, n_features)

        if not compute_derivatives:
            return features

        # Compute derivatives using autograd
        N = x.shape[0]
        n_eff = self.n_effective
        n_in = self.n_input

        # Initialize storage
        grad_features = torch.zeros(N, n_eff, n_in, device=self.device)
        laplace_features = torch.zeros(N, n_eff, device=self.device)

        # Compute derivatives for each feature
        for i in range(n_eff):
            # First derivative: ∂φᵢ/∂x
            grad_i = torch.autograd.grad(
                outputs=features[:, i].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]  # (N, n_input)

            grad_features[:, i, :] = grad_i

            # Second derivative (Laplacian): ∇²φᵢ = Σⱼ ∂²φᵢ/∂xⱼ²
            for j in range(n_in):
                grad2_ij = torch.autograd.grad(
                    outputs=grad_i[:, j].sum(),
                    inputs=x,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]  # (N, n_input)

                laplace_features[:, i] += grad2_ij[:, j]

        return features, grad_features, laplace_features

    def to(self, device):
        """Move features to device."""
        self.device = device
        self.W = self.W.to(device)
        self.b = self.b.to(device)
        return self


def compute_jacobian_features(features, x):
    """
    Compute Jacobian ∂φ/∂x for random features.

    This is a helper function that computes the full Jacobian matrix
    using autograd, similar to mre_pinn/fields.py:jacobian()

    Args:
        features: (N, n_features) feature values
        x: (N, n_input) input coordinates (must have requires_grad=True)

    Returns:
        jac: (N, n_features, n_input) Jacobian matrix
    """
    N, n_features = features.shape
    n_input = x.shape[1]

    jac = torch.zeros(N, n_features, n_input, device=features.device)

    for i in range(n_features):
        grad = torch.autograd.grad(
            outputs=features[:, i].sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        jac[:, i, :] = grad

    return jac


def compute_laplacian_features(features, x):
    """
    Compute Laplacian ∇²φ for random features.

    Similar to mre_pinn/fields.py:laplacian()

    Args:
        features: (N, n_features) feature values
        x: (N, n_input) input coordinates (must have requires_grad=True)

    Returns:
        lap: (N, n_features) Laplacian values
    """
    N, n_features = features.shape
    n_input = x.shape[1]

    lap = torch.zeros(N, n_features, device=features.device)

    for i in range(n_features):
        # First derivative
        grad = torch.autograd.grad(
            outputs=features[:, i].sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # (N, n_input)

        # Second derivative (trace of Hessian)
        for j in range(n_input):
            grad2 = torch.autograd.grad(
                outputs=grad[:, j].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]  # (N, n_input)

            lap[:, i] += grad2[:, j]

    return lap
