"""
MREPIELM model class - analog of MREPINN.

Similar architecture to MREPINN but uses random features instead of
trainable neural networks.

Key differences:
- Random features (not trained) for u and μ
- Weights solved via linear system (not gradient descent)
- Same normalization and interface as MREPINN
"""

import numpy as np
import torch

from mre_pinn.utils import as_complex, concat
from .features import RandomFeatures


class MREPIELM(torch.nn.Module):
    """
    PIELM model for MRE - dual random feature model.

    Architecture:
    - u_features: Random features for wave displacement field (3 complex components)
    - mu_features: Random features for elasticity field (1 complex value)

    Follows the same structure as MREPINN:
    - Same input normalization (input_loc, input_scale)
    - Same output normalization (u_loc, u_scale, mu_loc, mu_scale)
    - Same forward interface
    """

    def __init__(self, example, omega, n_features=1000,
                 frequency_scale=1.0, use_sin_cos=True, seed=None, device='cpu'):
        """
        Initialize MREPIELM.

        Args:
            example: MREExample object (same as MREPINN)
            omega: Angular frequency (rad/s)
            n_features: Number of random features (analog of n_hidden in PINN)
            frequency_scale: Scale of random frequencies
            use_sin_cos: If True, use [cos, sin] features (2x features)
            seed: Random seed for reproducibility
            device: 'cpu' or 'cuda'
        """
        super().__init__()

        self.device = device

        # Same normalization as MREPINN (from mre_pinn/model/pinn.py:13-36)
        metadata = example.metadata
        center_vals = np.array([v.values for v in metadata['center'].wave.values])
        extent_vals = np.array([v.values for v in metadata['extent'].wave.values])

        x_center = torch.as_tensor(center_vals, dtype=torch.float32, device=device)
        x_extent = torch.as_tensor(extent_vals, dtype=torch.float32, device=device)

        stats = example.describe()
        self.u_loc = torch.tensor(stats['mean'].loc['wave'], device=device)
        self.u_scale = torch.tensor(stats['std'].loc['wave'], device=device)
        self.mu_loc = torch.tensor(stats['mean'].loc['mre'], device=device)
        self.mu_scale = torch.tensor(stats['std'].loc['mre'], device=device)
        self.omega = torch.tensor(omega, device=device)

        # Anatomy field (optional)
        if 'anat' in example:
            self.a_loc = torch.tensor(stats['mean'].loc['anat'], device=device)
            self.a_scale = torch.tensor(stats['std'].loc['anat'], device=device)
        else:
            self.a_loc = torch.zeros(0, device=device)
            self.a_scale = torch.zeros(0, device=device)

        self.input_loc = x_center
        self.input_scale = x_extent

        # Store data properties
        self.is_complex_u = example.wave.field.is_complex
        self.is_complex_mu = example.mre.field.is_complex
        self.n_input = len(self.input_loc)  # 3 for x,y,z
        self.n_output_u = len(self.u_loc)   # 6 for complex (3 components * 2)
        self.n_output_mu = len(self.mu_loc) + len(self.a_loc)  # 2 for complex + anatomy

        # Random features (not trainable)
        self.u_features = RandomFeatures(
            n_input=self.n_input,
            n_features=n_features,
            frequency_scale=frequency_scale,
            use_sin_cos=use_sin_cos,
            seed=seed if seed is None else seed,
            device=device
        )

        self.mu_features = RandomFeatures(
            n_input=self.n_input,
            n_features=n_features,
            frequency_scale=frequency_scale,
            use_sin_cos=use_sin_cos,
            seed=seed if seed is None else seed + 1,  # Different seed for mu
            device=device
        )

        # Output weights (to be solved by training module)
        self.u_weights = None   # (n_effective, n_output_u)
        self.mu_weights = None  # (n_effective, n_output_mu)
        self.u_bias = None      # (1, n_output_u) - bias term like AutoDES
        self.mu_bias = None     # (1, n_output_mu) - bias term like AutoDES

    def normalize_input(self, x):
        """
        Normalize input coordinates (same as MREPINN).

        Args:
            x: (N, 3) spatial coordinates

        Returns:
            x_norm: (N, 3) normalized coordinates
        """
        x = (x - self.input_loc) / self.input_scale
        x = x * self.omega
        return x

    def forward(self, inputs, return_real=False):
        """
        Forward pass (after weights are solved).

        Same interface as MREPINN.forward()

        Args:
            inputs: Tuple of (x,) where x is (N, 3) spatial coordinates
            return_real: If True, return real representation (for compatibility)

        Returns:
            u_pred: (N, 3) complex displacement (or (N, 6) if return_real)
            mu_pred: (N, 1) complex elasticity (or (N, 2) if return_real)
            a_pred: (N, ...) anatomy (if present)
        """
        if self.u_weights is None or self.mu_weights is None:
            raise RuntimeError("Model weights not solved yet. Call MREPIELMModel.solve() first.")

        x, = inputs

        # Ensure correct device
        if x.device != self.device:
            x = x.to(self.device)

        # Normalize input (same as MREPINN)
        x = self.normalize_input(x)

        # Compute features (without derivatives for inference)
        phi_u = self.u_features(x, compute_derivatives=False)    # (N, n_effective)
        phi_mu = self.mu_features(x, compute_derivatives=False)  # (N, n_effective)

        # Linear prediction with bias (like AutoDES: phi @ c_feat + c_bias)
        u_pred = phi_u @ self.u_weights + self.u_bias      # (N, n_output_u)
        mu_a_pred = phi_mu @ self.mu_weights + self.mu_bias # (N, n_output_mu)

        # Denormalize (same as MREPINN:62-70)
        u_pred = u_pred * self.u_scale + self.u_loc

        mu_pred = mu_a_pred[:, :len(self.mu_loc)]
        a_pred = mu_a_pred[:, len(self.mu_loc):]

        mu_pred = mu_pred * self.mu_scale + self.mu_loc
        a_pred = a_pred * self.a_scale + self.a_loc

        # Convert to complex if needed (unless return_real=True)
        if not return_real:
            if self.is_complex_u:
                u_pred = as_complex(u_pred, polar=False)  # (N, 3) complex
            if self.is_complex_mu:
                mu_pred = as_complex(mu_pred, polar=True)  # (N, 1) complex

        return u_pred, mu_pred, a_pred

    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.device = device
        self.u_features.to(device)
        self.mu_features.to(device)

        # Move normalization parameters
        self.input_loc = self.input_loc.to(device)
        self.input_scale = self.input_scale.to(device)
        self.u_loc = self.u_loc.to(device)
        self.u_scale = self.u_scale.to(device)
        self.mu_loc = self.mu_loc.to(device)
        self.mu_scale = self.mu_scale.to(device)
        self.omega = self.omega.to(device)

        if self.a_loc.numel() > 0:
            self.a_loc = self.a_loc.to(device)
            self.a_scale = self.a_scale.to(device)

        if self.u_weights is not None:
            self.u_weights = self.u_weights.to(device)
        if self.mu_weights is not None:
            self.mu_weights = self.mu_weights.to(device)

        return self
