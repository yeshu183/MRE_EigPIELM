"""
PIELM training (solving) module - analog of MREPINNModel.

Key differences from PINN:
- No iterative training - solve linear system once
- Still needs to assemble system with data and PDE constraints
- Compatible interface with PINN for easy swapping
"""

import time
from functools import cache
import numpy as np
import xarray as xr
import torch

from mre_pinn.utils import minibatch, as_xarray
from mre_pinn.pde import laplacian
from mre_pinn.training.losses import msae_loss
from .solver import solve_linear_system
from .equations import construct_pde_matrix_coupled


class MREPIELMModel:
    """
    PIELM solver - analog of MREPINNModel.

    Key differences:
    - solve() instead of train() - one-shot linear system solve
    - Still has test() method for compatibility with TestEvaluator
    - No iterative training, no gradients
    """

    def __init__(
        self,
        example,
        net,
        pde,
        loss_weights=[1, 0, 0, 1e-8],
        n_points=4096,
        n_pde_points=None,
        regularization=1e-6,
        solver_method='ridge',
        device='cpu'
    ):
        """
        Initialize PIELM solver.

        Args:
            example: MREExample object (same as PINN)
            net: MREPIELM model
            pde: PDE equation (same as PINN)
            loss_weights: [u_weight, mu_weight, a_weight, pde_weight]
            n_points: Number of data points to sample
            n_pde_points: Number of PDE collocation points (default: same as n_points)
            regularization: Ridge regularization parameter λ
            solver_method: 'ridge', 'lstsq', or 'pinv'
            device: 'cpu' or 'cuda'
        """
        self.example = example
        self.net = net
        self.pde = pde

        self.anatomical = ('anat' in example)
        if example.wave.field.has_components:
            self.wave_dims = example.wave.field.n_components
        else:
            self.wave_dims = 1

        self.loss_weights = loss_weights
        self.n_points = n_points
        self.n_pde_points = n_pde_points if n_pde_points is not None else n_points
        self.regularization = regularization
        self.solver_method = solver_method
        self.device = device

        # Move model to device
        self.net.to(device)

        # Prepare data (load and sample once)
        self._prepare_data()

    @cache
    def get_raw_tensors(self, device):
        """
        Get raw tensors from example (same as MREPINNData.get_raw_tensors).

        Cached to avoid reloading data.
        """
        example = self.example

        # Get numpy arrays from data example
        x = example.wave.field.points()
        u = example.wave.field.values()
        mu = example.mre.field.values()
        mu_mask = example.mre_mask.values.reshape(-1)

        # Convert to tensors
        x = torch.tensor(x, device=device, dtype=torch.float32)
        u = torch.tensor(u, device=device)
        mu = torch.tensor(mu, device=device)
        mu_mask = torch.tensor(mu_mask, device=device, dtype=torch.bool)

        if self.anatomical:
            a = example.anat.field.values()
            a = torch.tensor(a, device=device, dtype=torch.float32)
        else:
            a = u[:, :0]  # Empty tensor

        return x, u, mu, mu_mask, a

    def _prepare_data(self):
        """
        Load and sample data for training.

        Similar to MREPINNData.get_tensors but samples once and stores.
        """
        print("Loading data...")
        x, u, mu, mu_mask, a = self.get_raw_tensors(self.device)

        # Apply mask
        x_masked = x[mu_mask]
        u_masked = u[mu_mask]
        mu_masked = mu[mu_mask]
        a_masked = a[mu_mask] if self.anatomical else a

        # Sample for data fitting
        n_available = x_masked.shape[0]
        n_data = min(self.n_points, n_available)
        data_sample = torch.randperm(n_available)[:n_data]

        self.x_data = x_masked[data_sample]
        self.u_data = u_masked[data_sample]
        self.mu_data = mu_masked[data_sample]
        self.a_data = a_masked[data_sample] if self.anatomical else a_masked

        # Sample for PDE collocation (can be different points)
        n_pde = min(self.n_pde_points, n_available)
        pde_sample = torch.randperm(n_available)[:n_pde]

        self.x_pde = x_masked[pde_sample]

        # Store full data for testing
        self.x_full = x
        self.u_full = u
        self.mu_full = mu
        self.a_full = a

        print(f"Data loaded: {n_data} data points, {n_pde} PDE points")

    def solve(self, use_pde=False):
        """
        Solve PIELM linear system.

        Args:
            use_pde: If True, include PDE constraints. If False, only fit data.

        This is the main method (analog of MREPINNModel.train()).

        Constructs and solves:
            [√w_data * Φ_data]     [√w_data * u_data]
            [√w_pde  * Φ_PDE ] W = [√w_pde  * 0     ]
        """
        print("="*60)
        print("PIELM Solver")
        print("="*60)

        u_weight, mu_weight, a_weight, pde_weight = self.loss_weights

        # Phase 1: Solve for u (wave field)
        print("\n[1/2] Solving for wave field (u)...")
        t_start = time.time()

        # Compute features at data points
        phi_u_data = self.net.u_features(
            self.net.normalize_input(self.x_data),
            compute_derivatives=False
        )  # (n_data, n_features)

        # Build data constraint: Φ_u W_u = u_data
        A_u = phi_u_data * np.sqrt(u_weight)
        b_u = self.u_data * np.sqrt(u_weight)

        # Add bias column (like AutoDES PIELM_solver_v2.ipynb Cell 5)
        ones_u = torch.ones((A_u.shape[0], 1), device=self.device)
        A_u_with_bias = torch.cat([A_u, ones_u], dim=1)

        if use_pde and pde_weight > 0:
            print("  Including PDE constraints...")

            # Compute μ values at PDE points for linearization
            # Use current mu_weights if available, otherwise use data
            if self.net.mu_weights is not None:
                # Use solved μ from previous iteration
                with torch.no_grad():
                    phi_mu_pde = self.net.mu_features(
                        self.net.normalize_input(self.x_pde),
                        compute_derivatives=False
                    )
                    mu_pde = phi_mu_pde @ self.net.mu_weights + self.net.mu_bias
                    # Convert to complex if needed
                    from mre_pinn.utils import as_complex
                    mu_pde_complex = as_complex(mu_pde, polar=True)
                    mu_pde_real = mu_pde_complex.real.abs()  # Use magnitude for linearization
            else:
                print("    Warning: No μ weights available for PDE constraints.")
                print("    Using data-only solve first. Call solve() again with use_pde=True after initial solve.")
                raise RuntimeError(
                    "PDE constraints require μ values for linearization. "
                    "First call solve(use_pde=False) to fit data, then call solve(use_pde=True)."
                )

            # Construct PDE constraint matrix
            try:
                A_pde_u, b_pde_u = construct_pde_matrix_coupled(
                    self.pde,
                    self.x_pde,
                    self.net.u_features,
                    self.net.mu_features,
                    u_prev=None,  # Not needed for current implementation
                    mu_prev=mu_pde_real,
                    omega=self.pde.omega,
                    rho=self.pde.rho,
                    device=self.device
                )

                # Weight PDE constraints
                A_pde_u = A_pde_u * np.sqrt(pde_weight)
                b_pde_u = b_pde_u * np.sqrt(pde_weight)

                # Combine data and PDE constraints
                A_u_with_bias = torch.cat([A_u_with_bias, A_pde_u], dim=0)
                b_u = torch.cat([b_u, b_pde_u], dim=0)

                print(f"    Added {A_pde_u.shape[0]} PDE constraint rows")

            except NotImplementedError as e:
                print(f"    PDE constraints not fully implemented: {e}")
                print(f"    Continuing with data-only solve...")
                pass

        # Solve
        print(f"  Solving system: A={A_u_with_bias.shape}, b={b_u.shape}")
        weights_with_bias = solve_linear_system(
            A_u_with_bias, b_u,
            regularization=self.regularization,
            method=self.solver_method
        )

        # Split weights and bias
        self.net.u_weights = weights_with_bias[:-1]  # All except last
        self.net.u_bias = weights_with_bias[-1:]     # Last row (bias)

        t_u = time.time() - t_start
        print(f"  [OK] Wave field solved in {t_u:.2f}s")

        # Phase 2: Solve for mu (elasticity field)
        print("\n[2/2] Solving for elasticity field (mu)...")
        t_start = time.time()

        # Compute features at data points
        phi_mu_data = self.net.mu_features(
            self.net.normalize_input(self.x_data),
            compute_derivatives=False
        )  # (n_data, n_features)

        # Build data constraint: Φ_μ W_μ = μ_data (+ anatomy if present)
        mu_a_data = torch.cat([self.mu_data, self.a_data], dim=-1) if self.anatomical else self.mu_data
        A_mu = phi_mu_data * np.sqrt(mu_weight)
        b_mu = mu_a_data * np.sqrt(mu_weight)

        if use_pde and pde_weight > 0:
            print("  Including PDE constraints...")
            # TODO: Phase 3 - add PDE constraints
            raise NotImplementedError("PDE constraints not yet implemented (Phase 3)")

        # Add bias column (like AutoDES PIELM_solver_v2.ipynb Cell 5)
        ones_mu = torch.ones((A_mu.shape[0], 1), device=self.device)
        A_mu_with_bias = torch.cat([A_mu, ones_mu], dim=1)

        # Solve
        print(f"  Solving system: A={A_mu_with_bias.shape}, b={b_mu.shape}")
        weights_with_bias = solve_linear_system(
            A_mu_with_bias, b_mu,
            regularization=self.regularization,
            method=self.solver_method
        )

        # Split weights and bias
        self.net.mu_weights = weights_with_bias[:-1]  # All except last
        self.net.mu_bias = weights_with_bias[-1:]     # Last row (bias)

        t_mu = time.time() - t_start
        print(f"  [OK] Elasticity field solved in {t_mu:.2f}s")

        print(f"\n{'='*60}")
        print(f"PIELM model solved! Total time: {t_u + t_mu:.2f}s")
        print(f"{'='*60}\n")

        # Compute and print training metrics
        self._compute_metrics()

    def _compute_metrics(self):
        """Compute and print training metrics (data fitting)."""
        print("Training metrics:")

        # Predict on training data (returns complex)
        with torch.no_grad():
            u_pred_complex, mu_pred_complex, a_pred = self.net((self.x_data,), return_real=False)

            # Convert to real representation for comparison with data
            # u_data is (N, 6): [real_x, real_y, real_z, imag_x, imag_y, imag_z]
            # u_pred is (N, 3) complex
            u_pred_real = torch.cat([u_pred_complex.real, u_pred_complex.imag], dim=1)

            # mu_data is (N, 2): [magnitude, phase]
            # mu_pred is (N, 1) complex in polar form
            mu_pred_real = torch.cat([mu_pred_complex.abs(), mu_pred_complex.angle()], dim=1)

            # Compute losses
            u_loss = msae_loss(self.u_data, u_pred_real).item()
            mu_loss = msae_loss(self.mu_data, mu_pred_real).item()

            print(f"  Wave field (u) MSAE:  {u_loss:.6e}")
            print(f"  Elasticity (mu) MSAE:  {mu_loss:.6e}")

            if self.anatomical:
                a_loss = msae_loss(self.a_data, a_pred).item()
                print(f"  Anatomy (a) MSAE:     {a_loss:.6e}")

    @minibatch
    def predict(self, x):
        """
        Predict on arbitrary points (same interface as MREPINNModel.predict).

        Args:
            x: (N, 3) spatial coordinates

        Returns:
            Tuple of (u_pred, mu_pred, a_pred, lu_pred, f_trac, f_body)
            All as CPU tensors (same format as PINN)
        """
        x.requires_grad = True
        u_pred, mu_pred, a_pred = self.net.forward(inputs=(x,))
        lu_pred = laplacian(u_pred, x)
        f_trac, f_body = self.pde.traction_and_body_forces(x, u_pred, mu_pred)

        return (
            u_pred.detach().cpu(),
            mu_pred.detach().cpu(),
            a_pred.detach().cpu(),
            lu_pred.detach().cpu(),
            f_trac.detach().cpu(),
            f_body.detach().cpu()
        )

    def test(self):
        """
        Test on full domain (same interface as MREPINNModel.test).

        Returns:
            Tuple ('train', (a, u, lu, pde, mu, direct, fem))
            Same format as PINN for compatibility with TestEvaluator
        """
        print("Testing on full domain...")

        # Get model predictions as tensors
        u_pred, mu_pred, a_pred, lu_pred, f_trac, f_body = \
            self.predict(self.x_full, batch_size=self.n_points)

        # Get ground truth xarrays
        u_true = self.example.wave
        mu_true = self.example.mre
        if 'anat' in self.example:
            a_true = self.example.anat
        else:
            a_true = u_true * 0
            a_pred = u_pred * 0
        mu_mask = self.example.mre_mask

        # Get baseline results if available
        mu_direct = self.example.direct if 'direct' in self.example else None
        mu_fem = self.example.fem if 'fem' in self.example else None
        Lu_true = self.example.Lu if 'Lu' in self.example else None

        # Apply mask level
        mask_level = 1.0
        mu_mask = ((mu_mask > 0) - 1) * mask_level + 1

        # Convert predicted tensors to xarrays
        u_shape, mu_shape, a_shape = u_true.shape, mu_true.shape, a_true.shape
        u_pred = as_xarray(u_pred.reshape(u_shape), like=u_true)
        lu_pred = as_xarray(lu_pred.reshape(u_shape), like=u_true)
        f_trac = as_xarray(f_trac.reshape(u_shape), like=u_true)
        f_body = as_xarray(f_body.reshape(u_shape), like=u_true)
        mu_pred = as_xarray(mu_pred.reshape(mu_shape), like=mu_true)
        a_pred = as_xarray(a_pred.reshape(a_shape), like=a_true)

        # Build anatomy result
        a_vars = ['a_pred', 'a_diff', 'a_true']
        a_dim = xr.DataArray(a_vars, dims=['variable'])
        a = xr.concat([
            mu_mask * a_pred,
            mu_mask * (a_true - a_pred),
            mu_mask * a_true
        ], dim=a_dim)
        a.name = 'anatomy'

        # Build wave field result
        u_vars = ['u_pred', 'u_diff', 'u_true']
        u_dim = xr.DataArray(u_vars, dims=['variable'])
        u = xr.concat([
            mu_mask * u_pred,
            mu_mask * (u_true - u_pred),
            mu_mask * u_true
        ], dim=u_dim)
        u.name = 'wave field'

        # Build Laplacian result
        if Lu_true is not None:
            lu_vars = ['lu_pred', 'lu_diff', 'Lu_true']
            lu_dim = xr.DataArray(lu_vars, dims=['variable'])
            lu = xr.concat([
                mu_mask * lu_pred,
                mu_mask * (Lu_true - lu_pred),
                mu_mask * Lu_true
            ], dim=lu_dim)
            lu.name = 'Laplacian'
        else:
            lu_vars = ['lu_pred']
            lu_dim = xr.DataArray(lu_vars, dims=['variable'])
            lu = xr.concat([mu_mask * lu_pred], dim=lu_dim)
            lu.name = 'Laplacian'

        # Build PDE result (traction + body forces)
        pde_vars = ['f_trac', 'f_body', 'f_sum']
        pde_dim = xr.DataArray(pde_vars, dims=['variable'])
        pde = xr.concat([
            mu_mask * f_trac,
            mu_mask * f_body,
            mu_mask * (f_trac + f_body)
        ], dim=pde_dim)
        pde.name = 'PDE residual'

        # Build elasticity result
        mu_vars = ['mu_pred', 'mu_diff', 'mu_true']
        mu_dim = xr.DataArray(mu_vars, dims=['variable'])
        mu_list = [
            mu_mask * mu_pred,
            mu_mask * (mu_true - mu_pred),
            mu_mask * mu_true
        ]

        # Add baselines if available
        if mu_direct is not None:
            mu_vars.append('mu_direct')
            mu_list.append(mu_mask * mu_direct)
        if mu_fem is not None:
            mu_vars.append('mu_fem')
            mu_list.append(mu_mask * mu_fem)

        mu_dim = xr.DataArray(mu_vars, dims=['variable'])
        mu = xr.concat(mu_list, dim=mu_dim)
        mu.name = 'elasticity'

        # Return in same format as PINN
        return ('train', (a, u, lu, pde, mu, mu_direct, mu_fem))

    def benchmark(self, n_iters=100):
        """
        Benchmark PIELM solve time.

        Note: Unlike PINN, PIELM solves once, not iteratively.
        This measures the time to solve n_iters times (for comparison).
        """
        print(f'# iterations: {n_iters}')

        solve_times = []
        for i in range(n_iters):
            t_start = time.time()
            self.solve(use_pde=False)  # Data fitting only
            t_end = time.time()
            solve_times.append(t_end - t_start)

        avg_time = np.mean(solve_times)
        std_time = np.std(solve_times)

        print(f'Average solve time: {avg_time:.4f}s ± {std_time:.4f}s')
        print(f'Total time: {avg_time * n_iters:.4f}s')
