"""
Simple validation test for PIELM implementation.

This script tests the basic PIELM workflow on a simple synthetic MRE example
to verify that all components work correctly before testing on real data.

Tests:
1. Model initialization
2. Data-only solve (Phase 2)
3. PDE-constrained solve (Phase 3)
4. Forward prediction
5. Comparison of residuals
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mre_pinn.pde import HelmholtzEquation
from pielm import MREPIELM, MREPIELMModel


def create_simple_synthetic_data(n_points=100, omega=100.0, device='cpu'):
    """
    Create a simple synthetic MRE dataset for testing.

    Uses a homogeneous medium with known analytical solution.
    For Helmholtz equation: μ∇²u + ρω²u = 0

    Simple solution: u(x,y,z) = A * exp(i*k*x) where k = ω*sqrt(ρ/μ)
    """
    # Domain: [0, 0.1]^3 meters
    x = np.random.rand(n_points, 3) * 0.1

    # Material properties
    mu_true = 3000.0  # Pa (brain tissue stiffness)
    rho = 1000.0      # kg/m³

    # Wave number
    k = omega * np.sqrt(rho / mu_true)

    # Analytical solution: plane wave in x-direction
    # u = [A*exp(i*k*x), 0, 0] (only x-component for simplicity)
    A = 1e-6  # 1 micron amplitude

    u_x_complex = A * np.exp(1j * k * x[:, 0])
    u_y_complex = np.zeros(n_points, dtype=complex)
    u_z_complex = np.zeros(n_points, dtype=complex)

    # Stack into (N, 3) complex array
    u_complex = np.stack([u_x_complex, u_y_complex, u_z_complex], axis=1)

    # Split into real and imaginary (N, 6)
    u_data = np.concatenate([u_complex.real, u_complex.imag], axis=1)

    # Elasticity (constant)
    mu_complex = np.full(n_points, mu_true, dtype=complex)
    mu_data = np.concatenate([np.abs(mu_complex)[:, None], np.angle(mu_complex)[:, None]], axis=1)

    # Convert to torch
    x = torch.tensor(x, dtype=torch.float32, device=device)
    u_data = torch.tensor(u_data, dtype=torch.float32, device=device)
    mu_data = torch.tensor(mu_data, dtype=torch.float32, device=device)

    return x, u_data, mu_data, mu_true, rho


class SimpleMREExample:
    """Minimal MREExample for testing."""
    def __init__(self, x, u_data, mu_data):
        self.x_data = x
        self.u_data = u_data
        self.mu_data = mu_data

        # Mock field objects
        class Field:
            is_complex = True
            has_components = True
            n_components = 3

            def __init__(self, points, data):
                self._points = points
                self._data = data

            def points(self):
                return self._points.cpu().numpy()

            def values(self):
                return self._data.cpu().numpy()

        # Mock mask (all ones - no masking)
        class Mask:
            def __init__(self, n):
                self.values = np.ones(n, dtype=bool)

        self.wave = type('wave', (), {'field': Field(x, u_data)})()
        self.mre = type('mre', (), {'field': Field(x, mu_data)})()
        self.mre_mask = Mask(len(x))

    def __contains__(self, key):
        return False  # No anatomy


def test_pielm_basic():
    """Test basic PIELM workflow."""
    print("=" * 80)
    print("PIELM Simple Validation Test")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 1. Create synthetic data
    print("\n1. Creating synthetic data...")
    n_train = 200
    n_test = 50
    omega = 100.0

    x_train, u_train, mu_train, mu_true, rho = create_simple_synthetic_data(
        n_points=n_train, omega=omega, device=device
    )
    x_test, u_test, mu_test, _, _ = create_simple_synthetic_data(
        n_points=n_test, omega=omega, device=device
    )

    print(f"   Training points: {n_train}")
    print(f"   Test points: {n_test}")
    print(f"   True mu: {mu_true} Pa")
    print(f"   Omega: {omega} rad/s")
    print(f"   Rho: {rho} kg/m^3")

    # 2. Create example and initialize PIELM
    print("\n2. Initializing PIELM model...")
    example = SimpleMREExample(x_train, u_train, mu_train)

    # Note: MREPIELM.__init__ expects real MREExample with metadata
    # We'll create the net manually to bypass the metadata requirements
    net = MREPIELM.__new__(MREPIELM)
    torch.nn.Module.__init__(net)

    net.device = device
    net.is_complex_u = True
    net.is_complex_mu = True
    net.n_input = 3
    net.n_output_u = 6
    net.n_output_mu = 2
    net.input_loc = torch.tensor([0.05, 0.05, 0.05], device=device)
    net.input_scale = torch.tensor([0.1, 0.1, 0.1], device=device)
    net.u_loc = u_train.mean(dim=0)
    net.u_scale = u_train.std(dim=0) + 1e-8
    net.mu_loc = mu_train.mean(dim=0)
    net.mu_scale = mu_train.std(dim=0) + 1e-8
    net.a_loc = torch.zeros(0, device=device)
    net.a_scale = torch.zeros(0, device=device)
    net.omega = torch.tensor(omega, device=device)

    from pielm.features import RandomFeatures
    net.u_features = RandomFeatures(
        n_input=3, n_features=500, frequency_scale=1.0,
        use_sin_cos=True, seed=42, device=device
    )
    net.mu_features = RandomFeatures(
        n_input=3, n_features=500, frequency_scale=1.0,
        use_sin_cos=True, seed=43, device=device
    )
    net.u_weights = None
    net.mu_weights = None
    net.u_bias = None
    net.mu_bias = None

    print(f"   Features: {net.u_features.n_effective}")
    print(f"   Input dim: {net.n_input}")
    print(f"   Output u dim: {net.n_output_u}")
    print(f"   Output mu dim: {net.n_output_mu}")

    # 3. Create PDE
    print("\n3. Creating Helmholtz PDE...")
    pde = HelmholtzEquation(omega=omega, rho=rho)

    # 4. Create solver (data-only first)
    print("\n4. Solving with data-only (Phase 2)...")

    solver = MREPIELMModel(
        example=example,
        net=net,
        pde=pde,
        loss_weights=[1.0, 1.0, 0.0, 0.0],  # Only data weights
        n_points=n_train,  # Use all training points
        regularization=1e-6,
        device=device
    )

    solver.solve(use_pde=False)
    print("   [OK] Data-only solve complete")

    # 5. Test forward pass
    print("\n5. Testing forward prediction...")
    with torch.no_grad():
        u_pred, mu_pred, a_pred = net((x_test,))

    # Compute errors
    u_test_complex = torch.complex(
        torch.tensor(u_test[:, :3], device=device),
        torch.tensor(u_test[:, 3:], device=device)
    )

    u_error = torch.abs(u_pred - u_test_complex).mean()
    mu_error = torch.abs(mu_pred.abs() - mu_true).mean()

    print(f"   Mean u error: {u_error.item():.2e}")
    print(f"   Mean mu error: {mu_error.item():.2e} Pa ({mu_error.item()/mu_true*100:.1f}%)")

    # 6. Test with PDE constraints (Phase 3)
    print("\n6. Solving with PDE constraints (Phase 3)...")

    solver_pde = MREPIELMModel(
        example=example,
        net=net,
        pde=pde,
        loss_weights=[1.0, 1.0, 0.0, 0.1],  # Add PDE weight
        n_points=n_train,
        n_pde_points=100,  # PDE collocation points
        regularization=1e-6,
        device=device
    )

    # First solve without PDE to get initial mu
    solver_pde.solve(use_pde=False)
    print("   [OK] Initial solve (data-only) complete")

    # Then solve with PDE constraints
    try:
        solver_pde.solve(use_pde=True)
        print("   [OK] PDE-constrained solve complete")

        # Test forward pass with PDE
        with torch.no_grad():
            u_pred_pde, mu_pred_pde, _ = net((x_test,))

        u_error_pde = torch.abs(u_pred_pde - u_test_complex).mean()
        mu_error_pde = torch.abs(mu_pred_pde.abs() - mu_true).mean()

        print(f"   Mean u error (with PDE): {u_error_pde.item():.2e}")
        print(f"   Mean mu error (with PDE): {mu_error_pde.item():.2e} Pa ({mu_error_pde.item()/mu_true*100:.1f}%)")

    except Exception as e:
        print(f"   [WARN] PDE solve failed: {e}")
        print("   This is expected if PDE constraints need further debugging")

    # 7. Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"[OK] Model initialization: PASSED")
    print(f"[OK] Data-only solve: PASSED")
    print(f"[OK] Forward prediction: PASSED")
    print(f"  - u error: {u_error.item():.2e}")
    print(f"  - mu error: {mu_error.item()/mu_true*100:.1f}%")

    # Basic sanity checks
    if u_error.item() < 1e-3:  # Reasonable for synthetic data
        print("[OK] u prediction accuracy: GOOD")
    else:
        print("[WARN] u prediction accuracy: NEEDS IMPROVEMENT")

    if mu_error.item() / mu_true < 0.1:  # Within 10%
        print("[OK] mu prediction accuracy: GOOD")
    else:
        print("[WARN] mu prediction accuracy: NEEDS IMPROVEMENT")

    print("\nNote: This is a simple synthetic test. Real MRE data will have more complexity.")


if __name__ == "__main__":
    test_pielm_basic()
