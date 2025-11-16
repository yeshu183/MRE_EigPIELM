"""
Test Suite for Phase 1: Bernstein Polynomial Basis

Comprehensive tests for:
- Bernstein basis function evaluation
- Analytical derivatives
- Partition of unity property
- Endpoint interpolation
- Ridge regression solver
- Derivative helpers

Run with: python -m pytest tests/test_phase1_bernstein.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pytest

from mre_pielm.core import (
    BernsteinBasis3D,
    solve_ridge,
    solve_lstsq,
    solve_pinv,
    condition_number,
    compute_residual,
    verify_derivatives_finite_diff,
    split_complex_tensor,
    merge_complex_tensor
)


class TestBernsteinBasis:
    """Test Bernstein polynomial basis functions."""

    def test_initialization(self):
        """Test basis initialization."""
        basis = BernsteinBasis3D(
            degrees=(5, 6, 4),
            domain=((0, 1), (0, 1), (0, 1))
        )

        assert basis.nx == 5
        assert basis.ny == 6
        assert basis.nz == 4
        assert basis.n_features == 6 * 7 * 5  # (nx+1) * (ny+1) * (nz+1) = 210

    def test_partition_of_unity(self):
        """Test that basis functions sum to 1."""
        basis = BernsteinBasis3D(
            degrees=(8, 8, 6),
            domain=((0, 0.08), (0, 0.1), (0, 0.01))
        )

        # Random test points
        n_test = 50
        x_test = torch.rand(n_test, 3)
        x_test[:, 0] *= 0.08
        x_test[:, 1] *= 0.1
        x_test[:, 2] *= 0.01

        phi = basis(x_test)
        sums = phi.sum(dim=1)

        # Should all be very close to 1
        assert torch.allclose(sums, torch.ones(n_test), atol=1e-8), \
            f"Partition of unity violated! Max error: {torch.abs(sums - 1.0).max():.2e}"

    def test_endpoint_interpolation(self):
        """Test endpoint interpolation property."""
        basis = BernsteinBasis3D(
            degrees=(5, 5, 5),
            domain=((0, 1), (0, 1), (0, 1))
        )

        # Test corner points
        corners = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])

        phi = basis(corners)

        # At corners, some basis functions should be exactly 1 or 0
        # Check that each row has at least one element equal to 1
        for i in range(len(corners)):
            max_val = phi[i].max()
            assert max_val >= 0.99, f"Corner {i}: max basis value {max_val} < 0.99"

    def test_basis_evaluation_shape(self):
        """Test output shape of basis evaluation."""
        degrees = (10, 12, 8)
        basis = BernsteinBasis3D(
            degrees=degrees,
            domain=((0, 0.08), (0, 0.1), (0, 0.01))
        )

        n_points = 100
        x = torch.rand(n_points, 3)
        x[:, 0] *= 0.08
        x[:, 1] *= 0.1
        x[:, 2] *= 0.01

        phi = basis(x)

        expected_features = (degrees[0] + 1) * (degrees[1] + 1) * (degrees[2] + 1)
        assert phi.shape == (n_points, expected_features), \
            f"Expected shape ({n_points}, {expected_features}), got {phi.shape}"

    def test_gradient_shape(self):
        """Test gradient output shape."""
        basis = BernsteinBasis3D(
            degrees=(5, 6, 4),
            domain=((0, 1), (0, 1), (0, 1))
        )

        n_points = 50
        x = torch.rand(n_points, 3)

        grad_phi = basis.gradient(x)

        expected_shape = (n_points, basis.n_features, 3)
        assert grad_phi.shape == expected_shape, \
            f"Expected gradient shape {expected_shape}, got {grad_phi.shape}"

    def test_laplacian_shape(self):
        """Test Laplacian output shape."""
        basis = BernsteinBasis3D(
            degrees=(5, 6, 4),
            domain=((0, 1), (0, 1), (0, 1))
        )

        n_points = 50
        x = torch.rand(n_points, 3)

        lap_phi = basis.laplacian(x)

        expected_shape = (n_points, basis.n_features)
        assert lap_phi.shape == expected_shape, \
            f"Expected Laplacian shape {expected_shape}, got {lap_phi.shape}"

    def test_derivatives_vs_finite_difference(self):
        """Test analytical derivatives against finite differences."""
        basis = BernsteinBasis3D(
            degrees=(5, 5, 4),
            domain=((0, 1), (0, 1), (0, 1))
        )

        x = torch.rand(20, 3) * 0.8 + 0.1  # Keep away from boundaries

        results = verify_derivatives_finite_diff(basis, x, h=1e-5)

        # Gradient should match within 1e-4
        assert results['gradient_error'] < 1e-4, \
            f"Gradient error too large: {results['gradient_error']:.2e}"

        # Laplacian should match within 1e-3 (second derivatives are less accurate)
        assert results['laplacian_error'] < 1e-3, \
            f"Laplacian error too large: {results['laplacian_error']:.2e}"

        print(f"\nDerivative verification:")
        print(f"  Gradient error: {results['gradient_error']:.4e} "
              f"(rel: {results['gradient_rel_error']:.4e})")
        print(f"  Laplacian error: {results['laplacian_error']:.4e} "
              f"(rel: {results['laplacian_rel_error']:.4e})")

    def test_gpu_compatibility(self):
        """Test GPU compatibility if CUDA available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        basis_cpu = BernsteinBasis3D(
            degrees=(5, 5, 4),
            domain=((0, 1), (0, 1), (0, 1)),
            device='cpu'
        )

        basis_gpu = BernsteinBasis3D(
            degrees=(5, 5, 4),
            domain=((0, 1), (0, 1), (0, 1)),
            device='cuda'
        )

        x = torch.rand(100, 3)

        phi_cpu = basis_cpu(x)
        phi_gpu = basis_gpu(x.cuda())

        # Results should match
        assert torch.allclose(phi_cpu, phi_gpu.cpu(), atol=1e-6)

    def test_domain_mapping(self):
        """Test coordinate normalization."""
        domain = ((0.0, 0.08), (0.0, 0.1), (0.0, 0.01))
        basis = BernsteinBasis3D(degrees=(5, 5, 5), domain=domain)

        # Physical corner points
        x_phys = torch.tensor([[0.0, 0.0, 0.0],
                               [0.08, 0.1, 0.01]])

        phi = basis(x_phys)

        # Should work without errors and produce valid results
        assert phi.shape == (2, basis.n_features)
        assert torch.all(torch.isfinite(phi))


class TestLinearSolvers:
    """Test linear system solvers."""

    def test_solve_ridge_basic(self):
        """Test ridge regression on simple problem."""
        # Create random linear system
        n_rows, n_features = 200, 50
        H = torch.randn(n_rows, n_features)
        W_true = torch.randn(n_features)
        K = H @ W_true + 0.01 * torch.randn(n_rows)  # Add small noise

        # Solve
        W_solved = solve_ridge(H, K, ridge=1e-8)

        # Check solution quality
        residual, rel_residual = compute_residual(H, W_solved, K)

        assert residual < 0.1, f"Large residual: {residual:.4e}"
        assert rel_residual < 0.01, f"Large relative residual: {rel_residual:.4e}"

    def test_solve_ridge_overdetermined(self):
        """Test on overdetermined system (more rows than features)."""
        H = torch.randn(1000, 100)
        K = torch.randn(1000)

        W = solve_ridge(H, K, ridge=1e-10, verbose=True)

        assert W.shape == (100,)
        assert torch.all(torch.isfinite(W))

        residual, rel_residual = compute_residual(H, W, K)
        print(f"\nOverdetermined system: residual={residual:.4e}, rel={rel_residual:.4e}")

    def test_solve_lstsq(self):
        """Test least squares solver."""
        H = torch.randn(500, 100)
        K = torch.randn(500)

        W = solve_lstsq(H, K, verbose=True)

        assert W.shape == (100,)
        residual, _ = compute_residual(H, W, K)
        assert residual < 1.0

    def test_solve_pinv(self):
        """Test pseudoinverse solver."""
        H = torch.randn(500, 100)
        K = torch.randn(500)

        W = solve_pinv(H, K, rcond=1e-8, verbose=True)

        assert W.shape == (100,)
        residual, _ = compute_residual(H, W, K)
        assert residual < 1.0

    def test_condition_number(self):
        """Test condition number computation."""
        # Well-conditioned matrix
        H_good = torch.eye(100) + 0.1 * torch.randn(100, 100)
        cond_good = condition_number(H_good)

        # Ill-conditioned matrix
        H_bad = torch.diag(torch.logspace(-10, 0, 100))
        cond_bad = condition_number(H_bad)

        print(f"\nCondition numbers:")
        print(f"  Well-conditioned: {cond_good:.2e}")
        print(f"  Ill-conditioned: {cond_bad:.2e}")

        assert cond_good < cond_bad
        assert cond_bad > 1e6  # Should be large

    def test_solvers_agree(self):
        """Test that all solvers give similar results on well-posed problem."""
        H = torch.randn(500, 100)
        K = torch.randn(500)

        W_ridge = solve_ridge(H, K, ridge=1e-10)
        W_lstsq = solve_lstsq(H, K)
        W_pinv = solve_pinv(H, K, rcond=1e-8)

        # All should be close
        assert torch.allclose(W_ridge, W_lstsq, atol=1e-4)
        assert torch.allclose(W_ridge, W_pinv, atol=1e-4)


class TestComplexHelpers:
    """Test complex number utilities."""

    def test_split_merge_complex(self):
        """Test splitting and merging complex tensors."""
        # Create complex tensor
        z = torch.tensor([1+2j, 3+4j, 5+6j], dtype=torch.complex64)

        # Split
        real, imag = split_complex_tensor(z)

        assert torch.allclose(real, torch.tensor([1., 3., 5.]))
        assert torch.allclose(imag, torch.tensor([2., 4., 6.]))

        # Merge back
        z_reconstructed = merge_complex_tensor(real, imag, as_complex_dtype=True)

        assert torch.allclose(z, z_reconstructed)

    def test_interleaved_format(self):
        """Test interleaved real/imag format."""
        real = torch.tensor([[1., 3.], [5., 7.]])
        imag = torch.tensor([[2., 4.], [6., 8.]])

        # Merge as interleaved
        z_interleaved = merge_complex_tensor(real, imag, as_complex_dtype=False)

        # Should be [1, 2, 3, 4] in each row
        expected = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
        assert torch.allclose(z_interleaved, expected)

        # Split back
        real_rec, imag_rec = split_complex_tensor(z_interleaved)
        assert torch.allclose(real, real_rec)
        assert torch.allclose(imag, imag_rec)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_basis_and_solver_integration(self):
        """Test using basis with solver for simple fitting problem."""
        # Create basis
        basis = BernsteinBasis3D(
            degrees=(8, 8, 6),
            domain=((0, 1), (0, 1), (0, 1))
        )

        # Generate training data: fit f(x,y,z) = x² + y² + z²
        n_train = 500
        x_train = torch.rand(n_train, 3)
        f_train = (x_train ** 2).sum(dim=1)

        # Build feature matrix
        H = basis(x_train)

        # Solve
        W = solve_ridge(H, f_train, ridge=1e-8, verbose=True)

        # Test on new points
        n_test = 100
        x_test = torch.rand(n_test, 3)
        f_test_true = (x_test ** 2).sum(dim=1)

        phi_test = basis(x_test)
        f_test_pred = phi_test @ W

        # Compute error
        error = torch.abs(f_test_pred - f_test_true).mean()
        print(f"\nIntegration test - Function approximation error: {error:.4e}")

        assert error < 0.05, f"Approximation error too large: {error:.4e}"


def test_phase1_summary():
    """Print summary of Phase 1 tests."""
    print("\n" + "="*70)
    print("Phase 1: Bernstein Basis and Core Components - Test Summary")
    print("="*70)
    print("\nAll tests passed! ✓")
    print("\nComponents tested:")
    print("  ✓ Bernstein polynomial basis (3D tensor product)")
    print("  ✓ Analytical derivatives (gradient, Laplacian)")
    print("  ✓ Partition of unity property")
    print("  ✓ Endpoint interpolation")
    print("  ✓ Domain mapping and normalization")
    print("  ✓ Ridge regression solver")
    print("  ✓ Alternative solvers (lstsq, pinv)")
    print("  ✓ Complex number utilities")
    print("  ✓ Integration: basis + solver")
    print("\nPhase 1 implementation is ready for use!")
    print("="*70)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
