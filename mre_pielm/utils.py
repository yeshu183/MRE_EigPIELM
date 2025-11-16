"""
Utility Functions for MRE-PIELM

Helper functions for data extraction, normalization, and coordinate handling.
"""

import torch
import numpy as np
import xarray as xr
from typing import Tuple, Dict, Any, Optional


def extract_domain_bounds(
    xarray_data: xr.DataArray
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Extract physical domain bounds from xarray data.

    Parameters
    ----------
    xarray_data : xr.DataArray
        xarray with spatial coordinates (x, y, z)

    Returns
    -------
    domain : tuple of tuples
        ((x_min, x_max), (y_min, y_max), (z_min, z_max))

    Examples
    --------
    >>> domain = extract_domain_bounds(example.wave)
    >>> print(domain)  # ((0.0, 0.08), (0.0, 0.1), (0.0, 0.01))
    """
    spatial_dims = [d for d in 'xyz' if d in xarray_data.dims]

    bounds = []
    for dim in spatial_dims:
        coords = xarray_data[dim].values
        bounds.append((float(coords.min()), float(coords.max())))

    return tuple(bounds)


def extract_normalization_stats(xarray_data: xr.DataArray) -> Dict[str, np.ndarray]:
    """
    Extract normalization statistics from xarray data.

    Parameters
    ----------
    xarray_data : xr.DataArray
        Data array to analyze

    Returns
    -------
    stats : dict
        Dictionary with 'mean', 'std', 'min', 'max' for each component
    """
    stats = {}

    if 'component' in xarray_data.dims:
        # Multi-component field
        n_components = len(xarray_data.component)
        stats['mean'] = np.array([xarray_data.sel(component=c).values.mean()
                                  for c in xarray_data.component.values])
        stats['std'] = np.array([xarray_data.sel(component=c).values.std()
                                 for c in xarray_data.component.values])
        stats['min'] = np.array([xarray_data.sel(component=c).values.min()
                                 for c in xarray_data.component.values])
        stats['max'] = np.array([xarray_data.sel(component=c).values.max()
                                 for c in xarray_data.component.values])
    else:
        # Scalar field
        stats['mean'] = np.array([xarray_data.values.mean()])
        stats['std'] = np.array([xarray_data.values.std()])
        stats['min'] = np.array([xarray_data.values.min()])
        stats['max'] = np.array([xarray_data.values.max()])

    return stats


def xarray_to_points_and_values(
    xarray_data: xr.DataArray,
    mask: Optional[xr.DataArray] = None,
    flatten: bool = True,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert xarray data to coordinate points and values.

    Parameters
    ----------
    xarray_data : xr.DataArray
        Data with spatial dimensions
    mask : xr.DataArray, optional
        Binary mask to filter points (only include mask > 0)
    flatten : bool, optional
        If True, flatten to (N, ...) shape (default: True)
    device : str, optional
        Device for tensors

    Returns
    -------
    coords : torch.Tensor, shape (N, 3)
        Spatial coordinates (x, y, z)
    values : torch.Tensor, shape (N, ...)
        Corresponding values
    """
    # Extract spatial coordinates
    spatial_dims = [d for d in 'xyz' if d in xarray_data.dims]
    coords = np.meshgrid(
        *[xarray_data[d].values for d in spatial_dims],
        indexing='ij'
    )
    coords = np.stack(coords, axis=-1)  # (Nx, Ny, Nz, 3)

    # Extract values
    values = xarray_data.values

    # Apply mask if provided
    if mask is not None:
        mask_values = mask.values > 0
        coords = coords[mask_values]
        values = values[mask_values]

    # Flatten if requested
    if flatten:
        if mask is None:
            # Reshape coordinates
            coords = coords.reshape(-1, coords.shape[-1])

            # Reshape values
            if 'component' in xarray_data.dims:
                # Move component to last dimension
                component_axis = xarray_data.dims.index('component')
                values = np.moveaxis(values, component_axis, -1)
                values = values.reshape(-1, values.shape[-1])
            else:
                values = values.reshape(-1)

    # Convert to torch tensors
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    if np.iscomplexobj(values):
        values_tensor = torch.tensor(values, dtype=torch.complex64, device=device)
    else:
        values_tensor = torch.tensor(values, dtype=torch.float32, device=device)

    return coords_tensor, values_tensor


def standardize_coords(
    coords: torch.Tensor,
    center: torch.Tensor,
    extent: torch.Tensor
) -> torch.Tensor:
    """
    Standardize coordinates to [-1, 1] or similar range.

    Normalization: x_std = (x - center) / extent

    Parameters
    ----------
    coords : torch.Tensor, shape (..., D)
        Coordinates to normalize
    center : torch.Tensor, shape (D,)
        Center values
    extent : torch.Tensor, shape (D,)
        Extent (range) values

    Returns
    -------
    coords_std : torch.Tensor
        Standardized coordinates
    """
    return (coords - center) / extent


def unstandardize_coords(
    coords_std: torch.Tensor,
    center: torch.Tensor,
    extent: torch.Tensor
) -> torch.Tensor:
    """
    Reverse standardization of coordinates.

    Parameters
    ----------
    coords_std : torch.Tensor
        Standardized coordinates
    center : torch.Tensor
        Center values
    extent : torch.Tensor
        Extent values

    Returns
    -------
    coords : torch.Tensor
        Physical coordinates
    """
    return coords_std * extent + center


def unstandardize_output(
    values_std: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor
) -> torch.Tensor:
    """
    Reverse standardization of output values.

    Denormalization: x = x_std * std + mean

    Parameters
    ----------
    values_std : torch.Tensor
        Standardized values
    mean : torch.Tensor
        Mean values
    std : torch.Tensor
        Standard deviation values

    Returns
    -------
    values : torch.Tensor
        Physical values
    """
    return values_std * std + mean


def sample_random_points(
    domain: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    n_points: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample random points uniformly in domain.

    Parameters
    ----------
    domain : tuple of tuples
        ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    n_points : int
        Number of points to sample
    device : str, optional
        Device for tensor

    Returns
    -------
    points : torch.Tensor, shape (n_points, 3)
        Random coordinates
    """
    points = torch.rand(n_points, 3, device=device)

    for i, (min_val, max_val) in enumerate(domain):
        points[:, i] = points[:, i] * (max_val - min_val) + min_val

    return points


def sample_grid_points(
    domain: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    n_points_per_dim: Tuple[int, int, int],
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample points on regular grid in domain.

    Parameters
    ----------
    domain : tuple of tuples
        ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    n_points_per_dim : tuple of int
        Number of points along each dimension (nx, ny, nz)
    device : str, optional
        Device for tensor

    Returns
    -------
    points : torch.Tensor, shape (nx*ny*nz, 3)
        Grid coordinates
    """
    grids = []
    for i, (min_val, max_val) in enumerate(domain):
        n = n_points_per_dim[i]
        grid = torch.linspace(min_val, max_val, n, device=device)
        grids.append(grid)

    # Create meshgrid
    mesh = torch.meshgrid(*grids, indexing='ij')
    points = torch.stack(mesh, dim=-1)
    points = points.reshape(-1, 3)

    return points


def compute_relative_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-10
) -> float:
    """
    Compute relative L2 error.

    Error = ||pred - target|| / ||target||

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    target : torch.Tensor
        Target values
    eps : float, optional
        Small value to avoid division by zero

    Returns
    -------
    error : float
        Relative L2 error
    """
    if torch.is_complex(pred):
        pred = torch.abs(pred)
        target = torch.abs(target)

    numerator = torch.norm(pred - target)
    denominator = torch.norm(target) + eps

    return (numerator / denominator).item()


def split_complex_components(
    u: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split complex field into real and imaginary parts.

    Parameters
    ----------
    u : torch.Tensor (complex)
        Complex-valued field

    Returns
    -------
    u_real : torch.Tensor
        Real part
    u_imag : torch.Tensor
        Imaginary part
    """
    if torch.is_complex(u):
        return u.real, u.imag
    else:
        raise ValueError("Input must be complex-valued")


def merge_complex_components(
    u_real: torch.Tensor,
    u_imag: torch.Tensor
) -> torch.Tensor:
    """
    Merge real and imaginary parts into complex field.

    Parameters
    ----------
    u_real : torch.Tensor
        Real part
    u_imag : torch.Tensor
        Imaginary part

    Returns
    -------
    u : torch.Tensor (complex)
        Complex-valued field
    """
    return torch.complex(u_real, u_imag)


def create_collocation_points(
    domain: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    n_collocation: int,
    sampling: str = 'random',
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create collocation points for PDE constraints.

    Parameters
    ----------
    domain : tuple of tuples
        Physical domain bounds
    n_collocation : int
        Number of collocation points
    sampling : str, optional
        Sampling strategy: 'random' or 'grid' (default: 'random')
    device : str, optional
        Device for tensor

    Returns
    -------
    x_colloc : torch.Tensor, shape (n_collocation, 3)
        Collocation coordinates

    Notes
    -----
    For PIELM, the paper uses ~1.5M collocation points for 2D Helmholtz.
    For 3D MRE, we might use fewer (10k-100k) depending on domain size.
    """
    if sampling == 'random':
        return sample_random_points(domain, n_collocation, device)
    elif sampling == 'grid':
        # Distribute points evenly across dimensions
        n_per_dim = int(np.ceil(n_collocation ** (1/3)))
        return sample_grid_points(domain, (n_per_dim, n_per_dim, n_per_dim), device)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling}")
