"""
Periodic Boundary Module
========================

Utilities for handling periodic boundary conditions.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union


def apply_periodic_boundary(
    coords: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    Apply periodic boundary conditions to coordinates.

    Parameters
    ----------
    coords : ndarray
        Coordinates (N, 3) or (3,).
    box_size : float
        Box size.

    Returns
    -------
    wrapped : ndarray
        Coordinates wrapped to [0, box_size).

    Examples
    --------
    >>> coords = np.array([[210.0, -5.0, 100.0]])
    >>> wrapped = apply_periodic_boundary(coords, 205.0)
    >>> print(wrapped)  # [[5.0, 200.0, 100.0]]
    """
    return np.mod(coords, box_size)


def wrap_coordinates(
    coords: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    Alias for apply_periodic_boundary.

    Parameters
    ----------
    coords : ndarray
        Coordinates.
    box_size : float
        Box size.

    Returns
    -------
    wrapped : ndarray
        Wrapped coordinates.
    """
    return apply_periodic_boundary(coords, box_size)


def periodic_distance(
    pos1: np.ndarray,
    pos2: np.ndarray,
    box_size: float,
) -> Union[float, np.ndarray]:
    """
    Compute distance between points with periodic boundaries.

    Parameters
    ----------
    pos1 : ndarray
        First position(s) (3,) or (N, 3).
    pos2 : ndarray
        Second position(s) (3,) or (N, 3).
    box_size : float
        Box size.

    Returns
    -------
    distance : float or ndarray
        Distance(s) accounting for periodic boundaries.

    Examples
    --------
    >>> d = periodic_distance([1, 1, 1], [204, 1, 1], 205.0)
    >>> print(d)  # 2.0 (wraps around)
    """
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    
    dr = pos1 - pos2
    dr = dr - box_size * np.round(dr / box_size)
    
    return np.sqrt(np.sum(dr**2, axis=-1))


def periodic_displacement(
    pos1: np.ndarray,
    pos2: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    Compute displacement vector with periodic boundaries.

    Parameters
    ----------
    pos1 : ndarray
        First position(s).
    pos2 : ndarray
        Second position(s).
    box_size : float
        Box size.

    Returns
    -------
    dr : ndarray
        Displacement vector(s) pos1 - pos2 with periodic wrap.
    """
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    
    dr = pos1 - pos2
    dr = dr - box_size * np.round(dr / box_size)
    
    return dr


def find_particles_in_sphere(
    coords: np.ndarray,
    center: np.ndarray,
    radius: float,
    box_size: float,
) -> np.ndarray:
    """
    Find indices of particles within a sphere (periodic).

    Parameters
    ----------
    coords : ndarray
        Particle coordinates (N, 3).
    center : ndarray
        Sphere center (3,).
    radius : float
        Sphere radius.
    box_size : float
        Box size for periodic boundaries.

    Returns
    -------
    indices : ndarray
        Indices of particles within sphere.

    Examples
    --------
    >>> idx = find_particles_in_sphere(coords, halo_center, R200c, box_size)
    >>> particles_in_halo = coords[idx]
    """
    distances = periodic_distance(coords, center, box_size)
    return np.where(distances <= radius)[0]


def shift_to_center(
    coords: np.ndarray,
    center: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    Shift coordinates so that center is at box center.

    Useful for visualization or when computing profiles.

    Parameters
    ----------
    coords : ndarray
        Particle coordinates (N, 3).
    center : ndarray
        Current center position (3,).
    box_size : float
        Box size.

    Returns
    -------
    shifted : ndarray
        Shifted coordinates with center at box_size/2.
    """
    # Compute displacement from center accounting for periodicity
    dr = periodic_displacement(coords, center, box_size)
    
    # Place at box center
    return dr + box_size / 2


def replicate_particles(
    coords: np.ndarray,
    masses: np.ndarray,
    box_size: float,
    n_replicas: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replicate particles for periodic boundary visualization.

    Creates copies of particles shifted by box_size in each direction.

    Parameters
    ----------
    coords : ndarray
        Particle coordinates (N, 3).
    masses : ndarray
        Particle masses (N,).
    box_size : float
        Box size.
    n_replicas : int
        Number of replicas in each direction (total 3^n_replicas copies).

    Returns
    -------
    coords_rep : ndarray
        Replicated coordinates.
    masses_rep : ndarray
        Replicated masses.
    """
    if n_replicas == 0:
        return coords, masses
    
    # Generate shifts
    shifts = []
    for dx in range(-n_replicas, n_replicas + 1):
        for dy in range(-n_replicas, n_replicas + 1):
            for dz in range(-n_replicas, n_replicas + 1):
                shifts.append([dx * box_size, dy * box_size, dz * box_size])
    
    shifts = np.array(shifts)
    
    # Replicate
    coords_list = []
    for shift in shifts:
        coords_list.append(coords + shift)
    
    coords_rep = np.vstack(coords_list)
    masses_rep = np.tile(masses, len(shifts))
    
    return coords_rep, masses_rep


def get_nearby_particles_periodic(
    tree,
    center: np.ndarray,
    radius: float,
    coords: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    Query KDTree for particles near a point with periodic boundaries.

    Handles particles that cross box boundaries by querying at
    shifted positions.

    Parameters
    ----------
    tree : cKDTree
        KD-tree built from particle coordinates.
    center : ndarray
        Query center (3,).
    radius : float
        Search radius.
    coords : ndarray
        Original particle coordinates (for distance verification).
    box_size : float
        Box size.

    Returns
    -------
    indices : ndarray
        Unique indices of nearby particles.

    Notes
    -----
    For large search radii (radius > box_size/2), consider using
    a brute-force periodic distance calculation instead.
    """
    # Check if we need to handle periodic boundaries
    needs_wrap = any(
        center[i] < radius or center[i] > box_size - radius
        for i in range(3)
    )
    
    if not needs_wrap:
        # Simple case - no wrapping needed
        return np.array(tree.query_ball_point(center, radius))
    
    # Generate shifted query centers for periodic images
    shifts = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                shifts.append([dx * box_size, dy * box_size, dz * box_size])
    
    # Query at all shifted positions
    all_indices = set()
    for shift in shifts:
        shifted_center = center + np.array(shift)
        indices = tree.query_ball_point(shifted_center, radius)
        all_indices.update(indices)
    
    indices = np.array(list(all_indices))
    
    # Verify distances with periodic boundary
    if len(indices) > 0:
        distances = periodic_distance(coords[indices], center, box_size)
        mask = distances <= radius
        indices = indices[mask]
    
    return indices
