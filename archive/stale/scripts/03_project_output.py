#!/usr/bin/env python3
"""
Project particles to 2D density maps and save outputs.

This script takes transformed particles and creates:
- 2D projected density maps (3 axes)
- Power spectra
- Lens plane HDF5 files for lux

Usage:
    python 03_project_output.py --mode replace --resolution 625
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_particles(particle_file: Path) -> dict:
    """Load transformed particles from HDF5."""
    with h5py.File(particle_file, 'r') as f:
        return {
            'positions': f['positions'][:],
            'masses': f['masses'][:],
        }


def project_to_2d(
    positions: np.ndarray,
    masses: np.ndarray,
    box_size: float,
    grid_size: int,
    axis: int = 2,
) -> np.ndarray:
    """
    Project particles onto a 2D grid using CIC mass assignment.
    
    Parameters
    ----------
    positions : (N, 3) array
    masses : (N,) array
    box_size : float
    grid_size : int
    axis : int
        Projection axis (0=x, 1=y, 2=z)
    
    Returns
    -------
    density : (grid_size, grid_size) array
        Surface density in Msun/h / (Mpc/h)^2
    """
    try:
        import MAS_library as MASL
        
        # Get 2D coordinates (perpendicular to projection axis)
        axes_2d = [i for i in range(3) if i != axis]
        pos_2d = positions[:, axes_2d].astype(np.float32)
        
        # Create 2D density field
        density = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Use MAS_library for CIC assignment
        MASL.MA(pos_2d, density, box_size, 'CIC', W=masses.astype(np.float32))
        
        # Convert to surface density (Msun/h per (Mpc/h)^2)
        pixel_area = (box_size / grid_size) ** 2
        density /= pixel_area
        
        return density
        
    except ImportError:
        logger.warning("MAS_library not available, using simple binning")
        return _simple_project(positions, masses, box_size, grid_size, axis)


def _simple_project(positions, masses, box_size, grid_size, axis):
    """Simple histogram-based projection (fallback)."""
    axes_2d = [i for i in range(3) if i != axis]
    pos_2d = positions[:, axes_2d]
    
    # Bin edges
    edges = np.linspace(0, box_size, grid_size + 1)
    
    # 2D histogram weighted by mass
    density, _, _ = np.histogram2d(
        pos_2d[:, 0], pos_2d[:, 1],
        bins=[edges, edges],
        weights=masses
    )
    
    # Convert to surface density
    pixel_area = (box_size / grid_size) ** 2
    density /= pixel_area
    
    return density.astype(np.float32)


def compute_power_spectrum(
    density: np.ndarray,
    box_size: float,
) -> tuple:
    """
    Compute 2D power spectrum of density field.
    
    Returns
    -------
    k : array
        Wavenumbers in h/Mpc
    Pk : array
        Power spectrum in (Mpc/h)^2
    """
    grid_size = density.shape[0]
    
    # FFT
    fft = np.fft.fft2(density)
    
    # Power spectrum
    power = np.abs(fft) ** 2
    
    # Wavenumber grid
    kfreq = np.fft.fftfreq(grid_size, d=box_size/grid_size)
    kx, ky = np.meshgrid(kfreq, kfreq)
    k_mag = np.sqrt(kx**2 + ky**2)
    
    # Bin power spectrum radially
    k_bins = np.linspace(0, kfreq.max(), 50)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    Pk = np.zeros(len(k_centers))
    counts = np.zeros(len(k_centers))
    
    for i in range(len(k_centers)):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if mask.sum() > 0:
            Pk[i] = power[mask].mean()
            counts[i] = mask.sum()
    
    # Normalize
    Pk *= (box_size / grid_size) ** 2
    
    # Filter valid bins
    valid = counts > 0
    
    return k_centers[valid] * 2 * np.pi, Pk[valid]


def save_lens_plane(
    density: np.ndarray,
    output_path: Path,
    box_size: float,
    redshift: float = 0.0,
    axis: int = 2,
):
    """
    Save density map in lux-compatible HDF5 format.
    
    The PreProjected format expects:
    - DensityMap: 2D array of surface density
    - BoxSize, GridSize, Redshift attributes
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Store density map
        f.create_dataset('DensityMap', data=density, 
                        compression='gzip', compression_opts=4)
        
        # Metadata
        f.attrs['BoxSize'] = box_size
        f.attrs['GridSize'] = density.shape[0]
        f.attrs['Redshift'] = redshift
        f.attrs['ProjectionAxis'] = axis
        f.attrs['Units'] = 'Msun/h / (Mpc/h)^2'
        f.attrs['Created'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    logger.info(f"Saved lens plane: {output_path}")


def compute_halo_profiles(
    particles: dict,
    matches: dict,
    box_size: float = 205.0,
    n_bins: int = 50,
    r_max_R200: float = 5.0,
) -> dict:
    """
    Compute radial density profiles for matched halos.
    
    Returns profiles in units of R_200 for easy comparison.
    """
    from scipy.spatial import cKDTree
    
    n_halos = len(matches['dmo_indices'])
    logger.info(f"Computing profiles for {n_halos} halos...")
    
    # Radial bins in units of R_200
    r_bins_R200 = np.logspace(-2, np.log10(r_max_R200), n_bins + 1)
    r_centers_R200 = np.sqrt(r_bins_R200[:-1] * r_bins_R200[1:])
    
    # Build KDTree
    tree = cKDTree(particles['positions'], boxsize=box_size)
    
    # Storage
    all_profiles = np.zeros((n_halos, n_bins), dtype=np.float64)
    all_enclosed = np.zeros((n_halos, n_bins), dtype=np.float64)
    
    for i in range(n_halos):
        if i % 500 == 0:
            logger.info(f"  {i}/{n_halos} profiles computed...")
        
        center = matches['dmo_positions'][i]
        R_200 = matches['dmo_radii'][i]
        
        # Physical radii
        r_bins_phys = r_bins_R200 * R_200
        r_max = r_bins_phys[-1]
        
        # Find particles
        indices = tree.query_ball_point(center, r=r_max)
        if len(indices) == 0:
            continue
        
        # Relative positions
        dx = particles['positions'][indices] - center
        dx = dx - box_size * np.round(dx / box_size)
        r = np.linalg.norm(dx, axis=1)
        m = particles['masses'][indices]
        
        # Bin particles
        bin_idx = np.digitize(r, r_bins_phys) - 1
        
        for j in range(n_bins):
            in_bin = (bin_idx == j)
            V_shell = (4.0/3.0) * np.pi * (r_bins_phys[j+1]**3 - r_bins_phys[j]**3)
            all_profiles[i, j] = np.sum(m[in_bin]) / V_shell
            all_enclosed[i, j] = np.sum(m[bin_idx <= j])
    
    return {
        'r_bins_R200': r_bins_R200,
        'r_centers_R200': r_centers_R200,
        'density_profiles': all_profiles,
        'enclosed_mass': all_enclosed,
        'halo_masses': matches['dmo_masses'],
        'halo_radii': matches['dmo_radii'],
        'halo_indices': matches['dmo_indices'],
    }


def save_profiles(profiles: dict, output_path: Path, mode: str):
    """Save halo profiles to HDF5."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.attrs['mode'] = mode
        f.attrs['n_halos'] = len(profiles['halo_masses'])
        
        for key, value in profiles.items():
            f.create_dataset(key, data=value, compression='gzip')
    
    logger.info(f"Saved {len(profiles['halo_masses'])} profiles to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=625)
    parser.add_argument('--snapshot', type=int, default=99)
    parser.add_argument('--grid-size', type=int, default=4096)
    parser.add_argument('--output-dir', type=str,
                        default='/mnt/home/mlee1/ceph/hydro_replace')
    parser.add_argument('--skip-profiles', action='store_true',
                        help='Skip halo profile computation')
    args = parser.parse_args()
    
    t_start = time.time()
    
    # Paths
    res = args.resolution
    base_dir = Path(args.output_dir) / f'L205n{res}TNG' / args.mode
    particle_file = base_dir / f'particles_snap{args.snapshot:03d}.h5'
    match_file = Path(args.output_dir) / f'L205n{res}TNG' / 'matches' / f'spatial_match_snap{args.snapshot:03d}.h5'
    
    box_size = 205.0  # Mpc/h
    
    logger.info("=" * 60)
    logger.info(f"PROJECTION + OUTPUT - {args.mode.upper()}")
    logger.info("=" * 60)
    logger.info(f"Particle file: {particle_file}")
    logger.info(f"Grid size: {args.grid_size}")
    
    if not particle_file.exists():
        logger.error(f"Particle file not found: {particle_file}")
        return
    
    # Load particles
    logger.info("Loading particles...")
    particles = load_particles(particle_file)
    logger.info(f"  Loaded {len(particles['positions'])} particles")
    
    # Project along each axis
    for axis in [0, 1, 2]:
        axis_name = ['x', 'y', 'z'][axis]
        logger.info(f"Projecting along {axis_name}-axis...")
        
        t_proj = time.time()
        density = project_to_2d(
            particles['positions'],
            particles['masses'],
            box_size,
            args.grid_size,
            axis=axis,
        )
        logger.info(f"  Projection took {time.time()-t_proj:.1f}s")
        
        # Save lens plane
        lens_file = base_dir / 'lens_planes' / f'lens_snap{args.snapshot:03d}_axis{axis}.h5'
        save_lens_plane(density, lens_file, box_size, axis=axis)
        
        # Compute and save power spectrum
        logger.info("  Computing power spectrum...")
        k, Pk = compute_power_spectrum(density, box_size)
        
        pk_file = base_dir / 'power_spectra' / f'pk2d_snap{args.snapshot:03d}_axis{axis}.npz'
        pk_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(pk_file, k=k, Pk=Pk)
        logger.info(f"  Saved power spectrum: {pk_file}")
    
    # Compute halo profiles (if matches exist and not skipped)
    if not args.skip_profiles and match_file.exists():
        logger.info("Computing halo profiles...")
        with h5py.File(match_file, 'r') as f:
            matches = {key: f[key][:] for key in f.keys()}
        
        profiles = compute_halo_profiles(particles, matches, box_size)
        
        profile_file = base_dir / 'profiles' / f'profiles_snap{args.snapshot:03d}.h5'
        save_profiles(profiles, profile_file, args.mode)
    elif not match_file.exists():
        logger.info("Skipping profiles (no match file)")
    
    t_elapsed = time.time() - t_start
    logger.info(f"Total time: {t_elapsed:.1f}s")


if __name__ == '__main__':
    main()
