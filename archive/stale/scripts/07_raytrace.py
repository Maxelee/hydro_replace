#!/usr/bin/env python
"""
07_raytrace.py - Generate Convergence Maps via Ray-Tracing

This script performs ray-tracing through TNG-300 matter distributions to
generate weak lensing convergence maps for:
    - TNG-300-Dark (DMO)
    - TNG-300 hydro (true baryonic effects)
    - BCM-modified DMO
    - Hydro-replaced DMO

Each configuration is ray-traced 10 times with different projections to
capture cosmic variance.

Usage:
    mpirun -np 16 python 07_raytrace.py --config raytrace
    python 07_raytrace.py --snapshot 99 --n_realizations 10
    
Outputs:
    - convergence_maps_DMO_r00-09.fits
    - convergence_maps_Hydro_r00-09.fits
    - convergence_maps_BCM_r00-09.fits
    - convergence_maps_Replace_r00-09.fits

Dependencies:
    - lux C++ ray-tracing code (external)
    - hydro_replace package (this project)
"""

import os
import sys
import argparse
import numpy as np
import h5py
import yaml
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hydro_replace.utils.logging_setup import setup_logging
from hydro_replace.utils.parallel import get_mpi_info, distribute_items
from hydro_replace.utils.io_helpers import ensure_dir
from hydro_replace.raytrace.raytrace_engine import LuxInterface, RayTraceConfig
from hydro_replace.raytrace.convergence_maps import ConvergenceMap

# MPI setup
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1

logger = setup_logging("07_raytrace", rank=rank)


def load_config() -> Tuple[dict, dict, dict]:
    """Load all configuration files."""
    config_dir = project_root / "config"
    
    with open(config_dir / "simulation_paths.yaml") as f:
        sim_config = yaml.safe_load(f)
    
    with open(config_dir / "analysis_params.yaml") as f:
        analysis_config = yaml.safe_load(f)
    
    with open(config_dir / "raytrace_config.yaml") as f:
        raytrace_config = yaml.safe_load(f)
    
    return sim_config, analysis_config, raytrace_config


def generate_random_projections(n_realizations: int, seed: int = 42) -> List[dict]:
    """
    Generate random projection configurations for ray-tracing.
    
    Each projection has:
        - projection axis (0, 1, or 2)
        - rotation angle around that axis
        - random translation offset
    
    Parameters
    ----------
    n_realizations : int
        Number of independent projections
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    projections : list of dict
        Each dict contains projection parameters
    """
    np.random.seed(seed)
    
    projections = []
    for i in range(n_realizations):
        proj = {
            'id': i,
            'axis': np.random.randint(0, 3),  # x, y, or z
            'rotation_deg': np.random.uniform(0, 360),
            'offset_frac': np.random.uniform(0, 1, size=3),  # Fractional box offset
        }
        projections.append(proj)
    
    return projections


def run_raytracing(matter_field: np.ndarray, config: RayTraceConfig, 
                   projection: dict, output_path: Path,
                   lux_interface: LuxInterface) -> ConvergenceMap:
    """
    Run ray-tracing for a single realization.
    
    Parameters
    ----------
    matter_field : np.ndarray
        3D density field (n_grid, n_grid, n_grid)
    config : RayTraceConfig
        Ray-tracing configuration
    projection : dict
        Projection parameters
    output_path : Path
        Where to save FITS output
    lux_interface : LuxInterface
        Interface to lux ray-tracing code
    
    Returns
    -------
    kappa_map : ConvergenceMap
        Weak lensing convergence map
    """
    logger.info(f"  Projection {projection['id']}: axis={projection['axis']}, "
                f"rotation={projection['rotation_deg']:.1f}°")
    
    # Apply projection (rotation + translation)
    projected_field = apply_projection(matter_field, projection)
    
    # Run lux ray-tracing
    try:
        kappa_data = lux_interface.trace_rays(
            density_field=projected_field,
            source_redshift=config.source_z,
            n_lens_planes=config.n_lens_planes,
        )
    except Exception as e:
        logger.error(f"Ray-tracing failed: {e}")
        # Return empty map on failure
        kappa_data = np.zeros((config.n_pixels, config.n_pixels))
    
    # Add shape noise if requested
    if config.add_shape_noise:
        shape_noise = np.random.normal(
            0, config.shape_noise_sigma, 
            size=kappa_data.shape
        )
        kappa_data += shape_noise
    
    # Create ConvergenceMap object
    kappa_map = ConvergenceMap(
        data=kappa_data,
        pixel_scale_arcmin=config.fov_deg * 60 / config.n_pixels,
        fov_deg=config.fov_deg
    )
    
    # Save as FITS
    kappa_map.save_fits(output_path)
    logger.info(f"  Saved: {output_path.name}")
    
    return kappa_map


def apply_projection(field: np.ndarray, projection: dict) -> np.ndarray:
    """
    Apply rotation and translation to 3D field.
    
    Parameters
    ----------
    field : np.ndarray
        Input 3D density field
    projection : dict
        Projection parameters
    
    Returns
    -------
    projected : np.ndarray
        Transformed field
    """
    from scipy.ndimage import rotate, shift
    
    n = field.shape[0]
    
    # Translation (periodic wrap)
    offset = (projection['offset_frac'] * n).astype(int)
    shifted = np.roll(field, offset, axis=(0, 1, 2))
    
    # Rotation around projection axis
    axes_map = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    rot_axes = axes_map[projection['axis']]
    
    rotated = rotate(
        shifted, 
        angle=projection['rotation_deg'],
        axes=rot_axes,
        reshape=False,
        order=1  # Bilinear interpolation
    )
    
    return rotated


def grid_particles(particles: dict, box_size: float, n_grid: int) -> np.ndarray:
    """
    Assign particles to a 3D grid using Cloud-in-Cell (CIC).
    
    Parameters
    ----------
    particles : dict
        Dictionary with 'positions' and 'masses' arrays
    box_size : float
        Box size in Mpc/h
    n_grid : int
        Number of grid cells per dimension
    
    Returns
    -------
    density : np.ndarray
        3D density field (n_grid, n_grid, n_grid)
    """
    try:
        import MAS_library as MASL
        
        # Create density field
        density = np.zeros((n_grid, n_grid, n_grid), dtype=np.float32)
        
        # Positions should be in [0, box_size]
        pos = particles['positions'].astype(np.float32)
        masses = particles['masses'].astype(np.float32)
        
        # Use MAS_library for CIC assignment
        MASL.MA(pos, density, box_size, 'CIC', W=masses, verbose=False)
        
        return density
        
    except ImportError:
        logger.warning("MAS_library not available, using numpy histogramdd")
        
        # Fallback to histogram
        pos = particles['positions']
        masses = particles['masses']
        
        bins = np.linspace(0, box_size, n_grid + 1)
        density, _ = np.histogramdd(
            pos, bins=[bins, bins, bins], weights=masses
        )
        
        return density.astype(np.float32)


def load_matter_field(sim_path: str, snapshot: int, grid_size: int,
                      modification: str = None, 
                      bcm_file: Path = None,
                      replace_file: Path = None) -> np.ndarray:
    """
    Load or compute matter density field for ray-tracing.
    
    Parameters
    ----------
    sim_path : str
        Path to simulation
    snapshot : int
        Snapshot number
    grid_size : int
        Grid resolution for density field
    modification : str, optional
        'bcm' or 'replace' for modified fields
    bcm_file : Path, optional
        Path to BCM-modified particle file
    replace_file : Path, optional
        Path to hydro-replaced particle file
    
    Returns
    -------
    density : np.ndarray
        3D density field
    """
    from hydro_replace.data.load_simulations import SimulationData
    
    sim = SimulationData(sim_path)
    box_size = sim.box_size
    
    if modification == 'bcm' and bcm_file is not None:
        logger.info(f"  Loading BCM-modified particles from {bcm_file}")
        with h5py.File(bcm_file, 'r') as f:
            particles = {
                'positions': f['positions'][:],
                'masses': f['masses'][:]
            }
    elif modification == 'replace' and replace_file is not None:
        logger.info(f"  Loading hydro-replaced particles from {replace_file}")
        with h5py.File(replace_file, 'r') as f:
            particles = {
                'positions': f['positions'][:],
                'masses': f['masses'][:]
            }
    else:
        logger.info(f"  Loading raw simulation particles from {sim_path}")
        # Load all particle types and combine for total matter
        particles = sim.load_all_particles(snapshot)
    
    logger.info(f"  Gridding {len(particles['positions'])} particles to {grid_size}³ grid")
    density = grid_particles(particles, box_size, grid_size)
    
    return density


def main():
    """Main ray-tracing pipeline."""
    
    parser = argparse.ArgumentParser(description="Ray-tracing for convergence maps")
    parser.add_argument("--snapshot", type=int, default=99, help="Snapshot number")
    parser.add_argument("--n_realizations", type=int, default=10, 
                        help="Number of realizations per configuration")
    parser.add_argument("--configurations", nargs='+', 
                        default=['DMO', 'Hydro', 'BCM', 'Replace'],
                        help="Which configurations to ray-trace")
    parser.add_argument("--grid_size", type=int, default=1024,
                        help="Grid size for density field")
    args = parser.parse_args()
    
    if rank == 0:
        logger.info("="*60)
        logger.info("Ray-Tracing Pipeline")
        logger.info("="*60)
    
    # Load configuration
    sim_config, analysis_config, raytrace_config = load_config()
    
    # Output directory
    output_dir = Path(raytrace_config['output']['map_dir'])
    ensure_dir(output_dir)
    
    # Initialize ray-tracing configuration
    rt_cfg = RayTraceConfig(
        fov_deg=raytrace_config['field_of_view_deg'],
        n_pixels=int(raytrace_config['field_of_view_deg'] * 60 / 
                    raytrace_config['pixel_scale_arcmin']),
        source_z=raytrace_config['source_redshift'],
        n_lens_planes=raytrace_config['n_lens_planes'],
        add_shape_noise=raytrace_config.get('add_shape_noise', True),
        shape_noise_sigma=raytrace_config.get('shape_noise_sigma', 0.26),
    )
    
    if rank == 0:
        logger.info(f"Ray-tracing configuration:")
        logger.info(f"  FOV: {rt_cfg.fov_deg}° x {rt_cfg.fov_deg}°")
        logger.info(f"  Pixels: {rt_cfg.n_pixels} x {rt_cfg.n_pixels}")
        logger.info(f"  Source z: {rt_cfg.source_z}")
        logger.info(f"  Lens planes: {rt_cfg.n_lens_planes}")
        logger.info(f"  Shape noise: σ_ε = {rt_cfg.shape_noise_sigma}")
    
    # Initialize lux interface
    lux_path = raytrace_config.get('lux_executable', '/mnt/home/mlee1/lux/lux')
    lux = LuxInterface(executable_path=lux_path)
    
    # Generate projections
    projections = generate_random_projections(args.n_realizations, seed=42)
    
    # Simulation paths
    paths = {
        'DMO': sim_config['tng300']['dmo_path'],
        'Hydro': sim_config['tng300']['hydro_path'],
    }
    
    # Data product paths (from earlier pipeline steps)
    data_dir = Path(analysis_config['output']['base_dir'])
    bcm_particles = data_dir / "BCM_modified_particles.h5"
    replace_particles = data_dir / "replaced_snapshot.h5"
    
    # Process each configuration
    for config_name in args.configurations:
        if rank == 0:
            logger.info(f"\n{'='*40}")
            logger.info(f"Configuration: {config_name}")
            logger.info(f"{'='*40}")
        
        # Load matter field
        if config_name == 'DMO':
            density = load_matter_field(
                paths['DMO'], args.snapshot, args.grid_size
            )
        elif config_name == 'Hydro':
            density = load_matter_field(
                paths['Hydro'], args.snapshot, args.grid_size
            )
        elif config_name == 'BCM':
            density = load_matter_field(
                paths['DMO'], args.snapshot, args.grid_size,
                modification='bcm', bcm_file=bcm_particles
            )
        elif config_name == 'Replace':
            density = load_matter_field(
                paths['DMO'], args.snapshot, args.grid_size,
                modification='replace', replace_file=replace_particles
            )
        else:
            logger.warning(f"Unknown configuration: {config_name}, skipping")
            continue
        
        # Distribute realizations across MPI ranks
        local_projections = distribute_items(projections, rank, size)
        
        if rank == 0:
            logger.info(f"Processing {args.n_realizations} realizations...")
        
        # Ray-trace each realization
        for proj in local_projections:
            output_file = output_dir / f"convergence_maps_{config_name}_r{proj['id']:02d}.fits"
            
            run_raytracing(
                density, rt_cfg, proj, output_file, lux
            )
        
        # Synchronize
        if comm is not None:
            comm.Barrier()
    
    if rank == 0:
        logger.info("\n" + "="*60)
        logger.info("Ray-Tracing Complete")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Total maps: {len(args.configurations) * args.n_realizations}")
        logger.info("="*60)


if __name__ == "__main__":
    main()
