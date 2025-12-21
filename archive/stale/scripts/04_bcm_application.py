#!/usr/bin/env python
"""
04_bcm_application.py - Apply Baryonic Correction Model to DMO Halos

This script applies the Arico+2020 BCM (via BaryonForge) to TNG-300-Dark halos
to generate BCM-modified profiles and particle distributions for comparison
with hydro simulations and hydro replacement.

Usage:
    mpirun -np 16 python 04_bcm_application.py
    
Outputs:
    - halo_profiles_BCM.h5: BCM-modified profiles for all halos
    - BCM_modified_particles.h5: Modified particle positions/masses

Dependencies:
    - BaryonForge (for Arico BCM implementation)
    - hydro_replace package (this project)
"""

import os
import sys
import numpy as np
import h5py
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hydro_replace.utils.logging_setup import setup_logging
from hydro_replace.utils.parallel import get_mpi_info, distribute_items, gather_arrays
from hydro_replace.utils.io_helpers import save_hdf5, load_hdf5, ensure_dir
from hydro_replace.data.load_simulations import SimulationData
from hydro_replace.data.halo_catalogs import HaloCatalog
from hydro_replace.bcm.arico_bcm import AricoBCM, BCMParameters
from hydro_replace.analysis.profiles import ProfileAnalyzer, DensityProfile

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

logger = setup_logging("04_bcm_application", rank=rank)


def load_config():
    """Load simulation and analysis configuration."""
    config_dir = project_root / "config"
    
    with open(config_dir / "simulation_paths.yaml") as f:
        sim_config = yaml.safe_load(f)
    
    with open(config_dir / "analysis_params.yaml") as f:
        analysis_config = yaml.safe_load(f)
    
    return sim_config, analysis_config


def compute_bcm_profile(bcm: AricoBCM, halo_mass: float, halo_radius: float,
                        radial_bins: np.ndarray, concentration: float = None) -> np.ndarray:
    """
    Compute BCM-modified density profile for a single halo.
    
    Parameters
    ----------
    bcm : AricoBCM
        BCM model instance
    halo_mass : float
        M_200c in M_sun/h
    halo_radius : float
        R_200c in Mpc/h
    radial_bins : np.ndarray
        Radii at which to compute profile (in R_200c units)
    concentration : float, optional
        NFW concentration (computed from mass-concentration relation if None)
    
    Returns
    -------
    rho_bcm : np.ndarray
        BCM-modified density profile in M_sun/h / (Mpc/h)^3
    """
    # Convert radii to physical units
    r_physical = radial_bins * halo_radius  # Mpc/h
    
    # Get BCM profile from BaryonForge
    try:
        rho_bcm = bcm.compute_modified_profile(
            r=r_physical,
            M_200c=halo_mass,
            R_200c=halo_radius,
            concentration=concentration
        )
    except Exception as e:
        logger.warning(f"BCM computation failed for M={halo_mass:.2e}: {e}")
        rho_bcm = np.full_like(radial_bins, np.nan)
    
    return rho_bcm


def apply_bcm_to_particles(bcm: AricoBCM, particles: dict, halo_center: np.ndarray,
                           halo_mass: float, halo_radius: float, 
                           box_size: float) -> dict:
    """
    Apply BCM displacement to DM particles around a halo.
    
    This modifies particle positions to mimic the baryonic effects predicted
    by the BCM model (contraction in core, expansion in outskirts).
    
    Parameters
    ----------
    bcm : AricoBCM
        BCM model instance
    particles : dict
        Dictionary with 'positions', 'masses', 'ids' arrays
    halo_center : np.ndarray
        (3,) array of halo center coordinates
    halo_mass : float
        M_200c in M_sun/h
    halo_radius : float
        R_200c in Mpc/h
    box_size : float
        Simulation box size in Mpc/h
    
    Returns
    -------
    modified_particles : dict
        Dictionary with modified particle properties
    """
    from hydro_replace.utils.periodic_boundary import periodic_distance_vector
    
    positions = particles['positions'].copy()
    masses = particles['masses'].copy()
    
    # Compute distances from halo center (periodic)
    delta = periodic_distance_vector(positions, halo_center, box_size)
    r = np.linalg.norm(delta, axis=1)
    
    # Normalize by R_200c
    r_norm = r / halo_radius
    
    # Get BCM radial displacement factor
    # dr/r = (r_bcm - r_dmo) / r_dmo
    try:
        displacement_factor = bcm.compute_radial_displacement(
            r_norm=r_norm,
            M_200c=halo_mass,
            R_200c=halo_radius
        )
    except Exception as e:
        logger.warning(f"Displacement computation failed: {e}")
        displacement_factor = np.zeros_like(r)
    
    # Apply displacement: r_new = r * (1 + dr/r)
    # Direction is radial (outward for positive displacement)
    mask = r > 0  # Avoid division by zero
    new_positions = positions.copy()
    
    if np.any(mask):
        unit_vectors = delta[mask] / r[mask, np.newaxis]
        radial_displacement = displacement_factor[mask] * r[mask]
        new_positions[mask] += radial_displacement[:, np.newaxis] * unit_vectors
    
    # Apply periodic boundaries
    new_positions = new_positions % box_size
    
    return {
        'positions': new_positions,
        'masses': masses,
        'ids': particles['ids'],
        'original_r': r,
        'displacement': displacement_factor * r
    }


def main():
    """Main BCM application pipeline."""
    
    if rank == 0:
        logger.info("="*60)
        logger.info("BCM Application Pipeline")
        logger.info("="*60)
    
    # Load configuration
    sim_config, analysis_config = load_config()
    
    # Output directory
    output_dir = Path(analysis_config['output']['base_dir'])
    ensure_dir(output_dir)
    
    # Initialize BCM model
    if rank == 0:
        logger.info("Initializing BCM model with TNG-calibrated parameters...")
    
    bcm_params = BCMParameters(
        M_c=1e14,           # Characteristic mass
        beta=0.6,           # Power-law slope
        mu=1.0,             # Contraction parameter
        theta_ej=4.0,       # Ejection radius in R_200c
        M_star_frac=0.03,   # Stellar mass fraction
        f_gas=0.15,         # Gas fraction normalization
    )
    bcm = AricoBCM(params=bcm_params)
    
    # Load DMO halo catalog
    if rank == 0:
        logger.info("Loading TNG-300-Dark halo catalog...")
    
    dmo_path = sim_config['tng300']['dmo_path']
    snapshot = analysis_config['snapshot']['default']
    
    sim_dmo = SimulationData(dmo_path)
    catalog_dmo = HaloCatalog(sim_dmo, snapshot=snapshot)
    
    # Get halos above mass threshold
    min_mass = analysis_config['mass_bins']['log_min']  # log10(M)
    mass_mask = catalog_dmo.M_200c >= 10**min_mass
    
    halo_ids = np.where(mass_mask)[0]
    n_halos = len(halo_ids)
    
    if rank == 0:
        logger.info(f"Processing {n_halos} halos with M > 10^{min_mass} M_sun/h")
    
    # Distribute halos across MPI ranks
    local_halo_ids = distribute_items(halo_ids, rank, size)
    n_local = len(local_halo_ids)
    
    if rank == 0:
        logger.info(f"Each rank processing ~{n_local} halos")
    
    # Radial bins for profiles
    r_min = analysis_config['profiles']['r_min_R200c']
    r_max = analysis_config['profiles']['r_max_R200c']
    n_bins = analysis_config['profiles']['n_radial_bins']
    radial_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins)
    
    # Storage for results
    local_profiles = []
    local_masses = []
    local_radii = []
    local_ids = []
    
    # Process each halo
    for i, halo_idx in enumerate(local_halo_ids):
        if i % 100 == 0:
            logger.info(f"Rank {rank}: Processing halo {i+1}/{n_local}")
        
        # Get halo properties
        M_200c = catalog_dmo.M_200c[halo_idx]  # M_sun/h
        R_200c = catalog_dmo.R_200c[halo_idx]  # Mpc/h
        
        # Compute BCM profile
        rho_bcm = compute_bcm_profile(
            bcm, M_200c, R_200c, radial_bins
        )
        
        local_profiles.append(rho_bcm)
        local_masses.append(M_200c)
        local_radii.append(R_200c)
        local_ids.append(halo_idx)
    
    # Convert to arrays
    local_profiles = np.array(local_profiles)
    local_masses = np.array(local_masses)
    local_radii = np.array(local_radii)
    local_ids = np.array(local_ids)
    
    # Gather all results to rank 0
    if comm is not None:
        all_profiles = gather_arrays(local_profiles, comm)
        all_masses = gather_arrays(local_masses, comm)
        all_radii = gather_arrays(local_radii, comm)
        all_ids = gather_arrays(local_ids, comm)
    else:
        all_profiles = local_profiles
        all_masses = local_masses
        all_radii = local_radii
        all_ids = local_ids
    
    # Save results on rank 0
    if rank == 0:
        logger.info("Saving BCM profiles...")
        
        output_file = output_dir / "halo_profiles_BCM.h5"
        
        with h5py.File(output_file, 'w') as f:
            # Metadata
            f.attrs['n_halos'] = len(all_ids)
            f.attrs['bcm_model'] = 'Arico+2020'
            f.attrs['simulation'] = 'TNG-300-Dark'
            f.attrs['snapshot'] = snapshot
            
            # BCM parameters
            bcm_grp = f.create_group('bcm_parameters')
            bcm_grp.attrs['M_c'] = bcm_params.M_c
            bcm_grp.attrs['beta'] = bcm_params.beta
            bcm_grp.attrs['mu'] = bcm_params.mu
            bcm_grp.attrs['theta_ej'] = bcm_params.theta_ej
            
            # Radial bins
            f.create_dataset('radial_bins_R200c', data=radial_bins)
            
            # Halo data
            f.create_dataset('halo_ids', data=all_ids)
            f.create_dataset('M_200c', data=all_masses)
            f.create_dataset('R_200c', data=all_radii)
            
            # Profiles (n_halos, n_radial_bins)
            f.create_dataset('rho_bcm', data=all_profiles)
        
        logger.info(f"BCM profiles saved to {output_file}")
        
        # Summary statistics
        valid_profiles = ~np.isnan(all_profiles).all(axis=1)
        n_valid = valid_profiles.sum()
        logger.info(f"Successfully computed {n_valid}/{len(all_ids)} profiles")
        
        logger.info("="*60)
        logger.info("BCM Application Complete")
        logger.info("="*60)


if __name__ == "__main__":
    main()
