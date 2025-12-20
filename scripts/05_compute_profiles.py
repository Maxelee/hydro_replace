#!/usr/bin/env python
"""
05_compute_profiles.py
======================

Stage 5: Profile Analysis

Compute density profiles for hydro, DMO, and replaced fields.

Usage
-----
    mpirun -np 16 python 05_compute_profiles.py --config config/analysis_params.yaml

Output
------
    - profiles/profiles_{config}.h5: Stacked density profiles
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hydro_replace.analysis import (
    compute_density_profile,
    ProfileAnalyzer,
    compute_stacked_profile,
)
from src.hydro_replace.utils import (
    setup_logging,
    is_root,
    barrier,
    distribute_items,
    gather_list,
    get_mpi_comm,
    load_hdf5,
    save_hdf5,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 5: Profile Analysis"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/analysis_params.yaml',
        help='Path to analysis configuration file',
    )
    parser.add_argument(
        '--sim-config',
        type=str,
        default='config/simulation_paths.yaml',
        help='Path to simulation configuration file',
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory with extracted particles',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for profiles',
    )
    parser.add_argument(
        '--n-bins',
        type=int,
        default=50,
        help='Number of radial bins',
    )
    parser.add_argument(
        '--r-min',
        type=float,
        default=0.01,
        help='Minimum radius in R/R200c',
    )
    parser.add_argument(
        '--r-max',
        type=float,
        default=5.0,
        help='Maximum radius in R/R200c',
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging level',
    )
    return parser.parse_args()


def main():
    """Main function for profile computation."""
    args = parse_args()
    
    setup_logging(level=args.log_level, include_rank=True)
    comm = get_mpi_comm()
    
    if is_root():
        logger.info("=" * 60)
        logger.info("Stage 5: Profile Analysis")
        logger.info("=" * 60)
    
    # Load configurations
    with open(args.sim_config, 'r') as f:
        sim_config = yaml.safe_load(f)
    
    with open(args.config, 'r') as f:
        analysis_config = yaml.safe_load(f)
    
    # Set paths
    base_output = Path(sim_config['output']['base_dir'])
    input_dir = Path(args.input_dir) if args.input_dir else base_output / 'extracted_halos'
    output_dir = Path(args.output_dir) if args.output_dir else base_output / 'profiles'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_root():
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
    
    # Load matched catalog
    matching_dir = base_output / 'matching'
    matched_data = load_hdf5(matching_dir / 'matched_catalog.h5')
    hydro_ids = matched_data['hydro_ids']
    
    hydro_cat = load_hdf5(matching_dir / 'halo_catalog_hydro.h5')
    
    box_size = sim_config['simulations']['tng300']['box_size']
    
    # Define radial bins (in units of R200c)
    r_bins_norm = np.logspace(np.log10(args.r_min), np.log10(args.r_max), args.n_bins + 1)
    r_centers_norm = np.sqrt(r_bins_norm[:-1] * r_bins_norm[1:])
    
    if is_root():
        logger.info(f"Radial bins: {args.n_bins} bins from {args.r_min} to {args.r_max} R/R200c")
    
    # Get mass bins
    mass_bins = analysis_config['mass_bins']['regular']
    
    # Process each mass bin
    for bin_idx in range(len(mass_bins) - 1):
        mass_min = mass_bins[bin_idx]
        mass_max = mass_bins[bin_idx + 1]
        bin_name = f"M{bin_idx}"
        
        if is_root():
            logger.info(f"\n--- Processing mass bin: {bin_name} ({mass_min:.2e} - {mass_max:.2e}) ---")
        
        # Filter halos by mass
        masses = hydro_cat['m200c']
        mask = (masses >= mass_min) & (masses < mass_max)
        halos_in_bin = hydro_ids[mask]
        r200c_in_bin = hydro_cat['r200c'][mask]
        m200c_in_bin = hydro_cat['m200c'][mask]
        
        if len(halos_in_bin) == 0:
            if is_root():
                logger.warning(f"No halos in mass bin {bin_name}")
            continue
        
        if is_root():
            logger.info(f"Found {len(halos_in_bin)} halos in mass bin")
        
        # Distribute halos
        local_indices = distribute_items(list(range(len(halos_in_bin))), comm)
        
        # Storage for profiles
        local_profiles_hydro = []
        local_profiles_dmo = []
        local_masses = []
        local_r200c = []
        
        for idx in local_indices:
            hid = halos_in_bin[idx]
            r200c = r200c_in_bin[idx]
            m200c = m200c_in_bin[idx]
            
            # Physical radii for this halo
            r_bins = r_bins_norm * r200c
            
            # Load extracted particles (using 5x R200c file)
            extracted_file = input_dir / f'halo_{hid:06d}_5R200.h5'
            
            if not extracted_file.exists():
                logger.debug(f"File not found: {extracted_file}")
                continue
            
            try:
                extracted = load_hdf5(extracted_file)
                center_hydro = extracted['center_hydro']
                center_dmo = extracted['center_dmo']
                
                # Hydro total matter (DM + gas + stars)
                coords_hydro = np.vstack([
                    extracted.get('hydro/dm/coords', np.empty((0, 3))),
                    extracted.get('hydro/gas/coords', np.empty((0, 3))),
                    extracted.get('hydro/stars/coords', np.empty((0, 3))),
                ])
                masses_hydro = np.concatenate([
                    extracted.get('hydro/dm/masses', np.array([])),
                    extracted.get('hydro/gas/masses', np.array([])),
                    extracted.get('hydro/stars/masses', np.array([])),
                ])
                
                # DMO particles
                coords_dmo = extracted['dmo/dm/coords']
                masses_dmo = extracted['dmo/dm/masses']
                
                # Compute profiles
                profile_hydro = compute_density_profile(
                    coords_hydro, masses_hydro, center_hydro, r_bins, box_size
                )
                profile_dmo = compute_density_profile(
                    coords_dmo, masses_dmo, center_dmo, r_bins, box_size
                )
                
                local_profiles_hydro.append(profile_hydro.density)
                local_profiles_dmo.append(profile_dmo.density)
                local_masses.append(m200c)
                local_r200c.append(r200c)
                
            except Exception as e:
                logger.warning(f"Failed to compute profile for halo {hid}: {e}")
                continue
        
        barrier()
        
        # Gather all profiles
        all_profiles_hydro = gather_list(local_profiles_hydro, comm)
        all_profiles_dmo = gather_list(local_profiles_dmo, comm)
        all_masses = gather_list(local_masses, comm)
        all_r200c = gather_list(local_r200c, comm)
        
        if is_root() and all_profiles_hydro:
            # Stack profiles
            profiles_hydro = np.array(all_profiles_hydro)
            profiles_dmo = np.array(all_profiles_dmo)
            
            # Compute mean and scatter
            mean_hydro = np.nanmean(profiles_hydro, axis=0)
            std_hydro = np.nanstd(profiles_hydro, axis=0)
            
            mean_dmo = np.nanmean(profiles_dmo, axis=0)
            std_dmo = np.nanstd(profiles_dmo, axis=0)
            
            # Compute suppression
            with np.errstate(divide='ignore', invalid='ignore'):
                suppression = mean_hydro / mean_dmo
            
            logger.info(f"Computed profiles for {len(all_profiles_hydro)} halos")
            
            # Save results
            output_file = output_dir / f'profiles_{bin_name}.h5'
            
            save_hdf5(
                output_file,
                {
                    # Radii
                    'r_bins_norm': r_bins_norm,
                    'r_centers_norm': r_centers_norm,
                    # Mean profiles
                    'rho_hydro_mean': mean_hydro,
                    'rho_hydro_std': std_hydro,
                    'rho_dmo_mean': mean_dmo,
                    'rho_dmo_std': std_dmo,
                    'suppression': suppression,
                    # Individual profiles
                    'rho_hydro_all': profiles_hydro,
                    'rho_dmo_all': profiles_dmo,
                    # Halo properties
                    'masses': np.array(all_masses),
                    'r200c': np.array(all_r200c),
                },
                attrs={
                    'mass_bin': bin_name,
                    'mass_min': mass_min,
                    'mass_max': mass_max,
                    'n_halos': len(all_profiles_hydro),
                    'n_bins': args.n_bins,
                    'r_min_norm': args.r_min,
                    'r_max_norm': args.r_max,
                }
            )
            
            logger.info(f"Saved: {output_file}")
        
        barrier()
    
    if is_root():
        logger.info("\nStage 5 complete!")
        logger.info(f"Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
