#!/usr/bin/env python
"""
01_prepare_data.py
==================

Stage 1: Data Preparation

Load simulation data and perform bijective matching between hydro and DMO halos.

Usage
-----
    mpirun -np 16 python 01_prepare_data.py --config config/simulation_paths.yaml

Output
------
    - matched_catalog.h5: Bijectively matched halo pairs
    - halo_catalog_hydro.h5: Filtered hydro halo catalog
    - halo_catalog_dmo.h5: Filtered DMO halo catalog
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hydro_replace import (
    SimulationData,
    HaloCatalog,
    BijectiveMatcher,
)
from src.hydro_replace.utils import (
    setup_logging,
    is_root,
    barrier,
    save_hdf5,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Data Preparation and Bijective Matching"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/simulation_paths.yaml',
        help='Path to simulation configuration file',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides config)',
    )
    parser.add_argument(
        '--min-mass',
        type=float,
        default=1e12,
        help='Minimum halo mass in Msun/h',
    )
    parser.add_argument(
        '--max-mass',
        type=float,
        default=1e15,
        help='Maximum halo mass in Msun/h',
    )
    parser.add_argument(
        '--snapshot',
        type=int,
        default=99,
        help='Snapshot number',
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level',
    )
    return parser.parse_args()


def main():
    """Main function for data preparation."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, include_rank=True)
    
    if is_root():
        logger.info("=" * 60)
        logger.info("Stage 1: Data Preparation")
        logger.info("=" * 60)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine output directory
    output_dir = Path(args.output_dir or config['output']['base_dir'])
    matching_dir = output_dir / 'matching'
    matching_dir.mkdir(parents=True, exist_ok=True)
    
    if is_root():
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Mass range: {args.min_mass:.2e} - {args.max_mass:.2e} Msun/h")
    
    # Initialize simulation data loaders
    if is_root():
        logger.info("Loading simulation data...")
    
    sim_hydro = SimulationData(
        base_path=config['simulations']['tng300']['hydro'],
        snapshot=args.snapshot,
        is_dmo=False,
    )
    
    sim_dmo = SimulationData(
        base_path=config['simulations']['tng300']['dmo'],
        snapshot=args.snapshot,
        is_dmo=True,
    )
    
    # Load and filter halo catalogs
    if is_root():
        logger.info("Loading halo catalogs...")
    
    cat_hydro = HaloCatalog.from_illustris(
        sim_hydro.base_path,
        sim_hydro.snapshot,
    )
    cat_hydro_filtered = cat_hydro.filter_by_mass(args.min_mass, args.max_mass)
    
    cat_dmo = HaloCatalog.from_illustris(
        sim_dmo.base_path,
        sim_dmo.snapshot,
    )
    cat_dmo_filtered = cat_dmo.filter_by_mass(args.min_mass, args.max_mass)
    
    if is_root():
        logger.info(f"Hydro halos: {cat_hydro_filtered.n_halos} in mass range")
        logger.info(f"DMO halos: {cat_dmo_filtered.n_halos} in mass range")
    
    barrier()
    
    # Perform bijective matching
    if is_root():
        logger.info("Performing bijective matching...")
    
    matcher = BijectiveMatcher(
        box_size=config['simulations']['tng300']['box_size'],
        dm_mass_hydro=config['simulations']['tng300']['dm_particle_mass_hydro'],
        dm_mass_dmo=config['simulations']['tng300']['dm_particle_mass_dmo'],
    )
    
    # Load most-bound particle IDs
    mbp_hydro = sim_hydro.load_halo_most_bound_ids(cat_hydro_filtered.halo_ids)
    mbp_dmo = sim_dmo.load_halo_most_bound_ids(cat_dmo_filtered.halo_ids)
    
    # Perform matching
    matched = matcher.match(
        halo_ids_hydro=cat_hydro_filtered.halo_ids,
        mbp_ids_hydro=mbp_hydro,
        halo_ids_dmo=cat_dmo_filtered.halo_ids,
        mbp_ids_dmo=mbp_dmo,
    )
    
    if is_root():
        logger.info(f"Matched {matched.n_matched} halo pairs")
        logger.info(f"Match fraction: {matched.n_matched / cat_hydro_filtered.n_halos:.1%}")
    
    barrier()
    
    # Save results
    if is_root():
        logger.info("Saving results...")
        
        # Save matched catalog
        matched.save_hdf5(matching_dir / 'matched_catalog.h5')
        
        # Save filtered halo catalogs with additional info
        save_hdf5(
            matching_dir / 'halo_catalog_hydro.h5',
            {
                'halo_ids': cat_hydro_filtered.halo_ids,
                'm200c': cat_hydro_filtered.m200c,
                'r200c': cat_hydro_filtered.r200c,
                'positions': cat_hydro_filtered.positions,
            },
            attrs={
                'n_halos': cat_hydro_filtered.n_halos,
                'mass_min': args.min_mass,
                'mass_max': args.max_mass,
                'snapshot': args.snapshot,
                'simulation': 'TNG300_hydro',
            }
        )
        
        save_hdf5(
            matching_dir / 'halo_catalog_dmo.h5',
            {
                'halo_ids': cat_dmo_filtered.halo_ids,
                'm200c': cat_dmo_filtered.m200c,
                'r200c': cat_dmo_filtered.r200c,
                'positions': cat_dmo_filtered.positions,
            },
            attrs={
                'n_halos': cat_dmo_filtered.n_halos,
                'mass_min': args.min_mass,
                'mass_max': args.max_mass,
                'snapshot': args.snapshot,
                'simulation': 'TNG300_DMO',
            }
        )
        
        logger.info("Stage 1 complete!")
        logger.info(f"Outputs saved to: {matching_dir}")


if __name__ == '__main__':
    main()
