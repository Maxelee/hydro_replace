#!/usr/bin/env python
"""
03_halo_replacement.py
======================

Stage 3: Halo Replacement

Replace DMO particles with hydro particles in matched halos.

Usage
-----
    mpirun -np 16 python 03_halo_replacement.py --config config/analysis_params.yaml

Output
------
    - replaced_fields/replaced_{config}_{mass_bin}.h5: Replaced particle fields
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hydro_replace import SimulationData
from src.hydro_replace.replacement import HaloReplacer, MassBinConfig
from src.hydro_replace.utils import (
    setup_logging,
    is_root,
    barrier,
    get_mpi_comm,
    load_hdf5,
    save_hdf5,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 3: Halo Replacement"
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
        help='Output directory for replaced fields',
    )
    parser.add_argument(
        '--radius-mult',
        type=float,
        default=5.0,
        help='Radius multiplier for replacement',
    )
    parser.add_argument(
        '--mass-bins',
        type=str,
        default='regular',
        choices=['regular', 'cumulative', 'all'],
        help='Mass bin configuration',
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging level',
    )
    return parser.parse_args()


def main():
    """Main function for halo replacement."""
    args = parse_args()
    
    setup_logging(level=args.log_level, include_rank=True)
    comm = get_mpi_comm()
    
    if is_root():
        logger.info("=" * 60)
        logger.info("Stage 3: Halo Replacement")
        logger.info("=" * 60)
    
    # Load configurations
    with open(args.sim_config, 'r') as f:
        sim_config = yaml.safe_load(f)
    
    with open(args.config, 'r') as f:
        analysis_config = yaml.safe_load(f)
    
    # Set paths
    base_output = Path(sim_config['output']['base_dir'])
    input_dir = Path(args.input_dir) if args.input_dir else base_output / 'extracted_halos'
    output_dir = Path(args.output_dir) if args.output_dir else base_output / 'replaced_fields'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_root():
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
    
    # Load matched catalog
    matching_dir = base_output / 'matching'
    matched_data = load_hdf5(matching_dir / 'matched_catalog.h5')
    hydro_ids = matched_data['hydro_ids']
    dmo_ids = matched_data['dmo_ids']
    
    hydro_cat = load_hdf5(matching_dir / 'halo_catalog_hydro.h5')
    
    # Define mass bins
    mass_bin_config = analysis_config['mass_bins']
    
    if args.mass_bins == 'regular':
        mass_bins = mass_bin_config['regular']
        mass_bin_names = [f"M{i}" for i in range(len(mass_bins) - 1)]
    elif args.mass_bins == 'cumulative':
        mass_bins = mass_bin_config['cumulative']
        mass_bin_names = [f"Mcum{i}" for i in range(len(mass_bins) - 1)]
    else:
        # Both
        mass_bins = mass_bin_config['regular'] + mass_bin_config['cumulative']
        mass_bin_names = (
            [f"M{i}" for i in range(len(mass_bin_config['regular']) - 1)] +
            [f"Mcum{i}" for i in range(len(mass_bin_config['cumulative']) - 1)]
        )
    
    if is_root():
        logger.info(f"Using mass bins: {args.mass_bins}")
        logger.info(f"Number of bins: {len(mass_bin_names)}")
    
    box_size = sim_config['simulations']['tng300']['box_size']
    
    # Initialize DMO simulation for base particles
    sim_dmo = SimulationData(
        base_path=sim_config['simulations']['tng300']['dmo'],
        snapshot=99,
        is_dmo=True,
    )
    
    # Load full DMO particle data
    if is_root():
        logger.info("Loading DMO particle data...")
    
    coords_dmo_full = sim_dmo.load_particles('dm', ['Coordinates'])['Coordinates']
    masses_dmo_full = sim_dmo.load_particles('dm', ['Masses'])['Masses']
    
    # Convert units
    coords_dmo_full *= sim_config['units']['length']  # kpc -> Mpc
    masses_dmo_full *= sim_config['units']['mass']  # 1e10 Msun -> Msun
    
    if is_root():
        logger.info(f"Loaded {len(coords_dmo_full)} DMO particles")
    
    barrier()
    
    # Process each mass bin
    for bin_idx, bin_name in enumerate(mass_bin_names):
        if is_root():
            logger.info(f"\n--- Processing mass bin: {bin_name} ---")
        
        # Get mass range for this bin
        if bin_idx < len(mass_bin_config['regular']) - 1:
            mass_min = mass_bin_config['regular'][bin_idx]
            mass_max = mass_bin_config['regular'][bin_idx + 1]
        else:
            cum_idx = bin_idx - (len(mass_bin_config['regular']) - 1)
            mass_min = mass_bin_config['cumulative'][cum_idx]
            mass_max = mass_bin_config['cumulative'][cum_idx + 1]
        
        if is_root():
            logger.info(f"Mass range: {mass_min:.2e} - {mass_max:.2e} Msun/h")
        
        # Filter halos by mass
        masses = hydro_cat['m200c']
        mask = (masses >= mass_min) & (masses < mass_max)
        halos_in_bin = hydro_ids[mask]
        
        if len(halos_in_bin) == 0:
            if is_root():
                logger.warning(f"No halos in mass bin {bin_name}")
            continue
        
        if is_root():
            logger.info(f"Found {len(halos_in_bin)} halos in mass bin")
        
        # Initialize replacer
        replacer = HaloReplacer(
            coords_dmo=coords_dmo_full.copy(),
            masses_dmo=masses_dmo_full.copy(),
            box_size=box_size,
        )
        
        # Process each halo in this mass bin
        for hid in halos_in_bin:
            # Load extracted particles
            extracted_file = input_dir / f'halo_{hid:06d}_{args.radius_mult:.0f}R200.h5'
            
            if not extracted_file.exists():
                logger.warning(f"Extracted file not found: {extracted_file}")
                continue
            
            extracted = load_hdf5(extracted_file)
            
            center_hydro = extracted['center_hydro']
            center_dmo = extracted['center_dmo']
            radius = extracted['extraction_radius']
            
            # Get hydro particles (all baryonic + DM)
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
            
            # Perform replacement
            replacer.replace_halo(
                center_dmo=center_dmo,
                radius_dmo=radius,
                coords_hydro=coords_hydro,
                masses_hydro=masses_hydro,
                center_hydro=center_hydro,
            )
        
        # Get replaced field
        result = replacer.get_result()
        
        if is_root():
            logger.info(f"Replacement complete: {result.n_particles} particles")
            logger.info(f"Replaced {result.n_replaced} DMO particles")
            
            # Save result
            output_file = output_dir / f'replaced_R{args.radius_mult:.0f}_{bin_name}.h5'
            
            save_hdf5(
                output_file,
                {
                    'coords': result.coords,
                    'masses': result.masses,
                },
                attrs={
                    'mass_bin': bin_name,
                    'mass_min': mass_min,
                    'mass_max': mass_max,
                    'radius_multiplier': args.radius_mult,
                    'n_particles': result.n_particles,
                    'n_replaced': result.n_replaced,
                    'n_halos': len(halos_in_bin),
                }
            )
            
            logger.info(f"Saved: {output_file}")
        
        barrier()
    
    if is_root():
        logger.info("\nStage 3 complete!")
        logger.info(f"Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
