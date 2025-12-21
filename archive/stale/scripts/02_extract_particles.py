#!/usr/bin/env python
"""
02_extract_particles.py
=======================

Stage 2: Particle Extraction

Extract particles around matched halos at multiple radius multipliers.

Usage
-----
    mpirun -np 16 python 02_extract_particles.py --config config/analysis_params.yaml

Output
------
    - extracted_halos/halo_{id}_{radius_mult}R200.h5: Extracted particle data
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
from src.hydro_replace.data import ParticleExtractor, ExtractedHalo
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
        description="Stage 2: Particle Extraction"
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
        help='Input directory with matched catalog',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for extracted particles',
    )
    parser.add_argument(
        '--radius-mult',
        type=float,
        nargs='+',
        default=[1.0, 3.0, 5.0],
        help='Radius multipliers for extraction',
    )
    parser.add_argument(
        '--max-halos',
        type=int,
        default=None,
        help='Maximum number of halos to process (for testing)',
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging level',
    )
    return parser.parse_args()


def main():
    """Main function for particle extraction."""
    args = parse_args()
    
    setup_logging(level=args.log_level, include_rank=True)
    comm = get_mpi_comm()
    
    if is_root():
        logger.info("=" * 60)
        logger.info("Stage 2: Particle Extraction")
        logger.info("=" * 60)
    
    # Load configurations
    with open(args.sim_config, 'r') as f:
        sim_config = yaml.safe_load(f)
    
    with open(args.config, 'r') as f:
        analysis_config = yaml.safe_load(f)
    
    # Set paths
    base_output = Path(sim_config['output']['base_dir'])
    input_dir = Path(args.input_dir) if args.input_dir else base_output / 'matching'
    output_dir = Path(args.output_dir) if args.output_dir else base_output / 'extracted_halos'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_root():
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Radius multipliers: {args.radius_mult}")
    
    # Load matched catalog
    matched_data = load_hdf5(input_dir / 'matched_catalog.h5')
    hydro_ids = matched_data['hydro_ids']
    dmo_ids = matched_data['dmo_ids']
    
    # Load halo catalogs
    hydro_cat = load_hdf5(input_dir / 'halo_catalog_hydro.h5')
    dmo_cat = load_hdf5(input_dir / 'halo_catalog_dmo.h5')
    
    # Limit halos if requested
    n_halos = len(hydro_ids)
    if args.max_halos is not None:
        n_halos = min(n_halos, args.max_halos)
    
    if is_root():
        logger.info(f"Processing {n_halos} matched halo pairs")
    
    barrier()
    
    # Initialize simulation loaders
    sim_hydro = SimulationData(
        base_path=sim_config['simulations']['tng300']['hydro'],
        snapshot=99,
        is_dmo=False,
    )
    
    sim_dmo = SimulationData(
        base_path=sim_config['simulations']['tng300']['dmo'],
        snapshot=99,
        is_dmo=True,
    )
    
    box_size = sim_config['simulations']['tng300']['box_size']
    
    # Initialize extractors
    extractor_hydro = ParticleExtractor(sim_hydro, box_size)
    extractor_dmo = ParticleExtractor(sim_dmo, box_size)
    
    # Distribute halos across ranks
    halo_indices = list(range(n_halos))
    local_indices = distribute_items(halo_indices, comm)
    
    if is_root():
        logger.info(f"Distributed {n_halos} halos across ranks")
    
    # Process local halos
    local_results = []
    
    for i, idx in enumerate(local_indices):
        hid_hydro = hydro_ids[idx]
        hid_dmo = dmo_ids[idx]
        
        # Get halo properties
        hydro_idx = np.where(hydro_cat['halo_ids'] == hid_hydro)[0][0]
        dmo_idx = np.where(dmo_cat['halo_ids'] == hid_dmo)[0][0]
        
        center_hydro = hydro_cat['positions'][hydro_idx]
        r200c_hydro = hydro_cat['r200c'][hydro_idx]
        m200c_hydro = hydro_cat['m200c'][hydro_idx]
        
        center_dmo = dmo_cat['positions'][dmo_idx]
        r200c_dmo = dmo_cat['r200c'][dmo_idx]
        
        if (i + 1) % 10 == 0:
            logger.debug(f"Processing halo {i+1}/{len(local_indices)}: M={m200c_hydro:.2e}")
        
        # Extract at each radius multiplier
        for r_mult in args.radius_mult:
            try:
                # Extract hydro particles
                extracted_hydro = extractor_hydro.extract_particles(
                    center=center_hydro,
                    radius=r200c_hydro * r_mult,
                    particle_types=['dm', 'gas', 'stars'],
                )
                
                # Extract DMO particles
                extracted_dmo = extractor_dmo.extract_particles(
                    center=center_dmo,
                    radius=r200c_dmo * r_mult,
                    particle_types=['dm'],
                )
                
                # Save to HDF5
                output_file = output_dir / f'halo_{hid_hydro:06d}_{r_mult:.0f}R200.h5'
                
                save_hdf5(
                    output_file,
                    {
                        # Hydro particles
                        'hydro/dm/coords': extracted_hydro.coords_dm,
                        'hydro/dm/masses': extracted_hydro.masses_dm,
                        'hydro/gas/coords': extracted_hydro.coords_gas,
                        'hydro/gas/masses': extracted_hydro.masses_gas,
                        'hydro/stars/coords': extracted_hydro.coords_stars,
                        'hydro/stars/masses': extracted_hydro.masses_stars,
                        # DMO particles
                        'dmo/dm/coords': extracted_dmo.coords_dm,
                        'dmo/dm/masses': extracted_dmo.masses_dm,
                    },
                    attrs={
                        'halo_id_hydro': hid_hydro,
                        'halo_id_dmo': hid_dmo,
                        'center_hydro': center_hydro,
                        'center_dmo': center_dmo,
                        'r200c_hydro': r200c_hydro,
                        'r200c_dmo': r200c_dmo,
                        'm200c_hydro': m200c_hydro,
                        'radius_multiplier': r_mult,
                        'extraction_radius': r200c_hydro * r_mult,
                    }
                )
                
                local_results.append({
                    'halo_id': hid_hydro,
                    'r_mult': r_mult,
                    'n_dm_hydro': len(extracted_hydro.coords_dm),
                    'n_gas': len(extracted_hydro.coords_gas),
                    'n_stars': len(extracted_hydro.coords_stars),
                    'n_dm_dmo': len(extracted_dmo.coords_dm),
                    'success': True,
                })
                
            except Exception as e:
                logger.warning(f"Failed to extract halo {hid_hydro}: {e}")
                local_results.append({
                    'halo_id': hid_hydro,
                    'r_mult': r_mult,
                    'success': False,
                    'error': str(e),
                })
    
    barrier()
    
    # Gather results
    all_results = gather_list(local_results, comm)
    
    if is_root():
        n_success = sum(1 for r in all_results if r.get('success', False))
        n_total = len(all_results)
        
        logger.info(f"Successfully extracted {n_success}/{n_total} halo-radius combinations")
        logger.info("Stage 2 complete!")
        logger.info(f"Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
