#!/usr/bin/env python
"""
06_compute_power.py
===================

Stage 6: Power Spectrum Computation

Compute 3D matter power spectra for hydro, DMO, and replaced fields.

Usage
-----
    python 06_compute_power.py --config config/analysis_params.yaml

Output
------
    - power_spectra/pk_{field}_{config}.h5: Power spectrum data
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
from src.hydro_replace.analysis import (
    compute_power_spectrum,
    compute_suppression,
    PowerSpectrum,
    PowerSpectrumAnalyzer,
)
from src.hydro_replace.utils import (
    setup_logging,
    is_root,
    barrier,
    load_hdf5,
    save_hdf5,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 6: Power Spectrum Computation"
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
        '--replaced-dir',
        type=str,
        default=None,
        help='Directory with replaced fields',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for power spectra',
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=1024,
        help='Grid size for FFT',
    )
    parser.add_argument(
        '--mas',
        type=str,
        default='CIC',
        choices=['NGP', 'CIC', 'TSC', 'PCS'],
        help='Mass assignment scheme',
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        help='Number of OpenMP threads',
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging level',
    )
    return parser.parse_args()


def main():
    """Main function for power spectrum computation."""
    args = parse_args()
    
    setup_logging(level=args.log_level)
    
    logger.info("=" * 60)
    logger.info("Stage 6: Power Spectrum Computation")
    logger.info("=" * 60)
    
    # Load configurations
    with open(args.sim_config, 'r') as f:
        sim_config = yaml.safe_load(f)
    
    with open(args.config, 'r') as f:
        analysis_config = yaml.safe_load(f)
    
    # Set paths
    base_output = Path(sim_config['output']['base_dir'])
    replaced_dir = Path(args.replaced_dir) if args.replaced_dir else base_output / 'replaced_fields'
    output_dir = Path(args.output_dir) if args.output_dir else base_output / 'power_spectra'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Replaced fields directory: {replaced_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Grid size: {args.grid_size}")
    logger.info(f"MAS: {args.mas}")
    
    box_size = sim_config['simulations']['tng300']['box_size']
    
    # Compute DMO power spectrum
    logger.info("\n--- Computing DMO power spectrum ---")
    
    sim_dmo = SimulationData(
        base_path=sim_config['simulations']['tng300']['dmo'],
        snapshot=99,
        is_dmo=True,
    )
    
    coords_dmo = sim_dmo.load_particles('dm', ['Coordinates'])['Coordinates']
    masses_dmo = sim_dmo.load_particles('dm', ['Masses'])['Masses']
    
    # Convert units
    coords_dmo *= sim_config['units']['length']
    masses_dmo *= sim_config['units']['mass']
    
    logger.info(f"Loaded {len(coords_dmo)} DMO particles")
    
    pk_dmo = compute_power_spectrum(
        coords_dmo,
        masses_dmo,
        box_size,
        grid_size=args.grid_size,
        mas=args.mas,
        threads=args.threads,
        label='DMO',
    )
    
    pk_dmo.save_hdf5(output_dir / 'pk_dmo.h5')
    logger.info(f"Saved DMO power spectrum")
    
    # Compute Hydro power spectrum
    logger.info("\n--- Computing Hydro power spectrum ---")
    
    sim_hydro = SimulationData(
        base_path=sim_config['simulations']['tng300']['hydro'],
        snapshot=99,
        is_dmo=False,
    )
    
    # Load all particle types
    coords_list = []
    masses_list = []
    
    for ptype in ['dm', 'gas', 'stars']:
        try:
            coords = sim_hydro.load_particles(ptype, ['Coordinates'])['Coordinates']
            masses = sim_hydro.load_particles(ptype, ['Masses'])['Masses']
            
            coords *= sim_config['units']['length']
            masses *= sim_config['units']['mass']
            
            coords_list.append(coords)
            masses_list.append(masses)
            
            logger.info(f"Loaded {len(coords)} {ptype} particles")
        except Exception as e:
            logger.warning(f"Could not load {ptype} particles: {e}")
    
    coords_hydro = np.vstack(coords_list)
    masses_hydro = np.concatenate(masses_list)
    
    pk_hydro = compute_power_spectrum(
        coords_hydro,
        masses_hydro,
        box_size,
        grid_size=args.grid_size,
        mas=args.mas,
        threads=args.threads,
        label='Hydro',
    )
    
    pk_hydro.save_hdf5(output_dir / 'pk_hydro.h5')
    logger.info(f"Saved Hydro power spectrum")
    
    # Compute suppression
    k, suppression_hydro = compute_suppression(pk_hydro, pk_dmo)
    
    logger.info(f"Hydro/DMO suppression at k=1: {suppression_hydro[np.argmin(np.abs(k-1))]:.4f}")
    
    # Clear memory
    del coords_dmo, masses_dmo, coords_hydro, masses_hydro
    
    # Process replaced fields
    logger.info("\n--- Computing replaced field power spectra ---")
    
    replaced_files = sorted(replaced_dir.glob('replaced_*.h5'))
    
    if not replaced_files:
        logger.warning(f"No replaced field files found in {replaced_dir}")
    
    suppressions = {'k': k, 'hydro': suppression_hydro}
    
    for replaced_file in replaced_files:
        logger.info(f"\nProcessing: {replaced_file.name}")
        
        # Load replaced field
        data = load_hdf5(replaced_file)
        coords = data['coords']
        masses = data['masses']
        
        label = replaced_file.stem.replace('replaced_', '')
        
        # Compute power spectrum
        pk_replaced = compute_power_spectrum(
            coords,
            masses,
            box_size,
            grid_size=args.grid_size,
            mas=args.mas,
            threads=args.threads,
            label=label,
        )
        
        # Save
        pk_replaced.save_hdf5(output_dir / f'pk_{label}.h5')
        
        # Compute suppression relative to DMO
        _, suppression = compute_suppression(pk_replaced, pk_dmo)
        suppressions[label] = suppression
        
        # Compare to hydro
        ratio = suppression / suppression_hydro
        
        logger.info(f"  Suppression at k=1: {suppression[np.argmin(np.abs(k-1))]:.4f}")
        logger.info(f"  Ratio to Hydro at k=1: {ratio[np.argmin(np.abs(k-1))]:.4f}")
    
    # Save all suppressions
    save_hdf5(
        output_dir / 'suppressions_all.h5',
        suppressions,
        attrs={
            'grid_size': args.grid_size,
            'mas': args.mas,
            'box_size': box_size,
        }
    )
    
    logger.info("\nStage 6 complete!")
    logger.info(f"Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
