#!/usr/bin/env python3
"""
Compute halo matches between DMO and hydro simulations using MPI.

This script runs the BijectiveMatcher with MPI parallelization and saves
the results to an HDF5 file that can be loaded by the main pipeline.

Usage:
    # Single process (slow)
    python compute_halo_matches.py --resolution 625 --snapshot 99

    # MPI parallel (fast)
    srun -n 16 python compute_halo_matches.py --resolution 625 --snapshot 99
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Setup logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try MPI import
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1

# Only rank 0 logs
if rank != 0:
    logging.disable(logging.CRITICAL)


def main():
    parser = argparse.ArgumentParser(description='Compute halo matches with MPI')
    parser.add_argument('--resolution', type=int, default=625,
                        help='Simulation resolution (625, 1250, 2500)')
    parser.add_argument('--snapshot', type=int, default=99,
                        help='Snapshot number')
    parser.add_argument('--mass-min', type=float, default=1e13,
                        help='Minimum halo mass (Msun/h)')
    parser.add_argument('--output-dir', type=str, 
                        default='/mnt/home/mlee1/ceph/hydro_replace',
                        help='Output directory')
    args = parser.parse_args()

    # Simulation paths
    res = args.resolution
    dmo_basepath = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L205n{res}TNG_DM/output'
    hydro_basepath = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L205n{res}TNG/output'
    
    # Output path
    output_dir = Path(args.output_dir) / f'L205n{res}TNG' / 'matches'
    output_file = output_dir / f'matched_snap{args.snapshot:03d}.h5'
    
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Computing halo matches for L205n{res}TNG snapshot {args.snapshot}")
        logger.info(f"  DMO: {dmo_basepath}")
        logger.info(f"  Hydro: {hydro_basepath}")
        logger.info(f"  Mass min: {args.mass_min:.1e} Msun/h")
        logger.info(f"  Output: {output_file}")
        logger.info(f"  MPI ranks: {size}")
    
    # Check if already computed
    if output_file.exists():
        if rank == 0:
            logger.info(f"Matches already exist at {output_file}, skipping")
        return
    
    # Import matcher after MPI is set up
    from hydro_replace.data.bijective_matching import BijectiveMatcher
    
    # Create matcher
    matcher = BijectiveMatcher(
        dmo_basePath=dmo_basepath,
        hydro_basePath=hydro_basepath,
        snapNum=args.snapshot,
        min_halo_mass=args.mass_min,
        min_particles=50,
        mass_unit=1e10,
    )
    
    # Run matching
    matched = matcher.run()
    
    # Save results (only rank 0)
    if rank == 0:
        matched.save(str(output_file))
        logger.info(f"Saved {len(matched)} matched halos to {output_file}")
        
        # Print summary
        logger.info(f"Match summary:")
        logger.info(f"  Matched halos: {len(matched)}")
        logger.info(f"  Mass range: {matched.dmo_masses.min():.2e} - {matched.dmo_masses.max():.2e} Msun/h")


if __name__ == '__main__':
    main()
