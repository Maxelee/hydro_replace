#!/usr/bin/env python
"""
Unified Analysis Pipeline: Cache → Statistics → Profiles → Maps

This script orchestrates the full analysis workflow:
1. Generate particle cache (if not exists)
2. Compute halo statistics (baryon fractions, mass conservation)
3. Generate stacked density profiles
4. Generate 2D density maps (DMO, Hydro, Replace)

Usage:
    # Full pipeline for 625 resolution, single snapshot
    python run_full_analysis.py --sim-res 625 --snap 99 --mass-min 12.5

    # Process all snapshots
    python run_full_analysis.py --sim-res 625 --snap all --mass-min 12.5

    # Skip cache generation (use existing)
    python run_full_analysis.py --sim-res 625 --snap 99 --mass-min 12.5 --skip-cache

    # Only generate maps (skip stats/profiles)
    python run_full_analysis.py --sim-res 625 --snap 99 --mass-min 12.5 --maps-only
"""

import subprocess
import argparse
import os
import sys
import time

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = '/mnt/home/mlee1/hydro_replace2'
OUTPUT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields'

# MPI configuration per resolution
MPI_CONFIG = {
    625: {'nodes': 4, 'ntasks': 4},      # 4 files → 4 ranks
    1250: {'nodes': 8, 'ntasks': 8},     # 8 files → 8 ranks
    2500: {'nodes': 16, 'ntasks': 16},   # 16 files → 16 ranks
}

# Standard 20 snapshots for ray-tracing
RT_SNAPSHOTS = [29, 31, 33, 35, 38, 41, 43, 46, 49, 52, 
                56, 59, 63, 67, 71, 76, 80, 85, 90, 96]

# ============================================================================
# Helper Functions
# ============================================================================

def run_command(cmd, description):
    """Run a shell command and print output."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n⚠️  Command failed with return code {result.returncode}")
        return False
    return True


def check_cache_exists(sim_res, snap):
    """Check if particle cache exists for given snapshot."""
    cache_file = f'{OUTPUT_BASE}/L205n{sim_res}TNG/particle_cache/cache_snap{snap:03d}.h5'
    return os.path.exists(cache_file)


def check_stats_exists(sim_res, snap):
    """Check if statistics file exists."""
    stats_file = f'{OUTPUT_BASE}/L205n{sim_res}TNG/analysis/halo_statistics_snap{snap:03d}.h5'
    return os.path.exists(stats_file)


def check_maps_exist(sim_res, snap, mass_min):
    """Check if density maps exist."""
    map_dir = f'{OUTPUT_BASE}/L205n{sim_res}TNG/fields_snap{snap:03d}'
    replace_file = f'{map_dir}/replace_mass{mass_min:.1f}_axis2.npz'
    return os.path.exists(replace_file)


def get_mpi_command(sim_res, script, args_str):
    """Build MPI command for running scripts."""
    config = MPI_CONFIG[sim_res]
    return f"mpirun -np {config['ntasks']} python {BASE_DIR}/scripts/{script} {args_str}"


# ============================================================================
# Pipeline Steps
# ============================================================================

def step_generate_cache(args):
    """Generate particle ID cache for halos."""
    print("\n" + "="*70)
    print("STEP 1: Generate Particle Cache")
    print("="*70)
    
    for snap in args.snapshots:
        if check_cache_exists(args.sim_res, snap) and not args.force_cache:
            print(f"  ✓ Cache exists for snap {snap}, skipping")
            continue
        
        cmd = get_mpi_command(
            args.sim_res,
            'generate_particle_cache.py',
            f'--sim-res {args.sim_res} --snap {snap}'
        )
        if not run_command(cmd, f"Generating cache for snapshot {snap}"):
            return False
    return True


def step_compute_statistics(args):
    """Compute halo statistics (baryon fractions, mass conservation)."""
    print("\n" + "="*70)
    print("STEP 2: Compute Halo Statistics")
    print("="*70)
    
    for snap in args.snapshots:
        if check_stats_exists(args.sim_res, snap) and not args.force_stats:
            print(f"  ✓ Statistics exist for snap {snap}, skipping")
            continue
        
        cmd = get_mpi_command(
            args.sim_res,
            'compute_halo_statistics_distributed.py',
            f'--sim-res {args.sim_res} --snap {snap}'
        )
        if not run_command(cmd, f"Computing statistics for snapshot {snap}"):
            return False
    return True


def step_generate_profiles(args):
    """Generate stacked density profiles."""
    print("\n" + "="*70)
    print("STEP 3: Generate Density Profiles")
    print("="*70)
    
    for snap in args.snapshots:
        cmd = get_mpi_command(
            args.sim_res,
            'generate_profiles_cached_new.py',
            f'--sim-res {args.sim_res} --snap {snap} --mass-min {args.mass_min}'
        )
        if not run_command(cmd, f"Generating profiles for snapshot {snap}"):
            return False
    return True


def step_generate_maps(args):
    """Generate 2D density maps (DMO, Hydro, Replace)."""
    print("\n" + "="*70)
    print("STEP 4: Generate 2D Density Maps")
    print("="*70)
    
    for snap in args.snapshots:
        if check_maps_exist(args.sim_res, snap, args.mass_min) and not args.force_maps:
            print(f"  ✓ Maps exist for snap {snap}, skipping")
            continue
        
        cmd = get_mpi_command(
            args.sim_res,
            'generate_maps_cached.py',
            f'--sim-res {args.sim_res} --snap {snap} --mass-min {args.mass_min}'
        )
        if not run_command(cmd, f"Generating maps for snapshot {snap}"):
            return False
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run full analysis pipeline: cache → stats → profiles → maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline for single snapshot
    python run_full_analysis.py --sim-res 625 --snap 99 --mass-min 12.5

    # All ray-tracing snapshots
    python run_full_analysis.py --sim-res 625 --snap all --mass-min 12.5

    # Skip cache generation (use existing)
    python run_full_analysis.py --sim-res 625 --snap 99 --skip-cache

    # Only generate maps
    python run_full_analysis.py --sim-res 625 --snap 99 --maps-only
        """
    )
    
    # Required arguments
    parser.add_argument('--sim-res', type=int, required=True, 
                       choices=[625, 1250, 2500],
                       help='Simulation resolution')
    parser.add_argument('--snap', type=str, default='99',
                       help='Snapshot(s): "99", "all", "rt" (20 ray-tracing snaps), or comma-separated')
    
    # Mass selection
    parser.add_argument('--mass-min', type=float, default=12.5,
                       help='Minimum log10(M200) for halo selection')
    
    # Pipeline control
    parser.add_argument('--skip-cache', action='store_true',
                       help='Skip cache generation (assume exists)')
    parser.add_argument('--skip-stats', action='store_true',
                       help='Skip statistics computation')
    parser.add_argument('--skip-profiles', action='store_true',
                       help='Skip profile generation')
    parser.add_argument('--maps-only', action='store_true',
                       help='Only generate maps (skip all other steps)')
    
    # Force regeneration
    parser.add_argument('--force-cache', action='store_true',
                       help='Force regenerate cache even if exists')
    parser.add_argument('--force-stats', action='store_true',
                       help='Force recompute statistics')
    parser.add_argument('--force-maps', action='store_true',
                       help='Force regenerate maps')
    
    args = parser.parse_args()
    
    # Parse snapshot selection
    if args.snap == 'all':
        # All standard snapshots
        args.snapshots = RT_SNAPSHOTS + [99]
    elif args.snap == 'rt':
        # Only ray-tracing snapshots
        args.snapshots = RT_SNAPSHOTS
    else:
        args.snapshots = [int(s) for s in args.snap.split(',')]
    
    # Print configuration
    print("="*70)
    print("UNIFIED ANALYSIS PIPELINE")
    print("="*70)
    print(f"Simulation: L205n{args.sim_res}TNG")
    print(f"Snapshots:  {args.snapshots}")
    print(f"Mass min:   10^{args.mass_min} Msun/h")
    print(f"Output:     {OUTPUT_BASE}/L205n{args.sim_res}TNG/")
    print("="*70)
    
    t_start = time.time()
    
    # Run pipeline steps
    if args.maps_only:
        if not step_generate_maps(args):
            sys.exit(1)
    else:
        if not args.skip_cache:
            if not step_generate_cache(args):
                sys.exit(1)
        
        if not args.skip_stats:
            if not step_compute_statistics(args):
                sys.exit(1)
        
        if not args.skip_profiles:
            if not step_generate_profiles(args):
                sys.exit(1)
        
        if not step_generate_maps(args):
            sys.exit(1)
    
    t_total = time.time() - t_start
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print(f"Total time: {t_total/60:.1f} minutes")
    print("="*70)


if __name__ == '__main__':
    main()
