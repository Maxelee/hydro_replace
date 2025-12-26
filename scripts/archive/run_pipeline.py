#!/usr/bin/env python
"""
Complete End-to-End Pipeline for Hydro Replace Analysis.

This script runs the full pipeline:
1. Generate particle cache (stores particle IDs for halos)
2. Compute halo statistics (baryon fractions, mass conservation)
3. Generate density profiles (stacked by mass bin)
4. Generate 2D density maps (DMO, Hydro, Replace)
5. Generate lens planes for ray-tracing
6. Run lux ray-tracing (lens potential + convergence maps)

All steps are configurable with mass_min, mass_max, and radius_factor.

Usage:
    # Full pipeline for 625 resolution
    python run_pipeline.py --sim-res 625 --snap 99 --mass-min 12.5

    # With mass range and custom radius
    python run_pipeline.py --sim-res 625 --snap 99 \\
        --mass-min 12.5 --mass-max 14.0 --radius-factor 3.0

    # All ray-tracing snapshots
    python run_pipeline.py --sim-res 625 --snap rt --mass-min 12.5

    # Skip specific steps
    python run_pipeline.py --sim-res 625 --snap 99 --mass-min 12.5 \\
        --skip-cache --skip-stats --lensplanes-only
"""

import subprocess
import argparse
import os
import sys
import time
import glob

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = '/mnt/home/mlee1/hydro_replace2'
FIELDS_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields'
LENSPLANES_BASE = '/mnt/home/mlee1/ceph/hydro_replace_lensplanes'
LUX_OUTPUT_BASE = '/mnt/home/mlee1/ceph/lux_out'
LUX_DIR = '/mnt/home/mlee1/lux'

# MPI configuration per resolution
MPI_CONFIG = {
    625: {'ntasks': 4, 'nodes': 4},
    1250: {'ntasks': 8, 'nodes': 8},
    2500: {'ntasks': 16, 'nodes': 16},
}

# Standard 20 snapshots for ray-tracing
RT_SNAPSHOTS = [29, 31, 33, 35, 38, 41, 43, 46, 49, 52,
                56, 59, 63, 67, 71, 76, 80, 85, 90, 96]

# Lux snapshot configuration
LUX_SNAP_LIST = "96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29"
LUX_SNAP_STACK = "false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true"


# ============================================================================
# Helper Functions
# ============================================================================

def run_command(cmd, description, check=True):
    """Run a shell command and print output."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if check and result.returncode != 0:
        print(f"\n⚠️  Command failed with return code {result.returncode}")
        return False
    return True


def get_mpi_cmd(sim_res, script, args_str):
    """Build MPI command."""
    config = MPI_CONFIG[sim_res]
    return f"mpirun -np {config['ntasks']} python {BASE_DIR}/scripts/{script} {args_str}"


def get_model_dir_name(mass_min, mass_max):
    """Get directory name for replace model."""
    if mass_max is not None:
        return f"replace_M{mass_min:.1f}-{mass_max:.1f}"
    else:
        return f"replace_Mgt{mass_min:.1f}"


def check_file_exists(filepath):
    """Check if file exists."""
    return os.path.exists(filepath)


# ============================================================================
# Pipeline Steps
# ============================================================================

def step_cache(args):
    """Step 1: Generate particle cache."""
    print("\n" + "="*70)
    print("STEP 1: PARTICLE CACHE")
    print("="*70)
    
    for snap in args.snapshots:
        cache_file = f'{FIELDS_BASE}/L205n{args.sim_res}TNG/particle_cache/cache_snap{snap:03d}.h5'
        
        if check_file_exists(cache_file) and not args.force:
            print(f"  ✓ Cache exists for snap {snap}, skipping")
            continue
        
        cmd = get_mpi_cmd(
            args.sim_res,
            'generate_particle_cache.py',
            f'--sim-res {args.sim_res} --snap {snap}'
        )
        if not run_command(cmd, f"Generating cache for snapshot {snap}"):
            return False
    return True


def step_statistics(args):
    """Step 2: Compute halo statistics."""
    print("\n" + "="*70)
    print("STEP 2: HALO STATISTICS")
    print("="*70)
    
    for snap in args.snapshots:
        stats_file = f'{FIELDS_BASE}/L205n{args.sim_res}TNG/analysis/halo_statistics_snap{snap:03d}.h5'
        
        if check_file_exists(stats_file) and not args.force:
            print(f"  ✓ Statistics exist for snap {snap}, skipping")
            continue
        
        cmd = get_mpi_cmd(
            args.sim_res,
            'compute_halo_statistics_distributed.py',
            f'--sim-res {args.sim_res} --snap {snap}'
        )
        if not run_command(cmd, f"Computing statistics for snapshot {snap}"):
            return False
    return True


def step_profiles(args):
    """Step 3: Generate density profiles."""
    print("\n" + "="*70)
    print("STEP 3: DENSITY PROFILES")
    print("="*70)
    
    for snap in args.snapshots:
        mass_args = f'--mass-min {args.mass_min}'
        if args.mass_max:
            mass_args += f' --mass-max {args.mass_max}'
        
        cmd = get_mpi_cmd(
            args.sim_res,
            'generate_profiles_cached_new.py',
            f'--sim-res {args.sim_res} --snap {snap} {mass_args}'
        )
        if not run_command(cmd, f"Generating profiles for snapshot {snap}"):
            return False
    return True


def step_maps(args):
    """Step 4: Generate 2D density maps."""
    print("\n" + "="*70)
    print("STEP 4: 2D DENSITY MAPS")
    print("="*70)
    
    for snap in args.snapshots:
        mass_args = f'--mass-min {args.mass_min}'
        if args.mass_max:
            mass_args += f' --mass-max {args.mass_max}'
        
        cmd = get_mpi_cmd(
            args.sim_res,
            'generate_maps_cached.py',
            f'--sim-res {args.sim_res} --snap {snap} {mass_args}'
        )
        if not run_command(cmd, f"Generating maps for snapshot {snap}"):
            return False
    return True


def step_lensplanes(args):
    """Step 5: Generate lens planes for ray-tracing."""
    print("\n" + "="*70)
    print("STEP 5: LENS PLANES")
    print("="*70)
    
    # Build snapshot argument
    if len(args.snapshots) == len(RT_SNAPSHOTS) and set(args.snapshots) == set(RT_SNAPSHOTS):
        snap_arg = 'rt'
    else:
        snap_arg = ','.join(str(s) for s in args.snapshots)
    
    # Build mass arguments
    mass_args = f'--mass-min {args.mass_min}'
    if args.mass_max:
        mass_args += f' --mass-max {args.mass_max}'
    
    cmd = get_mpi_cmd(
        args.sim_res,
        'generate_lensplanes_replace.py',
        f'--sim-res {args.sim_res} --snap {snap_arg} {mass_args} '
        f'--radius-factor {args.radius_factor} --seed {args.seed}'
    )
    if not run_command(cmd, "Generating replace lens planes"):
        return False
    
    return True


def step_raytracing(args):
    """Step 6: Run lux ray-tracing."""
    print("\n" + "="*70)
    print("STEP 6: RAY-TRACING (LUX)")
    print("="*70)
    
    model_dir = get_model_dir_name(args.mass_min, args.mass_max)
    sim_name = f'L205n{args.sim_res}TNG'
    
    input_dir = f'{LENSPLANES_BASE}/{sim_name}/seed{args.seed}/{model_dir}'
    output_dir = f'{LUX_OUTPUT_BASE}/{sim_name}/seed{args.seed}/{model_dir}'
    
    # Check if lens planes exist
    density_files = glob.glob(f'{input_dir}/density*.dat')
    if not density_files:
        print(f"  ERROR: No density files found in {input_dir}")
        print("  Run with --skip-raytracing or generate lens planes first")
        return False
    
    print(f"  Found {len(density_files)} density files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write lux configuration
    config_file = f'{output_dir}/lux_replace.ini'
    with open(config_file, 'w') as f:
        f.write(f"""# Lux configuration for {model_dir}
# Auto-generated by run_pipeline.py

input_dir = {input_dir}
LP_output_dir = {output_dir}
RT_output_dir = {output_dir}

simulation_format = PreProjected

LP_grid = {args.lp_grid}
LP_random_seed = {args.seed}
planes_per_snapshot = 2
projection_direction = 3
translation_rotation = true

snapshot_list = {LUX_SNAP_LIST}
snapshot_stack = {LUX_SNAP_STACK}

RT_grid = {args.rt_grid}
RT_random_seed = {args.seed}
RT_randomization = true
angle = 5.0

verbose = true
""")
    print(f"  Wrote config: {config_file}")
    
    # Copy config.dat if exists
    config_dat = f'{input_dir}/config.dat'
    if os.path.exists(config_dat):
        import shutil
        shutil.copy(config_dat, f'{output_dir}/config.dat')
        print(f"  Copied config.dat to output directory")
    
    # Create run directories
    for i in range(1, 101):
        os.makedirs(f'{output_dir}/run{i:03d}', exist_ok=True)
    print(f"  Created run001-run100 directories")
    
    # Run lux
    cmd = f'cd {LUX_DIR} && mpirun -np 16 ./lux {config_file}'
    if not run_command(cmd, "Running lux ray-tracing"):
        return False
    
    # Check output
    kappa_files = glob.glob(f'{output_dir}/run001/kappa_*.fits')
    if kappa_files:
        print(f"  ✓ Generated {len(kappa_files)} convergence maps")
    
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Complete end-to-end pipeline for hydro replace analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline for single snapshot
    python run_pipeline.py --sim-res 625 --snap 99 --mass-min 12.5

    # With mass range
    python run_pipeline.py --sim-res 625 --snap 99 \\
        --mass-min 12.5 --mass-max 14.0

    # All ray-tracing snapshots
    python run_pipeline.py --sim-res 625 --snap rt --mass-min 12.5

    # Only lens planes and ray-tracing
    python run_pipeline.py --sim-res 625 --snap rt --mass-min 12.5 \\
        --skip-cache --skip-stats --skip-profiles --skip-maps
        """
    )
    
    # Required arguments
    parser.add_argument('--sim-res', type=int, required=True,
                       choices=[625, 1250, 2500],
                       help='Simulation resolution')
    parser.add_argument('--snap', type=str, default='99',
                       help='Snapshot(s): "99", "rt" (20 ray-tracing), or comma-separated')
    
    # Mass/radius selection
    parser.add_argument('--mass-min', type=float, default=12.5,
                       help='Minimum log10(M200) for halo selection (default: 12.5)')
    parser.add_argument('--mass-max', type=float, default=None,
                       help='Maximum log10(M200) for halo selection (optional)')
    parser.add_argument('--radius-factor', type=float, default=5.0,
                       help='Replacement radius as multiple of R200 (default: 5.0)')
    
    # Ray-tracing options
    parser.add_argument('--seed', type=int, default=2020,
                       help='Random seed for randomization (default: 2020)')
    parser.add_argument('--lp-grid', type=int, default=4096,
                       help='Lens potential grid resolution (default: 4096)')
    parser.add_argument('--rt-grid', type=int, default=1024,
                       help='Ray-tracing grid resolution (default: 1024)')
    
    # Skip steps
    parser.add_argument('--skip-cache', action='store_true',
                       help='Skip particle cache generation')
    parser.add_argument('--skip-stats', action='store_true',
                       help='Skip halo statistics computation')
    parser.add_argument('--skip-profiles', action='store_true',
                       help='Skip profile generation')
    parser.add_argument('--skip-maps', action='store_true',
                       help='Skip 2D map generation')
    parser.add_argument('--skip-lensplanes', action='store_true',
                       help='Skip lens plane generation')
    
    # Convenience flags
    parser.add_argument('--lensplanes-only', action='store_true',
                       help='Only generate lens planes (skip cache, stats, profiles, maps)')
    
    # Force regeneration
    parser.add_argument('--force', action='store_true',
                       help='Force regenerate even if outputs exist')
    
    args = parser.parse_args()
    
    # Parse snapshots
    if args.snap == 'rt':
        args.snapshots = RT_SNAPSHOTS
    elif args.snap == 'all':
        args.snapshots = RT_SNAPSHOTS + [99]
    else:
        args.snapshots = [int(s) for s in args.snap.split(',')]
    
    # Handle convenience flags
    if args.lensplanes_only:
        args.skip_cache = True
        args.skip_stats = True
        args.skip_profiles = True
        args.skip_maps = True
    
    # Print configuration
    print("="*70)
    print("HYDRO REPLACE - FULL PIPELINE")
    print("="*70)
    print(f"Simulation:    L205n{args.sim_res}TNG")
    print(f"Snapshots:     {args.snapshots}")
    print(f"Mass range:    10^{args.mass_min} - 10^{args.mass_max or '∞'} Msun/h")
    print(f"Radius factor: {args.radius_factor}×R200")
    print(f"Random seed:   {args.seed}")
    print(f"LP grid:       {args.lp_grid}")
    print(f"RT grid:       {args.rt_grid}")
    print("="*70)
    print("Steps to run:")
    print(f"  1. Cache:       {'SKIP' if args.skip_cache else 'RUN'}")
    print(f"  2. Statistics:  {'SKIP' if args.skip_stats else 'RUN'}")
    print(f"  3. Profiles:    {'SKIP' if args.skip_profiles else 'RUN'}")
    print(f"  4. Maps:        {'SKIP' if args.skip_maps else 'RUN'}")
    print(f"  5. Lens planes: {'SKIP' if args.skip_lensplanes else 'RUN'}")
    print("="*70)
    
    t_start = time.time()
    
    # Run pipeline steps
    if not args.skip_cache:
        if not step_cache(args):
            print("\n❌ Pipeline failed at step 1 (cache)")
            sys.exit(1)
    
    if not args.skip_stats:
        if not step_statistics(args):
            print("\n❌ Pipeline failed at step 2 (statistics)")
            sys.exit(1)
    
    if not args.skip_profiles:
        if not step_profiles(args):
            print("\n❌ Pipeline failed at step 3 (profiles)")
            sys.exit(1)
    
    if not args.skip_maps:
        if not step_maps(args):
            print("\n❌ Pipeline failed at step 4 (maps)")
            sys.exit(1)
    
    if not args.skip_lensplanes:
        if not step_lensplanes(args):
            print("\n❌ Pipeline failed at step 5 (lens planes)")
            sys.exit(1)
    
    t_total = time.time() - t_start
    
    # Summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print(f"Total time: {t_total/60:.1f} minutes")
    print("="*70)
    
    model_dir = get_model_dir_name(args.mass_min, args.mass_max)
    sim_name = f'L205n{args.sim_res}TNG'
    
    print("\nOutput locations:")
    print(f"  Cache:       {FIELDS_BASE}/{sim_name}/particle_cache/")
    print(f"  Statistics:  {FIELDS_BASE}/{sim_name}/analysis/")
    print(f"  Maps:        {FIELDS_BASE}/{sim_name}/fields_snap*/")
    print(f"  Lens planes: {LENSPLANES_BASE}/{sim_name}/seed{args.seed}/{model_dir}/")
    print(f"  Convergence: {LUX_OUTPUT_BASE}/{sim_name}/seed{args.seed}/{model_dir}/")


if __name__ == '__main__':
    main()
