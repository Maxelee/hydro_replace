#!/usr/bin/env python
"""
Full ray-tracing pipeline orchestration.

This script orchestrates the complete pipeline:
  1. Generate lens planes for all models (DMO, Hydro, BCM, Replace)
  2. Generate lux configuration files
  3. Run lux ray-tracing for each model

Usage:
    # Test with low-res (L205n625TNG)
    python scripts/run_full_raytracing.py --sim-res 625 --test
    
    # Production with high-res (L205n2500TNG)
    python scripts/run_full_raytracing.py --sim-res 2500

Options:
    --sim-res     : Simulation resolution (625, 1250, 2500)
    --test        : Run in test mode (1 snapshot, 1024 grid)
    --skip-lensplanes : Skip lens plane generation
    --skip-lux    : Skip lux ray-tracing
    --models      : Specific models to run (default: all)
    --seed        : Random seed (default: 2020)
"""

import os
import sys
import subprocess
import argparse
import time
import glob
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================

LUX_EXECUTABLE = '/mnt/home/mlee1/lux/lux'
WORKSPACE = '/mnt/home/mlee1/hydro_replace2'
OUTPUT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_lensplanes'
LUX_OUTPUT_BASE = '/mnt/home/mlee1/ceph/lux_out'

# Models and their configurations
MODEL_CONFIGS = {
    'dmo': {'label': 'DMO', 'mass_min': None},
    'hydro': {'label': 'Hydro', 'mass_min': None},
    'replace_12.0': {'label': 'Replace (M>10^12)', 'model': 'replace', 'mass_min': 12.0},
    'replace_12.5': {'label': 'Replace (M>10^12.5)', 'model': 'replace', 'mass_min': 12.5},
    'replace_13.0': {'label': 'Replace (M>10^13)', 'model': 'replace', 'mass_min': 13.0},
    'bcm': {'label': 'BCM (all 3 models)', 'mass_min': None},
}

# SLURM configuration by resolution
SLURM_CONFIG = {
    625: {'nodes': 4, 'tasks': 64, 'time': '04:00:00'},
    1250: {'nodes': 8, 'tasks': 128, 'time': '08:00:00'},
    2500: {'nodes': 16, 'tasks': 256, 'time': '12:00:00'},
}


# ============================================================================
# Helper Functions
# ============================================================================

def run_command(cmd, description=None, check=True, shell=False):
    """Run a command and print output."""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")
    
    print(f"Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    
    result = subprocess.run(
        cmd, shell=shell, capture_output=True, text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result


def submit_slurm_job(script, env_vars, config, wait=True):
    """Submit a SLURM job and optionally wait for completion."""
    cmd = f"cd {WORKSPACE} && "
    cmd += " ".join(f"{k}={v}" for k, v in env_vars.items())
    cmd += f" sbatch -N {config['nodes']} -n {config['tasks']} -t {config['time']} {script}"
    
    result = run_command(cmd, shell=True, check=True)
    
    # Extract job ID from output
    for line in result.stdout.split('\n'):
        if 'Submitted batch job' in line:
            job_id = line.split()[-1]
            print(f"  Job ID: {job_id}")
            
            if wait:
                wait_for_job(job_id)
            return job_id
    
    return None


def wait_for_job(job_id, poll_interval=30):
    """Wait for a SLURM job to complete."""
    print(f"  Waiting for job {job_id} to complete...")
    
    while True:
        result = subprocess.run(
            ['squeue', '-j', job_id, '-h'],
            capture_output=True, text=True
        )
        
        if result.stdout.strip() == '':
            # Job no longer in queue - check completion status
            result = subprocess.run(
                ['sacct', '-j', job_id, '--format=State', '-n', '-P'],
                capture_output=True, text=True
            )
            states = result.stdout.strip().split('\n')
            
            if any('FAILED' in s or 'CANCELLED' in s for s in states):
                print(f"  Job {job_id} FAILED")
                return False
            
            print(f"  Job {job_id} COMPLETED")
            return True
        
        time.sleep(poll_interval)


def generate_lux_config(sim_res, model_key, seed=2020):
    """Generate lux configuration file for a model."""
    sim_name = f"L205n{sim_res}TNG"
    
    # Determine input/output directories based on model
    if model_key.startswith('replace_'):
        mass_min = float(model_key.split('_')[1])
        model_dir = f"replace_Mgt{mass_min:.1f}"
        input_dir = f"{OUTPUT_BASE}/{sim_name}/{model_dir}"
        output_dir = f"{LUX_OUTPUT_BASE}/{sim_name}/{model_dir}"
    elif model_key == 'bcm':
        # BCM has 3 sub-models
        configs = []
        for bcm_name in ['Arico20', 'Schneider19', 'Schneider25']:
            model_dir = f"bcm_{bcm_name}"
            input_dir = f"{OUTPUT_BASE}/{sim_name}/{model_dir}"
            output_dir = f"{LUX_OUTPUT_BASE}/{sim_name}/{model_dir}"
            configs.append((bcm_name, input_dir, output_dir))
        return configs  # Return list for BCM
    else:
        input_dir = f"{OUTPUT_BASE}/{sim_name}/{model_key}"
        output_dir = f"{LUX_OUTPUT_BASE}/{sim_name}/{model_key}"
    
    return [(model_key, input_dir, output_dir)]


def write_lux_ini(model_name, input_dir, output_dir, grid_res=4096, seed=2020):
    """Write a lux configuration file."""
    os.makedirs(output_dir, exist_ok=True)
    
    config_file = os.path.join(output_dir, f'lux_{model_name}.ini')
    
    # Snapshot configuration (matching SNAPSHOT_CONFIG in generate_lensplanes.py)
    snap_list = "96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29"
    snap_stack = "0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1"
    
    config_content = f"""# Lux configuration for {model_name}
# Auto-generated by run_full_raytracing.py

[path]
input_dir = {input_dir}
LP_output_dir = {output_dir}
kappa_output_dir = {output_dir}

[lens]
simulation_format = PreProjected
LP_grid = {grid_res}
LP_random_seed = {seed}
planes_per_snapshot = 2
projection_direction = 3
translation_rotation = 1

snap_list = {snap_list}
snap_stack = {snap_stack}

[ray]
kappa_grid = {grid_res}
source_z = 1.0
FoV = 3.5
kappa_output = 1

[cosmology]
Omega_m = 0.3089
h = 0.6774
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"  Wrote: {config_file}")
    return config_file


# ============================================================================
# Main Pipeline
# ============================================================================

def run_lensplane_generation(sim_res, models, seed, grid_res, test_mode=False):
    """Run lens plane generation for specified models."""
    print("\n" + "="*70)
    print("STEP 1: LENS PLANE GENERATION")
    print("="*70)
    
    slurm_config = SLURM_CONFIG[sim_res].copy()
    if test_mode:
        slurm_config['time'] = '01:00:00'
    
    job_ids = []
    
    for model_key in models:
        config = MODEL_CONFIGS.get(model_key, {})
        model = config.get('model', model_key.split('_')[0] if '_' in model_key else model_key)
        mass_min = config.get('mass_min')
        label = config.get('label', model_key)
        
        print(f"\n  Submitting: {label}")
        
        env_vars = {
            'SIM_RES': sim_res,
            'MODEL': model,
            'SNAP': '96' if test_mode else 'all',
            'SEED': seed,
            'GRID_RES': grid_res,
        }
        
        if mass_min is not None:
            env_vars['MASS_MIN'] = mass_min
        
        job_id = submit_slurm_job(
            'batch/run_lensplanes.sh',
            env_vars,
            slurm_config,
            wait=False  # Submit all, then wait
        )
        if job_id:
            job_ids.append((model_key, job_id))
        
        time.sleep(1)  # Brief pause between submissions
    
    # Wait for all jobs to complete
    print("\n  Waiting for all lens plane jobs to complete...")
    for model_key, job_id in job_ids:
        success = wait_for_job(job_id)
        if not success:
            print(f"  WARNING: Job for {model_key} failed!")


def run_lux_raytracing(sim_res, models, seed, grid_res):
    """Run lux ray-tracing for specified models."""
    print("\n" + "="*70)
    print("STEP 2: LUX RAY-TRACING")
    print("="*70)
    
    for model_key in models:
        configs = generate_lux_config(sim_res, model_key, seed)
        
        for model_name, input_dir, output_dir in configs:
            # Check that lens planes exist
            density_files = glob.glob(os.path.join(input_dir, 'density*.dat'))
            if len(density_files) == 0:
                print(f"\n  WARNING: No density files found in {input_dir}")
                print(f"           Skipping {model_name}")
                continue
            
            print(f"\n  Running lux for: {model_name}")
            print(f"    Input: {input_dir}")
            print(f"    Output: {output_dir}")
            
            # Write lux config
            config_file = write_lux_ini(model_name, input_dir, output_dir, grid_res, seed)
            
            # Run lux (MPI)
            cmd = f"cd {os.path.dirname(LUX_EXECUTABLE)} && mpirun -np 16 {LUX_EXECUTABLE} {config_file}"
            
            # Submit as SLURM job
            slurm_script = f"""#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J lux_{model_name}
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -o {output_dir}/lux.o%j
#SBATCH -e {output_dir}/lux.e%j
#SBATCH -t 02:00:00

module load openmpi hdf5

cd {os.path.dirname(LUX_EXECUTABLE)}
srun -n 16 {LUX_EXECUTABLE} {config_file}
"""
            
            script_file = os.path.join(output_dir, f'run_lux_{model_name}.sh')
            with open(script_file, 'w') as f:
                f.write(slurm_script)
            
            result = subprocess.run(
                ['sbatch', script_file],
                capture_output=True, text=True
            )
            print(f"    {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser(description='Full ray-tracing pipeline')
    parser.add_argument('--sim-res', type=int, default=625, choices=[625, 1250, 2500],
                        help='Simulation resolution')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (1 snapshot, 1024 grid)')
    parser.add_argument('--skip-lensplanes', action='store_true',
                        help='Skip lens plane generation')
    parser.add_argument('--skip-lux', action='store_true',
                        help='Skip lux ray-tracing')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to run (default: all)')
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed for reproducibility')
    parser.add_argument('--grid-res', type=int, default=None,
                        help='Grid resolution (default: 1024 for test, 4096 for production)')
    
    args = parser.parse_args()
    
    # Determine grid resolution
    if args.grid_res is None:
        grid_res = 1024 if args.test else 4096
    else:
        grid_res = args.grid_res
    
    # Determine models to run
    if args.models is None:
        models = list(MODEL_CONFIGS.keys())
    else:
        models = args.models
    
    print("="*70)
    print("FULL RAY-TRACING PIPELINE")
    print("="*70)
    print(f"  Simulation: L205n{args.sim_res}TNG")
    print(f"  Test mode: {args.test}")
    print(f"  Grid resolution: {grid_res}")
    print(f"  Random seed: {args.seed}")
    print(f"  Models: {', '.join(models)}")
    print("="*70)
    
    # Step 1: Generate lens planes
    if not args.skip_lensplanes:
        run_lensplane_generation(
            args.sim_res, models, args.seed, grid_res, args.test
        )
    else:
        print("\n  Skipping lens plane generation (--skip-lensplanes)")
    
    # Step 2: Run lux ray-tracing
    if not args.skip_lux:
        run_lux_raytracing(args.sim_res, models, args.seed, grid_res)
    else:
        print("\n  Skipping lux ray-tracing (--skip-lux)")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
