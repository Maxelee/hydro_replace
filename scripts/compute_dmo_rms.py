#!/usr/bin/env python
"""
Compute DMO RMS values for all realizations.

This script computes the smoothed convergence RMS for DMO at each 
(LP, run, source_z) combination and saves to a single HDF5 file.

Run once before the main statistics pipeline.

Usage:
    mpirun -np 64 python compute_dmo_rms.py
"""

import os
import sys
import numpy as np
import h5py
from mpi4py import MPI

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import (
    RT_BASE, STATS_BASE, N_LP, N_RUNS, KAPPA_TARGETS,
    SMOOTHING_ARCMIN, PIXEL_SCALE_ARCMIN
)
from stats_utils import load_kappa, smooth_kappa


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=" * 80)
        print("Computing DMO RMS values")
        print("=" * 80)
        print(f"MPI size: {size}")
        print(f"Output: {STATS_BASE}/dmo_rms.h5")
        print()
    
    # Create output directory
    if rank == 0:
        os.makedirs(STATS_BASE, exist_ok=True)
    comm.Barrier()
    
    # Build task list: (LP, run, z_idx, kappa_num)
    tasks = []
    z_names = sorted(KAPPA_TARGETS.keys())
    for lp in range(N_LP):
        for run in range(1, N_RUNS + 1):
            for z_idx, z_name in enumerate(z_names):
                kappa_num = KAPPA_TARGETS[z_name]
                tasks.append((lp, run, z_idx, kappa_num))
    
    n_tasks = len(tasks)
    if rank == 0:
        print(f"Total tasks: {n_tasks} ({N_LP} LP × {N_RUNS} runs × {len(z_names)} redshifts)")
        print()
    
    # Distribute tasks
    my_tasks = tasks[rank::size]
    
    # Initialize local results
    n_z = len(z_names)
    local_rms = np.zeros((N_LP, N_RUNS, n_z), dtype=np.float32)
    local_count = np.zeros((N_LP, N_RUNS, n_z), dtype=np.int32)
    
    # Process tasks
    for task_idx, (lp, run, z_idx, kappa_num) in enumerate(my_tasks):
        if task_idx % 100 == 0 and task_idx > 0:
            print(f"[Rank {rank}] Processed {task_idx}/{len(my_tasks)} tasks", flush=True)
        
        # Build path
        lp_str = f"LP_{lp:02d}"
        run_str = f"run{run:03d}"
        kappa_file = f"kappa{kappa_num:02d}.dat"
        kappa_path = os.path.join(RT_BASE, 'dmo', lp_str, run_str, kappa_file)
        
        # Check if file exists
        if not os.path.exists(kappa_path):
            print(f"[Rank {rank}] WARNING: Missing {kappa_path}", flush=True)
            continue
        
        try:
            # Load and smooth
            kappa = load_kappa(kappa_path)
            kappa_smooth = smooth_kappa(kappa, SMOOTHING_ARCMIN, PIXEL_SCALE_ARCMIN)
            
            # Compute RMS
            rms = np.std(kappa_smooth)
            
            # Store
            local_rms[lp, run - 1, z_idx] = rms
            local_count[lp, run - 1, z_idx] = 1
            
        except Exception as e:
            print(f"[Rank {rank}] ERROR processing {kappa_path}: {e}", flush=True)
            continue
    
    if rank == 0:
        print()
        print("All tasks completed. Gathering results...")
    
    # Gather results
    global_rms = np.zeros((N_LP, N_RUNS, n_z), dtype=np.float32)
    global_count = np.zeros((N_LP, N_RUNS, n_z), dtype=np.int32)
    
    comm.Reduce(local_rms, global_rms, op=MPI.SUM, root=0)
    comm.Reduce(local_count, global_count, op=MPI.SUM, root=0)
    
    # Save results
    if rank == 0:
        # Check for missing data
        missing = np.sum(global_count == 0)
        if missing > 0:
            print(f"WARNING: {missing} missing realizations")
        
        # Replace zeros with NaN
        global_rms[global_count == 0] = np.nan
        
        # Save to HDF5
        output_path = os.path.join(STATS_BASE, 'dmo_rms.h5')
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('rms', data=global_rms, dtype='float32', compression='gzip')
            f.create_dataset('count', data=global_count, dtype='int32', compression='gzip')
            
            # Metadata
            f.attrs['N_LP'] = N_LP
            f.attrs['N_RUNS'] = N_RUNS
            f.attrs['N_z'] = n_z
            f.attrs['z_names'] = [z.encode() for z in z_names]
            f.attrs['kappa_nums'] = [KAPPA_TARGETS[z] for z in z_names]
            f.attrs['smoothing_arcmin'] = SMOOTHING_ARCMIN
            f.attrs['description'] = 'DMO smoothed convergence RMS for all realizations'
        
        print()
        print(f"Results saved to: {output_path}")
        print(f"Shape: {global_rms.shape}")
        print(f"RMS range: [{np.nanmin(global_rms):.6f}, {np.nanmax(global_rms):.6f}]")
        print(f"Mean RMS: {np.nanmean(global_rms):.6f}")
        print()
        print("=" * 80)
        print("DMO RMS computation complete!")
        print("=" * 80)


if __name__ == '__main__':
    main()
