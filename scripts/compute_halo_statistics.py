#!/usr/bin/env python
"""
Compute baryon fractions and mass conservation for all matched halos.

This script processes all halos in the particle cache and computes:
- Baryon fractions (f_b, f_gas, f_stellar) at multiple radii
- Mass conservation ratios (M_hydro/M_dmo) at multiple radii

Output: HDF5 file with per-halo and stacked results.

Usage:
    python compute_halo_statistics.py --snap 99 --sim-res 2500
    
For MPI parallel (much faster):
    mpirun -np 32 python compute_halo_statistics.py --snap 99 --sim-res 2500
"""

import numpy as np
import h5py
import argparse
import os
import sys
import time

# Check for MPI
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    HAS_MPI = size > 1
except ImportError:
    rank = 0
    size = 1
    HAS_MPI = False

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from particle_access import MatchedHaloSnapshot, list_available_snapshots
from particle_analysis import (
    compute_baryon_fraction, 
    compute_mass_conservation,
    compute_radial_profile,
)


def main():
    parser = argparse.ArgumentParser(description='Compute halo statistics')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, default=2500, choices=[625, 1250, 2500])
    parser.add_argument('--mass-min', type=float, default=12.0,
                        help='Minimum log10(M200c/Msun)')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--compute-profiles', action='store_true',
                        help='Also compute radial profiles (slower)')
    
    args = parser.parse_args()
    
    # Radii at which to compute statistics
    radii_r200 = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    n_radii = len(radii_r200)
    
    if rank == 0:
        print("=" * 70)
        print("HALO STATISTICS COMPUTATION")
        print("=" * 70)
        print(f"Snapshot: {args.snap}")
        print(f"Resolution: L205n{args.sim_res}TNG")
        print(f"Mass minimum: 10^{args.mass_min} Msun/h")
        print(f"Radii: {radii_r200}")
        if HAS_MPI:
            print(f"MPI ranks: {size}")
        print("=" * 70)
    
    t_start = time.time()
    
    # ========================================================================
    # Load halo information (all ranks need this)
    # ========================================================================
    if rank == 0:
        print("\n[1/4] Loading halo cache...")
    
    # Each rank loads the cache (lightweight - just halo info)
    mh = MatchedHaloSnapshot(snapshot=args.snap, sim_res=args.sim_res, verbose=(rank == 0))
    
    # Get all halos above mass threshold
    halos = mh.get_halo_info(mass_range=[args.mass_min, 16.0])
    all_indices = sorted(halos.keys())
    n_halos = len(all_indices)
    
    if rank == 0:
        print(f"  Total halos: {n_halos}")
    
    # ========================================================================
    # Distribute halos across ranks
    # ========================================================================
    my_indices = [idx for i, idx in enumerate(all_indices) if i % size == rank]
    n_my_halos = len(my_indices)
    
    if rank == 0:
        print(f"\n[2/4] Distributing {n_halos} halos across {size} ranks...")
        print(f"  Rank 0 processing {n_my_halos} halos")
    
    # ========================================================================
    # Load snapshot data (this is the expensive part)
    # ========================================================================
    if rank == 0:
        print(f"\n[3/4] Loading snapshot data and processing halos...")
        t0 = time.time()
    
    # Pre-allocate results arrays for this rank
    my_results = {
        'dmo_indices': np.array(my_indices, dtype=np.int32),
        'log_masses': np.zeros(n_my_halos, dtype=np.float32),
        'positions': np.zeros((n_my_halos, 3), dtype=np.float32),
        'radii': np.zeros(n_my_halos, dtype=np.float32),
        
        # Baryon fractions at each radius
        'f_baryon': np.zeros((n_my_halos, n_radii), dtype=np.float32),
        'f_gas': np.zeros((n_my_halos, n_radii), dtype=np.float32),
        'f_stellar': np.zeros((n_my_halos, n_radii), dtype=np.float32),
        'm_total': np.zeros((n_my_halos, n_radii), dtype=np.float64),
        'm_gas': np.zeros((n_my_halos, n_radii), dtype=np.float64),
        'm_stellar': np.zeros((n_my_halos, n_radii), dtype=np.float64),
        'm_dm_hydro': np.zeros((n_my_halos, n_radii), dtype=np.float64),
        
        # Mass conservation
        'm_dmo': np.zeros((n_my_halos, n_radii), dtype=np.float64),
        'ratio_total': np.zeros((n_my_halos, n_radii), dtype=np.float32),
        'ratio_dm': np.zeros((n_my_halos, n_radii), dtype=np.float32),
    }
    
    # Process each halo
    for i, dmo_idx in enumerate(my_indices):
        if rank == 0 and i % 10 == 0:
            print(f"    Processing halo {i+1}/{n_my_halos}...")
        
        halo = halos[dmo_idx]
        my_results['log_masses'][i] = halo.log_mass
        my_results['positions'][i] = halo.position
        my_results['radii'][i] = halo.radius
        
        # Load particles
        try:
            dmo_data = mh.get_particles('dmo', dmo_idx, include_coords=True)
            hydro_data = mh.get_particles('hydro', dmo_idx, include_coords=True)
        except Exception as e:
            if rank == 0:
                print(f"    WARNING: Failed to load halo {dmo_idx}: {e}")
            continue
        
        # Compute baryon fractions
        bf_result = compute_baryon_fraction(hydro_data, radii_r200)
        my_results['f_baryon'][i] = bf_result.f_baryon
        my_results['f_gas'][i] = bf_result.f_gas
        my_results['f_stellar'][i] = bf_result.f_stellar
        my_results['m_total'][i] = bf_result.m_total
        my_results['m_gas'][i] = bf_result.m_gas
        my_results['m_stellar'][i] = bf_result.m_stellar
        my_results['m_dm_hydro'][i] = bf_result.m_dm
        
        # Compute mass conservation
        mc_result = compute_mass_conservation(dmo_data, hydro_data, radii_r200)
        my_results['m_dmo'][i] = mc_result.m_dmo
        my_results['ratio_total'][i] = mc_result.ratio_total
        my_results['ratio_dm'][i] = mc_result.ratio_dm
    
    if rank == 0:
        print(f"  Processing time: {time.time()-t0:.1f}s")
    
    # ========================================================================
    # Gather results to rank 0
    # ========================================================================
    if rank == 0:
        print(f"\n[4/4] Gathering results and saving...")
    
    if HAS_MPI:
        comm.Barrier()
        
        # Gather all results to rank 0
        all_results_list = comm.gather(my_results, root=0)
        
        if rank == 0:
            # Combine results from all ranks
            combined = {key: [] for key in my_results.keys()}
            for r_results in all_results_list:
                for key in combined.keys():
                    combined[key].append(r_results[key])
            
            # Concatenate
            for key in combined.keys():
                combined[key] = np.concatenate(combined[key], axis=0)
            
            # Sort by DMO index
            sort_order = np.argsort(combined['dmo_indices'])
            for key in combined.keys():
                combined[key] = combined[key][sort_order]
            
            final_results = combined
    else:
        final_results = my_results
    
    # ========================================================================
    # Save results (rank 0 only)
    # ========================================================================
    if rank == 0:
        output_dir = args.output_dir or os.path.join(
            '/mnt/home/mlee1/ceph/hydro_replace_fields',
            f'L205n{args.sim_res}TNG',
            'analysis'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f'halo_statistics_snap{args.snap:03d}.h5')
        
        with h5py.File(output_file, 'w') as f:
            # Metadata
            f.attrs['snapshot'] = args.snap
            f.attrs['sim_res'] = args.sim_res
            f.attrs['n_halos'] = len(final_results['dmo_indices'])
            f.attrs['mass_min_log'] = args.mass_min
            f.attrs['radii_r200'] = radii_r200
            f.attrs['cosmic_baryon_fraction'] = 0.0486 / 0.3089
            f.attrs['expected_dm_ratio'] = 0.00398342749867548 / 0.0047271638660809
            
            # Per-halo data
            grp = f.create_group('halos')
            for key, data in final_results.items():
                grp.create_dataset(key, data=data, compression='gzip')
        
        print(f"  Saved: {output_file}")
        print(f"  Total halos: {len(final_results['dmo_indices'])}")
        
        # Quick summary
        r200_idx = np.argmin(np.abs(radii_r200 - 1.0))
        fb_r200 = final_results['f_baryon'][:, r200_idx]
        rt_r200 = final_results['ratio_total'][:, r200_idx]
        
        print(f"\n  Summary at R200:")
        print(f"    Mean f_baryon = {np.mean(fb_r200):.4f} ± {np.std(fb_r200):.4f}")
        print(f"    Mean M_hydro/M_dmo = {np.mean(rt_r200):.4f} ± {np.std(rt_r200):.4f}")
        
        t_total = time.time() - t_start
        print(f"\n{'='*70}")
        print(f"Complete in {t_total:.1f}s")
        print(f"{'='*70}")
    
    # Cleanup
    mh.close()


if __name__ == '__main__':
    main()
