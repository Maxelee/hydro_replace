#!/usr/bin/env python
"""
Example: Complete Halo Analysis Pipeline

Demonstrates how to use particle_access and particle_analysis to:
1. Load matched halos
2. Compute baryon fractions vs halo mass
3. Check mass conservation
4. Generate radial profiles
5. Save results for plotting

Usage:
    python example_halo_analysis.py --snap 99 --sim-res 2500
    python example_halo_analysis.py --snap 99 --mass-min 13.0 --max-halos 50
"""

import numpy as np
import argparse
import os
import sys
import time

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from particle_access import MatchedHaloSnapshot, list_available_snapshots
from particle_analysis import HaloAnalyzer, compute_baryon_fraction, compute_mass_conservation


def main():
    parser = argparse.ArgumentParser(description='Halo analysis example')
    parser.add_argument('--snap', type=int, default=None,
                        help='Snapshot number (default: first available)')
    parser.add_argument('--sim-res', type=int, default=2500,
                        choices=[625, 1250, 2500])
    parser.add_argument('--mass-min', type=float, default=12.5,
                        help='Minimum log10(M200c/Msun)')
    parser.add_argument('--mass-max', type=float, default=15.0,
                        help='Maximum log10(M200c/Msun)')
    parser.add_argument('--max-halos', type=int, default=20,
                        help='Maximum halos to analyze')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for output files')
    parser.add_argument('--no-profiles', action='store_true',
                        help='Skip radial profile computation')
    
    args = parser.parse_args()
    
    # Find available snapshots
    available = list_available_snapshots(sim_res=args.sim_res)
    print(f"Available snapshots: {available}")
    
    if len(available) == 0:
        print("ERROR: No cache files found!")
        print("Run generate_particle_cache.py first.")
        sys.exit(1)
    
    snap = args.snap if args.snap is not None else available[0]
    if snap not in available:
        print(f"ERROR: Snapshot {snap} not in cache. Available: {available}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"HALO ANALYSIS PIPELINE")
    print(f"  Snapshot: {snap}")
    print(f"  Resolution: L205n{args.sim_res}TNG")
    print(f"  Mass range: [{args.mass_min}, {args.mass_max}]")
    print(f"  Max halos: {args.max_halos}")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    # ========================================================================
    # Initialize
    # ========================================================================
    print("[1/5] Loading matched halo snapshot...")
    mh = MatchedHaloSnapshot(snapshot=snap, sim_res=args.sim_res)
    
    # Get halos in mass range
    halos = mh.get_halo_info(mass_range=[args.mass_min, args.mass_max])
    print(f"  Found {len(halos)} halos in mass range")
    
    # Select subset (most massive)
    if len(halos) > args.max_halos:
        sorted_indices = sorted(halos.keys(), 
                               key=lambda x: halos[x].log_mass, 
                               reverse=True)[:args.max_halos]
        halos = {idx: halos[idx] for idx in sorted_indices}
        print(f"  Selected {len(halos)} most massive")
    
    # ========================================================================
    # Run Analysis
    # ========================================================================
    print(f"\n[2/5] Running analysis on {len(halos)} halos...")
    
    analyzer = HaloAnalyzer(mh, n_profile_bins=25)
    results = analyzer.analyze_all(
        mass_range=[args.mass_min, args.mass_max],
        max_halos=args.max_halos,
        compute_profiles=not args.no_profiles,
        verbose=True,
    )
    
    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print(f"\n[3/5] Computing summary statistics...")
    
    # Extract key quantities
    dmo_indices = []
    log_masses = []
    f_baryon_r200 = []
    f_gas_r200 = []
    f_stellar_r200 = []
    ratio_total_r200 = []
    ratio_dm_r200 = []
    
    r200_idx = np.argmin(np.abs(analyzer.radii_r200 - 1.0))
    
    for dmo_idx, result in results.items():
        dmo_indices.append(dmo_idx)
        log_masses.append(result.halo_info.log_mass)
        f_baryon_r200.append(result.baryon_fraction.f_baryon[r200_idx])
        f_gas_r200.append(result.baryon_fraction.f_gas[r200_idx])
        f_stellar_r200.append(result.baryon_fraction.f_stellar[r200_idx])
        ratio_total_r200.append(result.mass_conservation.ratio_total[r200_idx])
        ratio_dm_r200.append(result.mass_conservation.ratio_dm[r200_idx])
    
    log_masses = np.array(log_masses)
    f_baryon_r200 = np.array(f_baryon_r200)
    f_gas_r200 = np.array(f_gas_r200)
    f_stellar_r200 = np.array(f_stellar_r200)
    ratio_total_r200 = np.array(ratio_total_r200)
    ratio_dm_r200 = np.array(ratio_dm_r200)
    
    # ========================================================================
    # Print Results
    # ========================================================================
    print(f"\n[4/5] Results Summary")
    print("=" * 70)
    
    print("\n--- Baryon Fractions at R200 ---")
    print(f"{'Halo':<10} {'log M':<8} {'f_b':<8} {'f_gas':<8} {'f_star':<8}")
    print("-" * 50)
    for i, dmo_idx in enumerate(dmo_indices[:10]):  # First 10
        print(f"{dmo_idx:<10} {log_masses[i]:<8.2f} {f_baryon_r200[i]:<8.3f} "
              f"{f_gas_r200[i]:<8.3f} {f_stellar_r200[i]:<8.3f}")
    if len(dmo_indices) > 10:
        print(f"... ({len(dmo_indices) - 10} more halos)")
    
    print(f"\nMean f_baryon = {np.mean(f_baryon_r200):.3f} ± {np.std(f_baryon_r200):.3f}")
    print(f"Cosmic f_baryon = 0.157")
    
    print("\n--- Mass Conservation at R200 ---")
    print(f"{'Halo':<10} {'log M':<8} {'M_hyd/M_dmo':<12} {'M_dm/M_dmo':<12}")
    print("-" * 50)
    for i, dmo_idx in enumerate(dmo_indices[:10]):
        print(f"{dmo_idx:<10} {log_masses[i]:<8.2f} {ratio_total_r200[i]:<12.3f} "
              f"{ratio_dm_r200[i]:<12.3f}")
    if len(dmo_indices) > 10:
        print(f"... ({len(dmo_indices) - 10} more halos)")
    
    print(f"\nMean M_hydro/M_dmo = {np.mean(ratio_total_r200):.3f} ± {np.std(ratio_total_r200):.3f}")
    print(f"Mean M_dm/M_dmo = {np.mean(ratio_dm_r200):.3f} (expected: 0.843)")
    
    # Stack by mass
    print("\n--- Stacked by Mass Bin ---")
    stacked = analyzer.stack_results(results)
    
    print(f"\n{'Mass bin':<16} {'N':<6} {'f_b (R200)':<15} {'M_rat (R200)':<15}")
    print("-" * 55)
    for i in range(len(stacked['mass_bins']) - 1):
        m1, m2 = stacked['mass_bins'][i], stacked['mass_bins'][i+1]
        n = stacked['n_halos'][i]
        if n > 0:
            fb = stacked['f_baryon_mean'][i, r200_idx]
            fb_std = stacked['f_baryon_std'][i, r200_idx]
            mr = stacked['ratio_total_mean'][i, r200_idx]
            mr_std = stacked['ratio_total_std'][i, r200_idx]
            print(f"[{m1:.1f}, {m2:.1f})     {n:<6} {fb:.3f} ± {fb_std:.3f}     "
                  f"{mr:.3f} ± {mr_std:.3f}")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    print(f"\n[5/5] Saving results...")
    
    output_dir = args.output_dir or os.path.join(
        '/mnt/home/mlee1/ceph/hydro_replace_fields',
        f'L205n{args.sim_res}TNG',
        'analysis'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'halo_analysis_snap{snap:03d}.npz')
    
    # Prepare data for saving
    save_data = {
        # Metadata
        'snapshot': snap,
        'sim_res': args.sim_res,
        'radii_r200': analyzer.radii_r200,
        
        # Per-halo data
        'dmo_indices': np.array(dmo_indices),
        'log_masses': log_masses,
        'f_baryon_r200': f_baryon_r200,
        'f_gas_r200': f_gas_r200,
        'f_stellar_r200': f_stellar_r200,
        'ratio_total_r200': ratio_total_r200,
        'ratio_dm_r200': ratio_dm_r200,
        
        # Stacked data
        'mass_bins': stacked['mass_bins'],
        'n_halos_per_bin': stacked['n_halos'],
        'f_baryon_mean': stacked['f_baryon_mean'],
        'f_baryon_std': stacked['f_baryon_std'],
        'ratio_total_mean': stacked['ratio_total_mean'],
        'ratio_total_std': stacked['ratio_total_std'],
    }
    
    # Add full baryon fraction arrays at all radii
    f_baryon_all = np.zeros((len(results), len(analyzer.radii_r200)))
    ratio_total_all = np.zeros((len(results), len(analyzer.radii_r200)))
    
    for i, dmo_idx in enumerate(dmo_indices):
        f_baryon_all[i] = results[dmo_idx].baryon_fraction.f_baryon
        ratio_total_all[i] = results[dmo_idx].mass_conservation.ratio_total
    
    save_data['f_baryon_all'] = f_baryon_all
    save_data['ratio_total_all'] = ratio_total_all
    
    np.savez(output_file, **save_data)
    print(f"  Saved: {output_file}")
    
    # Cleanup
    mh.close()
    
    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Analysis complete in {t_total:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
