#!/usr/bin/env python
"""
Check progress of the binned lensplane and ray-tracing pipeline.

This utility scans output directories and reports completion status for:
- Lens plane generation (20 snapshots × 20 realizations × 40 planes per model)
- Lux conversion (20 realizations × 41 files per model)
- Ray-tracing (20 realizations × 100 runs × 40 kappa maps per model)

Usage:
    python check_progress.py                    # Full status report
    python check_progress.py --model dmo        # Check single model
    python check_progress.py --missing          # List missing files for reruns
    python check_progress.py --summary          # Brief summary only
"""

import os
import sys
import argparse
from itertools import combinations


# =============================================================================
# Configuration
# =============================================================================

LP_BASE = '/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG'
LUX_BASE = '/mnt/home/mlee1/ceph/hydro_replace_LP_lux/L205n2500TNG'
RT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG'

N_REALIZATIONS = 20
N_PLANES = 40  # 20 snapshots × 2 pps
N_RT_RUNS = 100
N_KAPPA_PLANES = 40

# Snapshot order
SNAPSHOTS = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]

# Mass and radius edges for binned configs
MASS_EDGES = [12.0, 12.5, 13.0, 13.5, 15.0]
RADIUS_EDGES = [0.0, 0.5, 1.0, 3.0, 5.0]


def generate_binned_model_names():
    """Generate all binned model names."""
    models = ['dmo', 'hydro']
    
    mass_bins = [(10**m_lo, 10**m_hi) for m_lo, m_hi in combinations(MASS_EDGES, 2)]
    radius_shells = list(combinations(RADIUS_EDGES, 2))
    
    for M_lo, M_hi in mass_bins:
        for R_inner, R_outer in radius_shells:
            label = f"hydro_replace_Ml_{M_lo:.2e}_Mu_{M_hi:.2e}_Ri_{R_inner}_Ro_{R_outer}".replace('+', '')
            models.append(label)
    
    return models


def check_lensplanes(model):
    """Check lens plane completion status for a model."""
    model_dir = os.path.join(LP_BASE, model)
    
    if not os.path.exists(model_dir):
        return {'complete': 0, 'total': N_REALIZATIONS * N_PLANES, 'missing': []}
    
    complete = 0
    missing = []
    
    for real in range(N_REALIZATIONS):
        lp_dir = os.path.join(model_dir, f'LP_{real:02d}')
        for plane in range(N_PLANES):
            filepath = os.path.join(lp_dir, f'lenspot{plane:02d}.dat')
            if os.path.exists(filepath):
                complete += 1
            else:
                missing.append((real, plane))
    
    return {
        'complete': complete,
        'total': N_REALIZATIONS * N_PLANES,
        'missing': missing
    }


def check_lux_conversion(model):
    """Check lux conversion status for a model."""
    model_dir = os.path.join(LUX_BASE, model)
    
    if not os.path.exists(model_dir):
        return {'complete': 0, 'total': N_REALIZATIONS * (N_PLANES + 1), 'missing': []}
    
    complete = 0
    missing = []
    
    for real in range(N_REALIZATIONS):
        lp_dir = os.path.join(model_dir, f'LP_{real:02d}')
        
        # Check config.dat
        config_path = os.path.join(lp_dir, 'config.dat')
        if os.path.exists(config_path):
            complete += 1
        else:
            missing.append((real, 'config.dat'))
        
        # Check lenspot files (1-indexed for lux)
        for plane in range(1, N_PLANES + 1):
            filepath = os.path.join(lp_dir, f'lenspot{plane:02d}.dat')
            if os.path.exists(filepath):
                complete += 1
            else:
                missing.append((real, f'lenspot{plane:02d}.dat'))
    
    return {
        'complete': complete,
        'total': N_REALIZATIONS * (N_PLANES + 1),
        'missing': missing
    }


def check_raytracing(model):
    """Check ray-tracing completion status for a model."""
    model_dir = os.path.join(RT_BASE, model)
    
    if not os.path.exists(model_dir):
        return {'complete': 0, 'total': N_REALIZATIONS * N_RT_RUNS * N_KAPPA_PLANES, 'missing': []}
    
    complete = 0
    missing = []
    
    for real in range(N_REALIZATIONS):
        lp_dir = os.path.join(model_dir, f'LP_{real:02d}')
        
        for run in range(1, N_RT_RUNS + 1):
            run_dir = os.path.join(lp_dir, f'run{run:03d}')
            
            for plane in range(1, N_KAPPA_PLANES + 1):
                filepath = os.path.join(run_dir, f'kappa_{plane:02d}.dat')
                if os.path.exists(filepath):
                    complete += 1
                else:
                    missing.append((real, run, plane))
    
    return {
        'complete': complete,
        'total': N_REALIZATIONS * N_RT_RUNS * N_KAPPA_PLANES,
        'missing': missing
    }


def print_model_status(model, verbose=False):
    """Print status for a single model."""
    lp = check_lensplanes(model)
    lux = check_lux_conversion(model)
    rt = check_raytracing(model)
    
    lp_pct = 100 * lp['complete'] / lp['total'] if lp['total'] > 0 else 0
    lux_pct = 100 * lux['complete'] / lux['total'] if lux['total'] > 0 else 0
    rt_pct = 100 * rt['complete'] / rt['total'] if rt['total'] > 0 else 0
    
    status = '✓' if lp_pct == 100 and lux_pct == 100 and rt_pct == 100 else '○'
    
    print(f"{status} {model[:60]:<60}")
    print(f"    LP: {lp['complete']:>5}/{lp['total']} ({lp_pct:5.1f}%)  "
          f"Lux: {lux['complete']:>5}/{lux['total']} ({lux_pct:5.1f}%)  "
          f"RT: {rt['complete']:>8}/{rt['total']} ({rt_pct:5.1f}%)")
    
    if verbose and (lp['missing'] or lux['missing'] or rt['missing']):
        if lp['missing']:
            print(f"    Missing LP: {len(lp['missing'])} files")
        if lux['missing']:
            print(f"    Missing Lux: {len(lux['missing'])} files")
        if rt['missing']:
            print(f"    Missing RT: {len(rt['missing'])} files")
    
    return {
        'lp': lp,
        'lux': lux,
        'rt': rt
    }


def print_summary(results):
    """Print summary statistics."""
    total_lp = sum(r['lp']['complete'] for r in results.values())
    total_lp_exp = sum(r['lp']['total'] for r in results.values())
    
    total_lux = sum(r['lux']['complete'] for r in results.values())
    total_lux_exp = sum(r['lux']['total'] for r in results.values())
    
    total_rt = sum(r['rt']['complete'] for r in results.values())
    total_rt_exp = sum(r['rt']['total'] for r in results.values())
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Models: {len(results)}")
    print(f"Lens planes:     {total_lp:>10} / {total_lp_exp} ({100*total_lp/total_lp_exp:.1f}%)")
    print(f"Lux conversion:  {total_lux:>10} / {total_lux_exp} ({100*total_lux/total_lux_exp:.1f}%)")
    print(f"Ray-tracing:     {total_rt:>10} / {total_rt_exp} ({100*total_rt/total_rt_exp:.1f}%)")
    
    # Count complete models
    complete = sum(1 for r in results.values() 
                   if r['lp']['complete'] == r['lp']['total'] 
                   and r['lux']['complete'] == r['lux']['total']
                   and r['rt']['complete'] == r['rt']['total'])
    print(f"\nComplete models: {complete} / {len(results)}")


def print_missing_configs(results):
    """Print missing configurations for targeted reruns."""
    print("\n" + "=" * 70)
    print("MISSING CONFIGURATIONS")
    print("=" * 70)
    
    for model, status in results.items():
        if status['lp']['missing']:
            # Group by snapshot
            snap_missing = {}
            for real, plane in status['lp']['missing']:
                snap_idx = plane // 2
                snap = SNAPSHOTS[snap_idx]
                if snap not in snap_missing:
                    snap_missing[snap] = set()
                snap_missing[snap].add(real)
            
            print(f"\n{model}:")
            for snap in sorted(snap_missing.keys()):
                reals = sorted(snap_missing[snap])
                print(f"  Snap {snap}: realizations {reals}")


def main():
    parser = argparse.ArgumentParser(description='Check pipeline progress')
    parser.add_argument('--model', type=str, help='Check single model')
    parser.add_argument('--missing', action='store_true', help='Show missing configs')
    parser.add_argument('--summary', action='store_true', help='Summary only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BINNED PIPELINE PROGRESS CHECK")
    print("=" * 70)
    print(f"LP base:  {LP_BASE}")
    print(f"Lux base: {LUX_BASE}")
    print(f"RT base:  {RT_BASE}")
    print("=" * 70)
    
    if args.model:
        models = [args.model]
    else:
        models = generate_binned_model_names()
    
    print(f"\nChecking {len(models)} models...")
    print("-" * 70)
    
    results = {}
    for model in models:
        results[model] = print_model_status(model, args.verbose)
    
    if not args.summary:
        print_summary(results)
    
    if args.missing:
        print_missing_configs(results)


if __name__ == '__main__':
    main()
