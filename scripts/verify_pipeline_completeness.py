#!/usr/bin/env python3
"""
Verify the completeness of all data products after recovery.

Checks:
1. All 34 models have 10 realizations
2. Each realization has 40 lensplanes (LP and LUX format)
3. Each realization has 100 ray-tracing runs with kappa40.dat
4. Total convergence maps per model >= 500 (goal)

Usage:
    python3 verify_pipeline_completeness.py
"""

import os
import glob
from collections import defaultdict

# Paths
LP_BASE = "/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG"
LUX_BASE = "/mnt/home/mlee1/ceph/hydro_replace_LP_lux/L205n2500TNG"
RT_BASE = "/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG"

# Expected counts
N_LENSPLANES = 40
N_REALIZATIONS = 10
N_RT_RUNS = 100
GOAL_KAPPA_MAPS = 500  # per model

# Build model list
MODELS = ["dmo", "hydro"]
MASS_CONFIGS = [
    "Ml_1.00e12_Mu_3.16e12", "Ml_1.00e12_Mu_inf",
    "Ml_3.16e12_Mu_1.00e13", "Ml_3.16e12_Mu_inf",
    "Ml_1.00e13_Mu_3.16e13", "Ml_1.00e13_Mu_inf",
    "Ml_3.16e13_Mu_1.00e15", "Ml_3.16e13_Mu_inf",
]
R_FACTORS = ["0.5", "1.0", "3.0", "5.0"]

for mass in MASS_CONFIGS:
    for r in R_FACTORS:
        MODELS.append(f"hydro_replace_{mass}_R_{r}")


def check_completeness():
    """Check completeness of all data products."""
    results = {
        'lp_complete': [],
        'lp_incomplete': [],
        'lux_complete': [],
        'lux_incomplete': [],
        'rt_complete': [],
        'rt_incomplete': [],
        'goal_met': [],
        'goal_not_met': [],
    }
    
    print("="*100)
    print(f"{'Model':<55} {'LP':<12} {'LUX':<12} {'RT':<12} {'Kappa':<12}")
    print("="*100)
    
    for model in MODELS:
        # Count LP files
        lp_count = 0
        for r in range(N_REALIZATIONS):
            lp_dir = os.path.join(LP_BASE, model, f"LP_{r:02d}")
            n = len(glob.glob(os.path.join(lp_dir, "lenspot*.dat")))
            lp_count += n
        lp_expected = N_LENSPLANES * N_REALIZATIONS
        lp_status = "OK" if lp_count == lp_expected else f"{lp_count}/{lp_expected}"
        
        # Count LUX files
        lux_count = 0
        for r in range(N_REALIZATIONS):
            lux_dir = os.path.join(LUX_BASE, model, f"LP_{r:02d}")
            n = len(glob.glob(os.path.join(lux_dir, "lenspot*.dat")))
            lux_count += n
        lux_status = "OK" if lux_count == lp_expected else f"{lux_count}/{lp_expected}"
        
        # Count RT runs with kappa40.dat
        rt_count = 0
        for r in range(N_REALIZATIONS):
            rt_dir = os.path.join(RT_BASE, model, f"LP_{r:02d}")
            n = len(glob.glob(os.path.join(rt_dir, "run*/kappa40.dat")))
            rt_count += n
        rt_expected = N_RT_RUNS * N_REALIZATIONS
        rt_status = "OK" if rt_count == rt_expected else f"{rt_count}/{rt_expected}"
        
        # Kappa goal
        kappa_status = "OK" if rt_count >= GOAL_KAPPA_MAPS else f"{rt_count}/{GOAL_KAPPA_MAPS}"
        
        print(f"{model:<55} {lp_status:<12} {lux_status:<12} {rt_status:<12} {kappa_status:<12}")
        
        # Categorize
        if lp_count == lp_expected:
            results['lp_complete'].append(model)
        else:
            results['lp_incomplete'].append((model, lp_count, lp_expected))
            
        if lux_count == lp_expected:
            results['lux_complete'].append(model)
        else:
            results['lux_incomplete'].append((model, lux_count, lp_expected))
            
        if rt_count == rt_expected:
            results['rt_complete'].append(model)
        else:
            results['rt_incomplete'].append((model, rt_count, rt_expected))
            
        if rt_count >= GOAL_KAPPA_MAPS:
            results['goal_met'].append(model)
        else:
            results['goal_not_met'].append((model, rt_count))
    
    print("="*100)
    
    # Summary
    print("\nSUMMARY")
    print("-"*50)
    print(f"LP complete:   {len(results['lp_complete'])}/{len(MODELS)}")
    print(f"LUX complete:  {len(results['lux_complete'])}/{len(MODELS)}")
    print(f"RT complete:   {len(results['rt_complete'])}/{len(MODELS)}")
    print(f"Goal met (≥{GOAL_KAPPA_MAPS}): {len(results['goal_met'])}/{len(MODELS)}")
    
    # Report issues
    if results['lp_incomplete']:
        print(f"\nLP INCOMPLETE ({len(results['lp_incomplete'])} models):")
        for model, count, expected in results['lp_incomplete']:
            print(f"  {model}: {count}/{expected}")
    
    if results['lux_incomplete']:
        print(f"\nLUX INCOMPLETE ({len(results['lux_incomplete'])} models):")
        for model, count, expected in results['lux_incomplete']:
            print(f"  {model}: {count}/{expected}")
    
    if results['goal_not_met']:
        print(f"\nGOAL NOT MET ({len(results['goal_not_met'])} models):")
        for model, count in results['goal_not_met']:
            print(f"  {model}: {count}/{GOAL_KAPPA_MAPS}")
    
    return results


if __name__ == "__main__":
    check_completeness()
