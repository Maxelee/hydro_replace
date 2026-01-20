#!/usr/bin/env python
"""
Quick data validation script - checks actual data availability.
"""

import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from constants import RT_BASE, LP_BASE

def check_rt_data():
    """Check ray-tracing data availability."""
    print("=" * 80)
    print("CHECKING RAY-TRACING DATA (Convergence Maps)")
    print("=" * 80)
    
    print(f"\nBase directory: {RT_BASE}")
    
    if not os.path.exists(RT_BASE):
        print(f"✗ Directory does not exist!")
        return
    
    # Check DMO
    dmo_dir = os.path.join(RT_BASE, 'dmo')
    if os.path.exists(dmo_dir):
        lp_dirs = glob.glob(os.path.join(dmo_dir, 'LP_*'))
        print(f"\n✓ DMO: {len(lp_dirs)} LP directories")
        
        if lp_dirs:
            # Check first LP
            run_dirs = glob.glob(os.path.join(lp_dirs[0], 'run*'))
            print(f"  LP_00: {len(run_dirs)} run directories")
            
            if run_dirs:
                kappa_files = glob.glob(os.path.join(run_dirs[0], 'kappa*.dat'))
                print(f"  run001: {len(kappa_files)} kappa files")
                if kappa_files:
                    print(f"  Example: {os.path.basename(kappa_files[0])}")
    else:
        print(f"✗ DMO directory not found")
    
    # Check Hydro
    hydro_dir = os.path.join(RT_BASE, 'hydro')
    if os.path.exists(hydro_dir):
        lp_dirs = glob.glob(os.path.join(hydro_dir, 'LP_*'))
        print(f"\n✓ Hydro: {len(lp_dirs)} LP directories")
    else:
        print(f"✗ Hydro directory not found")
    
    # Check Replace models
    replace_dirs = glob.glob(os.path.join(RT_BASE, 'hydro_replace_*'))
    print(f"\n✓ Replace models: {len(replace_dirs)} found")
    
    # Filter for models with Ml, Mu, Ri, Ro format
    valid_replace = []
    for d in replace_dirs:
        basename = os.path.basename(d)
        if '_Ml_' in basename and '_Mu_' in basename and '_Ri_' in basename and '_Ro_' in basename:
            valid_replace.append(basename)
    
    print(f"  Valid format (Ml_Mu_Ri_Ro): {len(valid_replace)}")
    if valid_replace:
        print(f"\n  First 5 models:")
        for model in valid_replace[:5]:
            print(f"    - {model}")


def check_lp_data():
    """Check lensplane data availability."""
    print("\n" + "=" * 80)
    print("CHECKING LENSPLANE DATA (Density Fields)")
    print("=" * 80)
    
    print(f"\nBase directory: {LP_BASE}")
    
    if not os.path.exists(LP_BASE):
        print(f"✗ Directory does not exist!")
        return
    
    # Check DMO
    dmo_dir = os.path.join(LP_BASE, 'dmo')
    if os.path.exists(dmo_dir):
        lp_dirs = glob.glob(os.path.join(dmo_dir, 'LP_*'))
        print(f"\n✓ DMO: {len(lp_dirs)} LP directories")
        
        if lp_dirs:
            lenspot_files = glob.glob(os.path.join(lp_dirs[0], 'lenspot*.dat'))
            print(f"  LP_00: {len(lenspot_files)} lenspot files")
            if lenspot_files:
                print(f"  Example: {os.path.basename(lenspot_files[0])}")
    else:
        print(f"✗ DMO directory not found")
    
    # Check Hydro
    hydro_dir = os.path.join(LP_BASE, 'hydro')
    if os.path.exists(hydro_dir):
        lp_dirs = glob.glob(os.path.join(hydro_dir, 'LP_*'))
        print(f"\n✓ Hydro: {len(lp_dirs)} LP directories")
    else:
        print(f"✗ Hydro directory not found")
    
    # Check Replace models
    replace_dirs = glob.glob(os.path.join(LP_BASE, 'hydro_replace_*'))
    print(f"\n✓ Replace models: {len(replace_dirs)} found")
    
    # Filter for valid format
    valid_replace = []
    for d in replace_dirs:
        basename = os.path.basename(d)
        if '_Ml_' in basename and '_Mu_' in basename and '_Ri_' in basename and '_Ro_' in basename:
            valid_replace.append(basename)
    
    print(f"  Valid format (Ml_Mu_Ri_Ro): {len(valid_replace)}")
    if valid_replace:
        print(f"\n  First 5 models:")
        for model in valid_replace[:5]:
            # Check if this model has data
            model_dir = os.path.join(LP_BASE, model)
            lp_dirs = glob.glob(os.path.join(model_dir, 'LP_*'))
            print(f"    - {model:50s} ({len(lp_dirs)} LPs)")


def get_available_replace_models():
    """Get list of Replace models that have both RT and LP data."""
    print("\n" + "=" * 80)
    print("MODELS WITH COMPLETE DATA (Both RT and LP)")
    print("=" * 80)
    
    # Get RT models
    rt_models = set()
    for d in glob.glob(os.path.join(RT_BASE, 'hydro_replace_*')):
        basename = os.path.basename(d)
        if '_Ml_' in basename and '_Mu_' in basename and '_Ri_' in basename and '_Ro_' in basename:
            rt_models.add(basename)
    
    # Get LP models
    lp_models = set()
    for d in glob.glob(os.path.join(LP_BASE, 'hydro_replace_*')):
        basename = os.path.basename(d)
        if '_Ml_' in basename and '_Mu_' in basename and '_Ri_' in basename and '_Ro_' in basename:
            lp_models.add(basename)
    
    # Find intersection
    complete_models = sorted(rt_models & lp_models)
    
    print(f"\nModels with RT data only: {len(rt_models - lp_models)}")
    print(f"Models with LP data only: {len(lp_models - rt_models)}")
    print(f"Models with both: {len(complete_models)}")
    
    if complete_models:
        print(f"\nComplete models (first 10):")
        for model in complete_models[:10]:
            print(f"  - {model}")
    
    return complete_models


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "DATA AVAILABILITY CHECK" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    
    check_rt_data()
    check_lp_data()
    complete_models = get_available_replace_models()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"RT base exists: {os.path.exists(RT_BASE)}")
    print(f"LP base exists: {os.path.exists(LP_BASE)}")
    print(f"Complete Replace models: {len(complete_models)}")
    
    if len(complete_models) > 0:
        print("\n✓ Data validation successful!")
        print(f"\nYou can use these models for testing:")
        for model in complete_models[:3]:
            print(f"  python scripts/compute_all_stats.py --model {model}")
    else:
        print("\n✗ No complete Replace models found!")
    
    print()


if __name__ == '__main__':
    main()
