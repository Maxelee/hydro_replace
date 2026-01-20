"""
diagnose_lensplane_completeness.py

Check which models have incomplete lens plane files in both:
1. Source LP directory (hydro_replace_LP) 
2. Lux-format LP directory (hydro_replace_LP_lux)

This helps trace where the pipeline is breaking.
"""

import os
from pathlib import Path
from collections import defaultdict

# Directories
LP_SOURCE = Path('/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG')
LP_LUX = Path('/mnt/home/mlee1/ceph/hydro_replace_LP_lux/L205n2500TNG')

EXPECTED_LENSPOTS = 40  # lenspot00.dat to lenspot39.dat
N_LPS = 10  # LP_00 to LP_09

# Model list
MODELS = ['dmo', 'hydro']
MASS_CONFIGS = [
    "Ml_1.00e12_Mu_3.16e12",
    "Ml_1.00e12_Mu_inf",
    "Ml_3.16e12_Mu_1.00e13",
    "Ml_3.16e12_Mu_inf",
    "Ml_1.00e13_Mu_3.16e13",
    "Ml_1.00e13_Mu_inf",
    "Ml_3.16e13_Mu_1.00e15",
    "Ml_3.16e13_Mu_inf",
]
R_FACTORS = ["0.5", "1.0", "3.0", "5.0"]

for mass_config in MASS_CONFIGS:
    for r_factor in R_FACTORS:
        MODELS.append(f"hydro_replace_{mass_config}_R_{r_factor}")


def count_lenspots(directory):
    """Count lenspot files in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob('lenspot*.dat')))


def find_missing_lenspots(directory):
    """Find which specific lenspot files are missing."""
    if not directory.exists():
        return list(range(EXPECTED_LENSPOTS))
    
    existing = set()
    for f in directory.glob('lenspot*.dat'):
        try:
            idx = int(f.stem.replace('lenspot', ''))
            existing.add(idx)
        except ValueError:
            pass
    
    expected = set(range(EXPECTED_LENSPOTS))
    return sorted(expected - existing)


def diagnose_all():
    """Run full diagnostics."""
    
    print("="*80)
    print("LENS PLANE COMPLETENESS DIAGNOSTICS")
    print("="*80)
    
    # Track issues
    source_incomplete = []
    lux_incomplete = []
    
    # Check each model
    for model in MODELS:
        model_issues_source = []
        model_issues_lux = []
        
        for lp_id in range(N_LPS):
            lp_name = f'LP_{lp_id:02d}'
            
            # Check source
            source_dir = LP_SOURCE / model / lp_name
            n_source = count_lenspots(source_dir)
            
            # Check lux format
            lux_dir = LP_LUX / model / lp_name
            n_lux = count_lenspots(lux_dir)
            
            if n_source < EXPECTED_LENSPOTS:
                missing = find_missing_lenspots(source_dir)
                model_issues_source.append((lp_id, n_source, missing))
            
            if n_lux < EXPECTED_LENSPOTS:
                missing = find_missing_lenspots(lux_dir)
                model_issues_lux.append((lp_id, n_lux, missing))
        
        if model_issues_source:
            source_incomplete.append((model, model_issues_source))
        if model_issues_lux:
            lux_incomplete.append((model, model_issues_lux))
    
    # Report source issues
    print("\n" + "="*80)
    print("SOURCE LP DIRECTORY ISSUES (hydro_replace_LP)")
    print("="*80)
    
    if not source_incomplete:
        print("✓ All source LP directories are complete!")
    else:
        print(f"\nIncomplete models: {len(source_incomplete)}/{len(MODELS)}")
        for model, issues in source_incomplete:
            print(f"\n{model}:")
            for lp_id, n_files, missing in issues:
                print(f"  LP_{lp_id:02d}: {n_files}/{EXPECTED_LENSPOTS} files")
                print(f"    Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    # Report lux issues
    print("\n" + "="*80)
    print("LUX-FORMAT LP DIRECTORY ISSUES (hydro_replace_LP_lux)")
    print("="*80)
    
    if not lux_incomplete:
        print("✓ All lux LP directories are complete!")
    else:
        print(f"\nIncomplete models: {len(lux_incomplete)}/{len(MODELS)}")
        for model, issues in lux_incomplete:
            print(f"\n{model}:")
            for lp_id, n_files, missing in issues:
                print(f"  LP_{lp_id:02d}: {n_files}/{EXPECTED_LENSPOTS} files")
                print(f"    Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Check for pattern - are the same models missing in both?
    source_models = set(m for m, _ in source_incomplete)
    lux_models = set(m for m, _ in lux_incomplete)
    
    both = source_models & lux_models
    source_only = source_models - lux_models
    lux_only = lux_models - source_models
    
    print(f"\nSource incomplete: {len(source_models)} models")
    print(f"Lux incomplete: {len(lux_models)} models")
    print(f"Both incomplete: {len(both)} models")
    
    if both:
        print(f"\n⚠️  Models incomplete in BOTH stages (upstream issue):")
        for m in sorted(both)[:10]:
            print(f"    {m}")
        if len(both) > 10:
            print(f"    ... and {len(both)-10} more")
    
    if lux_only:
        print(f"\n⚠️  Models incomplete ONLY in lux (conversion issue):")
        for m in sorted(lux_only):
            print(f"    {m}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if source_incomplete:
        print("\n1. SOURCE ISSUE: Re-run run_unified_2500_array.sh for missing snapshots")
        print("   The lens plane generation did not complete for some models.")
        print("   Check logs/unified_2500_* for errors.")
    
    if lux_only:
        print("\n2. CONVERSION ISSUE: Re-run convert_to_lensplanes.py for affected models")
    
    if not source_incomplete and not lux_incomplete:
        print("\n✓ All lens planes are complete. Issue may be elsewhere.")
    
    return source_incomplete, lux_incomplete


if __name__ == "__main__":
    diagnose_all()
