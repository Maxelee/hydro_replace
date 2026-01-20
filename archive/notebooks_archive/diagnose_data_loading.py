"""
diagnose_data_loading.py

Check how many convergence maps were successfully loaded for each model.
"""

import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

STATS_CACHE_DIR = Path('./statistics_cache')
RT_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG')

EXPECTED_PER_MODEL = 500  # 10 lens planes × 50 runs
Z_SNAP = 23

CUMULATIVE_MODELS = [
    'hydro_replace_Ml_1.00e12_Mu_inf_R_0.5',
    'hydro_replace_Ml_1.00e12_Mu_inf_R_1.0',
    'hydro_replace_Ml_1.00e12_Mu_inf_R_3.0',
    'hydro_replace_Ml_1.00e12_Mu_inf_R_5.0',
    'hydro_replace_Ml_3.16e12_Mu_inf_R_0.5',
    'hydro_replace_Ml_3.16e12_Mu_inf_R_1.0',
    'hydro_replace_Ml_3.16e12_Mu_inf_R_3.0',
    'hydro_replace_Ml_3.16e12_Mu_inf_R_5.0',
    'hydro_replace_Ml_1.00e13_Mu_inf_R_0.5',
    'hydro_replace_Ml_1.00e13_Mu_inf_R_1.0',
    'hydro_replace_Ml_1.00e13_Mu_inf_R_3.0',
    'hydro_replace_Ml_1.00e13_Mu_inf_R_5.0',
    'hydro_replace_Ml_3.16e13_Mu_inf_R_0.5',
    'hydro_replace_Ml_3.16e13_Mu_inf_R_1.0',
    'hydro_replace_Ml_3.16e13_Mu_inf_R_3.0',
    'hydro_replace_Ml_3.16e13_Mu_inf_R_5.0',
]

DIFFERENTIAL_MODELS = [
    'hydro_replace_Ml_1.00e12_Mu_3.16e12_R_0.5',
    'hydro_replace_Ml_1.00e12_Mu_3.16e12_R_1.0',
    'hydro_replace_Ml_1.00e12_Mu_3.16e12_R_3.0',
    'hydro_replace_Ml_1.00e12_Mu_3.16e12_R_5.0',
    'hydro_replace_Ml_3.16e12_Mu_1.00e13_R_0.5',
    'hydro_replace_Ml_3.16e12_Mu_1.00e13_R_1.0',
    'hydro_replace_Ml_3.16e12_Mu_1.00e13_R_3.0',
    'hydro_replace_Ml_3.16e12_Mu_1.00e13_R_5.0',
    'hydro_replace_Ml_1.00e13_Mu_3.16e13_R_0.5',
    'hydro_replace_Ml_1.00e13_Mu_3.16e13_R_1.0',
    'hydro_replace_Ml_1.00e13_Mu_3.16e13_R_3.0',
    'hydro_replace_Ml_1.00e13_Mu_3.16e13_R_5.0',
    'hydro_replace_Ml_3.16e13_Mu_1.00e15_R_0.5',
    'hydro_replace_Ml_3.16e13_Mu_1.00e15_R_1.0',
    'hydro_replace_Ml_3.16e13_Mu_1.00e15_R_3.0',
    'hydro_replace_Ml_3.16e13_Mu_1.00e15_R_5.0',
]

ALL_MODELS = ['dmo', 'hydro'] + CUMULATIVE_MODELS + DIFFERENTIAL_MODELS

# =============================================================================
# Diagnostic functions
# =============================================================================

def check_cached_statistics():
    """Check how many realizations were loaded for each model."""
    
    print("="*80)
    print("DIAGNOSTICS: Checking cached statistics")
    print("="*80)
    
    results = {}
    
    for model in ALL_MODELS:
        cache_file = STATS_CACHE_DIR / f'{model}_z{Z_SNAP:02d}_stats.h5'
        
        if not cache_file.exists():
            print(f"\n{model}:")
            print(f"  ❌ NOT CACHED: {cache_file}")
            results[model] = {'status': 'missing', 'n_realizations': 0}
            continue
        
        try:
            with h5py.File(cache_file, 'r') as f:
                n_realizations = f.attrs['n_realizations']
                n_failed = f.attrs.get('n_failed', 0)
                
                # Check actual data shapes
                C_ell_shape = f['C_ell'].shape
                peaks_shape = f['peaks'].shape
                minima_shape = f['minima'].shape
                
                # Get LP and run distribution
                LP_ids = f['LP_ids'][:]
                run_ids = f['run_ids'][:]
                
                unique_LPs = len(np.unique(LP_ids))
                runs_per_LP = []
                for LP in range(10):
                    n_runs = np.sum(LP_ids == LP)
                    runs_per_LP.append(n_runs)
            
            completion = 100 * n_realizations / EXPECTED_PER_MODEL
            
            print(f"\n{model}:")
            print(f"  ✓ Cached: {n_realizations}/{EXPECTED_PER_MODEL} ({completion:.1f}%)")
            print(f"    Failed: {n_failed}")
            print(f"    Lens planes with data: {unique_LPs}/10")
            print(f"    Runs per LP: {runs_per_LP}")
            print(f"    Data shapes: C_ell={C_ell_shape}, peaks={peaks_shape}, minima={minima_shape}")
            
            # Check for uniformity
            if len(set(runs_per_LP)) > 2:  # More than slight variation
                print(f"    ⚠️  WARNING: Non-uniform LP coverage!")
            
            results[model] = {
                'status': 'ok',
                'n_realizations': n_realizations,
                'n_failed': n_failed,
                'completion': completion,
                'unique_LPs': unique_LPs,
                'runs_per_LP': runs_per_LP
            }
            
        except Exception as e:
            print(f"\n{model}:")
            print(f"  ❌ ERROR reading cache: {e}")
            results[model] = {'status': 'error', 'n_realizations': 0}
    
    return results

def check_raw_files():
    """Check which raw kappa files actually exist on disk."""
    
    print("\n" + "="*80)
    print("DIAGNOSTICS: Checking raw kappa files (sampling)")
    print("="*80)
    
    # Sample a few models to check disk state
    sample_models = ['dmo', 'hydro', CUMULATIVE_MODELS[0], DIFFERENTIAL_MODELS[0]]
    
    for model in sample_models:
        print(f"\n{model}:")
        
        n_found = 0
        n_missing = 0
        missing_examples = []
        
        for LP_id in range(10):
            for run_id in range(1, 51):  # Check all 50 runs
                path = RT_BASE / model / f'LP_{LP_id:02d}' / f'run{run_id:03d}' / f'kappa{Z_SNAP:02d}.dat'
                
                if path.exists():
                    n_found += 1
                else:
                    n_missing += 1
                    if len(missing_examples) < 3:  # Show first 3 examples
                        missing_examples.append(str(path))
        
        total = n_found + n_missing
        print(f"  Found: {n_found}/{total} ({100*n_found/total:.1f}%)")
        
        if missing_examples:
            print(f"  Missing examples:")
            for ex in missing_examples:
                print(f"    - {ex}")

def plot_completion_summary(results):
    """Plot summary of data completion."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Completion percentage by model
    ax = axes[0]
    
    models_with_data = [m for m, r in results.items() if r['status'] == 'ok']
    completions = [results[m]['completion'] for m in models_with_data]
    
    colors = ['green' if c > 95 else 'orange' if c > 80 else 'red' for c in completions]
    
    y_pos = np.arange(len(models_with_data))
    ax.barh(y_pos, completions, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[:30] for m in models_with_data], fontsize=7)
    ax.set_xlabel('Completion (%)', fontsize=11)
    ax.set_title('Data Completion by Model', fontsize=12)
    ax.axvline(100, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Expected')
    ax.axvline(95, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='95% threshold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Panel 2: Distribution of realizations
    ax = axes[1]
    
    n_realizations = [results[m]['n_realizations'] for m in models_with_data]
    
    ax.hist(n_realizations, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(EXPECTED_PER_MODEL, color='red', linestyle='--', linewidth=2, 
               label=f'Expected ({EXPECTED_PER_MODEL})')
    ax.set_xlabel('Number of realizations loaded', fontsize=11)
    ax.set_ylabel('Number of models', fontsize=11)
    ax.set_title('Distribution of Realizations Across Models', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/data_completion_diagnostic.pdf')
    print("\nSaved diagnostic plot: figures/data_completion_diagnostic.pdf")
    
    return fig

def compute_expected_errors():
    """
    Estimate expected error bars based on cosmic variance.
    """
    print("\n" + "="*80)
    print("Expected statistical uncertainties")
    print("="*80)
    
    # For convergence power spectrum and peaks:
    # Error scales as 1/sqrt(N_realizations * N_independent_modes)
    
    # For C_ell at fixed ell:
    # - Each ell mode has ~ell modes in the ring
    # - Each realization is independent
    # Expected fractional error: ~sqrt(2/N_modes) * 1/sqrt(N_real)
    
    print("\nFor C_ell:")
    print("  At ell=1000 with 500 realizations:")
    N_modes_ell = 1000  # Rough estimate
    N_real = 500
    frac_err_ell = np.sqrt(2.0 / N_modes_ell) / np.sqrt(N_real)
    print(f"    Expected fractional error: {frac_err_ell:.3f} (~{100*frac_err_ell:.1f}%)")
    
    print("\nFor peak counts:")
    print("  Depends on peak density and field area")
    print("  Typical ~few hundred peaks per map")
    print("  With 500 realizations:")
    N_peaks_per_map = 200  # Rough estimate
    N_real = 500
    # Poisson error on mean
    frac_err_peaks = 1.0 / np.sqrt(N_peaks_per_map * N_real)
    print(f"    Expected fractional error: {frac_err_peaks:.3f} (~{100*frac_err_peaks:.1f}%)")
    
    print("\nNote: If error bars are much larger than this, it suggests:")
    print("  1. High variation between lens planes (LSS variance)")
    print("  2. Small baryonic effects relative to noise")
    print("  3. Possible issues with some simulations")

def summary_statistics():
    """Print summary statistics for quick assessment."""
    
    results = check_cached_statistics()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    n_total = len(ALL_MODELS)
    n_cached = sum(1 for r in results.values() if r['status'] == 'ok')
    n_missing = sum(1 for r in results.values() if r['status'] == 'missing')
    n_error = sum(1 for r in results.values() if r['status'] == 'error')
    
    print(f"\nModels: {n_total} total")
    print(f"  ✓ Cached successfully: {n_cached}")
    print(f"  ❌ Not cached: {n_missing}")
    print(f"  ⚠️  Error reading: {n_error}")
    
    if n_cached > 0:
        realizations = [r['n_realizations'] for r in results.values() if r['status'] == 'ok']
        print(f"\nRealizations per model:")
        print(f"  Mean: {np.mean(realizations):.1f}")
        print(f"  Min: {np.min(realizations)}")
        print(f"  Max: {np.max(realizations)}")
        print(f"  Median: {np.median(realizations):.1f}")
        
        completions = [r['completion'] for r in results.values() if r['status'] == 'ok']
        print(f"\nCompletion rates:")
        print(f"  Mean: {np.mean(completions):.1f}%")
        print(f"  Models with >95%: {sum(1 for c in completions if c > 95)}/{n_cached}")
        print(f"  Models with <80%: {sum(1 for c in completions if c < 80)}/{n_cached}")
    
    # Plot summary
    if n_cached > 0:
        plot_completion_summary(results)
    
    return results

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Running data loading diagnostics...\n")
    
    # Check cached files
    results = summary_statistics()
    
    # Check raw files (sampling)
    check_raw_files()
    
    # Expected errors
    compute_expected_errors()
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Analyze results and give recommendations
    realizations = [r['n_realizations'] for r in results.values() if r['status'] == 'ok']
    
    if len(realizations) > 0:
        mean_real = np.mean(realizations)
        
        if mean_real < 100:
            print("\n⚠️  WARNING: Very few realizations loaded (<100)")
            print("   - Check if ray-tracing completed successfully")
            print("   - Verify file paths in driver script")
            print("   - Run check_raw_files() to see which files are missing")
        elif mean_real < 400:
            print("\n⚠️  Low number of realizations (100-400)")
            print("   - Some maps may be missing or failed quality checks")
            print("   - Statistical errors will be larger than expected")
            print("   - Check logs for file reading errors")
        else:
            print("\n✓ Good: Most models have sufficient realizations")
            print("  - If error bars are still large, this indicates:")
            print("    a) Real cosmic variance between lens planes")
            print("    b) Small baryonic effects relative to DMO scatter")
    
    print("\nTo investigate further:")
    print("  1. Check specific model: results['model_name']")
    print("  2. Look at runs_per_LP to see if some LPs are missing")
    print("  3. Check driver script logs for file loading errors")
    print("  4. Verify RT output actually exists on disk")
