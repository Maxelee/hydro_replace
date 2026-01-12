"""
find_missing_lensing.py

Find all (model, LP) combinations that have incomplete ray-tracing output.
Generate a list of missing tasks for targeted SLURM array jobs.
"""

import os
from pathlib import Path

# Configuration
RT_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG')
N_RUNS = 50  # We expect 50 runs per LP (not 100)
N_LPS = 10   # 10 lens planes
Z_SNAP = 23  # kappa23.dat

# Model list (same as in batch script)
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

def count_complete_runs(model, lp_id):
    """Count how many runs have the final kappa file for a given (model, LP)."""
    lp_dir = RT_BASE / model / f'LP_{lp_id:02d}'
    
    if not lp_dir.exists():
        return 0
    
    complete = 0
    for run_id in range(1, N_RUNS + 1):
        kappa_file = lp_dir / f'run{run_id:03d}' / f'kappa{Z_SNAP:02d}.dat'
        if kappa_file.exists():
            complete += 1
    
    return complete

def find_all_missing():
    """Find all (model, LP) combinations that are not complete."""
    
    print("Scanning for missing ray-tracing runs...")
    print("=" * 80)
    
    missing = []
    summary = {}
    
    for model_idx, model in enumerate(MODELS):
        summary[model] = {'complete': 0, 'incomplete': 0, 'missing_lps': []}
        
        for lp_id in range(N_LPS):
            n_complete = count_complete_runs(model, lp_id)
            
            if n_complete == N_RUNS:
                summary[model]['complete'] += 1
            else:
                summary[model]['incomplete'] += 1
                summary[model]['missing_lps'].append((lp_id, n_complete))
                missing.append((model_idx, model, lp_id, n_complete))
    
    # Print summary
    print(f"\nSummary by model (expecting {N_RUNS} runs per LP, {N_LPS} LPs):")
    print("-" * 80)
    
    for model in MODELS:
        s = summary[model]
        if s['incomplete'] > 0:
            print(f"\n{model}:")
            print(f"  Complete LPs: {s['complete']}/{N_LPS}")
            print(f"  Incomplete LPs: {s['missing_lps']}")
        else:
            print(f"{model}: ✓ Complete ({N_LPS}/{N_LPS} LPs)")
    
    print("\n" + "=" * 80)
    print(f"TOTAL MISSING (model, LP) combinations: {len(missing)}")
    
    return missing, summary

def generate_array_indices(missing):
    """
    Generate SLURM array indices for missing (model, LP) combinations.
    
    The original array job uses: array_id = model_idx * N_LPS + lp_id
    """
    
    indices = []
    for model_idx, model, lp_id, n_complete in missing:
        array_idx = model_idx * N_LPS + lp_id
        indices.append(array_idx)
    
    return sorted(indices)

def format_array_string(indices):
    """Format indices into SLURM array specification."""
    if not indices:
        return ""
    
    # Group consecutive indices into ranges
    ranges = []
    start = indices[0]
    end = indices[0]
    
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{end}")
            start = idx
            end = idx
    
    # Add last range
    if start == end:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{end}")
    
    return ",".join(ranges)

def main():
    missing, summary = find_all_missing()
    
    if not missing:
        print("\n✓ All models and LPs are complete!")
        return
    
    # Generate array indices
    indices = generate_array_indices(missing)
    array_str = format_array_string(indices)
    
    print(f"\nSLURM array indices for missing jobs:")
    print(f"  #SBATCH --array={array_str}")
    print(f"\n  Total missing tasks: {len(indices)}")
    
    # Print list format too
    print(f"\n  As list: {indices}")
    
    # Show which specific (model, LP) each index corresponds to
    print(f"\n  Index mapping (model_idx * 10 + lp_id):")
    for model_idx, model, lp_id, n_complete in missing[:10]:  # Show first 10
        array_idx = model_idx * N_LPS + lp_id
        print(f"    {array_idx}: {model} LP_{lp_id:02d} ({n_complete}/{N_RUNS} complete)")
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more")
    
    # Write to file for easy copy-paste
    with open('missing_lensing_tasks.txt', 'w') as f:
        f.write(f"# Missing ray-tracing tasks found: {len(indices)}\n")
        f.write(f"# SLURM array specification:\n")
        f.write(f"#SBATCH --array={array_str}\n")
        f.write(f"\n# Detailed list:\n")
        for model_idx, model, lp_id, n_complete in missing:
            array_idx = model_idx * N_LPS + lp_id
            f.write(f"{array_idx}: {model} LP_{lp_id:02d} ({n_complete}/{N_RUNS} complete)\n")
    
    print(f"\n  Written to: missing_lensing_tasks.txt")

if __name__ == "__main__":
    main()
