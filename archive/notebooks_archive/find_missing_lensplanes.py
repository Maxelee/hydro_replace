"""
find_missing_lensplanes.py

Analyze missing lenspot files and generate a targeted re-run list.
Maps missing lenspot indices to snapshot numbers for the unified pipeline.
"""

import os
from pathlib import Path
from collections import defaultdict
import json

# Directories
LP_SOURCE = Path('/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG')

EXPECTED_LENSPOTS = 40  # lenspot00.dat to lenspot39.dat
N_LPS = 10  # LP_00 to LP_09
PLANES_PER_SNAPSHOT = 2

# Snapshot mapping: index -> snapshot number
# lenspot index = snapshot_index * 2 + plane_within_snapshot
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]

def lenspot_to_snapshot(lenspot_idx):
    """Convert lenspot index to (snapshot_number, snapshot_array_index)."""
    snap_idx = lenspot_idx // PLANES_PER_SNAPSHOT
    return SNAPSHOT_ORDER[snap_idx], snap_idx

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


def analyze_missing():
    """Analyze all missing lenspots and group by snapshot."""
    
    # Track: snapshot_idx -> list of (model, LP_id, missing_lenspots)
    snapshot_reruns = defaultdict(list)
    
    # Also track by model for summary
    model_missing = defaultdict(lambda: defaultdict(list))
    
    for model in MODELS:
        for lp_id in range(N_LPS):
            lp_name = f'LP_{lp_id:02d}'
            source_dir = LP_SOURCE / model / lp_name
            
            missing = find_missing_lenspots(source_dir)
            if not missing:
                continue
            
            # Group missing by snapshot
            for lenspot_idx in missing:
                snap_num, snap_idx = lenspot_to_snapshot(lenspot_idx)
                snapshot_reruns[snap_idx].append({
                    'model': model,
                    'lp_id': lp_id,
                    'lenspot_idx': lenspot_idx,
                    'snapshot': snap_num,
                })
                model_missing[model][lp_id].append(lenspot_idx)
    
    return snapshot_reruns, model_missing


def generate_rerun_info():
    """Generate detailed re-run information."""
    
    snapshot_reruns, model_missing = analyze_missing()
    
    print("="*80)
    print("MISSING LENSPLANE ANALYSIS")
    print("="*80)
    
    # Summary by snapshot
    print("\n## Missing by Snapshot ##\n")
    
    snapshot_needs_rerun = set()
    for snap_idx in sorted(snapshot_reruns.keys()):
        snap_num = SNAPSHOT_ORDER[snap_idx]
        items = snapshot_reruns[snap_idx]
        
        # Count unique (model, lp_id) combinations
        unique_combos = set((x['model'], x['lp_id']) for x in items)
        
        print(f"Snapshot {snap_num} (array index {snap_idx}): {len(unique_combos)} (model, LP) combinations need re-run")
        snapshot_needs_rerun.add(snap_idx)
    
    # Create SLURM array string
    array_indices = sorted(snapshot_needs_rerun)
    
    # Group consecutive indices
    ranges = []
    if array_indices:
        start = array_indices[0]
        end = array_indices[0]
        
        for idx in array_indices[1:]:
            if idx == end + 1:
                end = idx
            else:
                if start == end:
                    ranges.append(f"{start}")
                else:
                    ranges.append(f"{start}-{end}")
                start = idx
                end = idx
        
        if start == end:
            ranges.append(f"{start}")
        else:
            ranges.append(f"{start}-{end}")
    
    array_str = ",".join(ranges) if ranges else ""
    
    print(f"\n## SLURM Array Specification ##")
    print(f"#SBATCH --array={array_str}")
    print(f"Total snapshots to re-run: {len(snapshot_needs_rerun)}")
    
    # Summary by model
    print(f"\n## Models with Missing Data ##")
    print(f"Total models affected: {len(model_missing)}")
    
    for model in sorted(model_missing.keys()):
        lp_data = model_missing[model]
        total_missing = sum(len(v) for v in lp_data.values())
        print(f"  {model}: {total_missing} missing files across {len(lp_data)} LPs")
    
    # Write detailed info to JSON for the batch script
    rerun_data = {
        'array_string': array_str,
        'snapshots_to_rerun': list(snapshot_needs_rerun),
        'snapshot_order': SNAPSHOT_ORDER,
        'details': {}
    }
    
    for snap_idx, items in snapshot_reruns.items():
        snap_num = SNAPSHOT_ORDER[snap_idx]
        rerun_data['details'][str(snap_idx)] = {
            'snapshot': snap_num,
            'models': list(set(x['model'] for x in items)),
            'n_combinations': len(set((x['model'], x['lp_id']) for x in items)),
        }
    
    with open('missing_lensplane_reruns.json', 'w') as f:
        json.dump(rerun_data, f, indent=2)
    
    print(f"\nDetailed info written to: missing_lensplane_reruns.json")
    
    return array_str, snapshot_needs_rerun, model_missing


if __name__ == "__main__":
    array_str, snapshots, models = generate_rerun_info()
    
    print("\n" + "="*80)
    print("RECOMMENDED NEXT STEPS")
    print("="*80)
    print(f"""
1. Run the targeted re-run script:
   sbatch batch/run_unified_2500_missing.sh

2. After completion, re-run lensplane diagnostics:
   python notebooks/diagnose_lensplane_completeness.py

3. Then run the lux ray-tracing for missing models:
   sbatch batch/run_lux_missing.sh
""")
