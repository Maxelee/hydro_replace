#!/usr/bin/env python3
"""
Find all missing lensplanes and generate a recovery task list.

This script identifies:
1. Which model/realization/plane combinations are missing
2. Maps them back to the snapshots that need to be re-run
3. Generates a task list for batch recovery

Output: missing_lensplane_tasks.json
"""

import os
import glob
import json
from collections import defaultdict

# Paths
LP_BASE = "/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG"

# Configuration
N_LENSPLANES = 40
N_REALIZATIONS = 10
PPS = 2  # Planes per snapshot

# Snapshot order (same as in generate_all_unified.py)
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
SNAPSHOT_TO_INDEX = {snap: idx for idx, snap in enumerate(SNAPSHOT_ORDER)}

# Models
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


def plane_to_snapshot(plane_idx):
    """Convert plane index (0-39) to snapshot number."""
    snap_idx = plane_idx // PPS
    return SNAPSHOT_ORDER[snap_idx]


def find_missing_planes():
    """Find all missing lensplane files."""
    missing = []  # List of (model, realization, plane_idx, snapshot)
    
    for model in MODELS:
        for r in range(N_REALIZATIONS):
            real_dir = os.path.join(LP_BASE, model, f"LP_{r:02d}")
            
            # Get existing planes
            existing = set()
            for f in glob.glob(os.path.join(real_dir, "lenspot*.dat")):
                try:
                    idx = int(os.path.basename(f).replace("lenspot", "").replace(".dat", ""))
                    existing.add(idx)
                except ValueError:
                    pass
            
            # Find missing
            for p in range(N_LENSPLANES):
                if p not in existing:
                    snap = plane_to_snapshot(p)
                    missing.append({
                        'model': model,
                        'realization': r,
                        'plane_idx': p,
                        'snapshot': snap
                    })
    
    return missing


def group_by_snapshot(missing):
    """Group missing planes by snapshot for efficient regeneration."""
    by_snapshot = defaultdict(list)
    for item in missing:
        snap = item['snapshot']
        by_snapshot[snap].append(item)
    return dict(by_snapshot)


def generate_recovery_tasks(by_snapshot):
    """Generate recovery task list."""
    tasks = []
    
    for snap in sorted(by_snapshot.keys()):
        items = by_snapshot[snap]
        
        # Group by model - we need to re-run phase 5 for each model
        models_affected = defaultdict(list)
        for item in items:
            models_affected[item['model']].append(item['realization'])
        
        task = {
            'snapshot': snap,
            'snap_index': SNAPSHOT_ORDER.index(snap),
            'n_missing': len(items),
            'models': {m: sorted(set(reals)) for m, reals in models_affected.items()}
        }
        tasks.append(task)
    
    return tasks


def main():
    print("Finding missing lensplanes...")
    missing = find_missing_planes()
    
    print(f"Total missing: {len(missing)} plane files")
    
    by_snapshot = group_by_snapshot(missing)
    print(f"Snapshots with missing data: {len(by_snapshot)}")
    
    tasks = generate_recovery_tasks(by_snapshot)
    
    # Print summary
    print("\n" + "="*80)
    print("RECOVERY TASKS BY SNAPSHOT")
    print("="*80)
    for task in tasks:
        snap = task['snapshot']
        n_models = len(task['models'])
        print(f"Snapshot {snap} (index {task['snap_index']}): {task['n_missing']} planes across {n_models} models")
        for model, reals in task['models'].items():
            print(f"  {model}: realizations {reals}")
    
    # Save to JSON
    output = {
        'total_missing': len(missing),
        'missing_planes': missing,
        'recovery_tasks': tasks,
    }
    
    output_path = "/mnt/home/mlee1/hydro_replace2/missing_lensplane_tasks.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nTask list saved to: {output_path}")
    
    # Generate summary for batch script
    print("\n" + "="*80)
    print("SNAPSHOTS NEEDING REGENERATION")
    print("="*80)
    snaps_to_run = sorted(by_snapshot.keys())
    print(f"Snapshots: {snaps_to_run}")
    print(f"Array indices: {[SNAPSHOT_ORDER.index(s) for s in snaps_to_run]}")


if __name__ == "__main__":
    main()
