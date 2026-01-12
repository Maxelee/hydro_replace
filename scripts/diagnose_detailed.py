#!/usr/bin/env python3
"""
Detailed diagnosis: identify exactly which snapshot/plane combinations are missing.
Output an actionable list for regeneration.
"""

import os
import glob
import json
from collections import defaultdict

# Paths
LP_BASE = "/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG"
N_LENSPLANES = 40
N_REALIZATIONS = 10
PPS = 2  # planes per snapshot

# Snapshot order
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]

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

# Find missing planes
missing_by_snapshot = defaultdict(list)  # snapshot -> [(model, realization, plane_idx)]
missing_by_model = defaultdict(list)     # model -> [(realization, plane_idx, snapshot)]

for model in MODELS:
    for r in range(N_REALIZATIONS):
        real_dir = os.path.join(LP_BASE, model, f"LP_{r:02d}")
        if not os.path.exists(real_dir):
            # All planes missing for this realization
            for p in range(N_LENSPLANES):
                snap = plane_to_snapshot(p)
                missing_by_snapshot[snap].append((model, r, p))
                missing_by_model[model].append((r, p, snap))
            continue
        
        # Check each plane
        for p in range(N_LENSPLANES):
            path = os.path.join(real_dir, f"lenspot{p:02d}.dat")
            if not os.path.exists(path):
                snap = plane_to_snapshot(p)
                missing_by_snapshot[snap].append((model, r, p))
                missing_by_model[model].append((r, p, snap))

# Print summary
print("="*80)
print("MISSING PLANES BY SNAPSHOT")
print("="*80)
for snap in sorted(missing_by_snapshot.keys()):
    entries = missing_by_snapshot[snap]
    print(f"\nSnapshot {snap}: {len(entries)} missing planes")
    # Group by model
    by_model = defaultdict(list)
    for model, r, p in entries:
        by_model[model].append((r, p))
    for model in sorted(by_model.keys()):
        items = by_model[model]
        reals = sorted(set(r for r, p in items))
        print(f"  {model}: realizations {reals}")

print("\n" + "="*80)
print("MODELS WITH MISSING DATA")
print("="*80)
for model in sorted(missing_by_model.keys()):
    entries = missing_by_model[model]
    if not entries:
        continue
    # Group by snapshot
    by_snap = defaultdict(list)
    for r, p, snap in entries:
        by_snap[snap].append((r, p))
    print(f"\n{model}:")
    for snap in sorted(by_snap.keys()):
        items = by_snap[snap]
        reals = sorted(set(r for r, p in items))
        print(f"  snap {snap}: realizations {reals}")

# Create actionable job list
print("\n" + "="*80)
print("REGENERATION JOBS (snapshot, models needing regen)")
print("="*80)

jobs = []
for snap in sorted(missing_by_snapshot.keys()):
    entries = missing_by_snapshot[snap]
    models_affected = list(set(model for model, r, p in entries))
    # For replace models, we need to re-run the unified pipeline for that snapshot
    jobs.append({
        'snapshot': snap,
        'models': models_affected,
        'n_missing': len(entries)
    })
    print(f"Snapshot {snap}: {len(models_affected)} models, {len(entries)} total planes missing")

# Save jobs
with open("/mnt/home/mlee1/hydro_replace2/regeneration_jobs.json", "w") as f:
    json.dump({
        'jobs': jobs,
        'missing_by_snapshot': {str(k): v for k, v in missing_by_snapshot.items()},
    }, f, indent=2)

print("\nJobs saved to regeneration_jobs.json")
