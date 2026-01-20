#!/usr/bin/env python3
"""
Diagnose missing data products across the entire pipeline:
1. Lensplanes (hydro_replace_LP) - 40 files per model/realization
2. Lux format (hydro_replace_LP_lux) - 40 lenspot files + config.dat per model/realization
3. Ray-tracing (hydro_replace_RT) - 100 runs per model/realization, kappa_40.dat in each

Expected totals:
- 34 models × 10 realizations = 340 combinations
- Each combination needs 40 lensplanes, 100 runs with kappa outputs
- Goal: 500 convergence maps per model = 5 realizations × 100 runs
        (but pipeline produces 10 realizations × 100 runs = 1000 potential)
"""

import os
from collections import defaultdict

# Paths
LP_BASE = "/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG"
LUX_BASE = "/mnt/home/mlee1/ceph/hydro_replace_LP_lux/L205n2500TNG"
RT_BASE = "/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG"

# Expected counts
N_LENSPLANES = 40  # 20 snapshots × 2 planes each
N_REALIZATIONS = 10
N_RT_RUNS = 100

# Models
MODELS = ["dmo", "hydro"]
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

for mass in MASS_CONFIGS:
    for r in R_FACTORS:
        MODELS.append(f"hydro_replace_{mass}_R_{r}")

print(f"Total models: {len(MODELS)}")
print("="*80)

# Track missing items
missing_lp = []       # (model, realization, missing_planes)
missing_lux = []      # (model, realization, missing_planes) 
missing_rt = []       # (model, realization, missing_runs)
incomplete_rt = []    # (model, realization, run, missing_kappa)

# Summary stats
lp_stats = defaultdict(int)
lux_stats = defaultdict(int)
rt_stats = defaultdict(int)

print("\n" + "="*80)
print("CHECKING LENSPLANES (hydro_replace_LP)")
print("="*80)

for model in MODELS:
    model_dir = os.path.join(LP_BASE, model)
    if not os.path.exists(model_dir):
        print(f"MISSING MODEL DIR: {model}")
        for r in range(N_REALIZATIONS):
            missing_lp.append((model, r, list(range(N_LENSPLANES))))
        continue
    
    for r in range(N_REALIZATIONS):
        real_dir = os.path.join(model_dir, f"LP_{r:02d}")
        if not os.path.exists(real_dir):
            missing_lp.append((model, r, list(range(N_LENSPLANES))))
            continue
        
        # Check for lenspot files
        existing = set()
        for f in os.listdir(real_dir):
            if f.startswith("lenspot") and f.endswith(".dat"):
                try:
                    idx = int(f.replace("lenspot", "").replace(".dat", ""))
                    existing.add(idx)
                except:
                    pass
        
        missing_planes = [i for i in range(N_LENSPLANES) if i not in existing]
        if missing_planes:
            missing_lp.append((model, r, missing_planes))
        
        lp_stats[model] += len(existing)

# Print LP summary
print(f"\nTotal missing LP entries: {len(missing_lp)}")
if len(missing_lp) <= 20:
    for item in missing_lp:
        print(f"  {item[0]}/LP_{item[1]:02d}: missing {len(item[2])} planes: {item[2][:5]}...")

print("\n" + "="*80)
print("CHECKING LUX FORMAT (hydro_replace_LP_lux)")
print("="*80)

for model in MODELS:
    model_dir = os.path.join(LUX_BASE, model)
    if not os.path.exists(model_dir):
        print(f"MISSING MODEL DIR: {model}")
        for r in range(N_REALIZATIONS):
            missing_lux.append((model, r, list(range(N_LENSPLANES))))
        continue
    
    for r in range(N_REALIZATIONS):
        real_dir = os.path.join(model_dir, f"LP_{r:02d}")
        if not os.path.exists(real_dir):
            missing_lux.append((model, r, list(range(N_LENSPLANES))))
            continue
        
        # Check for lenspot files
        existing = set()
        for f in os.listdir(real_dir):
            if f.startswith("lenspot") and f.endswith(".dat"):
                try:
                    idx = int(f.replace("lenspot", "").replace(".dat", ""))
                    existing.add(idx)
                except:
                    pass
        
        missing_planes = [i for i in range(N_LENSPLANES) if i not in existing]
        if missing_planes:
            missing_lux.append((model, r, missing_planes))
        
        lux_stats[model] += len(existing)

print(f"\nTotal missing LUX entries: {len(missing_lux)}")
if len(missing_lux) <= 20:
    for item in missing_lux:
        print(f"  {item[0]}/LP_{item[1]:02d}: missing {len(item[2])} planes: {item[2][:5]}...")

print("\n" + "="*80)
print("CHECKING RAY-TRACING (hydro_replace_RT)")
print("="*80)

for model in MODELS:
    model_dir = os.path.join(RT_BASE, model)
    if not os.path.exists(model_dir):
        print(f"MISSING MODEL DIR: {model}")
        for r in range(N_REALIZATIONS):
            missing_rt.append((model, r, list(range(1, N_RT_RUNS+1))))
        continue
    
    for r in range(N_REALIZATIONS):
        real_dir = os.path.join(model_dir, f"LP_{r:02d}")
        if not os.path.exists(real_dir):
            missing_rt.append((model, r, list(range(1, N_RT_RUNS+1))))
            continue
        
        # Check for run directories with kappa_40.dat
        existing_runs = []
        missing_runs = []
        for run_idx in range(1, N_RT_RUNS+1):
            run_dir = os.path.join(real_dir, f"run{run_idx:03d}")
            kappa_file = os.path.join(run_dir, "kappa_40.dat")
            if os.path.exists(kappa_file):
                existing_runs.append(run_idx)
            else:
                missing_runs.append(run_idx)
        
        if missing_runs:
            missing_rt.append((model, r, missing_runs))
        
        rt_stats[model] += len(existing_runs)

print(f"\nTotal RT missing entries: {len(missing_rt)}")

print("\n" + "="*80)
print("SUMMARY BY MODEL")
print("="*80)
print(f"{'Model':<50} {'LP':<8} {'LUX':<8} {'RT':<10}")
print("-"*80)
for model in MODELS:
    lp = lp_stats.get(model, 0)
    lux = lux_stats.get(model, 0)
    rt = rt_stats.get(model, 0)
    lp_exp = N_LENSPLANES * N_REALIZATIONS
    rt_exp = N_RT_RUNS * N_REALIZATIONS
    
    lp_status = "OK" if lp == lp_exp else f"{lp}/{lp_exp}"
    lux_status = "OK" if lux == lp_exp else f"{lux}/{lp_exp}"
    rt_status = "OK" if rt == rt_exp else f"{rt}/{rt_exp}"
    
    print(f"{model:<50} {lp_status:<8} {lux_status:<8} {rt_status:<10}")

print("\n" + "="*80)
print("DETAILED MISSING DATA")
print("="*80)

# Group missing by type
print(f"\nMissing LP entries: {len(missing_lp)}")
print(f"Missing LUX entries: {len(missing_lux)}")
print(f"Missing RT entries: {len(missing_rt)}")

# Output actionable lists
print("\n" + "="*80)
print("REGENERATION REQUIREMENTS")
print("="*80)

# What needs LP regeneration?
lp_needs_regen = set()
for model, r, planes in missing_lp:
    lp_needs_regen.add((model, r))

# What needs LUX conversion?
lux_needs_conv = set()
for model, r, planes in missing_lux:
    if (model, r) not in lp_needs_regen:  # Only if LP exists
        lux_needs_conv.add((model, r))

# What needs RT?
rt_needs_run = set()
for model, r, runs in missing_rt:
    if (model, r) not in lp_needs_regen and (model, r) not in lux_needs_conv:
        rt_needs_run.add((model, r))

print(f"\n1. Need LP regeneration: {len(lp_needs_regen)} combinations")
print(f"2. Need LUX conversion (LP exists): {len(lux_needs_conv)} combinations") 
print(f"3. Need RT runs (LUX exists): {len(rt_needs_run)} combinations")

# Save detailed lists
import json
output = {
    "missing_lp": [(m, r, p) for m, r, p in missing_lp],
    "missing_lux": [(m, r, p) for m, r, p in missing_lux],
    "missing_rt": [(m, r, runs) for m, r, runs in missing_rt],
    "lp_needs_regen": list(lp_needs_regen),
    "lux_needs_conv": list(lux_needs_conv),
    "rt_needs_run": list(rt_needs_run),
}

with open("/mnt/home/mlee1/hydro_replace2/missing_data_report.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nDetailed report saved to: missing_data_report.json")
