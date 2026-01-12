#!/usr/bin/env python3
"""
Fast diagnosis of missing data using directory counts and glob.
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

print(f"Total models: {len(MODELS)}")

# Track issues
missing_lp = []
missing_lux = []
missing_rt = []
partial_rt = []

def count_lenspot_files(directory):
    """Count lenspot*.dat files in a directory"""
    if not os.path.exists(directory):
        return 0
    return len(glob.glob(os.path.join(directory, "lenspot*.dat")))

def count_complete_runs(directory):
    """Count run directories with kappa_40.dat"""
    if not os.path.exists(directory):
        return 0, []
    complete = []
    incomplete = []
    for i in range(1, N_RT_RUNS + 1):
        kappa_path = os.path.join(directory, f"run{i:03d}", "kappa_40.dat")
        if os.path.exists(kappa_path):
            complete.append(i)
        else:
            incomplete.append(i)
    return len(complete), incomplete

print("\n" + "="*90)
print(f"{'Model':<55} {'LP':<10} {'LUX':<10} {'RT runs':<15}")
print("="*90)

total_lp_missing = 0
total_lux_missing = 0
total_rt_missing = 0

for model in MODELS:
    lp_total = 0
    lux_total = 0
    rt_total = 0
    
    model_missing_lp = []
    model_missing_lux = []
    model_missing_rt = []
    
    for r in range(N_REALIZATIONS):
        # Check LP
        lp_dir = os.path.join(LP_BASE, model, f"LP_{r:02d}")
        lp_count = count_lenspot_files(lp_dir)
        lp_total += lp_count
        if lp_count < N_LENSPLANES:
            model_missing_lp.append((r, N_LENSPLANES - lp_count))
        
        # Check LUX
        lux_dir = os.path.join(LUX_BASE, model, f"LP_{r:02d}")
        lux_count = count_lenspot_files(lux_dir)
        lux_total += lux_count
        if lux_count < N_LENSPLANES:
            model_missing_lux.append((r, N_LENSPLANES - lux_count))
        
        # Check RT - fast check via directory count
        rt_dir = os.path.join(RT_BASE, model, f"LP_{r:02d}")
        if os.path.exists(rt_dir):
            run_dirs = [d for d in os.listdir(rt_dir) if d.startswith("run")]
            # Quick check: just count runs that have kappa_40.dat
            rt_count = len(glob.glob(os.path.join(rt_dir, "run*/kappa_40.dat")))
        else:
            rt_count = 0
        rt_total += rt_count
        if rt_count < N_RT_RUNS:
            model_missing_rt.append((r, N_RT_RUNS - rt_count))
    
    lp_exp = N_LENSPLANES * N_REALIZATIONS
    rt_exp = N_RT_RUNS * N_REALIZATIONS
    
    lp_status = "OK" if lp_total == lp_exp else f"{lp_total}/{lp_exp}"
    lux_status = "OK" if lux_total == lp_exp else f"{lux_total}/{lp_exp}"
    rt_status = "OK" if rt_total == rt_exp else f"{rt_total}/{rt_exp}"
    
    print(f"{model:<55} {lp_status:<10} {lux_status:<10} {rt_status:<15}")
    
    if model_missing_lp:
        missing_lp.append((model, model_missing_lp))
        total_lp_missing += sum(m[1] for m in model_missing_lp)
    if model_missing_lux:
        missing_lux.append((model, model_missing_lux))
        total_lux_missing += sum(m[1] for m in model_missing_lux)
    if model_missing_rt:
        missing_rt.append((model, model_missing_rt))
        total_rt_missing += sum(m[1] for m in model_missing_rt)

print("="*90)
print(f"\nSUMMARY:")
print(f"  Total missing LP files: {total_lp_missing}")
print(f"  Total missing LUX files: {total_lux_missing}")
print(f"  Total missing RT runs: {total_rt_missing}")

print(f"\n  Models with LP issues: {len(missing_lp)}")
print(f"  Models with LUX issues: {len(missing_lux)}")
print(f"  Models with RT issues: {len(missing_rt)}")

# Detailed breakdown
if missing_lp:
    print("\n" + "="*90)
    print("MISSING LP DETAILS (model, realization, missing_count):")
    print("="*90)
    for model, issues in missing_lp:
        for r, count in issues:
            print(f"  {model}/LP_{r:02d}: missing {count} files")

if missing_lux:
    print("\n" + "="*90)
    print("MISSING LUX DETAILS:")
    print("="*90)
    for model, issues in missing_lux:
        for r, count in issues:
            print(f"  {model}/LP_{r:02d}: missing {count} files")

if missing_rt:
    print("\n" + "="*90)
    print("MISSING RT DETAILS:")
    print("="*90)
    for model, issues in missing_rt:
        for r, count in issues:
            print(f"  {model}/LP_{r:02d}: missing {count} runs")

# Save actionable report
import json
report = {
    "missing_lp": [(m, i) for m, issues in missing_lp for i in issues],
    "missing_lux": [(m, i) for m, issues in missing_lux for i in issues],
    "missing_rt": [(m, i) for m, issues in missing_rt for i in issues],
    "summary": {
        "total_lp_missing": total_lp_missing,
        "total_lux_missing": total_lux_missing,
        "total_rt_missing": total_rt_missing,
    }
}

with open("/mnt/home/mlee1/hydro_replace2/missing_data_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n\nReport saved to: missing_data_report.json")
