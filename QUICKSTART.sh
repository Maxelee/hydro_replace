#!/bin/bash
# Quick start guide for unified statistics pipeline

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                  UNIFIED STATISTICS PIPELINE - QUICK START                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

STEP 1: Check Data Availability
────────────────────────────────────────────────────────────────────────────────
python scripts/check_data_availability.py

STEP 2: Test Components
────────────────────────────────────────────────────────────────────────────────
python scripts/test_pipeline.py

STEP 3: Compute DMO RMS (one-time, ~2-3 hours)
────────────────────────────────────────────────────────────────────────────────
sbatch batch/run_compute_dmo_rms.sh

# Check job status:
squeue -u $USER

# Monitor progress:
tail -f logs/dmo_rms_*.out

# Verify output exists:
ls -lh /mnt/home/mlee1/ceph/hydro_replace_stats/dmo_rms.h5

STEP 4: Run Statistics Pipeline
────────────────────────────────────────────────────────────────────────────────

Option A: Single model (for testing)
──────────────────────────────────────
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

mpirun -np 64 python scripts/compute_all_stats.py --model dmo

Option B: All models (array job)
──────────────────────────────────
sbatch batch/run_compute_stats.sh

# Check array job status:
squeue -u $USER

# Monitor a specific task (e.g., task 0 = dmo):
tail -f logs/stats_0_*.out

STEP 5: Analyze Results
────────────────────────────────────────────────────────────────────────────────
jupyter notebook notebooks/response_analysis_template.ipynb

# Or in Python:
python << 'PYTHON'
import h5py
import numpy as np

# Load peak counts for dmo at z~1.0
with h5py.File('/mnt/home/mlee1/ceph/hydro_replace_stats/dmo/stats.h5', 'r') as f:
    peaks = f['peaks'][:, 1, :]  # All realizations, z_idx=1
    sn_bins = f['sn_bin_edges'][:]
    print(f"Shape: {peaks.shape}")
    print(f"S/N range: [{sn_bins[0]}, {sn_bins[-1]}]")
PYTHON

DOCUMENTATION
────────────────────────────────────────────────────────────────────────────────
Full guide:      docs/unified_stats_pipeline.md
Implementation:  docs/IMPLEMENTATION_SUMMARY.md
Custom instr.:   See top-level .copilot-instructions.md

TROUBLESHOOTING
────────────────────────────────────────────────────────────────────────────────
Problem: "DMO RMS file not found"
Solution: Run step 3 first (compute_dmo_rms.py)

Problem: "No module named 'Pk_library'"
Solution: Activate venv: source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

Problem: "Missing kappa/lenspot files"
Solution: Check data with: python scripts/check_data_availability.py

Problem: Job fails with memory error
Solution: Edit batch scripts to reduce --ntasks-per-node and increase --mem

AVAILABLE MODELS FOR TESTING
────────────────────────────────────────────────────────────────────────────────
Models with complete data (RT + LP):
  - dmo
  - hydro
  - hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.0_Ro_0.5
  - hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.0_Ro_1.0
  - hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.0_Ro_3.0
  - hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.0_Ro_5.0

Models with LP data only: 96 additional Replace models
(Can compute P(k) but not C_ℓ/peaks until RT data is generated)

EOF
