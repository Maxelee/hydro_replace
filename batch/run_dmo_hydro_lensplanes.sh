#!/bin/bash
#SBATCH --job-name=dmo_hydro_lp
#SBATCH --output=logs/dmo_hydro_lp_%A_%a.o
#SBATCH --error=logs/dmo_hydro_lp_%A_%a.e
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=6:00:00
#SBATCH --partition=cca
#SBATCH --array=0-19

# =============================================================================
# Regenerate DMO and Hydro lensplanes with n_realizations=20
# =============================================================================
#
# This script generates ONLY DMO and Hydro lensplanes (Phase 4) with settings
# that match the currently running Replace lensplane jobs:
#   - n_realizations: 20
#   - seed: 2020
#   - planes_per_snapshot: 2
#   - grid_res: 4096
#
# The transforms are deterministic based on seed, so DMO/Hydro will have
# matching transforms to the Replace lensplanes.
#
# Array tasks: 20 snapshots (one per task)
#
# Usage:
#   sbatch run_dmo_hydro_lensplanes.sh              # Run all snapshots
#   sbatch --array=0 run_dmo_hydro_lensplanes.sh   # Only snapshot 96 (z~0)
#
# =============================================================================

set -e

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Snapshot order (from z≈0 to z≈2) - must match SNAPSHOT_ORDER in generate_all_unified.py
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

# Get snapshot for this array task
SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "DMO + Hydro Lensplane Generation"
echo "========================================"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Snapshot: $SNAP"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Started: $(date)"
echo ""
echo "Settings (must match Replace jobs):"
echo "  n_realizations: 20"
echo "  seed: 2020"
echo "  pps: 2"
echo "  grid: 4096"
echo "========================================"

# Run Phase 4 only (DMO + Hydro lensplanes)
# Note: --phase4-only uses optimized path that skips profiles and doesn't need KDTree
srun python3 -u scripts/generate_all_unified.py \
    --snap $SNAP \
    --sim-res 2500 \
    --lensplane-grid 4096 \
    --phase4-only

echo ""
echo "========================================"
echo "Done with snapshot $SNAP"
echo "Finished: $(date)"
echo "========================================"
