#!/bin/bash
#SBATCH --job-name=dmo_hydro_regen
#SBATCH --output=logs/dmo_hydro_regen_%A_%a.o
#SBATCH --error=logs/dmo_hydro_regen_%A_%a.e
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --time=8:00:00
#SBATCH --partition=cca
#SBATCH --array=0-19

# =============================================================================
# Regenerate DMO and Hydro lensplanes with n_realizations=20
# =============================================================================
#
# This script regenerates DMO and Hydro lensplanes to match the Replace
# lensplanes which use n_realizations=20 and the transforms.h5 file.
#
# The existing DMO/Hydro lensplanes were generated with n_realizations=10,
# causing a mismatch in random transforms with the Replace lensplanes.
#
# Array tasks: 20 snapshots (one per task)
#
# Usage:
#   sbatch run_dmo_hydro_regen.sh
#
# =============================================================================

set -e

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Snapshot order (from z≈0 to z≈2)
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "DMO/Hydro Lensplane Regeneration"
echo "========================================"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Snapshot: $SNAP"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Started: $(date)"
echo "========================================"

# First, backup existing lensplanes and transforms (only for first snapshot)
# if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
#     LP_BASE="/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG"
    
#     # Backup transforms.h5 (was created with n_realizations=10)
#     if [ -f "$LP_BASE/transforms.h5" ]; then
#         echo "Backing up existing transforms.h5..."
#         mv "$LP_BASE/transforms.h5" "$LP_BASE/transforms_old_10real.h5" 2>/dev/null || true
#     fi
    
#     if [ -d "$LP_BASE/dmo" ]; then
#         echo "Backing up existing DMO lensplanes..."
#         mv "$LP_BASE/dmo" "$LP_BASE/dmo_old_10real" 2>/dev/null || true
#     fi
#     if [ -d "$LP_BASE/hydro" ]; then
#         echo "Backing up existing Hydro lensplanes..."
#         mv "$LP_BASE/hydro" "$LP_BASE/hydro_old_10real" 2>/dev/null || true
#     fi
    
#     echo "Backup complete."
# fi

# Wait for task 0 to finish backup
# sleep 10

# Run unified pipeline for Phase 4 only (DMO + Hydro lensplanes)
# Using default n_realizations=20 from LENSPLANE_CONFIG
srun python3 -u scripts/generate_all_unified.py \
    --snap $SNAP \
    --sim-res 2500 \
    --mass-min 12.0 \
    --radius-mult 5.0 \
    --grid 1024 \
    --enable-lensplanes \
    --lensplane-grid 4096 \
    --phase4-only

echo ""
echo "========================================"
echo "Done with snapshot $SNAP"
echo "Finished: $(date)"
echo "========================================"
