#!/bin/bash
#SBATCH --job-name=lp_recovery
#SBATCH --output=logs/lp_recovery_%A_%a.o
#SBATCH --error=logs/lp_recovery_%A_%a.e
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --time=12:00:00
#SBATCH --partition=cca
#SBATCH --array=0,1,2,3,4,6,8,9,10,11,12,14,15,16,17,19

# ============================================================================
# Lensplane Recovery Job
# ============================================================================
# Re-run unified pipeline for snapshots with missing lensplanes.
# Only runs phase 5 (Replace lensplanes) since DMO/Hydro are complete.
#
# Missing data identified across 16 snapshots:
#   Snapshots: [29, 33, 35, 38, 41, 46, 49, 52, 56, 59, 67, 76, 80, 85, 90, 96]
#   Array indices: [19, 17, 16, 15, 14, 12, 11, 10, 9, 8, 6, 4, 3, 2, 1, 0]
#
# This script uses array indices matching the snapshot order.
# ============================================================================

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Map array index to snapshot number
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)
SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Lensplane Recovery Pipeline"
echo "Resolution: 2500"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Snapshot: $SNAP"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Started: $(date)"
echo "========================================"

# Run unified pipeline with ONLY phase 5 (Replace lensplanes)
# --incremental ensures we only generate missing files
srun python3 -u scripts/generate_all_unified.py \
    --snap $SNAP \
    --sim-res 2500 \
    --mass-min 12.0 \
    --radius-mult 5.0 \
    --grid 1024 \
    --enable-lensplanes \
    --lensplane-grid 4096 \
    --phase5-only \
    --incremental

echo ""
echo "Finished: $(date)"
echo "========================================"
