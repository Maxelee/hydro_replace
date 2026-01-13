#!/bin/bash
#SBATCH --job-name=unified_binned
#SBATCH --output=logs/unified_binned_%A_%a.o
#SBATCH --error=logs/unified_binned_%A_%a.e
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --time=06:00:00
#SBATCH --partition=cca
#SBATCH --array=0-199

# =============================================================================
# Array job for BINNED mode unified pipeline on L205n2500TNG
# =============================================================================
#
# This script runs the binned mass×radius shell replacement pipeline.
# 
# Configuration:
#   - 100 total configs (10 mass bins × 10 radius shells)
#   - 20 snapshots in the lightcone
#   - Split into 10 chunks of 10 configs each per snapshot
#   - Total array tasks: 20 snapshots × 10 chunks = 200 tasks
#
# Task mapping:
#   SNAP_IDX = TASK_ID / 10   (0-19, maps to snapshot number)
#   CHUNK    = TASK_ID % 10   (0-9, maps to config range)
#
# Each chunk processes configs [CHUNK*10, CHUNK*10+10)
#
# Usage:
#   sbatch run_unified_2500_binned_array.sh              # Run all
#   sbatch --array=0-9 run_unified_2500_binned_array.sh  # Only snapshot 96 (all chunks)
#   sbatch --array=0,10,20 run_unified_2500_binned_array.sh  # Chunk 0 of first 3 snaps
#
# =============================================================================

set -e

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Snapshot order (from z≈0 to z≈2)
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

# Configuration
N_CHUNKS=10
CONFIGS_PER_CHUNK=10

# Map array task ID to (snapshot_index, chunk)
SNAP_IDX=$((SLURM_ARRAY_TASK_ID / N_CHUNKS))
CHUNK=$((SLURM_ARRAY_TASK_ID % N_CHUNKS))
SNAP=${SNAPSHOTS[$SNAP_IDX]}

# Config range for this chunk
CONFIG_START=$((CHUNK * CONFIGS_PER_CHUNK))
CONFIG_END=$((CONFIG_START + CONFIGS_PER_CHUNK))

echo "========================================"
echo "Unified Pipeline - BINNED MODE"
echo "========================================"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Snapshot index: $SNAP_IDX -> Snapshot: $SNAP"
echo "Chunk: $CHUNK -> Configs [$CONFIG_START, $CONFIG_END)"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Started: $(date)"
echo "========================================"

# Run unified pipeline in binned mode with config range
srun python3 -u scripts/generate_all_unified.py \
    --snap $SNAP \
    --sim-res 2500 \
    --mass-min 12.0 \
    --radius-mult 5.0 \
    --grid 1024 \
    --enable-lensplanes \
    --lensplane-grid 4096 \
    --phase5-only \
    --binned-mode \
    --config-start $CONFIG_START \
    --config-end $CONFIG_END \
    --skip-existing

echo ""
echo "========================================"
echo "Done with snapshot $SNAP, chunk $CHUNK"
echo "Configs [$CONFIG_START, $CONFIG_END)"
echo "Finished: $(date)"
echo "========================================"
