#!/bin/bash
#SBATCH --job-name=bcm_lensplanes
#SBATCH --partition=cca
#SBATCH --time=8:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --output=/mnt/home/mlee1/hydro_replace2/logs/bcm_lensplanes_%A_%a.out
#SBATCH --error=/mnt/home/mlee1/hydro_replace2/logs/bcm_lensplanes_%A_%a.err
#SBATCH --array=0-99

# ============================================================================
# BCM Lensplane Generation Pipeline (Preempt-safe with checkpointing)
# ============================================================================
# 
# Generates BCM-corrected lensplanes for all 5 BCM model variants.
# Uses checkpoint files to allow graceful restart after preemption.
#
# Array mapping: 100 tasks = 5 models × 20 snapshots
#   - Tasks 0-19:   schneider19 (snap indices 0-19)
#   - Tasks 20-39:  schneider25 (snap indices 0-19)
#   - Tasks 40-59:  arico20 (snap indices 0-19)
#   - Tasks 60-79:  schneider19_sharma (snap indices 0-19)
#   - Tasks 80-99:  arico20_sharma (snap indices 0-19)
#
# Checkpointing:
#   - Checkpoint file: .ouroboros/checkpoints/bcm_lensplanes_{model}_{snap}.done
#   - If checkpoint exists, task skips (already completed)
#   - After successful completion, checkpoint file is created
#
# Usage:
#   sbatch run_unified_2500_bcm_array.sh           # Run all
#   sbatch run_unified_2500_bcm_array.sh           # Re-run skips completed
# ============================================================================

# Exit on error but allow handling
set -e

# Base directories
WORK_DIR="/mnt/home/mlee1/hydro_replace2"
CHECKPOINT_DIR="${WORK_DIR}/.ouroboros/checkpoints"
OUTPUT_BASE="/mnt/home/mlee1/ceph/hydro_replace_LP_bcm/L205n2500TNG"

# Create checkpoint directory
mkdir -p "${CHECKPOINT_DIR}"

# Define BCM models (5 variants)
BCM_MODELS=(
    "schneider19"
    "schneider25"
    "arico20"
    "schneider19_sharma"
    "arico20_high_mass"
)

# Define snapshots (20 snapshots, matching unified pipeline order)
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)

# Calculate model and snapshot indices from array task ID
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 20))
SNAP_IDX=$((SLURM_ARRAY_TASK_ID % 20))

# Get model name and snapshot number
BCM_MODEL=${BCM_MODELS[$MODEL_IDX]}
SNAP=${SNAPSHOTS[$SNAP_IDX]}

# Checkpoint file for this task
CHECKPOINT_FILE="${CHECKPOINT_DIR}/bcm_lensplanes_${BCM_MODEL}_snap${SNAP}.done"

# ============================================================================
# Signal handler for preemption (SIGTERM/SIGUSR1)
# ============================================================================
cleanup_handler() {
    echo ""
    echo "============================================================================"
    echo "PREEMPTION SIGNAL RECEIVED"
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${BCM_MODEL} snap ${SNAP}"
    echo "This task will be requeued and will restart from the beginning."
    echo "Time: $(date)"
    echo "============================================================================"
    
    # Don't create checkpoint - task is incomplete
    # The job will be requeued automatically with --requeue
    exit 0
}

# Trap preemption signals
trap 'cleanup_handler' SIGTERM SIGUSR1 SIGINT

# ============================================================================
# Check if already completed
# ============================================================================
echo "============================================================================"
echo "BCM Lensplane Generation (Preempt-safe)"
echo "============================================================================"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${BCM_MODEL}"
echo "Snapshot: ${SNAP}"
echo "Checkpoint file: ${CHECKPOINT_FILE}"
echo "Start time: $(date)"
echo "============================================================================"

# Check for existing checkpoint
# if [[ -f "${CHECKPOINT_FILE}" ]]; then
#     echo ""
#     echo "*** CHECKPOINT FOUND - Task already completed ***"
#     echo "Skipping: ${BCM_MODEL} snapshot ${SNAP}"
#     echo ""
    
#     # Verify the output actually exists
#     EXPECTED_OUTPUT="${OUTPUT_BASE}/${BCM_MODEL}/LP_19/lenspot01.dat"
#     if [[ -f "${EXPECTED_OUTPUT}" ]]; then
#         echo "Output verified: ${EXPECTED_OUTPUT} exists"
#         echo "============================================================================"
#         exit 0
#     else
#         echo "WARNING: Checkpoint exists but output missing!"
#         echo "Removing invalid checkpoint and re-running..."
#         rm -f "${CHECKPOINT_FILE}"
#     fi
# fi

# ============================================================================
# Run the BCM pipeline
# ============================================================================
echo ""
echo "Starting BCM pipeline..."
echo ""

# Load required modules
module purge
module load python openmpi python-mpi hdf5

# Activate virtual environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Navigate to scripts directory
cd "${WORK_DIR}/scripts"

# Run with mpirun
srun -n ${SLURM_NTASKS} python generate_all_unified_bcm.py \
    --snap ${SNAP} \
    --sim-res 2500 \
    --bcm-model ${BCM_MODEL} \
    --mass-min 12 \
    --enable-lensplanes \
    --skip-2d-maps

# ============================================================================
# Create checkpoint on success
# ============================================================================
RETVAL=$?

if [[ ${RETVAL} -eq 0 ]]; then
    # Verify output exists before creating checkpoint
    EXPECTED_OUTPUT="${OUTPUT_BASE}/${BCM_MODEL}/LP_19/lenspot01.dat"
    if [[ -f "${EXPECTED_OUTPUT}" ]]; then
        echo ""
        echo "Creating checkpoint: ${CHECKPOINT_FILE}"
        echo "completed=$(date -Iseconds)" > "${CHECKPOINT_FILE}"
        echo "model=${BCM_MODEL}" >> "${CHECKPOINT_FILE}"
        echo "snapshot=${SNAP}" >> "${CHECKPOINT_FILE}"
        echo "job_id=${SLURM_JOB_ID}" >> "${CHECKPOINT_FILE}"
        echo "array_task_id=${SLURM_ARRAY_TASK_ID}" >> "${CHECKPOINT_FILE}"
        
        echo ""
        echo "============================================================================"
        echo "SUCCESS: ${BCM_MODEL} snapshot ${SNAP}"
        echo "Checkpoint created: ${CHECKPOINT_FILE}"
        echo "End time: $(date)"
        echo "============================================================================"
    else
        echo "ERROR: Output file not found: ${EXPECTED_OUTPUT}"
        echo "Not creating checkpoint."
        exit 1
    fi
else
    echo ""
    echo "============================================================================"
    echo "FAILED: ${BCM_MODEL} snapshot ${SNAP}"
    echo "Exit code: ${RETVAL}"
    echo "Not creating checkpoint."
    echo "End time: $(date)"
    echo "============================================================================"
    exit ${RETVAL}
fi
