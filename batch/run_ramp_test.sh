#!/bin/bash
#SBATCH --job-name=ramp_test
#SBATCH --partition=cca
#SBATCH --time=12:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --output=/mnt/home/mlee1/hydro_replace2/logs/ramp_test_%j.out
#SBATCH --error=/mnt/home/mlee1/hydro_replace2/logs/ramp_test_%j.err

# ============================================================================
# Ramp Boundary Test for Replace Models
# ============================================================================
#
# This script tests the effect of hard radial cutoffs vs smooth cosine ramps.
#
# Configuration:
#   - Snapshot: 96 (z ≈ 0.04)
#   - Mass bin: M > 10^13 M⊙/h (cumulative)
#   - Alpha values: 0.5, 1.0, 3.0, 5.0 R₂₀₀
#   - Delta values: 0 (hard), 0.1, 0.2, 0.5 R₂₀₀
#   - Total: 4 × 4 = 16 configurations
#
# Output: /mnt/home/mlee1/ceph/hydro_replace_LP_ramp_test/L205n2500TNG/
#
# ============================================================================

set -e

WORK_DIR="/mnt/home/mlee1/hydro_replace2"
cd "${WORK_DIR}"

# Load modules
module load python openmpi python-mpi hdf5

# Activate environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

echo "============================================================================"
echo "RAMP BOUNDARY TEST"
echo "============================================================================"
echo "Start time: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Tasks: ${SLURM_NTASKS}"
echo "============================================================================"

# Run the test
srun -n ${SLURM_NTASKS} python3 -u scripts/generate_ramp_test.py \
    --snap 96 \
    --sim-res 2500 \
    --alpha 1.0 \
    --delta 0.0 \

echo "============================================================================"
echo "End time: $(date)"
echo "============================================================================"
