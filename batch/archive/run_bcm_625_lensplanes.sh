#!/bin/bash
#SBATCH -J bcm_625_lp
#SBATCH -o logs/bcm_625_lensplanes_%j.o
#SBATCH -e logs/bcm_625_lensplanes_%j.e
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -p cca

# BCM Lens plane generation for L205n625TNG
#
# Generates lens planes for all BCM models for ray-tracing.
#
# Prerequisites:
#   - BCM maps must exist for all snapshots (from BCM array job)
#
# Usage:
#   sbatch --dependency=afterok:<BCM_ARRAY_JOBID> batch/run_bcm_625_lensplanes.sh

set -e

# Environment
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Configuration
SIM_RES=625
MASS_MIN=12.5
RADIUS_FACTOR=5.0
NTASKS=4
SEED=2020

BCM_MODELS=("Arico20" "Schneider19" "Schneider25")

# All 21 snapshots
SNAPSHOTS="99,29,31,33,35,38,41,43,46,49,52,56,59,63,67,71,76,80,85,90,96"

echo "=========================================="
echo "BCM LENS PLANES - L205n${SIM_RES}TNG"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes:  $SLURM_JOB_NODELIST"
echo "Start:  $(date)"
echo ""

echo "Configuration:"
echo "  Resolution:    L205n${SIM_RES}TNG"
echo "  Mass min:      10^${MASS_MIN} Msun/h"
echo "  Radius factor: ${RADIUS_FACTOR}×R200"
echo "  Seed:          ${SEED}"
echo "  BCM Models:    ${BCM_MODELS[@]}"
echo "  Snapshots:     ${SNAPSHOTS}"
echo ""

LENSPLANES_DIR=/mnt/home/mlee1/ceph/hydro_replace_lensplanes/L205n${SIM_RES}TNG

for BCM_MODEL in "${BCM_MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Generating lens planes: ${BCM_MODEL}"
    echo "=========================================="
    
    MODEL_LOWER=$(echo ${BCM_MODEL} | tr '[:upper:]' '[:lower:]')
    OUTPUT_DIR=${LENSPLANES_DIR}/seed${SEED}/bcm_${MODEL_LOWER}_Mgt${MASS_MIN}
    
    # Check if already done
    N_EXISTING=$(ls ${OUTPUT_DIR}/density*.dat 2>/dev/null | wc -l)
    if [ "${N_EXISTING}" -ge 42 ]; then
        echo "  Lens planes exist (${N_EXISTING} files), skipping"
        continue
    fi
    
    time mpirun -np ${NTASKS} python scripts/generate_lensplanes_bcm.py \
        --sim-res ${SIM_RES} \
        --snap ${SNAPSHOTS} \
        --bcm-model ${BCM_MODEL} \
        --mass-min ${MASS_MIN} \
        --radius-factor ${RADIUS_FACTOR} \
        --seed ${SEED}
done

echo ""
echo "=========================================="
echo "BCM LENS PLANES COMPLETE"
echo "=========================================="
echo "End: $(date)"

echo ""
echo "Output summary:"
for BCM_MODEL in "${BCM_MODELS[@]}"; do
    MODEL_LOWER=$(echo ${BCM_MODEL} | tr '[:upper:]' '[:lower:]')
    OUTPUT_DIR=${LENSPLANES_DIR}/seed${SEED}/bcm_${MODEL_LOWER}_Mgt${MASS_MIN}
    N_FILES=$(ls ${OUTPUT_DIR}/density*.dat 2>/dev/null | wc -l)
    echo "  ${BCM_MODEL}: ${N_FILES} lens plane files"
done

echo ""
echo "Done!"
