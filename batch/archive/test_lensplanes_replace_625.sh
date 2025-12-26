#!/bin/bash
#SBATCH -J lensplanes_replace
#SBATCH -o logs/lensplanes_replace_%j.o
#SBATCH -e logs/lensplanes_replace_%j.e
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH -p cca

# Test lens plane generation with halo replacement on 625 resolution
#
# Usage:
#   sbatch batch/test_lensplanes_replace_625.sh
#
# Or with custom mass range:
#   MASS_MIN=13.0 MASS_MAX=14.0 sbatch batch/test_lensplanes_replace_625.sh

set -e
echo "=========================================="
echo "LENS PLANES WITH HALO REPLACEMENT"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start:  $(date)"
echo ""

# Environment
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Configuration (can override with environment variables)
SIM_RES=${SIM_RES:-625}
SNAP=${SNAP:-99}
MASS_MIN=${MASS_MIN:-12.5}
MASS_MAX=${MASS_MAX:-""}
RADIUS_FACTOR=${RADIUS_FACTOR:-5.0}
SEED=${SEED:-2020}
NTASKS=4

echo "Configuration:"
echo "  Resolution:    L205n${SIM_RES}TNG"
echo "  Snapshot:      ${SNAP}"
echo "  Mass range:    10^${MASS_MIN} - 10^${MASS_MAX:-∞} Msun/h"
echo "  Radius factor: ${RADIUS_FACTOR}×R200"
echo "  Seed:          ${SEED}"
echo "  MPI tasks:     ${NTASKS}"
echo ""

# Build command
CMD="mpirun -np ${NTASKS} python scripts/generate_lensplanes_replace.py \
    --sim-res ${SIM_RES} \
    --snap ${SNAP} \
    --mass-min ${MASS_MIN} \
    --radius-factor ${RADIUS_FACTOR} \
    --seed ${SEED}"

# Add mass-max if specified
if [ -n "${MASS_MAX}" ]; then
    CMD="${CMD} --mass-max ${MASS_MAX}"
fi

echo "Command: ${CMD}"
echo ""

# Run
time ${CMD}

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End: $(date)"

# List outputs
if [ -n "${MASS_MAX}" ]; then
    DIR_NAME="replace_M${MASS_MIN}-${MASS_MAX}"
else
    DIR_NAME="replace_Mgt${MASS_MIN}"
fi
OUTPUT_DIR=/mnt/home/mlee1/ceph/hydro_replace_lensplanes/L205n${SIM_RES}TNG/seed${SEED}/${DIR_NAME}

echo ""
echo "Output files:"
ls -lh ${OUTPUT_DIR}/*.dat 2>/dev/null | head -10 || echo "  (none found)"
