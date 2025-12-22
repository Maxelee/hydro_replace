#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J lensplanes
#SBATCH -n 64
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH -o logs/lensplanes_%A_%a.o%j
#SBATCH -e logs/lensplanes_%A_%a.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH -t 06:00:00

# =============================================================================
# Batch script for generating lens planes
#
# Usage:
#   # Test with low-res (single snapshot)
#   SIM_RES=625 MODEL=dmo SNAP=99 sbatch batch/run_lensplanes.sh
#
#   # All snapshots, all models
#   SIM_RES=625 MODEL=all SNAP=all sbatch batch/run_lensplanes.sh
#
#   # Production with specific mass cuts (array job)
#   SIM_RES=2500 MODEL=replace MASS_MIN=12.0 SNAP=all sbatch batch/run_lensplanes.sh
#   SIM_RES=2500 MODEL=replace MASS_MIN=12.5 SNAP=all sbatch batch/run_lensplanes.sh
#   SIM_RES=2500 MODEL=replace MASS_MIN=13.0 SNAP=all sbatch batch/run_lensplanes.sh
#
# Environment variables:
#   SIM_RES:  Simulation resolution (625, 1250, 2500) [default: 625]
#   MODEL:    Model to generate (dmo, hydro, replace, bcm, all) [default: all]
#   SNAP:     Snapshots (all, or comma-separated) [default: all]
#   MASS_MIN: Minimum halo mass log10(M_sun/h) [default: 12.5]
#   MASS_MAX: Maximum halo mass log10(M_sun/h) [default: none]
#   SEED:     Random seed [default: 2020]
# =============================================================================

# Default values
SIM_RES=${SIM_RES:-625}
MODEL=${MODEL:-all}
SNAP=${SNAP:-all}
MASS_MIN=${MASS_MIN:-12.5}
MASS_MAX=${MASS_MAX:-}
SEED=${SEED:-2020}
GRID_RES=${GRID_RES:-4096}

# Load modules
module load python openmpi python-mpi hdf5

# Activate virtual environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Navigate to project directory
cd /mnt/home/mlee1/hydro_replace2

# Create logs directory
mkdir -p logs

# Build command
CMD="python scripts/generate_lensplanes.py \
    --sim-res ${SIM_RES} \
    --model ${MODEL} \
    --snap ${SNAP} \
    --mass-min ${MASS_MIN} \
    --seed ${SEED} \
    --grid-res ${GRID_RES} \
    --skip-existing"

# Add mass-max if specified
if [ -n "${MASS_MAX}" ]; then
    CMD="${CMD} --mass-max ${MASS_MAX}"
fi

# Print configuration
echo "=============================================="
echo "Lens Plane Generation"
echo "=============================================="
echo "Simulation: L205n${SIM_RES}TNG"
echo "Model: ${MODEL}"
echo "Snapshots: ${SNAP}"
echo "Mass cut: log10(M) >= ${MASS_MIN}"
if [ -n "${MASS_MAX}" ]; then
    echo "Mass max: log10(M) < ${MASS_MAX}"
fi
echo "Random seed: ${SEED}"
echo "Nodes: ${SLURM_NNODES:-1}"
echo "Tasks: ${SLURM_NTASKS:-1}"
echo "=============================================="
echo ""
echo "Command: ${CMD}"
echo ""

# Run
srun -n ${SLURM_NTASKS:-64} ${CMD}

echo ""
echo "=============================================="
echo "Job completed at $(date)"
echo "=============================================="
