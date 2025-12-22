#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J lensplanes_2500
#SBATCH -n 32
#SBATCH -N 8
#SBATCH --exclusive
#SBATCH -o /mnt/home/mlee1/hydro_replace2/logs/lensplanes_2500_%A_%a.o
#SBATCH -e /mnt/home/mlee1/hydro_replace2/logs/lensplanes_2500_%A_%a.e
#SBATCH -t 24:00:00
#SBATCH --array=0-17

# ==============================================================================
# Generate lens planes for L205n2500TNG - ALL models and mass cuts
# 
# Array indices (18 total):
#   0: DMO
#   1: Hydro
#   2-5: Replace (mass thresholds 12.5, 13.0, 13.5, 14.0)
#   6-9: BCM-Arico20 (mass thresholds 12.5, 13.0, 13.5, 14.0)
#   10-13: BCM-Schneider19 (mass thresholds 12.5, 13.0, 13.5, 14.0)
#   14-17: BCM-Schneider25 (mass thresholds 12.5, 13.0, 13.5, 14.0)
#
# Each job generates 20 seed realizations (seeds 2020-2039) for ray-tracing
# ==============================================================================

# Model configuration by array index
declare -a MODELS=(
    "dmo" "hydro"
    "replace" "replace" "replace" "replace"
    "bcm" "bcm" "bcm" "bcm"
    "bcm" "bcm" "bcm" "bcm"
    "bcm" "bcm" "bcm" "bcm"
)
declare -a MASS_MINS=(
    "12.5" "12.5"
    "12.5" "13.0" "13.5" "14.0"
    "12.5" "13.0" "13.5" "14.0"
    "12.5" "13.0" "13.5" "14.0"
    "12.5" "13.0" "13.5" "14.0"
)
declare -a BCM_MODELS=(
    "" ""
    "" "" "" ""
    "Arico20" "Arico20" "Arico20" "Arico20"
    "Schneider19" "Schneider19" "Schneider19" "Schneider19"
    "Schneider25" "Schneider25" "Schneider25" "Schneider25"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
MASS_MIN=${MASS_MINS[$SLURM_ARRAY_TASK_ID]}
BCM_MODEL=${BCM_MODELS[$SLURM_ARRAY_TASK_ID]}

SIM_RES=2500
GRID_RES=${GRID_RES:-4096}
SEED=${SEED:-2020}
NUM_SEEDS=${NUM_SEEDS:-20}

# Setup environment
module purge
module load python openmpi python-mpi
module load hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

echo "=============================================="
echo "Generating lens planes for L205n${SIM_RES}TNG"
echo "=============================================="
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
if [ -n "$BCM_MODEL" ]; then
    echo "BCM variant: $BCM_MODEL"
fi
echo "Mass cut: M > 10^${MASS_MIN} Msun/h"
echo "Grid: $GRID_RES"
echo "Seeds: $SEED to $((SEED + NUM_SEEDS - 1)) ($NUM_SEEDS realizations)"
echo "MPI ranks: $SLURM_NTASKS"
echo "=============================================="
echo ""

# Build command
if [ -n "$BCM_MODEL" ]; then
    # BCM model - specify which one
    CMD="python -u scripts/generate_lensplanes.py \
        --sim-res ${SIM_RES} \
        --model bcm \
        --bcm-models ${BCM_MODEL} \
        --snap all \
        --mass-min ${MASS_MIN} \
        --seed ${SEED} \
        --num-seeds ${NUM_SEEDS} \
        --grid-res ${GRID_RES} \
        --skip-existing"
else
    # DMO, Hydro, or Replace
    CMD="python -u scripts/generate_lensplanes.py \
        --sim-res ${SIM_RES} \
        --model ${MODEL} \
        --snap all \
        --mass-min ${MASS_MIN} \
        --seed ${SEED} \
        --num-seeds ${NUM_SEEDS} \
        --grid-res ${GRID_RES} \
        --skip-existing"
fi

echo "Command: $CMD"
echo ""

# Run with MPI
mpirun -np $SLURM_NTASKS $CMD

echo ""
echo "=============================================="
if [ -n "$BCM_MODEL" ]; then
    echo "Completed lens planes for BCM-${BCM_MODEL} (M > 10^${MASS_MIN})"
else
    echo "Completed lens planes for $MODEL (M > 10^${MASS_MIN})"
fi
echo "=============================================="
