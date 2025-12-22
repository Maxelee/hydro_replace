#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J lensplanes
#SBATCH -n 64
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH -o /mnt/home/mlee1/hydro_replace2/logs/lensplanes_%A_%a.o
#SBATCH -e /mnt/home/mlee1/hydro_replace2/logs/lensplanes_%A_%a.e
#SBATCH -t 24:00:00
#SBATCH --array=0-9

# ==============================================================================
# Generate lens planes for all models after maps complete
# 
# Key optimization: Data is loaded ONCE per snapshot, then multiple seed
# realizations are generated from the same data in memory. This is much
# faster than running separate jobs for each seed.
#
# Array indices:
#   0: DMO
#   1: Hydro
#   2-5: Replace (mass thresholds 12.5, 13.0, 13.5, 14.0)
#   6-9: BCM (all 3 BCM models, mass thresholds 12.5, 13.0, 13.5, 14.0)
#
# Output structure: .../seed{N}/model_name/density*.dat
#
# Submit with dependency on maps jobs:
#   sbatch --dependency=afterok:MAPS_JOB1:MAPS_JOB2 run_lensplanes_all.sh
# ==============================================================================

# Model configuration by array index
declare -a MODELS=("dmo" "hydro" "replace" "replace" "replace" "replace" "bcm" "bcm" "bcm" "bcm")
declare -a MASS_MINS=("12.5" "12.5" "12.5" "13.0" "13.5" "14.0" "12.5" "13.0" "13.5" "14.0")

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
MASS_MIN=${MASS_MINS[$SLURM_ARRAY_TASK_ID]}

SIM_RES=${SIM_RES:-2500}
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
echo "Generating lens planes"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Mass cut: M > 10^${MASS_MIN} Msun/h"
echo "Resolution: $SIM_RES, Grid: $GRID_RES"
echo "Seeds: $SEED to $((SEED + NUM_SEEDS - 1)) ($NUM_SEEDS realizations)"
echo "=============================================="

# Build command - now with --num-seeds for multiple realizations
CMD="python -u scripts/generate_lensplanes.py \
    --sim-res ${SIM_RES} \
    --model ${MODEL} \
    --snap all \
    --mass-min ${MASS_MIN} \
    --seed ${SEED} \
    --num-seeds ${NUM_SEEDS} \
    --grid-res ${GRID_RES} \
    --skip-existing"

echo "Command: $CMD"
echo ""

# Run with MPI
mpirun -np $SLURM_NTASKS $CMD

echo ""
echo "=============================================="
echo "Completed lens planes for $MODEL (M > 10^${MASS_MIN})"
echo "=============================================="
