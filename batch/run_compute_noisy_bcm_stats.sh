#!/bin/bash
#SBATCH --job-name=noisy_bcm_%a
#SBATCH --output=logs/noisy_bcm_stats_%a_%j.out
#SBATCH --error=logs/noisy_bcm_stats_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --partition=cca
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --array=0-1

# Compute BCM statistics WITH survey noise + smoothing.
#
# Submit with survey/smoothing as environment variables:
#   sbatch --export=SURVEY=LSST,SMOOTHING=1.0 batch/run_compute_noisy_bcm_stats.sh
#   sbatch --export=SURVEY=DES,SMOOTHING=2.0  batch/run_compute_noisy_bcm_stats.sh
#
# Or use batch/submit_noisy_jobs.sh to launch all 6 configurations.

set -e

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

mkdir -p logs

# Read survey and smoothing from environment (with defaults)
SURVEY=${SURVEY:-LSST}
SMOOTHING=${SMOOTHING:-1.0}

MODELS=("arico20")

# Data paths
LP_BASE="/mnt/home/mlee1/ceph/hydro_replace_LP_bcm/L205n2500TNG"
RT_BASE="/mnt/home/mlee1/ceph/hydro_replace_RT_bcm/L205n2500TNG"

# Validate array index
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set. Run with sbatch --array=N or set manually for testing."
    exit 1
fi

if [[ $SLURM_ARRAY_TASK_ID -ge ${#MODELS[@]} ]]; then
    echo "ERROR: Array task ID $SLURM_ARRAY_TASK_ID exceeds model count ${#MODELS[@]}"
    exit 1
fi

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Check source data
LP_DIR="${LP_BASE}/${MODEL}"
RT_DIR="${RT_BASE}/${MODEL}"

if [[ ! -d "$LP_DIR" ]] && [[ ! -d "$RT_DIR" ]]; then
    echo "ERROR: Neither lensplane nor ray-tracing data found for model: $MODEL"
    echo "  Checked: $LP_DIR"
    echo "  Checked: $RT_DIR"
    exit 1
fi

echo "========================================"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Survey: $SURVEY"
echo "Smoothing theta_G: $SMOOTHING arcmin"
echo "Start time: $(date)"
echo "========================================"
echo ""

if ! srun -n 64 python scripts/compute_all_stats.py \
    --model "$MODEL" \
    --bcm \
    --survey "$SURVEY" \
    --smoothing-scale "$SMOOTHING" \
    --no-bispectrum; then
    echo ""
    echo "========================================"
    echo "ERROR: Python script failed with exit code $?"
    echo "Model: $MODEL | Survey: $SURVEY | Smoothing: $SMOOTHING"
    echo "End time: $(date)"
    echo "========================================"
    exit 1
fi

echo ""
echo "========================================"
echo "End time: $(date)"
echo "Job complete!"
echo "========================================"
