#!/bin/bash
#SBATCH --job-name=stats_%a
#SBATCH --output=logs/stats_%a_%j.out
#SBATCH --error=logs/stats_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --partition=cca
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --array=0-1

# Exit on error (but allow checking exit codes explicitly)
set -e

# Array job to compute statistics for all models
# Array indices 0-101 cover all 102 models with complete data:

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Create logs directory
mkdir -p logs

# Define model list - MUST match lensplane directory names exactly
# Format: hydro_replace_Ml_{mass}_Mu_{mass}_Ri_{radius}_Ro_{radius}
# Mass format: X.XXeYY (e.g., 1.00e12, NOT 1.00e+12)
# All 100 Replace models with complete RT and LP data
MODELS=("arico20"
)





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

# Get model for this array task
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Validate model name format (should not contain e+, ee, or other typos)

# Check that source data exists (lensplanes or ray-traced maps)
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
echo "Lensplane dir exists: $([ -d "$LP_DIR" ] && echo "YES" || echo "NO")"
echo "Ray-tracing dir exists: $([ -d "$RT_DIR" ] && echo "YES" || echo "NO")"
echo "Start time: $(date)"
echo "========================================"
echo ""

# Run the script with error checking
if ! srun -n 64 python scripts/compute_all_stats.py --model "$MODEL" --bcm --no-bispectrum; then
    echo ""
    echo "========================================"
    echo "ERROR: Python script failed with exit code $?"
    echo "Model: $MODEL"
    echo "End time: $(date)"
    echo "========================================"
    exit 1
fi

echo ""
echo "========================================"
echo "End time: $(date)"
echo "Job complete!"
echo "========================================"
