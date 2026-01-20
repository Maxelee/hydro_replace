#!/bin/bash
#SBATCH --job-name=stats_%a
#SBATCH --output=logs/stats_%a_%j.out
#SBATCH --error=logs/stats_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --partition=cca
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --array=0-33

# Array job to compute statistics for all models
# Array indices 0-33 cover the main models:
#   0: dmo
#   1: hydro
#   2-17: 16 cumulative Replace models
#   18-33: 16 discrete tile Replace models

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Create logs directory
mkdir -p logs

# Define model list
MODELS=(
    "dmo"
    "hydro"
    # Cumulative models: 4 mass × 4 radii = 16
    "hydro_replace_Ml_1.00e+12_Mu_1.00e+15_Ri_0.0_Ro_0.5"
    "hydro_replace_Ml_1.00e+12_Mu_1.00e+15_Ri_0.0_Ro_1.0"
    "hydro_replace_Ml_1.00e+12_Mu_1.00e+15_Ri_0.0_Ro_3.0"
    "hydro_replace_Ml_1.00e+12_Mu_1.00e+15_Ri_0.0_Ro_5.0"
    "hydro_replace_Ml_3.16e+12_Mu_1.00e+15_Ri_0.0_Ro_0.5"
    "hydro_replace_Ml_3.16e+12_Mu_1.00e+15_Ri_0.0_Ro_1.0"
    "hydro_replace_Ml_3.16e+12_Mu_1.00e+15_Ri_0.0_Ro_3.0"
    "hydro_replace_Ml_3.16e+12_Mu_1.00e+15_Ri_0.0_Ro_5.0"
    "hydro_replace_Ml_1.00e+13_Mu_1.00e+15_Ri_0.0_Ro_0.5"
    "hydro_replace_Ml_1.00e+13_Mu_1.00e+15_Ri_0.0_Ro_1.0"
    "hydro_replace_Ml_1.00e+13_Mu_1.00e+15_Ri_0.0_Ro_3.0"
    "hydro_replace_Ml_1.00e+13_Mu_1.00e+15_Ri_0.0_Ro_5.0"
    "hydro_replace_Ml_3.16e+13_Mu_1.00e+15_Ri_0.0_Ro_0.5"
    "hydro_replace_Ml_3.16e+13_Mu_1.00e+15_Ri_0.0_Ro_1.0"
    "hydro_replace_Ml_3.16e+13_Mu_1.00e+15_Ri_0.0_Ro_3.0"
    "hydro_replace_Ml_3.16e+13_Mu_1.00e+15_Ri_0.0_Ro_5.0"
    # Discrete tile models: 4 mass × 4 radii = 16
    "hydro_replace_Ml_1.00e+12_Mu_3.16e+12_Ri_0.0_Ro_0.5"
    "hydro_replace_Ml_1.00e+12_Mu_3.16e+12_Ri_0.5_Ro_1.0"
    "hydro_replace_Ml_1.00e+12_Mu_3.16e+12_Ri_1.0_Ro_3.0"
    "hydro_replace_Ml_1.00e+12_Mu_3.16e+12_Ri_3.0_Ro_5.0"
    "hydro_replace_Ml_3.16e+12_Mu_1.00e+13_Ri_0.0_Ro_0.5"
    "hydro_replace_Ml_3.16e+12_Mu_1.00e+13_Ri_0.5_Ro_1.0"
    "hydro_replace_Ml_3.16e+12_Mu_1.00e+13_Ri_1.0_Ro_3.0"
    "hydro_replace_Ml_3.16e+12_Mu_1.00e+13_Ri_3.0_Ro_5.0"
    "hydro_replace_Ml_1.00e+13_Mu_3.16e+13_Ri_0.0_Ro_0.5"
    "hydro_replace_Ml_1.00e+13_Mu_3.16e+13_Ri_0.5_Ro_1.0"
    "hydro_replace_Ml_1.00e+13_Mu_3.16e+13_Ri_1.0_Ro_3.0"
    "hydro_replace_Ml_1.00e+13_Mu_3.16e+13_Ri_3.0_Ro_5.0"
    "hydro_replace_Ml_3.16e+13_Mu_1.00e+15_Ri_0.0_Ro_0.5"
    "hydro_replace_Ml_3.16e+13_Mu_1.00e+15_Ri_0.5_Ro_1.0"
    "hydro_replace_Ml_3.16e+13_Mu_1.00e+15_Ri_1.0_Ro_3.0"
    "hydro_replace_Ml_3.16e+13_Mu_1.00e+15_Ri_3.0_Ro_5.0"
)

# Get model for this array task
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Start time: $(date)"
echo "========================================"
echo ""

# Run the script
mpirun -np 64 python scripts/compute_all_stats.py --model "$MODEL"

echo ""
echo "========================================"
echo "End time: $(date)"
echo "Job complete!"
echo "========================================"
