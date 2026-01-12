#!/bin/bash
#SBATCH --job-name=lux_convert_recovery
#SBATCH --output=logs/lux_convert_recovery_%A_%a.o
#SBATCH --error=logs/lux_convert_recovery_%A_%a.e
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=cca
#SBATCH --mem=180G
#SBATCH --array=0-78

# ============================================================================
# Convert Recovered Lensplanes to LUX Format
# ============================================================================
# After running run_lensplane_recovery.sh, this script converts the
# recovered lensplanes to lux format for ray-tracing.
#
# This processes only the model/realization combinations that had missing data.
# There are 79 such combinations identified (indexed 0-78).
# ============================================================================

set -e

module purge
module load python openmpi/4.1.8 python-mpi hdf5/mpi-1.12.3 fftw gsl
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Directories
WORK_DIR="/mnt/home/mlee1/hydro_replace2/scripts"
INPUT_BASE="/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG"
OUTPUT_BASE="/mnt/home/mlee1/ceph/hydro_replace_LP_lux/L205n2500TNG"
LP_GRID=4096

cd "$WORK_DIR"

# Define the 79 model/realization combinations that need conversion
# Format: model:realization
COMBINATIONS=(
    "hydro_replace_Ml_1.00e12_Mu_3.16e12_R_0.5:4"
    "hydro_replace_Ml_1.00e12_Mu_3.16e12_R_0.5:5"
    "hydro_replace_Ml_1.00e12_Mu_3.16e12_R_0.5:6"
    "hydro_replace_Ml_1.00e12_Mu_3.16e12_R_0.5:7"
    "hydro_replace_Ml_1.00e12_Mu_3.16e12_R_0.5:8"
    "hydro_replace_Ml_1.00e12_Mu_3.16e12_R_0.5:9"
    "hydro_replace_Ml_1.00e12_Mu_inf_R_3.0:9"
    "hydro_replace_Ml_1.00e12_Mu_inf_R_5.0:5"
    "hydro_replace_Ml_1.00e12_Mu_inf_R_5.0:6"
    "hydro_replace_Ml_1.00e12_Mu_inf_R_5.0:7"
    "hydro_replace_Ml_1.00e12_Mu_inf_R_5.0:8"
    "hydro_replace_Ml_1.00e12_Mu_inf_R_5.0:9"
    "hydro_replace_Ml_3.16e12_Mu_1.00e13_R_0.5:7"
    "hydro_replace_Ml_3.16e12_Mu_1.00e13_R_0.5:8"
    "hydro_replace_Ml_3.16e12_Mu_1.00e13_R_0.5:9"
    "hydro_replace_Ml_3.16e12_Mu_1.00e13_R_1.0:1"
    "hydro_replace_Ml_3.16e12_Mu_1.00e13_R_1.0:2"
    "hydro_replace_Ml_3.16e12_Mu_1.00e13_R_1.0:3"
    "hydro_replace_Ml_3.16e12_Mu_1.00e13_R_1.0:4"
    "hydro_replace_Ml_3.16e12_Mu_1.00e13_R_1.0:5"
    "hydro_replace_Ml_3.16e12_Mu_1.00e13_R_1.0:6"
    "hydro_replace_Ml_3.16e12_Mu_1.00e13_R_1.0:7"
    "hydro_replace_Ml_3.16e12_Mu_1.00e13_R_1.0:8"
    "hydro_replace_Ml_3.16e12_Mu_1.00e13_R_1.0:9"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_1.0:3"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_1.0:4"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_1.0:5"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_1.0:6"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_1.0:7"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_1.0:8"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_1.0:9"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_3.0:4"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_3.0:5"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_3.0:6"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_3.0:7"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_3.0:8"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_3.0:9"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_5.0:1"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_5.0:2"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_5.0:3"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_5.0:4"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_5.0:5"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_5.0:6"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_5.0:7"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_5.0:8"
    "hydro_replace_Ml_3.16e12_Mu_inf_R_5.0:9"
    "hydro_replace_Ml_1.00e13_Mu_3.16e13_R_1.0:3"
    "hydro_replace_Ml_1.00e13_Mu_3.16e13_R_1.0:4"
    "hydro_replace_Ml_1.00e13_Mu_3.16e13_R_1.0:5"
    "hydro_replace_Ml_1.00e13_Mu_3.16e13_R_1.0:6"
    "hydro_replace_Ml_1.00e13_Mu_3.16e13_R_1.0:7"
    "hydro_replace_Ml_1.00e13_Mu_3.16e13_R_1.0:8"
    "hydro_replace_Ml_1.00e13_Mu_3.16e13_R_1.0:9"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_0.5:6"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_0.5:7"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_0.5:8"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_0.5:9"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_1.0:4"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_1.0:5"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_1.0:6"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_1.0:7"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_1.0:8"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_1.0:9"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_5.0:2"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_5.0:3"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_5.0:4"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_5.0:5"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_5.0:6"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_5.0:7"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_5.0:8"
    "hydro_replace_Ml_1.00e13_Mu_inf_R_5.0:9"
    "hydro_replace_Ml_3.16e13_Mu_1.00e15_R_5.0:2"
    "hydro_replace_Ml_3.16e13_Mu_1.00e15_R_5.0:3"
    "hydro_replace_Ml_3.16e13_Mu_1.00e15_R_5.0:4"
    "hydro_replace_Ml_3.16e13_Mu_1.00e15_R_5.0:5"
    "hydro_replace_Ml_3.16e13_Mu_1.00e15_R_5.0:6"
    "hydro_replace_Ml_3.16e13_Mu_1.00e15_R_5.0:7"
    "hydro_replace_Ml_3.16e13_Mu_1.00e15_R_5.0:8"
    "hydro_replace_Ml_3.16e13_Mu_1.00e15_R_5.0:9"
)

# Get this task's combination
if [ $SLURM_ARRAY_TASK_ID -ge ${#COMBINATIONS[@]} ]; then
    echo "Task ID exceeds combination count, exiting"
    exit 0
fi

COMBO="${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${COMBO%:*}"
REALIZATION="${COMBO#*:}"

echo "=============================================="
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Realization: $REALIZATION"
echo "Started at: $(date)"
echo "=============================================="

# Convert this specific model/realization
srun -n 40 python3 -u convert_to_lensplanes.py \
    --input-dir "$INPUT_BASE" \
    --output-dir "$OUTPUT_BASE" \
    --model "$MODEL" \
    --realization $REALIZATION \
    --grid $LP_GRID

echo ""
echo "=============================================="
echo "Conversion complete at: $(date)"
echo "=============================================="
