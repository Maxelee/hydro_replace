#!/bin/bash
#SBATCH --job-name=rt_recovery
#SBATCH --output=logs/rt_recovery_%A_%a.o
#SBATCH --error=logs/rt_recovery_%A_%a.e
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=cca
#SBATCH --mem=180G
#SBATCH --array=0-78

# ============================================================================
# Ray-Tracing Recovery for Recovered Lensplanes
# ============================================================================
# Run ray-tracing only for the 79 model/realization combinations
# that had missing lensplanes and have now been recovered.
# ============================================================================

set -e

module purge
module load python openmpi/4.1.8 python-mpi hdf5/mpi-1.12.3 fftw gsl
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Directories
OUTPUT_BASE="/mnt/home/mlee1/ceph/hydro_replace_LP_lux/L205n2500TNG"
RT_OUTPUT_BASE="/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG"

# Configuration
N_RT_RUNS=100
RT_GRID=1024
LP_GRID=4096

# Same combinations as the conversion script
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

# Check if input lensplanes exist
LP_DIR="$OUTPUT_BASE/$MODEL/LP_$(printf '%02d' $REALIZATION)"
if [ ! -f "$LP_DIR/lenspot39.dat" ]; then
    echo "ERROR: Input lensplanes incomplete at $LP_DIR"
    echo "Please run run_lux_convert_recovery.sh first"
    exit 1
fi

RT_DIR="$RT_OUTPUT_BASE/$MODEL/LP_$(printf '%02d' $REALIZATION)"

echo "Running ray-tracing for $MODEL/LP_$(printf '%02d' $REALIZATION)..."

# Create temporary ini file
INI_FILE="/tmp/lux_recovery_${MODEL}_LP$(printf '%02d' $REALIZATION)_$$.ini"
cat > "$INI_FILE" << EOF
LP_output_dir=$LP_DIR
RT_output_dir=$RT_DIR
LP_grid=$LP_GRID
RT_grid=$RT_GRID
planes_per_snapshot=2
angle=5.0
RT_random_seed=$((1992 + REALIZATION * 100))
RT_randomization=True
snapshot_list=96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29
snapshot_stack=false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true
verbose=True
EOF

mkdir -p "$RT_DIR"

# Create run directories (lux expects these to exist)
for run in $(seq -f "%03g" 1 $N_RT_RUNS); do
    mkdir -p "$RT_DIR/run$run"
done

# Run lux with MPI
srun -n 40 /mnt/home/mlee1/lux/lux "$INI_FILE" > "$RT_DIR/lux.log" 2>&1

rm -f "$INI_FILE"

echo ""
echo "=============================================="
echo "Ray-tracing complete at: $(date)"
echo "=============================================="
