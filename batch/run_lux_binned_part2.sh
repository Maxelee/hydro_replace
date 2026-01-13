#!/bin/bash
#SBATCH --job-name=lux_binned_2
#SBATCH --output=logs/lux_binned_2_%A_%a.out
#SBATCH --error=logs/lux_binned_2_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=024:00:00
#SBATCH -p cca
#SBATCH --mem=180G
#SBATCH --array=510-1019

# =============================================================================
# Lux Ray-Tracing Pipeline - Part 2 of 4
# =============================================================================
#
# This script processes (model, realization) combinations 510-1019 of 2040 total.
# See run_lux_binned_part1.sh for full documentation.
#
# =============================================================================

set -e

# Load modules
module purge
module load python openmpi/4.1.8 python-mpi hdf5/mpi-1.12.3 fftw gsl
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Directories
WORK_DIR="/mnt/home/mlee1/hydro_replace2/scripts"
INPUT_BASE="/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG"
OUTPUT_BASE="/mnt/home/mlee1/ceph/hydro_replace_LP_lux/L205n2500TNG"
RT_OUTPUT_BASE="/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG"

# Configuration
N_REALIZATIONS=20
N_RT_RUNS=100
RT_GRID=1024
LP_GRID=4096

cd "$WORK_DIR"
mkdir -p logs

# =============================================================================
# Build model list (same as part1)
# =============================================================================

MODELS=("dmo" "hydro")

MASS_EDGES=("1.00e12" "3.16e12" "1.00e13" "3.16e13" "1.00e15")
RADIUS_EDGES=("0.0" "0.5" "1.0" "3.0" "5.0")

for (( i=0; i<${#MASS_EDGES[@]}-1; i++ )); do
    for (( j=i+1; j<${#MASS_EDGES[@]}; j++ )); do
        M_lo="${MASS_EDGES[$i]}"
        M_hi="${MASS_EDGES[$j]}"
        for (( k=0; k<${#RADIUS_EDGES[@]}-1; k++ )); do
            for (( l=k+1; l<${#RADIUS_EDGES[@]}; l++ )); do
                R_inner="${RADIUS_EDGES[$k]}"
                R_outer="${RADIUS_EDGES[$l]}"
                MODELS+=("hydro_replace_Ml_${M_lo}_Mu_${M_hi}_Ri_${R_inner}_Ro_${R_outer}")
            done
        done
    done
done

N_MODELS=${#MODELS[@]}

# =============================================================================
# Determine which (model, realization) this task handles
# =============================================================================

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / N_REALIZATIONS))
REALIZATION=$((SLURM_ARRAY_TASK_ID % N_REALIZATIONS))

if [ $MODEL_IDX -ge $N_MODELS ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds model count ($N_MODELS), exiting"
    exit 0
fi

MODEL="${MODELS[$MODEL_IDX]}"

echo "=============================================="
echo "Lux Pipeline - Part 2"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL (index $MODEL_IDX)"
echo "Realization: $REALIZATION"
echo "Started at: $(date)"
echo "=============================================="

# =============================================================================
# Check if input data exists
# =============================================================================

INPUT_DIR="$INPUT_BASE/$MODEL/LP_$(printf '%02d' $REALIZATION)"
if [ ! -d "$INPUT_DIR" ]; then
    echo "Input directory not found: $INPUT_DIR"
    echo "Skipping this task"
    exit 0
fi

# =============================================================================
# STEP 1: Convert mass planes to lux format
# =============================================================================

OUTPUT_DIR="$OUTPUT_BASE/$MODEL/LP_$(printf '%02d' $REALIZATION)"

if [ -f "$OUTPUT_DIR/config.dat" ] && [ -f "$OUTPUT_DIR/lenspot40.dat" ]; then
    echo "Conversion already complete for $MODEL/LP_$(printf '%02d' $REALIZATION)"
else
    echo "Converting $MODEL/LP_$(printf '%02d' $REALIZATION)..."
    
    srun -n 40 python3 -u /mnt/home/mlee1/hydro_replace2/scripts/convert_to_lensplanes.py \
        --input-dir "$INPUT_BASE" \
        --output-dir "$OUTPUT_BASE" \
        --model "$MODEL" \
        --realization $REALIZATION \
        --grid $LP_GRID
    
    echo "Conversion complete"
fi

# =============================================================================
# STEP 2: Run ray-tracing
# =============================================================================

RT_DIR="$RT_OUTPUT_BASE/$MODEL/LP_$(printf '%02d' $REALIZATION)"

if [ -f "$RT_DIR/run$(printf '%03d' $N_RT_RUNS)/kappa_40.dat" ]; then
    echo "Ray-tracing already complete for $MODEL/LP_$(printf '%02d' $REALIZATION)"
else
    echo "Running ray-tracing for $MODEL/LP_$(printf '%02d' $REALIZATION)..."
    
    INI_FILE="/tmp/lux_${MODEL}_LP$(printf '%02d' $REALIZATION)_$$.ini"
    cat > "$INI_FILE" << EOF
LP_output_dir=$OUTPUT_DIR
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
    
    for run in $(seq -f "%03g" 1 $N_RT_RUNS); do
        mkdir -p "$RT_DIR/run$run"
    done
    
    srun -n 40 /mnt/home/mlee1/lux/lux "$INI_FILE" > "$RT_DIR/lux.log" 2>&1
    
    rm -f "$INI_FILE"
    echo "Ray-tracing complete"
fi

echo ""
echo "=============================================="
echo "Task $SLURM_ARRAY_TASK_ID complete"
echo "Finished at: $(date)"
echo "=============================================="
