#!/bin/bash
#SBATCH --job-name=lux_binned_unified
#SBATCH --output=logs/lux_binned_unified_%A_%a.out
#SBATCH --error=logs/lux_binned_unified_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=024:00:00
#SBATCH -p cca
#SBATCH --mem=180G
#SBATCH --array=0-2039

# =============================================================================
# Lux Ray-Tracing Pipeline - UNIFIED (All 2040 combinations in parallel)
# =============================================================================
#
# This script processes all (model, realization) combinations 0-2039 in 
# parallel via SLURM array job.
# 
# Models:
#   - 2 base models (dmo, hydro)
#   - 100 binned Replace configs (10 mass bins × 10 radius shells)
#   = 102 models total
#
# Realizations: 20 (LP_00 through LP_19)
#
# Total combinations: 102 × 20 = 2040
# Array jobs: 0-2039 (all running in parallel)
#
# Each array task:
#   1. Converts 40 lensplanes (distributed across 40 MPI ranks)
#   2. Runs 100 ray-tracing realizations (distributed across 40 MPI ranks)
#
# =============================================================================

set -e

# Load modules
module purge
module load python openmpi python-mpi hdf5
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
# Build model list
# =============================================================================

# Base models
MODELS=("dmo" "hydro")

# Binned Replace configurations
# Mass edges: 12.0, 12.5, 13.0, 13.5, 15.0 (5 points -> 10 combinations)
# Radius edges: 0.0, 0.5, 1.0, 3.0, 5.0 (5 points -> 10 combinations)
# Format: hydro_replace_Ml_{M_lo}_Mu_{M_hi}_Ri_{R_inner}_Ro_{R_outer}

MASS_EDGES=("1.00e12" "3.16e12" "1.00e13" "3.16e13" "1.00e15")
RADIUS_EDGES=("0.0" "0.5" "1.0" "3.0" "5.0")

# Generate all mass bin combinations
for (( i=0; i<${#MASS_EDGES[@]}-1; i++ )); do
    for (( j=i+1; j<${#MASS_EDGES[@]}; j++ )); do
        M_lo="${MASS_EDGES[$i]}"
        M_hi="${MASS_EDGES[$j]}"
        
        # Generate all radius shell combinations
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

# Task ID -> (model_idx, realization)
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / N_REALIZATIONS))
REALIZATION=$((SLURM_ARRAY_TASK_ID % N_REALIZATIONS))

if [ $MODEL_IDX -ge $N_MODELS ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds model count ($N_MODELS), exiting"
    exit 0
fi

MODEL="${MODELS[$MODEL_IDX]}"

echo "=============================================="
echo "Lux Pipeline - Unified (Array Task $SLURM_ARRAY_TASK_ID / 2039)"
echo "Model: $MODEL (index $MODEL_IDX / $((N_MODELS-1)))"
echo "Realization: $REALIZATION / $((N_REALIZATIONS-1))"
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

# Check if already converted (check for config.dat and last lens plane)
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

# Check if already done (check for last RT run's output)
if [ -f "$RT_DIR/run$(printf '%03d' $N_RT_RUNS)/kappa_40.dat" ]; then
    echo "Ray-tracing already complete for $MODEL/LP_$(printf '%02d' $REALIZATION)"
else
    echo "Running ray-tracing for $MODEL/LP_$(printf '%02d' $REALIZATION)..."
    
    # Create temporary ini file
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
    
    # Create run directories (lux expects these to exist)
    for run in $(seq -f "%03g" 1 $N_RT_RUNS); do
        mkdir -p "$RT_DIR/run$run"
    done
    
    # Run lux with MPI
    module purge
    module restore lux
    srun -n 40 /mnt/home/mlee1/lux/lux "$INI_FILE" > "$RT_DIR/lux.log" 2>&1
    
    rm -f "$INI_FILE"
    echo "Ray-tracing complete"
fi

# =============================================================================
# STEP 3: Cleanup - Remove intermediate lenspot files
# =============================================================================

echo "Cleaning up lenspot files..."
rm -f "$OUTPUT_DIR"/lenspot*.dat
echo "Cleanup complete"

echo ""
echo "=============================================="
echo "Task $SLURM_ARRAY_TASK_ID complete"
echo "Finished at: $(date)"
echo "=============================================="
