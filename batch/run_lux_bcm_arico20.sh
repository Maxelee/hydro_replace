#!/bin/bash
#SBATCH --job-name=lux_bcm_arico20
#SBATCH --output=logs/lux_bcm_arico20_%A_%a.out
#SBATCH --error=logs/lux_bcm_arico20_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=024:00:00
#SBATCH -p cca
#SBATCH --mem=180G
#SBATCH --array=0-19

# =============================================================================
# Lux Ray-Tracing Pipeline - BCM arico20 Model
# =============================================================================
#
# This script processes all 20 LP realizations for the arico20 BCM model.
# 
# Realizations: 20 (LP_00 through LP_19)
# Array jobs: 0-19
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
INPUT_BASE="/mnt/home/mlee1/ceph/hydro_replace_LP_bcm/L205n2500TNG"
OUTPUT_BASE="/mnt/home/mlee1/ceph/hydro_replace_LP_lux_bcm/L205n2500TNG"
RT_OUTPUT_BASE="/mnt/home/mlee1/ceph/hydro_replace_RT_bcm/L205n2500TNG"

# Model
MODEL="arico20"

# Configuration
N_RT_RUNS=100
RT_GRID=1024
LP_GRID=4096

cd "$WORK_DIR"
mkdir -p logs

# =============================================================================
# Determine which realization this task handles
# =============================================================================

REALIZATION=$SLURM_ARRAY_TASK_ID

echo "=============================================="
echo "Lux Pipeline - BCM arico20 (Array Task $SLURM_ARRAY_TASK_ID / 19)"
echo "Model: $MODEL"
echo "Realization: LP_$(printf '%02d' $REALIZATION)"
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

# Verify lensplanes exist
if [ ! -f "$INPUT_DIR/lenspot01.dat" ]; then
    echo "Lensplane files not found in: $INPUT_DIR"
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
if [ -f "$RT_DIR/run$(printf '%03d' $N_RT_RUNS)/kappa40.dat" ]; then
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

echo ""
echo "=============================================="
echo "Task $SLURM_ARRAY_TASK_ID complete"
echo "Finished at: $(date)"
echo "=============================================="
