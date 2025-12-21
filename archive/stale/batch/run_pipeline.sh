#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J pipeline
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=180G
#SBATCH -t 02:00:00
#SBATCH -o logs/pipeline.o%j
#SBATCH -e logs/pipeline.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=END,FAIL

# =============================================================================
# OPTIMIZED HYDRO REPLACE PIPELINE
# =============================================================================
# 
# This script runs the full pipeline with proper parallelization:
#   1. Spatial matching (fast, ~seconds)
#   2. Particle transform (MPI parallel, ~minutes)
#   3. Projection + output (serial, ~30s per mode)
#
# Usage:
#   sbatch batch/run_pipeline.sh
#   sbatch --export=RES=2500,MODES="dmo hydro replace" batch/run_pipeline.sh
#
# =============================================================================

set -e

# Load modules
module purge
module load python openmpi python-mpi hdf5

# Activate environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2
mkdir -p logs

# =============================================================================
# PARAMETERS
# =============================================================================
RES=${RES:-625}
SNAP=${SNAP:-99}
MASS_MIN=${MASS_MIN:-1e13}
RADIUS=${RADIUS:-5.0}
GRID_SIZE=${GRID_SIZE:-4096}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/hydro_replace}

# Modes to run (space-separated)
# Options: dmo hydro replace bcm-Arico20 bcm-Schneider19 bcm-Schneider25
MODES=${MODES:-"dmo hydro replace bcm-Arico20"}

echo "========================================================================"
echo "OPTIMIZED HYDRO REPLACE PIPELINE"
echo "========================================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Resolution: ${RES}^3"
echo "Snapshot: $SNAP"
echo "Mass min: $MASS_MIN Msun/h"
echo "Radius: ${RADIUS} x R_200"
echo "Grid size: $GRID_SIZE"
echo "MPI tasks: $SLURM_NTASKS"
echo "Modes: $MODES"
echo "========================================================================"
echo ""

total_start=$(date +%s)

# =============================================================================
# STEP 1: SPATIAL MATCHING (fast, runs once)
# =============================================================================
echo "========================================================================"
echo "STEP 1: SPATIAL HALO MATCHING"
echo "========================================================================"
step1_start=$(date +%s)

python scripts/01_spatial_match.py \
    --resolution $RES \
    --snapshot $SNAP \
    --mass-min $MASS_MIN \
    --output-dir $OUTPUT_DIR

step1_end=$(date +%s)
echo "Step 1 completed in $((step1_end - step1_start))s"
echo ""

# =============================================================================
# STEP 2 & 3: TRANSFORM + PROJECT (for each mode)
# =============================================================================
for MODE in $MODES; do
    echo "========================================================================"
    echo "Processing mode: $MODE"
    echo "========================================================================"
    mode_start=$(date +%s)
    
    # Parse BCM model if present
    if [[ $MODE == bcm-* ]]; then
        BCM_MODEL=${MODE#bcm-}
        TRANSFORM_MODE="bcm"
    else
        BCM_MODEL=""
        TRANSFORM_MODE=$MODE
    fi
    
    # Step 2: Transform (serial, uses optimized numpy/scipy)
    echo ""
    echo "--- Step 2: Particle Transform ---"
    
    if [ -n "$BCM_MODEL" ]; then
        python -u scripts/02_transform.py \
            --mode $TRANSFORM_MODE \
            --bcm-model $BCM_MODEL \
            --resolution $RES \
            --snapshot $SNAP \
            --radius $RADIUS \
            --output-dir $OUTPUT_DIR
    else
        python -u scripts/02_transform.py \
            --mode $TRANSFORM_MODE \
            --resolution $RES \
            --snapshot $SNAP \
            --radius $RADIUS \
            --output-dir $OUTPUT_DIR
    fi
    
    # Step 3: Project + Output (serial)
    echo ""
    echo "--- Step 3: Projection + Output ---"
    
    python scripts/03_project_output.py \
        --mode $MODE \
        --resolution $RES \
        --snapshot $SNAP \
        --grid-size $GRID_SIZE \
        --output-dir $OUTPUT_DIR
    
    mode_end=$(date +%s)
    echo ""
    echo "Mode $MODE completed in $((mode_end - mode_start))s"
    echo ""
done

# =============================================================================
# SUMMARY
# =============================================================================
total_end=$(date +%s)
total_elapsed=$((total_end - total_start))

echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
echo "Total runtime: ${total_elapsed}s ($(echo "scale=1; $total_elapsed/60" | bc) min)"
echo "End time: $(date)"
echo ""
echo "Output directory: $OUTPUT_DIR/L205n${RES}TNG/"
echo ""
echo "Modes processed:"
for MODE in $MODES; do
    echo "  - $MODE"
done
echo ""
echo "Next: Run lux ray-tracing with PreProjected format"
echo "========================================================================"
