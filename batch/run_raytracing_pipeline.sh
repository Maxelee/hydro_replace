#!/bin/bash
#
# Full ray-tracing pipeline for hydro_replace project.
#
# This script:
#   1. Generates lens planes for all models
#   2. Generates lux configuration files
#   3. Runs lux ray-tracing
#
# Usage:
#   ./batch/run_raytracing_pipeline.sh 625      # Test with low-res
#   ./batch/run_raytracing_pipeline.sh 2500     # Production with high-res
#
# Options:
#   SIM_RES=625|1250|2500     # Simulation resolution (positional arg or env)
#   GRID_RES=1024|4096        # Grid resolution (default: 4096)
#   TEST=1                    # Test mode (1 snapshot only)
#   SKIP_LENSPLANES=1         # Skip lens plane generation
#   SKIP_LUX=1                # Skip lux ray-tracing
#

set -e

# Configuration
SIM_RES=${1:-${SIM_RES:-625}}
GRID_RES=${GRID_RES:-4096}
SEED=${SEED:-2020}
TEST=${TEST:-0}

WORKSPACE=/mnt/home/mlee1/hydro_replace2
OUTPUT_BASE=/mnt/home/mlee1/ceph/hydro_replace_lensplanes
LUX_OUTPUT=/mnt/home/mlee1/ceph/lux_out
LUX_DIR=/mnt/home/mlee1/lux

SIM_NAME="L205n${SIM_RES}TNG"

echo "=============================================="
echo "FULL RAY-TRACING PIPELINE"
echo "=============================================="
echo "Simulation: ${SIM_NAME}"
echo "Grid resolution: ${GRID_RES}"
echo "Random seed: ${SEED}"
echo "Test mode: ${TEST}"
echo "=============================================="

cd $WORKSPACE

# Determine SLURM resources based on resolution
if [ "$SIM_RES" == "625" ]; then
    NODES=4
    TASKS=64
    TIME="04:00:00"
elif [ "$SIM_RES" == "1250" ]; then
    NODES=8
    TASKS=128
    TIME="08:00:00"
else
    NODES=16
    TASKS=256
    TIME="12:00:00"
fi

# Test mode: reduce resources and only 1 snapshot
if [ "$TEST" == "1" ]; then
    SNAP=96
    TIME="01:00:00"
    # Keep GRID_RES as specified (default 4096) - don't override in test mode
else
    SNAP=all
fi

echo ""
echo "SLURM config: ${NODES} nodes, ${TASKS} tasks, ${TIME} time"
echo ""

# ============================================
# STEP 1: Generate Lens Planes
# ============================================

if [ "${SKIP_LENSPLANES:-0}" != "1" ]; then
    echo "=============================================="
    echo "STEP 1: GENERATING LENS PLANES"
    echo "=============================================="
    
    mkdir -p logs
    
    # Array to store job IDs
    JOBS=()
    
    # DMO
    echo "Submitting DMO..."
    JOB=$(SIM_RES=$SIM_RES MODEL=dmo SNAP=$SNAP SEED=$SEED GRID_RES=$GRID_RES \
          sbatch -N $NODES -n $TASKS -t $TIME batch/run_lensplanes.sh | awk '{print $NF}')
    JOBS+=("$JOB")
    echo "  Job ID: $JOB"
    
    # Hydro
    echo "Submitting Hydro..."
    JOB=$(SIM_RES=$SIM_RES MODEL=hydro SNAP=$SNAP SEED=$SEED GRID_RES=$GRID_RES \
          sbatch -N $NODES -n $TASKS -t $TIME batch/run_lensplanes.sh | awk '{print $NF}')
    JOBS+=("$JOB")
    echo "  Job ID: $JOB"
    
    # Replace with different mass cuts
    for MASS_MIN in 12.0 12.5 13.0; do
        echo "Submitting Replace (M > 10^${MASS_MIN})..."
        JOB=$(SIM_RES=$SIM_RES MODEL=replace MASS_MIN=$MASS_MIN SNAP=$SNAP SEED=$SEED GRID_RES=$GRID_RES \
              sbatch -N $NODES -n $TASKS -t $TIME batch/run_lensplanes.sh | awk '{print $NF}')
        JOBS+=("$JOB")
        echo "  Job ID: $JOB"
    done
    
    # BCM (all three models in one job)
    echo "Submitting BCM (all 3 models)..."
    JOB=$(SIM_RES=$SIM_RES MODEL=bcm SNAP=$SNAP SEED=$SEED GRID_RES=$GRID_RES \
          sbatch -N $NODES -n $TASKS -t $TIME batch/run_lensplanes.sh | awk '{print $NF}')
    JOBS+=("$JOB")
    echo "  Job ID: $JOB"
    
    echo ""
    echo "Waiting for lens plane jobs to complete..."
    echo "Job IDs: ${JOBS[*]}"
    
    # Wait for all jobs to complete
    for JOB in "${JOBS[@]}"; do
        while squeue -j $JOB -h 2>/dev/null | grep -q $JOB; do
            sleep 30
        done
        
        # Check if job completed successfully
        STATE=$(sacct -j $JOB --format=State -n -P | head -1)
        if [[ "$STATE" == *"COMPLETED"* ]]; then
            echo "  Job $JOB: COMPLETED"
        else
            echo "  Job $JOB: $STATE (WARNING: may have failed)"
        fi
    done
    
    echo "All lens plane jobs finished."
else
    echo "Skipping lens plane generation (SKIP_LENSPLANES=1)"
fi

# ============================================
# STEP 2: Generate Lux Configurations
# ============================================

echo ""
echo "=============================================="
echo "STEP 2: GENERATING LUX CONFIGURATIONS"
echo "=============================================="

# Snapshot list for lux (comma-separated)
SNAP_LIST="96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29"
SNAP_STACK="false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true"

# Function to write lux config (simple key-value format, only valid params)
write_lux_config() {
    local MODEL=$1
    local INPUT_DIR=$2
    local OUTPUT_DIR=$3
    local CONFIG_FILE=$4
    
    mkdir -p $OUTPUT_DIR
    
    cat > $CONFIG_FILE << EOF
# Lux configuration for ${MODEL}
# Auto-generated by run_raytracing_pipeline.sh

# Input/Output directories
input_dir = ${INPUT_DIR}
LP_output_dir = ${OUTPUT_DIR}
RT_output_dir = ${OUTPUT_DIR}

# Simulation format
simulation_format = PreProjected

# Lens potential calculation
LP_grid = ${GRID_RES}
LP_random_seed = ${SEED}
planes_per_snapshot = 2
projection_direction = 3
translation_rotation = true

# Snapshot configuration
snapshot_list = ${SNAP_LIST}
snapshot_stack = ${SNAP_STACK}

# Ray-tracing
RT_grid = ${GRID_RES}
RT_random_seed = ${SEED}
RT_randomization = false
angle = 3.5

verbose = true
EOF
    
    echo "  Wrote: $CONFIG_FILE"
}

# Generate configs for all models
MODELS=("dmo" "hydro")
for MODEL in "${MODELS[@]}"; do
    INPUT_DIR="${OUTPUT_BASE}/${SIM_NAME}/${MODEL}"
    OUTPUT_DIR="${LUX_OUTPUT}/${SIM_NAME}/${MODEL}"
    CONFIG_FILE="${OUTPUT_DIR}/lux_${MODEL}.ini"
    write_lux_config "$MODEL" "$INPUT_DIR" "$OUTPUT_DIR" "$CONFIG_FILE"
done

# Replace models
for MASS_MIN in 12.0 12.5 13.0; do
    MODEL_DIR="replace_Mgt${MASS_MIN}"
    INPUT_DIR="${OUTPUT_BASE}/${SIM_NAME}/${MODEL_DIR}"
    OUTPUT_DIR="${LUX_OUTPUT}/${SIM_NAME}/${MODEL_DIR}"
    CONFIG_FILE="${OUTPUT_DIR}/lux_replace.ini"
    write_lux_config "replace_M${MASS_MIN}" "$INPUT_DIR" "$OUTPUT_DIR" "$CONFIG_FILE"
done

# BCM models
for BCM_NAME in Arico20 Schneider19 Schneider25; do
    MODEL_DIR="bcm_${BCM_NAME,,}"  # lowercase
    INPUT_DIR="${OUTPUT_BASE}/${SIM_NAME}/${MODEL_DIR}"
    OUTPUT_DIR="${LUX_OUTPUT}/${SIM_NAME}/${MODEL_DIR}"
    CONFIG_FILE="${OUTPUT_DIR}/lux_bcm.ini"
    write_lux_config "bcm_${BCM_NAME}" "$INPUT_DIR" "$OUTPUT_DIR" "$CONFIG_FILE"
done

# ============================================
# STEP 3: Run Lux Ray-Tracing
# ============================================

if [ "${SKIP_LUX:-0}" != "1" ]; then
    echo ""
    echo "=============================================="
    echo "STEP 3: RUNNING LUX RAY-TRACING"
    echo "=============================================="
    
    # Array to store lux job IDs
    LUX_JOBS=()
    
    # Function to submit lux job
    submit_lux_job() {
        local MODEL=$1
        local CONFIG_FILE=$2
        local OUTPUT_DIR=$3
        local INPUT_DIR=$4
        
        # Check if density files exist
        local DENSITY_COUNT=$(ls ${INPUT_DIR}/density*.dat 2>/dev/null | wc -l)
        if [ "$DENSITY_COUNT" -eq 0 ]; then
            echo "  WARNING: No density files for ${MODEL} in ${INPUT_DIR}, skipping"
            return
        fi
        
        echo "  Found $DENSITY_COUNT density files for ${MODEL}"
        
        # Copy config.dat from input_dir to output_dir (lux raytracing needs it there)
        if [ -f "${INPUT_DIR}/config.dat" ]; then
            cp "${INPUT_DIR}/config.dat" "${OUTPUT_DIR}/config.dat"
            echo "  Copied config.dat to output directory"
        else
            echo "  WARNING: No config.dat found in ${INPUT_DIR}"
        fi
        
        # Create run directories for lux (it creates run001 to run100)
        for i in $(seq -f "%03g" 1 100); do
            mkdir -p "${OUTPUT_DIR}/run${i}"
        done
        echo "  Created run001-run100 directories"
        
        # Create SLURM script
        local SCRIPT="${OUTPUT_DIR}/run_lux.sh"
        cat > $SCRIPT << EOF
#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J lux_${MODEL}
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -o ${OUTPUT_DIR}/lux.o%j
#SBATCH -e ${OUTPUT_DIR}/lux.e%j
#SBATCH -t 02:00:00

module load openmpi hdf5

cd ${LUX_DIR}
srun -n 16 ./lux ${CONFIG_FILE}
EOF
        
        JOB=$(sbatch $SCRIPT | awk '{print $NF}')
        LUX_JOBS+=("$JOB")
        echo "  ${MODEL}: Job ID $JOB"
    }
    
    # Submit lux jobs for all models
    for MODEL in dmo hydro; do
        INPUT_DIR="${OUTPUT_BASE}/${SIM_NAME}/${MODEL}"
        OUTPUT_DIR="${LUX_OUTPUT}/${SIM_NAME}/${MODEL}"
        CONFIG_FILE="${OUTPUT_DIR}/lux_${MODEL}.ini"
        submit_lux_job "$MODEL" "$CONFIG_FILE" "$OUTPUT_DIR" "$INPUT_DIR"
    done
    
    for MASS_MIN in 12.0 12.5 13.0; do
        MODEL_DIR="replace_Mgt${MASS_MIN}"
        INPUT_DIR="${OUTPUT_BASE}/${SIM_NAME}/${MODEL_DIR}"
        OUTPUT_DIR="${LUX_OUTPUT}/${SIM_NAME}/${MODEL_DIR}"
        CONFIG_FILE="${OUTPUT_DIR}/lux_replace.ini"
        submit_lux_job "replace_M${MASS_MIN}" "$CONFIG_FILE" "$OUTPUT_DIR" "$INPUT_DIR"
    done
    
    for BCM_NAME in Arico20 Schneider19 Schneider25; do
        MODEL_DIR="bcm_${BCM_NAME,,}"  # lowercase
        INPUT_DIR="${OUTPUT_BASE}/${SIM_NAME}/${MODEL_DIR}"
        OUTPUT_DIR="${LUX_OUTPUT}/${SIM_NAME}/${MODEL_DIR}"
        CONFIG_FILE="${OUTPUT_DIR}/lux_bcm.ini"
        submit_lux_job "bcm_${BCM_NAME}" "$CONFIG_FILE" "$OUTPUT_DIR" "$INPUT_DIR"
    done
    
    echo ""
    echo "Lux jobs submitted: ${LUX_JOBS[*]}"
    echo "Check status with: squeue -u \$USER"
else
    echo "Skipping lux ray-tracing (SKIP_LUX=1)"
fi

echo ""
echo "=============================================="
echo "PIPELINE SUBMISSION COMPLETE"
echo "=============================================="
echo ""
echo "Output directories:"
echo "  Lens planes: ${OUTPUT_BASE}/${SIM_NAME}/"
echo "  Lux output:  ${LUX_OUTPUT}/${SIM_NAME}/"
