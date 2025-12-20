#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J all_modes_test
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o logs/all_modes_test.o%j
#SBATCH -e logs/all_modes_test.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Run All Pipeline Modes - Comprehensive Test
# =============================================================================
# Tests all modes on low-resolution TNG (625^3) for validation:
#   1. hydro    - Pure hydrodynamic simulation (truth)
#   2. dmo      - Pure dark matter only (baseline)
#   3. replace  - DMO with hydro particles in halos
#   4. bcm-arico    - BCM with Arico20 model
#   5. bcm-schneider- BCM with Schneider25 model
#   6. bcm-mead     - BCM with Mead20 model (if available)
#
# Usage:
#   sbatch batch/run_all_modes.sh
#
#   # Or with custom resolution:
#   sbatch --export=RES=1250 batch/run_all_modes.sh
# =============================================================================

set -e

# Load modules
module purge
module load python openmpi hdf5
module restore lux  # Restore saved module collection for lux dependencies

# Activate virtual environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Move to project directory
cd /mnt/home/mlee1/hydro_replace2

# Create logs directory
mkdir -p logs

# Parameters
RES=${RES:-625}           # Low resolution for testing
SNAP=${SNAP:-99}          # z=0 snapshot
MASS_MIN=${MASS_MIN:-1e12}
MASS_MAX=${MASS_MAX:-1e16}
RADIUS=${RADIUS:-5.0}

echo "========================================================================"
echo "COMPREHENSIVE PIPELINE TEST - ALL MODES"
echo "========================================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Resolution: ${RES}^3 (L205n${RES}TNG)"
echo "Snapshot: $SNAP"
echo "Mass range: $MASS_MIN - $MASS_MAX Msun/h"
echo "Radius: ${RADIUS} x R_200"
echo "========================================================================"
echo ""

# Function to run a mode and time it
run_mode() {
    local mode=$1
    local bcm_model=$2
    
    echo "========================================================================"
    if [ -z "$bcm_model" ]; then
        echo "Running mode: $mode"
    else
        echo "Running mode: $mode (BCM model: $bcm_model)"
    fi
    echo "Start time: $(date)"
    echo "========================================================================"
    
    start_time=$(date +%s)
    
    if [ -z "$bcm_model" ]; then
        python -u scripts/hydro_replace_pipeline.py \
            --resolution $RES \
            --mode $mode \
            --snapshot $SNAP \
            --mass-min $MASS_MIN \
            --mass-max $MASS_MAX \
            --radius $RADIUS
    else
        python -u scripts/hydro_replace_pipeline.py \
            --resolution $RES \
            --mode $mode \
            --bcm-model $bcm_model \
            --snapshot $SNAP \
            --mass-min $MASS_MIN \
            --mass-max $MASS_MAX \
            --radius $RADIUS
    fi
    
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    
    echo ""
    echo "Completed $mode in ${elapsed}s"
    echo ""
}

# Track total time
total_start=$(date +%s)

# =============================================================================
# 1. HYDRO (truth)
# =============================================================================
run_mode "hydro"

# =============================================================================
# 2. DMO (baseline)
# =============================================================================
run_mode "dmo"

# =============================================================================
# 3. REPLACE (hydro replacement)
# =============================================================================
run_mode "replace"

# =============================================================================
# 4. BCM - Arico20
# =============================================================================
run_mode "bcm" "Arico20"

# =============================================================================
# 5. BCM - Schneider25
# =============================================================================
run_mode "bcm" "Schneider25"

# =============================================================================
# 6. BCM - Schneider19 
# =============================================================================
run_mode "bcm" "Schneider19"

# =============================================================================
# Summary
# =============================================================================
total_end=$(date +%s)
total_elapsed=$((total_end - total_start))

echo "========================================================================"
echo "ALL MODES COMPLETED"
echo "========================================================================"
echo "Total runtime: ${total_elapsed}s ($(echo "scale=1; $total_elapsed/60" | bc) min)"
echo "End time: $(date)"
echo ""
echo "Output directories:"
echo "  /mnt/home/mlee1/ceph/hydro_replace/L205n${RES}TNG/"
echo "    ├── hydro/"
echo "    ├── dmo/"
echo "    ├── replace/"
echo "    └── bcm_*/"
echo ""
echo "Next steps:"
echo "  1. Compare power spectra: compare_pk.py"
echo "  2. Compare profiles: compare_profiles.py"
echo "  3. Run lux ray-tracing (separate job)"
echo "========================================================================"
