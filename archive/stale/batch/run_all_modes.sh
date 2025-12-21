#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J all_modes_test
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 06:00:00
#SBATCH -o logs/all_modes_test.o%j
#SBATCH -e logs/all_modes_test.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Run All Pipeline Modes - Comprehensive Test
# =============================================================================
# Tests all modes on low-resolution TNG (625^3) for validation.
# ORDER: BCM first (most likely to fail), then replace, dmo, hydro
#
# Modes:
#   1. bcm-Arico20     - BCM with Arico20 model (test first)
#   2. bcm-Schneider25 - BCM with Schneider25 model
#   3. bcm-Schneider19 - BCM with Schneider19 model
#   4. replace         - DMO with hydro particles in halos
#   5. dmo             - Pure dark matter only (baseline)
#   6. hydro           - Pure hydrodynamic simulation (truth)
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
MASS_MIN=${MASS_MIN:-1e13}
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
echo "MODE ORDER: BCM (most likely to fail) → Replace → DMO → Hydro"
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
# 1. BCM - Arico20 (FIRST - most complex, most likely to fail)
# =============================================================================
run_mode "bcm" "Arico20"

# =============================================================================
# 2. BCM - Schneider25
# =============================================================================
run_mode "bcm" "Schneider25"

# =============================================================================
# 3. BCM - Schneider19 
# =============================================================================
run_mode "bcm" "Schneider19"

# =============================================================================
# 4. REPLACE (hydro replacement)
# =============================================================================
run_mode "replace"

# =============================================================================
# 5. DMO (baseline)
# =============================================================================
run_mode "dmo"

# =============================================================================
# 6. HYDRO (truth)
# =============================================================================
run_mode "hydro"

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
echo "    ├── bcm-Arico20/lens_planes/"
echo "    ├── bcm-Schneider25/lens_planes/"
echo "    ├── bcm-Schneider19/lens_planes/"
echo "    ├── replace/lens_planes/"
echo "    ├── dmo/lens_planes/"
echo "    └── hydro/lens_planes/"
echo ""
echo "Next steps:"
echo "  1. Compare power spectra: compare_pk.py"
echo "  2. Compare profiles: compare_profiles.py"
echo "  3. Run lux ray-tracing with PreProjected format"
echo "========================================================================"
