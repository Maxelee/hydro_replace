#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J test_pipeline
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/test_pipeline.o%j
#SBATCH -e logs/test_pipeline.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Test Hydro Replace Pipeline
# =============================================================================
# Quick test on a single snapshot with DMO mode (no halo matching needed)
#
# Usage:
#   sbatch batch/test_pipeline.sh
#   sbatch --export=MODE=hydro,SNAP=99 batch/test_pipeline.sh
# =============================================================================

set -e

# Load modules
module load python openmpi hdf5

# Activate virtual environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Move to project directory
cd /mnt/home/mlee1/hydro_replace2

# Create logs directory if needed
mkdir -p logs

# Default parameters (can be overridden via --export)
MODE=${MODE:-dmo}
SNAP=${SNAP:-99}
MASS_MIN=${MASS_MIN:-1e12}
MASS_MAX=${MASS_MAX:-1e16}
RADIUS=${RADIUS:-5.0}

echo "=============================================="
echo "Hydro Replace Pipeline Test"
echo "=============================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Mode: $MODE"
echo "Snapshot: $SNAP"
echo "Mass range: $MASS_MIN - $MASS_MAX Msun/h"
echo "Radius: ${RADIUS} x R_200"
echo "=============================================="

# Run pipeline
python -u scripts/hydro_replace_pipeline.py \
    --mode $MODE \
    --snapshot $SNAP \
    --mass-min $MASS_MIN \
    --mass-max $MASS_MAX \
    --radius $RADIUS

echo "=============================================="
echo "Test completed at $(date)"
echo "=============================================="
