#!/bin/bash
#SBATCH --job-name=profiles_fast
#SBATCH --output=logs/profiles_fast_%j.o
#SBATCH --error=logs/profiles_fast_%j.e
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=cca

# ============================================================================
# Fast Profile Generation
# ============================================================================
# Uses illustris_python's loadHalo to load particles belonging to halos
# directly, which is ~100x faster than scanning all snapshot files.
# ============================================================================

set -e
echo "=========================================="
echo "Fast Profile Generation"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Configuration
SIM_RES=${SIM_RES:-2500}
SNAPSHOT=${SNAPSHOT:-99}
MASS_MIN=${MASS_MIN:-12.5}
MASS_MAX=${MASS_MAX:-}
SKIP_BCM=${SKIP_BCM:-1}

echo "Configuration:"
echo "  SIM_RES: $SIM_RES"
echo "  SNAPSHOT: $SNAPSHOT"
echo "  MASS_MIN: $MASS_MIN"
echo "  MASS_MAX: $MASS_MAX"
echo "  SKIP_BCM: $SKIP_BCM"
echo ""

# Environment setup
module purge
module load python openmpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

echo "Environment:"
echo "  Python: $(which python)"
echo "  MPI tasks: $SLURM_NTASKS"
echo ""

cd /mnt/home/mlee1/hydro_replace2

# Build command
CMD="python scripts/generate_profiles_fast.py --snap $SNAPSHOT --sim-res $SIM_RES --mass-min $MASS_MIN"

if [ -n "$MASS_MAX" ]; then
    CMD="$CMD --mass-max $MASS_MAX"
fi

if [ "$SKIP_BCM" == "1" ]; then
    CMD="$CMD --skip-bcm"
fi

echo "Running: $CMD"
echo ""

mpirun $CMD

echo ""
echo "End time: $(date)"
echo "=========================================="
