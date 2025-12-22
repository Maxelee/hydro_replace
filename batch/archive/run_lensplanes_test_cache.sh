#!/bin/bash
#SBATCH --job-name=lp_test_cache
#SBATCH --output=logs/lp_test_cache_%j.o
#SBATCH --error=logs/lp_test_cache_%j.e
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00
#SBATCH --partition=cca

# =============================================================================
# TEST LENS PLANE GENERATION WITH PARTICLE CACHE
# 
# This is a quick test to verify that:
# 1. The particle cache is loaded correctly
# 2. Replace/BCM lens planes work with cache (no KD-tree fallback)
# 3. Output files are written correctly
#
# Uses: snapshot 29 (highest-z, should complete fastest), single seed
# Models: Replace + BCM (Arico20 only for quick test)
# =============================================================================

set -e

# Load modules
module load python openmpi hdf5 fftw gsl boost/mpi-1.84.0
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Configuration
SIM_RES=2500
SNAPSHOT=29
MASS_MIN=12.5
SEED=2020
GRID_RES=4096

echo "============================================================"
echo "LENS PLANE TEST WITH PARTICLE CACHE"
echo "============================================================"
echo "Simulation:  L205n${SIM_RES}TNG"
echo "Snapshot:    ${SNAPSHOT}"
echo "Mass cut:    log10(M) >= ${MASS_MIN}"
echo "Grid:        ${GRID_RES}"
echo "Seed:        ${SEED}"
echo ""
echo "Testing models: replace, bcm_arico20"
echo "============================================================"

# Check if particle cache exists
CACHE_FILE="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/particle_cache/cache_snap0${SNAPSHOT}.h5"
if [ -f "$CACHE_FILE" ]; then
    echo "✓ Particle cache exists: $CACHE_FILE"
    ls -lh "$CACHE_FILE"
else
    echo "✗ Particle cache NOT FOUND: $CACHE_FILE"
    echo ""
    echo "Waiting for particle cache job to complete..."
    echo "Check status with: squeue -u $USER | grep particle"
    echo ""
    echo "Once cache is ready, rerun this script."
    exit 1
fi

# Check if matches file exists
MATCHES_FILE="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/matches/matches_snap0${SNAPSHOT}.npz"
if [ -f "$MATCHES_FILE" ]; then
    echo "✓ Matches file exists: $MATCHES_FILE"
else
    echo "✗ Matches file NOT FOUND: $MATCHES_FILE"
    exit 1
fi

echo ""
echo "Starting test..."
echo ""

# Run with Replace model (tests cache loading and halo replacement)
echo "=========================================="
echo "Test 1: Replace model with cache"
echo "=========================================="
time mpirun -np 128 python scripts/generate_lensplanes.py \
    --sim-res ${SIM_RES} \
    --model replace \
    --snap ${SNAPSHOT} \
    --mass-min ${MASS_MIN} \
    --grid-res ${GRID_RES} \
    --seed ${SEED} \
    --num-seeds 1

echo ""
echo "=========================================="
echo "Test 2: BCM (Arico20) model with cache"
echo "=========================================="
time mpirun -np 128 python scripts/generate_lensplanes.py \
    --sim-res ${SIM_RES} \
    --model bcm \
    --snap ${SNAPSHOT} \
    --mass-min ${MASS_MIN} \
    --grid-res ${GRID_RES} \
    --seed ${SEED} \
    --num-seeds 1 \
    --bcm-models Arico20

# Verify outputs
echo ""
echo "=========================================="
echo "Verifying outputs..."
echo "=========================================="

OUTPUT_DIR="/mnt/home/mlee1/ceph/hydro_replace_lensplanes/L205n${SIM_RES}TNG/seed${SEED}"

# Check Replace output
REPLACE_FILE="${OUTPUT_DIR}/replace_Mgt${MASS_MIN}/density01.dat"
if [ -f "$REPLACE_FILE" ]; then
    echo "✓ Replace output exists: $REPLACE_FILE"
    ls -lh "$REPLACE_FILE"
else
    echo "✗ Replace output NOT FOUND: $REPLACE_FILE"
fi

# Check BCM output
BCM_FILE="${OUTPUT_DIR}/bcm_arico20_Mgt${MASS_MIN}/density01.dat"
if [ -f "$BCM_FILE" ]; then
    echo "✓ BCM Arico20 output exists: $BCM_FILE"
    ls -lh "$BCM_FILE"
else
    echo "✗ BCM output NOT FOUND: $BCM_FILE"
fi

echo ""
echo "============================================================"
echo "TEST COMPLETE"
echo ""
echo "If timing looks good (< 5 min per seed), proceed with:"
echo "  sbatch batch/run_lensplanes_production.sh"
echo "============================================================"
