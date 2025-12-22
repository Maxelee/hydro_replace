#!/bin/bash
#SBATCH --job-name=cache_625
#SBATCH --output=logs/cache_625_%j.o
#SBATCH --error=logs/cache_625_%j.e
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00
#SBATCH --partition=cca

# =============================================================================
# Test particle cache generation on L205n625TNG (low resolution)
# =============================================================================
# This is a quick test to validate the caching framework before running on
# higher resolution simulations.
# =============================================================================

set -e

echo "=============================================================="
echo "PARTICLE CACHE GENERATION TEST - L205n625TNG"
echo "=============================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks: ${SLURM_NTASKS}"
echo "=============================================================="

# Environment setup
module purge
module load python openmpi python-mpi hdf5

source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
cd /mnt/home/mlee1/hydro_replace2

SNAPSHOT=99
SIM_RES=625

# Check that matches file exists
MATCHES_FILE="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/matches/matches_snap0${SNAPSHOT}.npz"
if [ ! -f "$MATCHES_FILE" ]; then
    echo "ERROR: Matches file not found!"
    echo "  Expected: $MATCHES_FILE"
    echo "  Run generate_matches_fast.py first."
    exit 1
fi
echo "✓ Matches file found: $MATCHES_FILE"

# Check output directory
OUTPUT_DIR="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/particle_cache"
echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Starting particle cache generation..."
echo ""

srun python scripts/generate_particle_cache.py \
    --sim-res ${SIM_RES} \
    --snap ${SNAPSHOT}

echo ""
echo "=============================================================="
echo "Checking output..."
echo "=============================================================="

CACHE_FILE="${OUTPUT_DIR}/cache_snap0${SNAPSHOT}.h5"
if [ -f "$CACHE_FILE" ]; then
    echo "✓ Cache file created: $CACHE_FILE"
    ls -lh "$CACHE_FILE"
    
    # Show structure with h5ls
    echo ""
    echo "Cache structure:"
    h5ls -r "$CACHE_FILE" | head -30
else
    echo "✗ Cache file NOT found: $CACHE_FILE"
    exit 1
fi

echo ""
echo "=============================================================="
echo "Cache generation complete!"
echo "=============================================================="
