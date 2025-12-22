#!/bin/bash
#SBATCH --job-name=stats_625
#SBATCH --output=logs/stats_625_%j.o
#SBATCH --error=logs/stats_625_%j.e
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00
#SBATCH --partition=cca

# =============================================================================
# Test halo statistics on L205n625TNG (low resolution)
# =============================================================================
# Computes baryon fractions and mass conservation for all matched halos.
# Run this AFTER test_cache_625.sh completes successfully.
# =============================================================================

set -e

echo "=============================================================="
echo "HALO STATISTICS TEST - L205n625TNG"
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

# Check that cache file exists
CACHE_FILE="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/particle_cache/cache_snap0${SNAPSHOT}.h5"
if [ ! -f "$CACHE_FILE" ]; then
    echo "ERROR: Particle cache not found!"
    echo "  Expected: $CACHE_FILE"
    echo "  Run test_cache_625.sh first."
    exit 1
fi
echo "✓ Particle cache found: $CACHE_FILE"
ls -lh "$CACHE_FILE"

echo ""
echo "Starting halo statistics computation..."
echo ""

srun python scripts/compute_halo_statistics.py \
    --snap ${SNAPSHOT} \
    --sim-res ${SIM_RES} \
    --mass-min 11.5

echo ""
echo "=============================================================="
echo "Checking output..."
echo "=============================================================="

OUTPUT_FILE="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/analysis/halo_statistics_snap0${SNAPSHOT}.h5"
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ Output file created: $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
    
    # Show structure
    echo ""
    echo "Output structure:"
    h5ls -r "$OUTPUT_FILE"
else
    echo "✗ Output file NOT found: $OUTPUT_FILE"
    exit 1
fi

echo ""
echo "=============================================================="
echo "Halo statistics complete!"
echo "=============================================================="
