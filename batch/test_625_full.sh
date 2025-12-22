#!/bin/bash
#SBATCH --job-name=test_625_full
#SBATCH --output=logs/test_625_full_%j.o
#SBATCH --error=logs/test_625_full_%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=cca

# =============================================================================
# Full Test Pipeline on L205n625TNG (low resolution)
# =============================================================================
# Runs both:
#   1. Particle cache generation (new format with hydro_at_dmo, hydro_at_hydro)
#   2. Halo statistics computation (baryon fractions, mass conservation)
#
# MEMORY NOTE: Statistics computation loads full snapshots into each rank's
# memory. 625 resolution needs ~2-3GB per rank, so we use fewer ranks with
# total 64GB memory (4 tasks sharing it).
# =============================================================================

set -e

echo "=============================================================="
echo "FULL TEST PIPELINE - L205n625TNG"
echo "=============================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks: ${SLURM_NTASKS}"
date
echo "=============================================================="

# Environment setup
module purge
module load python openmpi python-mpi hdf5

source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
cd /mnt/home/mlee1/hydro_replace2

SNAPSHOT=99
SIM_RES=625

# =============================================================================
# Step 1: Generate Particle Cache
# =============================================================================
echo ""
echo "=============================================================="
echo "STEP 1: Particle Cache Generation"
echo "=============================================================="

# Check that matches file exists
MATCHES_FILE="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/matches/matches_snap0${SNAPSHOT}.npz"
if [ ! -f "$MATCHES_FILE" ]; then
    echo "ERROR: Matches file not found: $MATCHES_FILE"
    exit 1
fi
echo "✓ Matches file: $MATCHES_FILE"

# Clean up any existing cache (to test fresh generation)
CACHE_DIR="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/particle_cache"
CACHE_FILE="${CACHE_DIR}/cache_snap0${SNAPSHOT}.h5"
mkdir -p "$CACHE_DIR"

# Only regenerate cache if it doesn't exist (to save time on reruns)
if [ -f "$CACHE_FILE" ]; then
    echo "✓ Cache file already exists: $CACHE_FILE"
    ls -lh "$CACHE_FILE"
    echo "Skipping cache generation..."
else
    echo ""
    echo "Starting particle cache generation..."
    T_START=$(date +%s)

    srun python scripts/generate_particle_cache.py \
        --sim-res ${SIM_RES} \
        --snap ${SNAPSHOT}

    T_END=$(date +%s)
    echo "Cache generation took $((T_END - T_START)) seconds"

    # Verify cache was created
    if [ ! -f "$CACHE_FILE" ]; then
        echo "ERROR: Cache file not created!"
        exit 1
    fi
    echo ""
    echo "✓ Cache file created: $CACHE_FILE"
    ls -lh "$CACHE_FILE"
fi

echo ""
echo "Cache structure:"
h5ls -r "$CACHE_FILE" | head -40

# =============================================================================
# Step 2: Compute Halo Statistics
# =============================================================================
echo ""
echo "=============================================================="
echo "STEP 2: Halo Statistics Computation"
echo "=============================================================="

echo "Starting halo statistics..."
T_START=$(date +%s)

srun python scripts/compute_halo_statistics.py \
    --snap ${SNAPSHOT} \
    --sim-res ${SIM_RES} \
    --mass-min 11.5

T_END=$(date +%s)
echo "Statistics computation took $((T_END - T_START)) seconds"

# Verify output
OUTPUT_FILE="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/analysis/halo_statistics_snap0${SNAPSHOT}.h5"
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "ERROR: Statistics file not created!"
    exit 1
fi

echo ""
echo "✓ Statistics file created: $OUTPUT_FILE"
ls -lh "$OUTPUT_FILE"

echo ""
echo "Output structure:"
h5ls -r "$OUTPUT_FILE"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================================="
echo "PIPELINE COMPLETE"
echo "=============================================================="
date
echo ""
echo "Cache file: $CACHE_FILE"
echo "Statistics: $OUTPUT_FILE"
echo ""
echo "Next steps:"
echo "  - Inspect results in a notebook"
echo "  - If successful, run on L205n2500TNG"
echo "=============================================================="
