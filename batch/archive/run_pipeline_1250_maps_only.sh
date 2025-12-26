#!/bin/bash
#SBATCH -J maps_1250
#SBATCH -o logs/maps_1250_%A_%a.o
#SBATCH -e logs/maps_1250_%A_%a.e
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -p cca
#SBATCH --array=0-20

# Array job for L205n1250TNG - MAPS ONLY
#
# This skips statistics and profiles, goes straight to map generation.
# Use this after the fix to generate_maps_cached.py
#
# Usage:
#   sbatch batch/run_pipeline_1250_maps_only.sh

set -e

# Environment
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Configuration
SIM_RES=1250
MASS_MIN=12.5
RADIUS_FACTOR=5.0
NTASKS=32

# All 21 snapshots: 99 + 20 ray-tracing snapshots
SNAPSHOTS=(99 29 31 33 35 38 41 43 46 49 52 56 59 63 67 71 76 80 85 90 96)

# Get snapshot for this array task
SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "MAPS ONLY - L205n${SIM_RES}TNG, snap ${SNAP}"
echo "=========================================="
echo "Job ID:     $SLURM_ARRAY_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Nodes:      $SLURM_JOB_NODELIST"
echo "Start:      $(date)"
echo ""

OUTPUT_DIR=/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG

# Create output directories
mkdir -p ${OUTPUT_DIR}/snap$(printf "%03d" ${SNAP})/projected

# Check prerequisites
CACHE_FILE=${OUTPUT_DIR}/particle_cache/cache_snap$(printf "%03d" ${SNAP}).h5
if [ ! -f "${CACHE_FILE}" ]; then
    echo "ERROR: Cache file not found: ${CACHE_FILE}"
    echo "Cannot generate maps without particle cache."
    exit 1
fi

# ==========================================
# Generate 2D maps (DMO, Hydro, Replace)
# ==========================================
echo "=========================================="
echo "Generating 2D maps"
echo "=========================================="

MAP_FILE=${OUTPUT_DIR}/snap$(printf "%03d" ${SNAP})/projected/replace_M${MASS_MIN/./p}.npz
if [ -f "${MAP_FILE}" ]; then
    echo "  Replace map exists, skipping"
else
    echo "  Generating maps..."
    time mpirun -np ${NTASKS} python scripts/generate_maps_cached_lowmem.py \
        --sim-res ${SIM_RES} \
        --snap ${SNAP} \
        --mass-min ${MASS_MIN}
fi

echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "SNAPSHOT ${SNAP} COMPLETE"
echo "=========================================="
echo "End: $(date)"
echo ""
echo "Output files:"
ls -lh ${OUTPUT_DIR}/snap$(printf "%03d" ${SNAP})/projected/*.npz 2>/dev/null || echo "  (no maps)"
echo ""
echo "Done!"
