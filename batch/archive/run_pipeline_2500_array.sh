#!/bin/bash
#SBATCH -J pipe_2500
#SBATCH -o logs/pipe_2500_%A_%a.o
#SBATCH -e logs/pipe_2500_%A_%a.e
#SBATCH -N 8
#SBATCH --ntasks-per-node=2
#SBATCH --exclusive
#SBATCH -t 16:00:00
#SBATCH -p cca
#SBATCH --array=0-20

# Array job for L205n2500TNG - process each snapshot in parallel
#
# This runs per snapshot:
#   0. Generate halo matches
#   1. Particle cache generation (M > 10^12.5, 5×R200)
#   2. Halo statistics computation
#   3. Profile generation
#   4. 2D map generation (DMO, Hydro, Replace)
#
# Parameters:
#   - mass_min = 12.5 (10^12.5 Msun/h)
#   - mass_max = None (no upper limit)
#   - radius_factor = 5.0 (5×R200)
#
# Usage:
#   sbatch batch/run_pipeline_2500_array.sh
#
# After completion, run lens planes:
#   sbatch --dependency=afterok:<JOBID> batch/run_pipeline_2500_lensplanes.sh

set -e

# Environment
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Configuration
SIM_RES=2500
MASS_MIN=12.5
RADIUS_FACTOR=5.0
NTASKS=16

# All 21 snapshots: 99 + 20 ray-tracing snapshots
SNAPSHOTS=(99 29 31 33 35 38 41 43 46 49 52 56 59 63 67 71 76 80 85 90 96)

# Get snapshot for this array task
SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "PIPELINE - L205n${SIM_RES}TNG, snap ${SNAP}"
echo "=========================================="
echo "Job ID:     $SLURM_ARRAY_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Nodes:      $SLURM_JOB_NODELIST"
echo "Start:      $(date)"
echo ""

echo "Configuration:"
echo "  Resolution:    L205n${SIM_RES}TNG"
echo "  Snapshot:      ${SNAP}"
echo "  Mass min:      10^${MASS_MIN} Msun/h"
echo "  Radius factor: ${RADIUS_FACTOR}×R200"
echo "  MPI tasks:     ${NTASKS}"
echo ""

OUTPUT_DIR=/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG

# Create output directories
mkdir -p ${OUTPUT_DIR}/{matches,particle_cache,analysis,profiles}

# ==========================================
# Step 0: Generate halo matches
# ==========================================
echo "=========================================="
echo "STEP 0: Generate halo matches"
echo "=========================================="

MATCH_FILE=${OUTPUT_DIR}/matches/matches_snap$(printf "%03d" ${SNAP}).npz
if [ -f "${MATCH_FILE}" ]; then
    echo "  Matches exist, skipping"
else
    echo "  Generating matches..."
    time python scripts/generate_matches_fast.py \
        --snap ${SNAP} \
        --resolution ${SIM_RES}
fi

echo ""

# ==========================================
# Step 1: Generate particle cache
# ==========================================
echo "=========================================="
echo "STEP 1: Generate particle cache"
echo "=========================================="

CACHE_FILE=${OUTPUT_DIR}/particle_cache/cache_snap$(printf "%03d" ${SNAP}).h5
if [ -f "${CACHE_FILE}" ]; then
    echo "  Cache exists, skipping"
else
    echo "  Generating cache..."
    time mpirun -np ${NTASKS} python scripts/generate_particle_cache.py \
        --sim-res ${SIM_RES} \
        --snap ${SNAP}
fi

echo ""

# ==========================================
# Step 2: Compute halo statistics
# ==========================================
echo "=========================================="
echo "STEP 2: Compute halo statistics"
echo "=========================================="

STATS_FILE=${OUTPUT_DIR}/analysis/halo_statistics_snap$(printf "%03d" ${SNAP}).h5
if [ -f "${STATS_FILE}" ]; then
    echo "  Statistics exist, skipping"
else
    echo "  Computing statistics..."
    # Use lowmem version to avoid MPI broadcast of large particle ID lists
    time mpirun -np ${NTASKS} python scripts/compute_halo_statistics_lowmem.py \
        --sim-res ${SIM_RES} \
        --snap ${SNAP} \
        --mass-min ${MASS_MIN}
fi

echo ""

# ==========================================
# Step 3: Generate profiles
# ==========================================
echo "=========================================="
echo "STEP 3: Generate profiles"
echo "=========================================="

PROFILE_FILE=${OUTPUT_DIR}/profiles/profiles_snap$(printf "%03d" ${SNAP}).h5
if [ -f "${PROFILE_FILE}" ]; then
    echo "  Profiles exist, skipping"
else
    echo "  Generating profiles..."
    time mpirun -np ${NTASKS} python scripts/generate_profiles_cached_new.py \
        --sim-res ${SIM_RES} \
        --snap ${SNAP} \
        --mass-min ${MASS_MIN}
fi

echo ""

# ==========================================
# Step 4: Generate 2D maps
# ==========================================
echo "=========================================="
echo "STEP 4: Generate 2D maps"
echo "=========================================="

MAP_FILE=${OUTPUT_DIR}/snap$(printf "%03d" ${SNAP})/projected/replace_M${MASS_MIN/./p}.npz
if [ -f "${MAP_FILE}" ]; then
    echo "  Maps exist, skipping"
else
    echo "  Generating maps..."
    # Use low-memory version for 2500 resolution
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
ls -lh ${OUTPUT_DIR}/matches/matches_snap$(printf "%03d" ${SNAP}).npz 2>/dev/null || echo "  (no matches)"
ls -lh ${OUTPUT_DIR}/particle_cache/cache_snap$(printf "%03d" ${SNAP}).h5 2>/dev/null || echo "  (no cache)"
ls -lh ${OUTPUT_DIR}/analysis/halo_statistics_snap$(printf "%03d" ${SNAP}).h5 2>/dev/null || echo "  (no stats)"
ls -lh ${OUTPUT_DIR}/profiles/profiles_snap$(printf "%03d" ${SNAP}).h5 2>/dev/null || echo "  (no profiles)"
ls -lh ${OUTPUT_DIR}/snap$(printf "%03d" ${SNAP})/projected/*.npz 2>/dev/null | head -5 || echo "  (no maps)"
echo ""
echo "Done!"
