#!/bin/bash
#SBATCH -J pipe_625_full
#SBATCH -o logs/pipe_625_full_%j.o
#SBATCH -e logs/pipe_625_full_%j.e
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -p cca

# Full pipeline for L205n625TNG - all 21 snapshots
#
# This runs:
#   1. Particle cache generation (M > 10^12.5, 5×R200)
#   2. Halo statistics computation
#   3. Profile generation
#   4. 2D map generation (DMO, Hydro, Replace)
#   5. Lens plane generation
#
# Parameters:
#   - mass_min = 12.5 (10^12.5 Msun/h)
#   - mass_max = None (no upper limit)
#   - radius_factor = 5.0 (5×R200)
#   - 21 snapshots: 99 + 20 ray-tracing snapshots
#
# Usage:
#   sbatch batch/run_pipeline_625_full.sh

set -e
echo "=========================================="
echo "FULL PIPELINE - L205n625TNG, 21 snapshots"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes:  $SLURM_JOB_NODELIST"
echo "Start:  $(date)"
echo ""

# Environment
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Configuration
SIM_RES=625
MASS_MIN=12.5
RADIUS_FACTOR=5.0
NTASKS=4
SEED=2020

# All 21 snapshots: 99 + 20 ray-tracing snapshots
SNAPSHOTS=(99 29 31 33 35 38 41 43 46 49 52 56 59 63 67 71 76 80 85 90 96)

echo "Configuration:"
echo "  Resolution:    L205n${SIM_RES}TNG"
echo "  Mass min:      10^${MASS_MIN} Msun/h"
echo "  Mass max:      None"
echo "  Radius factor: ${RADIUS_FACTOR}×R200"
echo "  MPI tasks:     ${NTASKS}"
echo "  Snapshots:     ${SNAPSHOTS[@]}"
echo ""

OUTPUT_DIR=/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG
LENSPLANES_DIR=/mnt/home/mlee1/ceph/hydro_replace_lensplanes/L205n${SIM_RES}TNG

# ==========================================
# Step 0: Generate halo matches for all snapshots
# ==========================================
echo "=========================================="
echo "STEP 0: Generate halo matches"
echo "=========================================="

for SNAP in "${SNAPSHOTS[@]}"; do
    MATCH_FILE=${OUTPUT_DIR}/matches/matches_snap$(printf "%03d" ${SNAP}).npz
    if [ -f "${MATCH_FILE}" ]; then
        echo "  Snapshot ${SNAP}: matches exist, skipping"
    else
        echo "  Snapshot ${SNAP}: generating matches..."
        time python scripts/generate_matches_fast.py \
            --snap ${SNAP} \
            --resolution ${SIM_RES}
    fi
done

echo ""

# ==========================================
# Step 1: Generate particle cache for all snapshots
# ==========================================
echo "=========================================="
echo "STEP 1: Generate particle cache"
echo "=========================================="

for SNAP in "${SNAPSHOTS[@]}"; do
    CACHE_FILE=${OUTPUT_DIR}/particle_cache/cache_snap$(printf "%03d" ${SNAP}).h5
    if [ -f "${CACHE_FILE}" ]; then
        echo "  Snapshot ${SNAP}: cache exists, skipping"
    else
        echo "  Snapshot ${SNAP}: generating cache..."
        time mpirun -np ${NTASKS} python scripts/generate_particle_cache.py \
            --sim-res ${SIM_RES} \
            --snap ${SNAP}
    fi
done

echo ""

# ==========================================
# Step 2: Compute halo statistics for all snapshots
# ==========================================
echo "=========================================="
echo "STEP 2: Compute halo statistics"
echo "=========================================="

for SNAP in "${SNAPSHOTS[@]}"; do
    STATS_FILE=${OUTPUT_DIR}/analysis/halo_statistics_snap$(printf "%03d" ${SNAP}).h5
    if [ -f "${STATS_FILE}" ]; then
        echo "  Snapshot ${SNAP}: statistics exist, skipping"
    else
        echo "  Snapshot ${SNAP}: computing statistics..."
        time mpirun -np ${NTASKS} python scripts/compute_halo_statistics_distributed.py \
            --sim-res ${SIM_RES} \
            --snap ${SNAP} \
            --mass-min ${MASS_MIN}
    fi
done

echo ""

# ==========================================
# Step 3: Generate profiles for all snapshots
# ==========================================
echo "=========================================="
echo "STEP 3: Generate profiles"
echo "=========================================="

for SNAP in "${SNAPSHOTS[@]}"; do
    PROFILE_FILE=${OUTPUT_DIR}/profiles/profiles_snap$(printf "%03d" ${SNAP}).h5
    if [ -f "${PROFILE_FILE}" ]; then
        echo "  Snapshot ${SNAP}: profiles exist, skipping"
    else
        echo "  Snapshot ${SNAP}: generating profiles..."
        time mpirun -np ${NTASKS} python scripts/generate_profiles_cached_new.py \
            --sim-res ${SIM_RES} \
            --snap ${SNAP} \
            --mass-min ${MASS_MIN}
    fi
done

echo ""

# ==========================================
# Step 4: Generate 2D maps for all snapshots
# ==========================================
echo "=========================================="
echo "STEP 4: Generate 2D maps"
echo "=========================================="

for SNAP in "${SNAPSHOTS[@]}"; do
    MAP_FILE=${OUTPUT_DIR}/snap$(printf "%03d" ${SNAP})/projected/replace_M${MASS_MIN/./p}.npz
    if [ -f "${MAP_FILE}" ]; then
        echo "  Snapshot ${SNAP}: maps exist, skipping"
    else
        echo "  Snapshot ${SNAP}: generating maps..."
        time mpirun -np ${NTASKS} python scripts/generate_maps_cached.py \
            --sim-res ${SIM_RES} \
            --snap ${SNAP} \
            --mass-min ${MASS_MIN}
    fi
done

echo ""

# ==========================================
# Step 5: Generate lens planes
# ==========================================
echo "=========================================="
echo "STEP 5: Generate lens planes"
echo "=========================================="

# Convert snapshots array to comma-separated string
SNAP_LIST=$(IFS=,; echo "${SNAPSHOTS[*]}")

time mpirun -np ${NTASKS} python scripts/generate_lensplanes_replace.py \
    --sim-res ${SIM_RES} \
    --snap ${SNAP_LIST} \
    --mass-min ${MASS_MIN} \
    --radius-factor ${RADIUS_FACTOR} \
    --seed ${SEED}

echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "End: $(date)"

echo ""
echo "Output files:"

echo "  Match files:"
ls -lh ${OUTPUT_DIR}/matches/matches_snap*.npz 2>/dev/null | wc -l | xargs -I {} echo "    {} match files"

echo "  Cache files:"
ls -lh ${OUTPUT_DIR}/particle_cache/cache_snap*.h5 2>/dev/null | wc -l | xargs -I {} echo "    {} cache files"

echo "  Statistics files:"
ls -lh ${OUTPUT_DIR}/analysis/halo_statistics_snap*.h5 2>/dev/null | wc -l | xargs -I {} echo "    {} statistics files"

echo "  Profile files:"
ls -lh ${OUTPUT_DIR}/profiles/profiles_snap*.h5 2>/dev/null | wc -l | xargs -I {} echo "    {} profile files"

echo "  Map files:"
ls ${OUTPUT_DIR}/snap*/projected/*.npz 2>/dev/null | wc -l | xargs -I {} echo "    {} map files"

echo "  Lens planes:"
ls ${LENSPLANES_DIR}/seed${SEED}/replace_Mgt${MASS_MIN}/density*.dat 2>/dev/null | wc -l | xargs -I {} echo "    {} lens plane files"

echo ""
echo "Done!"
