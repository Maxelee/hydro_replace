#!/bin/bash
#SBATCH -J bcm_625
#SBATCH -o logs/bcm_625_%A_%a.o
#SBATCH -e logs/bcm_625_%A_%a.e
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH -p cca
#SBATCH --array=0-20

# Array job for BCM pipeline - L205n625TNG
#
# This runs per snapshot:
#   1. BCM statistics (mass enclosed at various radii)
#   2. BCM profiles (stacked density profiles by mass bin)
#   3. BCM 2D maps (projected density)
#
# Prerequisites:
#   - Halo matches must exist (from Replace pipeline)
#   - DMO particle cache must exist (from Replace pipeline)
#
# Parameters:
#   - mass_min = 12.5 (10^12.5 Msun/h)
#   - mass_max = None (no upper limit)
#   - BCM models: Arico20, Schneider19, Schneider25
#
# Usage:
#   sbatch batch/run_bcm_625_array.sh
#
# After completion, run lens planes:
#   sbatch --dependency=afterok:<JOBID> batch/run_bcm_625_lensplanes.sh

set -e

# Environment
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Configuration
SIM_RES=625
MASS_MIN=12.5
NTASKS=4

# BCM models to run
BCM_MODELS=("Arico20" "Schneider19" "Schneider25")

# All 21 snapshots
SNAPSHOTS=(99 29 31 33 35 38 41 43 46 49 52 56 59 63 67 71 76 80 85 90 96)

# Get snapshot for this array task
SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "BCM PIPELINE - L205n${SIM_RES}TNG, snap ${SNAP}"
echo "=========================================="
echo "Job ID:     $SLURM_ARRAY_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Nodes:      $SLURM_JOB_NODELIST"
echo "Start:      $(date)"
echo ""

echo "Configuration:"
echo "  Resolution:  L205n${SIM_RES}TNG"
echo "  Snapshot:    ${SNAP}"
echo "  Mass min:    10^${MASS_MIN} Msun/h"
echo "  BCM Models:  ${BCM_MODELS[@]}"
echo "  MPI tasks:   ${NTASKS}"
echo ""

OUTPUT_DIR=/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG

# Check prerequisites
CACHE_FILE=${OUTPUT_DIR}/particle_cache/cache_snap$(printf "%03d" ${SNAP}).h5
if [ ! -f "${CACHE_FILE}" ]; then
    echo "ERROR: Cache file not found: ${CACHE_FILE}"
    echo "Run the Replace pipeline first to generate DMO particle cache."
    exit 1
fi

# Process each BCM model
for BCM_MODEL in "${BCM_MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing BCM Model: ${BCM_MODEL}"
    echo "=========================================="
    
    # ==========================================
    # Step 1: BCM Statistics
    # ==========================================
    echo ""
    echo "--- Step 1: BCM Statistics ---"
    
    STATS_FILE=${OUTPUT_DIR}/analysis/bcm_statistics_${BCM_MODEL}_snap$(printf "%03d" ${SNAP}).h5
    if [ -f "${STATS_FILE}" ]; then
        echo "  Statistics exist, skipping"
    else
        echo "  Computing BCM statistics..."
        time mpirun -np ${NTASKS} python scripts/compute_bcm_statistics.py \
            --sim-res ${SIM_RES} \
            --snap ${SNAP} \
            --bcm-model ${BCM_MODEL} \
            --mass-min ${MASS_MIN}
    fi
    
    # ==========================================
    # Step 2: BCM Profiles
    # ==========================================
    echo ""
    echo "--- Step 2: BCM Profiles ---"
    
    PROFILE_FILE=${OUTPUT_DIR}/profiles/profiles_bcm_${BCM_MODEL}_snap$(printf "%03d" ${SNAP}).h5
    if [ -f "${PROFILE_FILE}" ]; then
        echo "  Profiles exist, skipping"
    else
        echo "  Generating BCM profiles..."
        time mpirun -np ${NTASKS} python scripts/generate_profiles_bcm_cached.py \
            --sim-res ${SIM_RES} \
            --snap ${SNAP} \
            --bcm-model ${BCM_MODEL} \
            --mass-min ${MASS_MIN}
    fi
    
    # ==========================================
    # Step 3: BCM 2D Maps
    # ==========================================
    echo ""
    echo "--- Step 3: BCM 2D Maps ---"
    
    MAP_FILE=${OUTPUT_DIR}/snap$(printf "%03d" ${SNAP})/projected/bcm_${BCM_MODEL,,}_M${MASS_MIN/./p}.npz
    if [ -f "${MAP_FILE}" ]; then
        echo "  Maps exist, skipping"
    else
        echo "  Generating BCM maps..."
        time mpirun -np ${NTASKS} python scripts/generate_maps_bcm_cached.py \
            --sim-res ${SIM_RES} \
            --snap ${SNAP} \
            --bcm-model ${BCM_MODEL} \
            --mass-min ${MASS_MIN}
    fi
done

echo ""
echo "=========================================="
echo "SNAPSHOT ${SNAP} - ALL BCM MODELS COMPLETE"
echo "=========================================="
echo "End: $(date)"

echo ""
echo "Output files:"
for BCM_MODEL in "${BCM_MODELS[@]}"; do
    echo "  ${BCM_MODEL}:"
    ls -lh ${OUTPUT_DIR}/analysis/bcm_statistics_${BCM_MODEL}_snap$(printf "%03d" ${SNAP}).h5 2>/dev/null || echo "    (no stats)"
    ls -lh ${OUTPUT_DIR}/profiles/profiles_bcm_${BCM_MODEL}_snap$(printf "%03d" ${SNAP}).h5 2>/dev/null || echo "    (no profiles)"
done

echo ""
echo "Done!"
