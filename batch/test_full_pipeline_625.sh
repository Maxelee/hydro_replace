#!/bin/bash
#SBATCH -J test_pipeline
#SBATCH -o logs/test_pipeline_%j.o
#SBATCH -e logs/test_pipeline_%j.e
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -p cca

# Test the full analysis pipeline on 625 resolution, single snapshot
#
# This tests:
#   1. Particle cache generation (M > 10^12.5)
#   2. Halo statistics computation
#   3. Profile generation  
#   4. 2D map generation (DMO, Hydro, Replace)
#
# Usage:
#   sbatch batch/test_full_pipeline_625.sh

set -e
echo "=========================================="
echo "FULL PIPELINE TEST - L205n625TNG, snap 99"
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
SNAP=99
MASS_MIN=12.5
NTASKS=4

echo "Configuration:"
echo "  Resolution: L205n${SIM_RES}TNG"
echo "  Snapshot:   ${SNAP}"
echo "  Mass min:   10^${MASS_MIN} Msun/h"
echo "  MPI tasks:  ${NTASKS}"
echo ""

OUTPUT_DIR=/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG

MATCH_FILE=${OUTPUT_DIR}/matches/matches_snap$(printf "%03d" ${SNAP}).npz
if [ -f "${MATCH_FILE}" ]; then
    echo "  Snapshot ${SNAP}: matches exist, skipping"
else
    echo "  Snapshot ${SNAP}: generating matches..."
    time python scripts/generate_matches_fast.py \
        --snap ${SNAP} \
        --resolution ${SIM_RES}
fi

# Step 1: Generate particle cache
echo "=========================================="
echo "STEP 1: Generate particle cache"
echo "=========================================="
time mpirun -np ${NTASKS} python scripts/generate_particle_cache.py \
    --sim-res ${SIM_RES} \
    --snap ${SNAP}

echo ""

# Step 2: Compute halo statistics
echo "=========================================="
echo "STEP 2: Compute halo statistics"
echo "=========================================="
time mpirun -np ${NTASKS} python scripts/compute_halo_statistics_distributed.py \
    --sim-res ${SIM_RES} \
    --snap ${SNAP}

echo ""

# Step 3: Generate profiles (if script exists)
echo "=========================================="
echo "STEP 3: Generate profiles"
echo "=========================================="
if [ -f "scripts/generate_profiles_cached_new.py" ]; then
    time mpirun -np ${NTASKS} python scripts/generate_profiles_cached_new.py \
        --sim-res ${SIM_RES} \
        --snap ${SNAP} \
        --mass-min ${MASS_MIN}
else
    echo "Profile script not found, skipping"
fi

echo ""

# Step 4: Generate maps
echo "=========================================="
echo "STEP 4: Generate 2D maps"
echo "=========================================="
time mpirun -np ${NTASKS} python scripts/generate_maps_cached.py \
    --sim-res ${SIM_RES} \
    --snap ${SNAP} \
    --mass-min ${MASS_MIN}

echo ""

# Summary
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "End: $(date)"

# List outputs
echo ""
echo "Output files:"
OUTPUT_DIR=/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG

echo "  Cache:"
ls -lh ${OUTPUT_DIR}/particle_cache/cache_snap*.h5 2>/dev/null || echo "    (none)"

echo "  Statistics:"
ls -lh ${OUTPUT_DIR}/analysis/halo_statistics_snap*.h5 2>/dev/null || echo "    (none)"

echo "  Maps:"
ls -lh ${OUTPUT_DIR}/fields_snap${SNAP:0:3}/*.npz 2>/dev/null | head -10 || echo "    (none)"

echo ""
echo "Done!"
