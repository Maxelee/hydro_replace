#!/bin/bash
#SBATCH -J test_direct
#SBATCH -o logs/test_direct_%j.o
#SBATCH -e logs/test_direct_%j.e
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH -p cca

# Test script: Compare direct map generation vs cache-based approach
#
# This tests the direct spatial query method on snap99 of L205n1250TNG
# to compare performance against the cache-based generate_maps_cached_lowmem.py
#
# Usage:
#   sbatch batch/test_maps_direct.sh

set -e

# Environment
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Configuration
SIM_RES=1250
SNAP=99
MASS_MIN=12.5
RADIUS_MULT=5.0
NTASKS=32

echo "=========================================="
echo "DIRECT MAP GENERATION TEST"
echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Nodes:      $SLURM_JOB_NODELIST"
echo "Start:      $(date)"
echo ""
echo "Configuration:"
echo "  Resolution:    L205n${SIM_RES}TNG"
echo "  Snapshot:      ${SNAP}"
echo "  Mass min:      10^${MASS_MIN} Msun/h"
echo "  Radius mult:   ${RADIUS_MULT}×R200"
echo "  MPI tasks:     ${NTASKS}"
echo ""

# Check that matches file exists
MATCH_FILE=/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/matches/matches_snap$(printf "%03d" ${SNAP}).npz
if [ ! -f "${MATCH_FILE}" ]; then
    echo "ERROR: Matches file not found: ${MATCH_FILE}"
    echo "Please run generate_matches_fast.py first"
    exit 1
fi
echo "Matches file: ${MATCH_FILE}"
echo ""

# Run direct map generation
echo "=========================================="
echo "Running direct map generation..."
echo "=========================================="
time mpirun -np ${NTASKS} python scripts/generate_maps_direct.py \
    --sim-res ${SIM_RES} \
    --snap ${SNAP} \
    --mass-min ${MASS_MIN} \
    --radius-mult ${RADIUS_MULT}

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End: $(date)"
echo ""

# Show output files
OUTPUT_DIR=/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/snap$(printf "%03d" ${SNAP})/projected_direct
echo "Output files:"
ls -lh ${OUTPUT_DIR}/*.npz 2>/dev/null || echo "  (no output files found)"
echo ""

# Compare with cached version if it exists
CACHED_DIR=/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/snap$(printf "%03d" ${SNAP})/projected
if [ -d "${CACHED_DIR}" ]; then
    echo "Cached version output (for comparison):"
    ls -lh ${CACHED_DIR}/*.npz 2>/dev/null | head -5
fi

echo ""
echo "Done!"
