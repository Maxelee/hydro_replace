#!/bin/bash
#SBATCH -J unified_test
#SBATCH -o logs/unified_test_%j.o
#SBATCH -e logs/unified_test_%j.e
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH -p cca

# Test the unified pipeline on L205n1250TNG snap99
# This should be faster than cache + separate scripts

set -e

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

SIM_RES=1250
SNAP=99
MASS_MIN=12.5
NTASKS=32

echo "=========================================="
echo "UNIFIED PIPELINE TEST"
echo "=========================================="
echo "Simulation: L205n${SIM_RES}TNG"
echo "Snapshot:   ${SNAP}"
echo "Mass min:   10^${MASS_MIN} Msun/h"
echo "MPI tasks:  ${NTASKS}"
echo "Start:      $(date)"
echo "=========================================="
echo ""

# Backup existing outputs (if any)
OUTPUT_DIR=/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG
SNAP_DIR=${OUTPUT_DIR}/snap$(printf "%03d" ${SNAP})/projected

if [ -f "${SNAP_DIR}/dmo.npz" ]; then
    echo "Backing up existing outputs..."
    mkdir -p ${SNAP_DIR}/backup_$(date +%Y%m%d_%H%M%S)
    cp ${SNAP_DIR}/*.npz ${SNAP_DIR}/backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
fi

echo ""
echo "Running unified pipeline..."
echo ""

time mpirun -np ${NTASKS} python scripts/generate_all_unified.py \
    --sim-res ${SIM_RES} \
    --snap ${SNAP} \
    --mass-min ${MASS_MIN} \
    --radius-mult 5.0

echo ""
echo "=========================================="
echo "TEST COMPLETE"
echo "=========================================="
echo "End: $(date)"
echo ""
echo "Output files:"
ls -lh ${OUTPUT_DIR}/profiles/profiles_snap$(printf "%03d" ${SNAP}).h5 2>/dev/null || echo "  (no profiles)"
ls -lh ${OUTPUT_DIR}/analysis/halo_statistics_snap$(printf "%03d" ${SNAP}).h5 2>/dev/null || echo "  (no stats)"
ls -lh ${SNAP_DIR}/*.npz 2>/dev/null | head -5 || echo "  (no maps)"
