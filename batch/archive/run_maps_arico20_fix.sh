#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J fix_arico20
#SBATCH -n 64
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH -o /mnt/home/mlee1/hydro_replace2/logs/fix_arico20_%A_%a.o
#SBATCH -e /mnt/home/mlee1/hydro_replace2/logs/fix_arico20_%A_%a.e
#SBATCH -t 04:00:00
#SBATCH --array=0-20

# ==============================================================================
# Regenerate only Arico20 BCM maps with corrected parameters
#
# This script fixes the Stage 2 maps that were generated with wrong parameters.
# Only regenerates bcm_Arico20_Mgt*.npz files, leaves other maps intact.
#
# Snapshots: 21 total (29, 31, 33, ..., 96, 99)
# ==============================================================================

# Snapshot list (same as run_maps_completed.sh)
SNAPSHOTS=(29 31 33 35 38 41 43 46 49 52 56 59 63 67 71 76 80 85 90 96 99)
SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

SIM_RES=${SIM_RES:-2500}
GRID_RES=${GRID_RES:-4096}

# Setup environment
module purge
module load python openmpi python-mpi
module load hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

echo "=============================================="
echo "Regenerating Arico20 BCM maps (fixed params)"
echo "Snapshot: $SNAP"
echo "Resolution: $SIM_RES, Grid: $GRID_RES"
echo "=============================================="

# First, remove the old Arico20 BCM files for this snapshot
OUTPUT_DIR="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/snap${SNAP}/projected"
echo "Removing old Arico20 files from $OUTPUT_DIR..."
rm -f ${OUTPUT_DIR}/bcm_Arico20_Mgt*.npz

# Run with only-bcm and only Arico20 model
mpirun -np $SLURM_NTASKS python -u scripts/generate_all.py \
    --sim-res ${SIM_RES} \
    --snap ${SNAP} \
    --grid-res ${GRID_RES} \
    --only-bcm \
    --bcm-models Arico20

echo ""
echo "=============================================="
echo "Completed Arico20 BCM maps for snapshot $SNAP"
echo "=============================================="
