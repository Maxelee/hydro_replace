#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J gen_bcm_maps
#SBATCH -n 32
#SBATCH -N 4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -o /mnt/home/mlee1/hydro_replace2/logs/bcm_maps_%A_%a.o
#SBATCH -e /mnt/home/mlee1/hydro_replace2/logs/bcm_maps_%A_%a.e
#SBATCH -t 12:00:00
#SBATCH --array=0-16

# ==============================================================================
# Generate BCM maps for snapshots that are missing them
# 
# Uses FEWER MPI ranks (32 instead of 64) to give more memory per process
# for BaryonForge lookup table generation.
#
# Snapshots needing BCM: 029,031,033,035,038,041,043,046,052,056,063,067,071,080,085,090,096
# ==============================================================================

# Snapshots missing BCM maps
SNAPSHOTS=(29 31 33 35 38 41 43 46 52 56 63 67 71 80 85 90 96)
SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

SIM_RES=${SIM_RES:-2500}
GRID_RES=${GRID_RES:-1024}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/hydro_replace_fields}

# BCM models to run
BCM_MODELS="Arico20 Schneider19 Schneider25"

# Mass thresholds
MASS_THRESHOLDS="12.5 13.0 13.5 14.0"

# Setup environment
module purge
module load python openmpi python-mpi
module load hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2/scripts

echo "=============================================="
echo "Generating BCM maps for snapshot $SNAP"
echo "Resolution: $SIM_RES, Grid: $GRID_RES"
echo "Using $SLURM_NTASKS MPI ranks for better memory"
echo "=============================================="

# Check if matches exist
MATCHES_FILE="${OUTPUT_DIR}/L205n${SIM_RES}TNG/matches/matches_snap$(printf '%03d' ${SNAP}).npz"
if [ ! -f "$MATCHES_FILE" ]; then
    echo "ERROR: Matches file not found: $MATCHES_FILE"
    exit 1
fi

# Create output directory
SNAP_DIR="${OUTPUT_DIR}/L205n${SIM_RES}TNG/snap$(printf '%03d' ${SNAP})/projected"
mkdir -p "$SNAP_DIR"

# ============================================================================
# Generate BCM for each mass threshold - run sequentially for memory
# ============================================================================
for MASS_MIN in $MASS_THRESHOLDS; do
    echo ""
    echo ">>> Mass threshold M > 10^${MASS_MIN} Msun/h"
    
    # Generate BCM maps - one model at a time to reduce memory
    for BCM_MODEL in $BCM_MODELS; do
        echo "    Generating BCM-${BCM_MODEL} maps..."
        mpirun -np $SLURM_NTASKS python -u generate_all.py \
            --snap $SNAP \
            --sim-res $SIM_RES \
            --grid-res $GRID_RES \
            --mass-min $MASS_MIN \
            --bcm-model $BCM_MODEL \
            --skip-existing \
            --skip-dmo-hydro \
            --skip-replace \
            --output-dir $OUTPUT_DIR
        
        # Brief pause between models
        sleep 5
    done
done

echo ""
echo "=============================================="
echo "Completed BCM maps for snapshot $SNAP"
echo "=============================================="
