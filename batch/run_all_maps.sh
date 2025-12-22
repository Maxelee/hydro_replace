#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J gen_maps
#SBATCH -n 64
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH -o /mnt/home/mlee1/hydro_replace2/logs/maps_%A_%a.o
#SBATCH -e /mnt/home/mlee1/hydro_replace2/logs/maps_%A_%a.e
#SBATCH -t 08:00:00
#SBATCH --array=0-20

# ==============================================================================
# Generate 2D projected maps for all snapshots
# 
# For each snapshot, generates:
#   - DMO map (once)
#   - Hydro map (once)
#   - Replace maps (4 mass thresholds: 12.5, 13.0, 13.5, 14.0)
#   - BCM maps (3 models × 4 mass thresholds)
#
# Uses array job to parallelize across 21 snapshots
# ==============================================================================

# All snapshots (20 ray-tracing + snap 99)
SNAPSHOTS=(29 31 33 35 38 41 43 46 49 52 56 59 63 67 71 76 80 85 90 96 99)
SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

SIM_RES=${SIM_RES:-2500}
GRID_RES=${GRID_RES:-1024}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/hydro_replace_fields}

# BCM models to run
BCM_MODELS="Arico20 Schneider19 Schneider25"

# Mass thresholds
MASS_THRESHOLDS="12.5 13.0 13.5 14.0"

# Setup environment
module load modules/2.4-20250724
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2/scripts

echo "=============================================="
echo "Generating 2D maps for snapshot $SNAP"
echo "Resolution: $SIM_RES, Grid: $GRID_RES"
echo "=============================================="

# Check if matches exist
MATCHES_FILE="${OUTPUT_DIR}/L205n${SIM_RES}TNG/matches/matches_snap$(printf '%03d' ${SNAP}).npz"
if [ ! -f "$MATCHES_FILE" ]; then
    echo "ERROR: Matches file not found: $MATCHES_FILE"
    echo "Run generate_matches_fast.py first!"
    exit 1
fi

# Create output directory
SNAP_DIR="${OUTPUT_DIR}/L205n${SIM_RES}TNG/snap$(printf '%03d' ${SNAP})/projected"
mkdir -p "$SNAP_DIR"

# ============================================================================
# STEP 1: Generate DMO and Hydro maps (no mass threshold, run once)
# ============================================================================
echo ""
echo ">>> STEP 1: Generating DMO and Hydro maps..."

mpirun -np $SLURM_NTASKS python -u generate_all.py \
    --snap $SNAP \
    --sim-res $SIM_RES \
    --grid-res $GRID_RES \
    --mass-min 12.5 \
    --skip-existing \
    --skip-bcm \
    --skip-replace \
    --output-dir $OUTPUT_DIR

# ============================================================================
# STEP 2: Generate Replace and BCM for each mass threshold
# ============================================================================
for MASS_MIN in $MASS_THRESHOLDS; do
    echo ""
    echo ">>> STEP 2: Generating Replace and BCM with M > 10^${MASS_MIN}..."
    
    mpirun -np $SLURM_NTASKS python -u generate_all.py \
        --snap $SNAP \
        --sim-res $SIM_RES \
        --grid-res $GRID_RES \
        --mass-min $MASS_MIN \
        --skip-existing \
        --skip-dmo-hydro \
        --bcm-models $BCM_MODELS \
        --output-dir $OUTPUT_DIR
done

echo ""
echo "=============================================="
echo "Completed snapshot $SNAP"
echo "=============================================="
