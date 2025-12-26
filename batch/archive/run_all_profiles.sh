#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J gen_profiles
#SBATCH -n 128
#SBATCH -N 8
#SBATCH --exclusive
#SBATCH -o /mnt/home/mlee1/hydro_replace2/logs/profiles_%A_%a.o
#SBATCH -e /mnt/home/mlee1/hydro_replace2/logs/profiles_%A_%a.e
#SBATCH -t 12:00:00
#SBATCH --array=0-20

# ==============================================================================
# Generate 3D radial density profiles for all snapshots
# 
# Uses the FIXED generate_profiles.py that correctly loads ALL particle files
# for each halo (not just one file per rank).
#
# For each snapshot, generates profiles for:
#   - DMO halos
#   - Hydro halos (matched)
#   - Replace halos (for 4 mass thresholds)
#   - BCM halos (3 models × 4 mass thresholds)
#
# Uses array job to parallelize across 21 snapshots
# ==============================================================================

# All snapshots (20 ray-tracing + snap 99)
SNAPSHOTS=(29 31 33 35 38 41 43 46 49 52 56 59 63 67 71 76 80 85 90 96 99)
SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

SIM_RES=${SIM_RES:-2500}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/hydro_replace_fields}

# Mass threshold for profile generation (high mass halos only for speed)
MASS_MIN=${MASS_MIN:-13.0}

# Setup environment
module load modules/2.4-20250724
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2/scripts

echo "=============================================="
echo "Generating profiles for snapshot $SNAP"
echo "Resolution: $SIM_RES"
echo "Mass threshold: log10(M) > $MASS_MIN"
echo "=============================================="

# Check if matches exist
MATCHES_FILE="${OUTPUT_DIR}/L205n${SIM_RES}TNG/matches/matches_snap$(printf '%03d' ${SNAP}).npz"
if [ ! -f "$MATCHES_FILE" ]; then
    echo "ERROR: Matches file not found: $MATCHES_FILE"
    echo "Run generate_matches_fast.py first!"
    exit 1
fi

# Run profile generation
mpirun -np $SLURM_NTASKS python -u generate_profiles.py \
    --snap $SNAP \
    --sim-res $SIM_RES \
    --mass-min $MASS_MIN \
    --output-dir $OUTPUT_DIR

echo ""
echo "=============================================="
echo "Completed profiles for snapshot $SNAP"
echo "=============================================="
