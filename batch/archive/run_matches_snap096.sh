#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J gen_match_96
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -o /mnt/home/mlee1/hydro_replace2/logs/matches_snap096_%j.o
#SBATCH -e /mnt/home/mlee1/hydro_replace2/logs/matches_snap096_%j.e
#SBATCH -t 04:00:00

# ==============================================================================
# Generate matches for snapshot 96 (missing)
# ==============================================================================

SNAP=96
SIM_RES=${SIM_RES:-2500}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/hydro_replace_fields}

# Setup environment
module purge
module load python openmpi python-mpi
module load hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2/scripts

echo "=============================================="
echo "Generating matches for snapshot $SNAP"
echo "Resolution: $SIM_RES"
echo "=============================================="

python -u generate_matches_fast.py \
    --snap $SNAP \
    --resolution $SIM_RES \
    --output-dir $OUTPUT_DIR

echo ""
echo "=============================================="
echo "Completed matches for snapshot $SNAP"
echo "=============================================="
