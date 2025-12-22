#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J gen_matches
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o /mnt/home/mlee1/hydro_replace2/logs/matches_%A_%a.o
#SBATCH -e /mnt/home/mlee1/hydro_replace2/logs/matches_%A_%a.e
#SBATCH -t 01:00:00
#SBATCH --array=0-16

# ==============================================================================
# Generate matches for all missing snapshots
# Uses array job to parallelize across snapshots
# ==============================================================================

# Snapshots to process (missing from current data)
SNAPSHOTS=(29 31 33 35 38 41 43 46 52 56 63 67 71 80 85 90 96)
SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

SIM_RES=${SIM_RES:-2500}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/hydro_replace_fields}

# Setup environment
module load modules/2.4-20250724
module load python hdf5
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

echo "Done with snapshot $SNAP"
