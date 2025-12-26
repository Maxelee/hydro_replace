#!/bin/bash
#SBATCH --job-name=BCM_2500
#SBATCH --output=logs/BCM_2500_%A_%a.o
#SBATCH --error=logs/BCM_2500_%A_%a.e
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --partition=cca
#SBATCH --array=0-19

# Array job for unified pipeline with lensplanes on L205n2500TNG
# Each task processes one of the 20 snapshots in the lightcone
#
# SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
# Array index 0 -> snap 96 (z=0.02)
# Array index 19 -> snap 29 (z=2.32)

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Map array index to snapshot number
SNAPSHOTS=(96 90 85 80 76 71 67 63 59 56 52 49 46 43 41 38 35 33 31 29)
SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Unified Pipeline with Lensplanes"
echo "Resolution: 2500"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Snapshot: $SNAP"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "========================================"

# Run unified pipeline with lensplane generation
# - Grid: 2500 for 2D maps (matches particle resolution)
# - Lensplane grid: 4096 for high-resolution ray-tracing
srun -n 32 python3 -u /mnt/home/mlee1/hydro_replace2/scripts/generate_all_unified_bcm.py --snap $SNAP --sim-res 2500 --models='schneider19,schneider25,arico20' --enable-lensplanes --lensplane-grid=4096 --grid=1024

echo ""
echo "Done with snapshot $SNAP (array task $SLURM_ARRAY_TASK_ID)"
echo "========================================"
