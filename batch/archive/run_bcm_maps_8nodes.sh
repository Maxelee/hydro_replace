#!/bin/bash
#SBATCH --job-name=gen_bcm_maps
#SBATCH --output=logs/bcm_maps_%A_%a.o
#SBATCH --error=logs/bcm_maps_%A_%a.e
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --partition=cca
#SBATCH --array=0-15

# BCM map generation for L205n2500TNG - 20 snapshots × 3 BCM models
# Using 8 nodes (instead of 4) to double memory per rank
# Each array task processes ~1 snapshot with all 3 BCM models

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# 16 snapshots for ray-tracing (excluding 49, 52, 76, 96 which may already exist)
SNAPSHOTS=(29 31 33 35 38 41 43 46 56 59 63 67 71 80 85 90)

# Each array task handles 1 snapshot with 3 BCM models sequentially
SNAP_IDX=$SLURM_ARRAY_TASK_ID
SNAP=${SNAPSHOTS[$SNAP_IDX]}

# BCM models to run
BCM_MODELS=("Arico20" "Schneider19" "Schneider25")

echo "=========================================="
echo "BCM Map Generation: L205n2500TNG snap $SNAP"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Nodes: $SLURM_NNODES (8 nodes = more memory)"
echo "Tasks: $SLURM_NTASKS"
echo "Started: $(date)"
echo "=========================================="

for BCM in "${BCM_MODELS[@]}"; do
    echo ""
    echo ">>> Running BCM model: $BCM"
    echo ""
    
    srun python scripts/generate_all.py \
        --sim-res 2500 \
        --snap $SNAP \
        --grid-res 1024 \
        --mass-min 13.0 \
        --skip-existing \
        --skip-replace \
        --skip-dmo-hydro \
        --bcm-models $BCM
done

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
