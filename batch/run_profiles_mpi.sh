#!/bin/bash
#SBATCH --job-name=profiles_mpi
#SBATCH --output=logs/profiles_%A_%a.o
#SBATCH --error=logs/profiles_%A_%a.e
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --partition=gen
#SBATCH --array=0-2
#SBATCH --dependency=afterok:2363284

# Profile generation for all resolutions
# Depends on BCM jobs completing first

module purge
module load python openmpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Map array index to simulation resolution
RESOLUTIONS=(625 1250 2500)
RES=${RESOLUTIONS[$SLURM_ARRAY_TASK_ID]}

# Adjust resources based on resolution
if [ "$RES" == "2500" ]; then
    # L205n2500 needs more time and resources
    SNAPSHOT=96
else
    SNAPSHOT=99
fi

echo "=========================================="
echo "Profile Generation: L205n${RES}TNG"
echo "Snapshot: ${SNAPSHOT}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Started: $(date)"
echo "=========================================="

srun python scripts/generate_profiles_mpi.py \
    --sim-res $RES \
    --snapshot $SNAPSHOT \
    --mode both

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
