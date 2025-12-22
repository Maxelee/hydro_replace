#!/bin/bash
#SBATCH --job-name=profiles_bcm
#SBATCH --output=logs/profiles_bcm_%A_%a.o
#SBATCH --error=logs/profiles_bcm_%A_%a.e
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --partition=cca
#SBATCH --array=0-2

# BCM Profile generation for all resolutions
# Depends on BCM maps completing first (uses same halo matches)
# Uses 32 ranks (not 64) due to BaryonForge memory requirements

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Map array index to simulation resolution
RESOLUTIONS=(625 1250 2500)
RES=${RESOLUTIONS[$SLURM_ARRAY_TASK_ID]}

# Adjust resources based on resolution
SNAPSHOT=99

echo "=========================================="
echo "BCM Profile Generation: L205n${RES}TNG"
echo "Snapshot: ${SNAPSHOT}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks: ${SLURM_NTASKS}"
echo "Started: $(date)"
echo "=========================================="

# Run with all 3 BCM models
srun python scripts/generate_profiles_bcm.py \
    --sim-res $RES \
    --snapshot $SNAPSHOT \
    --bcm-models Arico20 Schneider19 Schneider25

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
