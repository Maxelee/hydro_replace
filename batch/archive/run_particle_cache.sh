#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J particle_cache
#SBATCH -n 128
#SBATCH -N 8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /mnt/home/mlee1/hydro_replace2/logs/particle_cache_%A_%a.o
#SBATCH -e /mnt/home/mlee1/hydro_replace2/logs/particle_cache_%A_%a.e
#SBATCH -t 6:00:00
#SBATCH --array=0-20

# ==============================================================================
# Generate particle ID lookup tables for halos
#
# This creates a cache that dramatically speeds up:
# - Profile generation
# - Lens plane generation (Replace + BCM)
# - Any operation requiring particles around halos
#
# Array job: one task per snapshot
# ==============================================================================

# Snapshot list
SNAPSHOTS=(29 31 33 35 38 41 43 46 49 52 56 59 63 67 71 76 80 85 90 96 99)

SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}
SIM_RES=${SIM_RES:-2500}

# Setup environment
module purge
module load python openmpi python-mpi
module load hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

echo "=============================================="
echo "Particle Cache Generation"
echo "=============================================="
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Snapshot: $SNAP"
echo "Simulation: L205n${SIM_RES}TNG"
echo "MPI ranks: $SLURM_NTASKS"
echo "=============================================="
echo ""

# Run cache generation
mpirun -np $SLURM_NTASKS python -u scripts/generate_particle_cache.py \
    --sim-res ${SIM_RES} \
    --snap ${SNAP}

echo ""
echo "=============================================="
echo "Snapshot $SNAP complete!"
echo "=============================================="
