#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J profiles_v2
#SBATCH -n 64
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /mnt/home/mlee1/hydro_replace2/logs/profiles_v2_%A_%a.o
#SBATCH -e /mnt/home/mlee1/hydro_replace2/logs/profiles_v2_%A_%a.e
#SBATCH -t 4:00:00
#SBATCH --array=0-20

# ==============================================================================
# Production profile generation with memory-optimized v2 script
#
# Array job: one task per snapshot (21 snapshots total)
# Each task generates profiles for DMO, Hydro, Replace, and BCM models
#
# Output: profiles_Mgt{mass_min}.h5 in each snapshot directory
# ==============================================================================

# Snapshot list (20 ray-tracing snapshots + snap099)
SNAPSHOTS=(29 31 33 35 38 41 43 46 49 52 56 59 63 67 71 76 80 85 90 96 99)

SNAP=${SNAPSHOTS[$SLURM_ARRAY_TASK_ID]}
SIM_RES=${SIM_RES:-2500}
MODE=${MODE:-both}

# Setup environment
module purge
module load python openmpi python-mpi
module load hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

echo "=============================================="
echo "Profile Generation v2 (Memory Optimized)"
echo "=============================================="
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Snapshot: $SNAP"
echo "Simulation: L205n${SIM_RES}TNG"
echo "Mode: $MODE"
echo "MPI ranks: $SLURM_NTASKS"
echo "=============================================="
echo ""

# Run profile generation for this snapshot
mpirun -np $SLURM_NTASKS python -u scripts/generate_profiles_mpi_v2.py \
    --sim-res ${SIM_RES} \
    --snapshot ${SNAP} \
    --mode ${MODE}

echo ""
echo "=============================================="
echo "Snapshot $SNAP complete!"
echo "=============================================="
