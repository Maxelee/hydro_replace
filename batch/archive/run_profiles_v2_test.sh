#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J profiles_v2_test
#SBATCH -n 64
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /mnt/home/mlee1/hydro_replace2/logs/profiles_v2_test_%j.o
#SBATCH -e /mnt/home/mlee1/hydro_replace2/logs/profiles_v2_test_%j.e
#SBATCH -t 2:00:00

# ==============================================================================
# Test new memory-optimized profile generation script (v2)
#
# This tests generate_profiles_mpi_v2.py which uses per-halo KDTree queries
# instead of batch queries to avoid memory explosion.
#
# Test with single snapshot (snap099) before running all snapshots.
# ==============================================================================

SIM_RES=${SIM_RES:-2500}
TEST_SNAP=${TEST_SNAP:-99}
MODE=${MODE:-both}

# Setup environment
module purge
module load python openmpi python-mpi
module load hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

echo "=============================================="
echo "Testing Profile Generation v2 (Memory Optimized)"
echo "=============================================="
echo "Simulation: L205n${SIM_RES}TNG"
echo "Test snapshot: $TEST_SNAP"
echo "Mode: $MODE"
echo "MPI ranks: $SLURM_NTASKS"
echo "Nodes: $SLURM_NNODES"
echo "=============================================="
echo ""

# Run the test
mpirun -np $SLURM_NTASKS python -u scripts/generate_profiles_mpi_v2.py \
    --sim-res ${SIM_RES} \
    --snapshot ${TEST_SNAP} \
    --mode ${MODE}

echo ""
echo "=============================================="
echo "Test complete!"
echo "=============================================="
