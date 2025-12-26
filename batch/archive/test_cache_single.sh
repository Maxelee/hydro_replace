#!/bin/bash
#SBATCH --job-name=cache_test
#SBATCH --output=logs/cache_test_%j.o
#SBATCH --error=logs/cache_test_%j.e
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --time=3:00:00
#SBATCH --partition=cca

# Single snapshot test for particle cache
# 8 nodes, 32 tasks = 4 tasks/node with 16G each = 64G/node

set -e
module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
cd /mnt/home/mlee1/hydro_replace2

echo "Testing particle cache generation for snapshot 99"
echo "Nodes: 8, Tasks: 32, Memory: 16G/task"
echo ""

srun -n32 python scripts/generate_particle_cache.py --sim-res 2500 --snap 99

echo ""
echo "Checking output..."
ls -lh /mnt/home/mlee1/ceph/hydro_replace_fields/L205n2500TNG/particle_cache/
