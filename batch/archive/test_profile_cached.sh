#!/bin/bash
#SBATCH --job-name=prof_test
#SBATCH --output=logs/profile_test_%j.o
#SBATCH --error=logs/profile_test_%j.e
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=30:00
#SBATCH --partition=cca

# Single snapshot test for profiles using particle cache

set -e
module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
cd /mnt/home/mlee1/hydro_replace2

SNAPSHOT=99
SIM_RES=2500

echo "Testing profile generation with cache for snapshot ${SNAPSHOT}"
echo "Nodes: 4, Tasks: 32, Memory: 8G/task"
echo ""

# Check cache exists
CACHE_FILE="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/particle_cache/cache_snap0${SNAPSHOT}.h5"
if [ ! -f "$CACHE_FILE" ]; then
    echo "ERROR: Particle cache not found!"
    echo "  Expected: $CACHE_FILE"
    echo "  Run test_cache_single.sh first."
    exit 1
fi

echo "✓ Particle cache found: $CACHE_FILE"
ls -lh "$CACHE_FILE"
echo ""

mpirun -np 32 python scripts/generate_profiles_cached.py \
    --sim-res ${SIM_RES} \
    --snapshot ${SNAPSHOT} \
    --mode both

echo ""
echo "Checking output..."
ls -lh /mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/profiles_cached_snap0${SNAPSHOT}.h5 2>/dev/null || echo "Output file not found"
