#!/bin/bash
#SBATCH --job-name=lp_test
#SBATCH --output=logs/lp_test_%j.o
#SBATCH --error=logs/lp_test_%j.e
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=30:00
#SBATCH --partition=cca

# Single snapshot lens plane test (snap 99, Replace model only)

set -e
module purge
module load modules/2.3-20240529
module load python openmpi/4.1.6 hdf5 fftw gsl boost/mpi-1.84.0
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
cd /mnt/home/mlee1/hydro_replace2

SNAP=99

echo "=== LENS PLANE TEST: snap ${SNAP}, Replace model ==="
echo "Nodes: 4, Tasks: 64"
date

# Check if cache exists
CACHE="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n2500TNG/particle_cache/cache_snap0${SNAP}.h5"
if [ -f "$CACHE" ]; then
    echo "✓ Cache found: $CACHE"
    ls -lh "$CACHE"
else
    echo "✗ Cache NOT found - run test_cache_single.sh first"
    exit 1
fi

echo ""
echo "Running lens plane generation..."
time mpirun -np 64 python scripts/generate_lensplanes.py \
    --sim-res 2500 \
    --model replace \
    --snap ${SNAP} \
    --mass-min 12.5 \
    --seed 2020 \
    --num-seeds 1

echo ""
echo "=== Checking output ==="
ls -lh /mnt/home/mlee1/ceph/hydro_replace_lensplanes/L205n2500TNG/seed2020/replace_Mgt12.5/ 2>/dev/null || echo "No output found"

echo "=== DONE ==="
date
