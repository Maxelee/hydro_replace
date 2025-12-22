#!/bin/bash
#SBATCH --job-name=prof_test
#SBATCH --output=logs/prof_test_%j.o
#SBATCH --error=logs/prof_test_%j.e
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=30:00
#SBATCH --partition=cca

# Single snapshot profile test (snap 99)

set -e
module purge
module load modules/2.3-20240529
module load python openmpi/4.1.6 hdf5 fftw gsl boost/mpi-1.84.0
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
cd /mnt/home/mlee1/hydro_replace2

SNAP=99

echo "=== PROFILE TEST: snap ${SNAP} ==="
echo "Nodes: 2, Tasks: 32"
date

time mpirun -np 32 python scripts/generate_profiles_mpi_v2.py \
    --sim-res 2500 \
    --snapshot ${SNAP}

echo ""
echo "=== Checking output ==="
ls -lh /mnt/home/mlee1/ceph/hydro_replace_fields/L205n2500TNG/profiles/ 2>/dev/null | tail -5

echo "=== DONE ==="
date
