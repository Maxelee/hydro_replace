#!/bin/bash
#SBATCH --job-name=test_unified
#SBATCH --output=logs/test_unified.o%j
#SBATCH --error=logs/test_unified.e%j
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --partition=cca

# Test unified pipeline on snap99 with L205n625TNG

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

echo "========================================"
echo "Testing Unified Pipeline"
echo "Resolution: 625"
echo "Snapshot: 99"
echo "Output: L205n625TNG_unified_test"
echo "========================================"

srun python3 -u scripts/generate_all_unified.py \
    --snap 99 \
    --sim-res 625 \
    --mass-min 12.5 \
    --radius-mult 5.0 \
    --grid 1024 \
    --output-suffix "_unified_test"

echo "Done!"
