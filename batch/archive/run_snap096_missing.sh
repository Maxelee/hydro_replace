#!/bin/bash
#SBATCH --job-name=snap096_full
#SBATCH --output=logs/snap096_full_%j.o
#SBATCH --error=logs/snap096_full_%j.e
#SBATCH --partition=cca
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=0

# Generate matches and maps for snap096 (the only missing snapshot)

echo "=========================================="
echo "Snap096 Full Pipeline"
echo "Started: $(date)"
echo "=========================================="

module purge
module load modules/2.4-20250724
module load python openmpi hdf5 fftw gsl

source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
cd /mnt/home/mlee1/hydro_replace2

# Step 1: Generate matches for snap096 (single task - not MPI)
echo ""
echo ">>> Step 1: Generating matches for snap096"
python scripts/generate_matches_fast.py --snap 96 --resolution 2500

# Step 2: Generate DMO, Hydro, Replace, and BCM maps
echo ""
echo ">>> Step 2: Generating 2D maps for snap096"
srun python scripts/generate_all.py --sim-res 2500 --snap 96 --grid 1024 --mass-cut 13.0

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
