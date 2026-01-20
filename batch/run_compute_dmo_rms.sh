#!/bin/bash
#SBATCH --job-name=dmo_rms
#SBATCH --output=logs/dmo_rms_%j.out
#SBATCH --error=logs/dmo_rms_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --partition=cca
#SBATCH --time=04:00:00
#SBATCH --mem=64GB

# One-time job to compute DMO RMS values
# Run this BEFORE running compute_stats jobs

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Create logs directory
mkdir -p logs

# Run the script
echo "Computing DMO RMS values..."
echo "Start time: $(date)"
echo ""

mpirun -np 64 python scripts/compute_dmo_rms.py

echo ""
echo "End time: $(date)"
echo "Job complete!"
