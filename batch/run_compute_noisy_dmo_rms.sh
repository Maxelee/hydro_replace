#!/bin/bash
#SBATCH --job-name=dmo_rms_noisy
#SBATCH --output=logs/dmo_rms_noisy_%a_%j.out
#SBATCH --error=logs/dmo_rms_noisy_%a_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --partition=cca
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --array=0-5

# Compute DMO RMS for all 6 survey × smoothing configurations:
#   0: LSST, 1.0 arcmin
#   1: LSST, 2.0 arcmin
#   2: LSST, 3.0 arcmin
#   3: DES,  1.0 arcmin
#   4: DES,  2.0 arcmin
#   5: DES,  3.0 arcmin

set -e

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

mkdir -p logs

# Survey and smoothing configurations
SURVEYS=("LSST" "LSST" "LSST" "DES" "DES" "DES")
SMOOTHINGS=("1.0" "2.0" "3.0" "1.0" "2.0" "3.0")

SURVEY=${SURVEYS[$SLURM_ARRAY_TASK_ID]}
SMOOTHING=${SMOOTHINGS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Survey: $SURVEY"
echo "Smoothing theta_G: $SMOOTHING arcmin"
echo "Start time: $(date)"
echo "========================================"
echo ""

mpirun -np 64 python scripts/compute_dmo_rms.py --survey "$SURVEY" --smoothing-scale "$SMOOTHING"

echo ""
echo "========================================"
echo "End time: $(date)"
echo "Job complete!"
echo "========================================"
