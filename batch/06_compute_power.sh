#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J compute_power
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -o logs/06_compute_power.o%j
#SBATCH -e logs/06_compute_power.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH -t 06:00:00

# Stage 6: Power Spectrum Computation
# - Compute P(k) from pixelized 3D density fields
# - Compare DMO, Hydro, and Replaced
#
# Expected runtime: ~2-6 hours
# Memory: ~64-128 GB for 4096^3 grids

set -e

echo "=============================================="
echo "Stage 6: Power Spectrum Computation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
cd /mnt/home/mlee1/hydro_replace2

mkdir -p /mnt/home/mlee1/ceph/hydro_replace/power_spectra
mkdir -p logs

srun -n $SLURM_NTASKS python3 -u scripts/06_compute_power.py \
    --config config/simulation_paths.yaml \
    --log-level INFO

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
