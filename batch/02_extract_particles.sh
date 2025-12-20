#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J extract_particles
#SBATCH -n 64
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH -o logs/02_extract_particles.o%j
#SBATCH -e logs/02_extract_particles.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH -t 12:00:00

# Stage 2: Particle Extraction
# - Extract particles within 5R_vir for all matched halos
# - Store positions, velocities, masses, IDs
# - MPI parallel across halos
#
# Expected runtime: ~6-12 hours
# Memory: High (loading particle data)

set -e

echo "=============================================="
echo "Stage 2: Particle Extraction"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "=============================================="

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
cd /mnt/home/mlee1/hydro_replace2

mkdir -p /mnt/home/mlee1/ceph/hydro_replace/extracted_halos
mkdir -p logs

srun -n $SLURM_NTASKS python3 -u scripts/02_extract_particles.py \
    --config config/simulation_paths.yaml \
    --radius-mult 5.0 \
    --log-level INFO

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
