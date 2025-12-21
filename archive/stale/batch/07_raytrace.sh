#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J raytrace_lux
#SBATCH -n 32
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -o logs/07_raytrace.o%j
#SBATCH -e logs/07_raytrace.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH -t 2-00:00:00

# Stage 7: Ray-Tracing with lux
# - Generate lens planes from density fields
# - Compute convergence maps
#
# Expected runtime: ~1-2 days
# Uses lux C++ code

set -e

echo "=============================================="
echo "Stage 7: Ray-Tracing (lux)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "=============================================="

module load openmpi hdf5 fftw gsl boost

# lux executable and config
LUX_DIR=/mnt/home/mlee1/lux
LUX_EXE=$LUX_DIR/lux
LUX_CONFIG=$LUX_DIR/lux.ini

# Output directories
mkdir -p /mnt/home/mlee1/ceph/lux_out/LP_output
mkdir -p /mnt/home/mlee1/ceph/lux_out/RT_output
mkdir -p logs

cd $LUX_DIR

echo "Running lux ray-tracing..."
srun -n $SLURM_NTASKS $LUX_EXE $LUX_CONFIG

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
