#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J halo_replace
#SBATCH -n 64
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH -o logs/03_halo_replacement.o%j
#SBATCH -e logs/03_halo_replacement.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH -t 1-00:00:00

# Stage 3: Halo Replacement
# - Replace DMO particles with hydro particles in halo regions
# - Multiple mass bins and radius configurations
# - MPI parallel
#
# Expected runtime: ~12-24 hours
# Memory: Very high (full snapshot manipulation)

set -e

echo "=============================================="
echo "Stage 3: Halo Replacement"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "=============================================="

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
cd /mnt/home/mlee1/hydro_replace2

mkdir -p /mnt/home/mlee1/ceph/hydro_replace/replaced_snapshots
mkdir -p logs

# Run for different radius multipliers
for RADIUS in 1 3 5; do
    echo "Running replacement with radius=${RADIUS}×R_200c..."
    srun -n $SLURM_NTASKS python3 -u scripts/03_halo_replacement.py \
        --config config/simulation_paths.yaml \
        --radius-mult $RADIUS \
        --log-level INFO
done

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
