#!/bin/bash
#SBATCH --job-name=mass_frac
#SBATCH --output=logs/mass_fractions_%j.o
#SBATCH --error=logs/mass_fractions_%j.e
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH --partition=cca

# =============================================================================
# Compute exact mass filling fractions for Replace configurations
# =============================================================================
#
# This script computes the actual DMO mass excised for each (mass_bin, radius_shell)
# configuration at snapshot 96 (z=0.04).
#
# Usage:
#   sbatch run_mass_fractions.sh
#
# Output:
#   /mnt/home/mlee1/ceph/hydro_replace_fields/L205n2500TNG/mass_fractions_snap096.npz
#
# =============================================================================

set -e

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

echo "========================================"
echo "Mass Filling Fraction Computation"
echo "========================================"
echo "Started: $(date)"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "========================================"

srun python3 -u scripts/compute_mass_fractions.py \
    --snap 96 \
    --sim-res 2500 \
    --mass-min 12.0

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
