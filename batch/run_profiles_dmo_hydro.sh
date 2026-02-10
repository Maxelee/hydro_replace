#!/bin/bash
#SBATCH --job-name=profiles_individual
#SBATCH --output=logs/profiles_individual_%j.o
#SBATCH --error=logs/profiles_individual_%j.e
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=2:00:00
#SBATCH --partition=cca

# =============================================================================
# Generate individual halo profiles for DMO and Hydro at snapshot 96
# =============================================================================
#
# This script computes density profiles for each halo individually.
# Output: profiles_individual_snap096.h5 with:
#   - individual_dmo_density: (n_halos, 30) DMO density profiles
#   - individual_dmo_mass: (n_halos, 30) DMO mass per radial bin
#   - individual_hydro_density: (n_halos, 30) Hydro total density
#   - individual_hydro_density_dm/gas/stars: Component profiles
#   - halo_log_masses, halo_radii, halo_positions: Halo properties
#
# Usage:
#   sbatch run_profiles_dmo_hydro.sh
#
# =============================================================================

set -e

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

SNAP=96  # z ~ 0.04

echo "========================================"
echo "Individual Profile Generation (DMO + Hydro)"
echo "========================================"
echo "Snapshot: $SNAP"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "========================================"

# Run the individual profile generation script
mpirun -np $SLURM_NTASKS python scripts/generate_individual_profiles.py \
    --snap $SNAP \
    --sim-res 2500

echo "========================================"
echo "Complete: $(date)"
echo "========================================"
echo ""
echo "Profile file saved to:"
echo "  /mnt/home/mlee1/ceph/hydro_replace_fields/L205n2500TNG/profiles/profiles_individual_snap${SNAP}.h5"
