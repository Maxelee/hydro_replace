#!/bin/bash
#SBATCH --job-name=halo_stats
#SBATCH --constraint=icelake
#SBATCH --output=logs/halo_stats_%j.o
#SBATCH --error=logs/halo_stats_%j.e
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH -p cca

# =============================================================================
# Compute Halo Statistics (Baryon Fractions & Mass Conservation)
# =============================================================================
# This script computes baryon fractions and mass conservation for all matched
# halos in the particle cache.
#
# MEMORY NOTE: Each rank loads the full snapshot (~20-50 GB for 2500 res).
# We use few ranks with exclusive node access to maximize memory.
#
# Usage:
#   sbatch run_halo_statistics.sh              # Default: snap 99, res 2500
#   sbatch run_halo_statistics.sh 99 2500      # Explicit snap and resolution
#   MASS_MIN=13.0 sbatch run_halo_statistics.sh  # Custom mass threshold
# =============================================================================

set -e

# Parse arguments
SNAP=${1:-99}
SIM_RES=${2:-2500}
MASS_MIN=${MASS_MIN:-12.0}

echo "=============================================================="
echo "HALO STATISTICS COMPUTATION"
echo "=============================================================="
echo "Snapshot: ${SNAP}"
echo "Resolution: L205n${SIM_RES}TNG"
echo "Mass minimum: 10^${MASS_MIN} Msun/h"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks: ${SLURM_NTASKS}"
echo "=============================================================="

# Environment setup
module purge
module load python openmpi python-mpi hdf5

# Activate virtual environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Change to project directory
cd /mnt/home/mlee1/hydro_replace2

# Check cache file exists
CACHE_FILE="/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG/particle_cache/cache_snap$(printf '%03d' ${SNAP}).h5"
if [ ! -f "${CACHE_FILE}" ]; then
    echo "ERROR: Cache file not found: ${CACHE_FILE}"
    echo "Run generate_particle_cache.py first."
    exit 1
fi
echo "Cache file: ${CACHE_FILE}"

# Create logs directory
mkdir -p logs

# Run the computation
echo ""
echo "Starting MPI job with ${SLURM_NTASKS} ranks..."
echo ""

srun -n 32 python -u scripts/compute_halo_statistics.py \
    --snap ${SNAP} \
    --sim-res ${SIM_RES} \
    --mass-min ${MASS_MIN}

echo ""
echo "=============================================================="
echo "Job complete!"
echo "=============================================================="
