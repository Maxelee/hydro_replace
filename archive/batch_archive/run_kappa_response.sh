#!/bin/bash
#SBATCH --job-name=kappa_response
#SBATCH --output=logs/kappa_response_%j.o
#SBATCH --error=logs/kappa_response_%j.e
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --time=4:00:00
#SBATCH --partition=cca

# =============================================================================
# Convergence (κ) Baryonic Response Analysis with Fixed Kappa RMS
# =============================================================================
# This script runs the MPI-parallelized driver to compute weak lensing 
# statistics (C_ell, peaks, minima) and response fractions for all models.
#
# Key improvements in this run:
# - Uses constant KAPPA_RMS_DMO = 0.0107 for SNR normalization (from DMO ensemble)
# - Peak SNR bins: -2 to 6 (16 bins) - captures baryonic suppression
# - Minima SNR bins: -6 to 2 (16 bins) - captures void statistics
# - Forces recomputation to overwrite old cache that used per-map sigma
#
# Output locations:
#   - statistics_cache/    : HDF5 files with C_ell, peaks, minima per model
#   - response_analysis_output/ : Response fraction results and plots
# =============================================================================

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2/notebooks

echo "========================================"
echo "Convergence (κ) Baryonic Response Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Date: $(date)"
echo "========================================"

# Back up old cache (optional - comment out if not needed)
if [ -d "statistics_cache" ]; then
    BACKUP_DIR="statistics_cache_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up old cache to $BACKUP_DIR"
    mv statistics_cache "$BACKUP_DIR"
fi

if [ -d "response_analysis_output" ]; then
    BACKUP_DIR="response_analysis_output_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up old output to $BACKUP_DIR"
    mv response_analysis_output "$BACKUP_DIR"
fi

# Create fresh directories
mkdir -p statistics_cache
mkdir -p response_analysis_output

# Run with force recompute to regenerate all statistics
echo ""
echo "Running driver_baryonic_response_mpi.py with --force-recompute"
echo "Using KAPPA_RMS_DMO = 0.0107 for consistent SNR normalization"
echo ""

srun python3 -u driver_baryonic_response_mpi.py \
    --stage both \
    --force-recompute \
    --z 23

echo ""
echo "========================================"
echo "Done!"
echo "End time: $(date)"
echo "========================================"
