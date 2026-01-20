#!/bin/bash
#SBATCH --job-name=kappa_response_kappabins
#SBATCH --output=logs/kappa_response_kappabins_%j.o
#SBATCH --error=logs/kappa_response_kappabins_%j.e
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --time=4:00:00
#SBATCH --partition=cca

# =============================================================================
# Convergence (κ) Baryonic Response Analysis with KAPPA BINS
# =============================================================================
# This script runs the MPI-parallelized driver using KAPPA bins instead of SNR.
#
# Statistics computed:
# - C_ell: Angular power spectrum
# - Peaks: Peak counts binned by kappa (-0.05 to 0.2, 25 bins)
# - Minima: Minima counts binned by kappa (-0.07 to 0.07, 28 bins)
# - PDF: Probability distribution function (-0.1 to 0.25, 70 bins)
# - V0, V1, V2: Minkowski functionals (46 thresholds from -0.08 to 0.15)
#
# Output locations:
#   - statistics_cache_kappabins/    : HDF5 files with all statistics per model
#   - response_analysis_output_kappabins/ : Response fraction results
# =============================================================================

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2/notebooks

echo "========================================"
echo "Convergence (κ) Baryonic Response Analysis"
echo "Using KAPPA BINS (not SNR)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Date: $(date)"
echo "========================================"
echo ""
echo "Bin configuration:"
echo "  Peaks: kappa from -0.05 to 0.2 (25 bins)"
echo "  Minima: kappa from -0.07 to 0.07 (28 bins)"
echo "  PDF: kappa from -0.1 to 0.25 (70 bins)"
echo "  Minkowski functionals: 46 thresholds from -0.08 to 0.15"
echo ""

# Create directories (the script also does this but just in case)
mkdir -p statistics_cache_kappabins
mkdir -p response_analysis_output_kappabins

# Run with force recompute to regenerate all statistics
echo "Running driver_baryonic_response_mpi_kappabins.py with --force-recompute"
echo ""

srun python3 -u driver_baryonic_response_mpi_kappabins.py \
    --stage both \
    --force-recompute \
    --z 23

echo ""
echo "========================================"
echo "Done!"
echo "End time: $(date)"
echo "========================================"
