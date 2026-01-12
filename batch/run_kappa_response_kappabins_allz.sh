#!/bin/bash
#SBATCH --job-name=kappa_allz
#SBATCH --output=logs/kappa_allz_%A_%a.o
#SBATCH --error=logs/kappa_allz_%A_%a.e
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --time=4:00:00
#SBATCH --partition=cca
#SBATCH --array=0-40

# =============================================================================
# Convergence (κ) Baryonic Response Analysis - ALL REDSHIFTS
# =============================================================================
# Array job to process all 41 redshift slices (z=0 to z=40)
# Each array task processes one redshift slice
#
# Statistics computed at each redshift:
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

# Get redshift index from array task ID
Z_INDEX=$SLURM_ARRAY_TASK_ID

echo "========================================"
echo "Convergence (κ) Baryonic Response Analysis"
echo "REDSHIFT INDEX: z=${Z_INDEX}"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Date: $(date)"
echo "========================================"
echo ""

# Create directories
mkdir -p statistics_cache_kappabins
mkdir -p response_analysis_output_kappabins

# Run for this specific redshift
# Only force recompute if cache doesn't exist for this z
echo "Running driver_baryonic_response_mpi_kappabins.py for z=${Z_INDEX}"
echo ""

srun python3 -u driver_baryonic_response_mpi_kappabins.py \
    --stage both \
    --z ${Z_INDEX}

echo ""
echo "========================================"
echo "Done with z=${Z_INDEX}!"
echo "End time: $(date)"
echo "========================================"
