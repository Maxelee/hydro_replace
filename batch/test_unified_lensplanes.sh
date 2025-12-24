#!/bin/bash
#SBATCH --job-name=unified_LP_test
#SBATCH --output=logs/unified_LP_test.o%j
#SBATCH --error=logs/unified_LP_test.e%j
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --partition=cca

# Test unified pipeline with lensplane generation on L205n625TNG
# This tests the full pipeline including Phase 4-5 lensplane generation

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

echo "========================================"
echo "Testing Unified Pipeline with Lensplanes"
echo "Resolution: 625"
echo "Snapshot: 96 (index 0 in lightcone)"
echo "Output: L205n625TNG"
echo "========================================"

# Run with lensplanes enabled on snapshot 96 (first in SNAPSHOT_ORDER)
# Using smaller lensplane grid (1024) for faster testing
srun python3 -u scripts/generate_all_unified.py \
    --snap 96 \
    --sim-res 625 \
    --mass-min 12.5 \
    --radius-mult 5.0 \
    --grid 625 \
    --enable-lensplanes \
    --lensplane-grid 1024

echo "Done!"
echo ""
echo "Expected output structure:"
echo "  /mnt/home/mlee1/ceph/hydro_replace_LP/L205n625TNG/dmo/LP_00/lenspot00.dat"
echo "  /mnt/home/mlee1/ceph/hydro_replace_LP/L205n625TNG/dmo/LP_00/lenspot01.dat"
echo "  /mnt/home/mlee1/ceph/hydro_replace_LP/L205n625TNG/hydro/LP_00/lenspot00.dat"
echo "  ..."