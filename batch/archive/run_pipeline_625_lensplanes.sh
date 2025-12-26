#!/bin/bash
#SBATCH -J pipe_625_lp
#SBATCH -o logs/pipe_625_lensplanes_%j.o
#SBATCH -e logs/pipe_625_lensplanes_%j.e
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH -p cca

# Lens plane generation for L205n625TNG - runs after array job completes
#
# This generates lens planes for all 21 snapshots using the replace method.
#
# Usage:
#   sbatch --dependency=afterok:<ARRAY_JOBID> batch/run_pipeline_625_lensplanes.sh

set -e

# Environment
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Configuration
SIM_RES=625
MASS_MIN=12.5
RADIUS_FACTOR=5.0
NTASKS=4
SEED=2020

# All 21 snapshots
SNAPSHOTS="99,29,31,33,35,38,41,43,46,49,52,56,59,63,67,71,76,80,85,90,96"

echo "=========================================="
echo "LENS PLANES - L205n${SIM_RES}TNG"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes:  $SLURM_JOB_NODELIST"
echo "Start:  $(date)"
echo ""

echo "Configuration:"
echo "  Resolution:    L205n${SIM_RES}TNG"
echo "  Mass min:      10^${MASS_MIN} Msun/h"
echo "  Radius factor: ${RADIUS_FACTOR}×R200"
echo "  Seed:          ${SEED}"
echo "  MPI tasks:     ${NTASKS}"
echo "  Snapshots:     ${SNAPSHOTS}"
echo ""

OUTPUT_DIR=/mnt/home/mlee1/ceph/hydro_replace_fields/L205n${SIM_RES}TNG
LENSPLANES_DIR=/mnt/home/mlee1/ceph/hydro_replace_lensplanes/L205n${SIM_RES}TNG

# ==========================================
# Step 5: Generate lens planes
# ==========================================
echo "=========================================="
echo "STEP 5: Generate lens planes"
echo "=========================================="

time mpirun -np ${NTASKS} python scripts/generate_lensplanes_replace.py \
    --sim-res ${SIM_RES} \
    --snap ${SNAPSHOTS} \
    --mass-min ${MASS_MIN} \
    --radius-factor ${RADIUS_FACTOR} \
    --seed ${SEED}

echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "LENS PLANES COMPLETE"
echo "=========================================="
echo "End: $(date)"

echo ""
echo "Output files:"
echo "  Lens planes:"
ls ${LENSPLANES_DIR}/seed${SEED}/replace_Mgt${MASS_MIN}/density*.dat 2>/dev/null | wc -l | xargs -I {} echo "    {} lens plane files"

echo ""
echo "Full pipeline summary:"
echo "  Match files:"
ls ${OUTPUT_DIR}/matches/matches_snap*.npz 2>/dev/null | wc -l | xargs -I {} echo "    {} match files"
echo "  Cache files:"
ls ${OUTPUT_DIR}/particle_cache/cache_snap*.h5 2>/dev/null | wc -l | xargs -I {} echo "    {} cache files"
echo "  Statistics files:"
ls ${OUTPUT_DIR}/analysis/halo_statistics_snap*.h5 2>/dev/null | wc -l | xargs -I {} echo "    {} statistics files"
echo "  Profile files:"
ls ${OUTPUT_DIR}/profiles/profiles_snap*.h5 2>/dev/null | wc -l | xargs -I {} echo "    {} profile files"
echo "  Map files:"
ls ${OUTPUT_DIR}/snap*/projected/*.npz 2>/dev/null | wc -l | xargs -I {} echo "    {} map files"

echo ""
echo "Done!"
