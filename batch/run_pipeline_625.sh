#!/bin/bash
#SBATCH -J full_pipeline
#SBATCH -o logs/full_pipeline_%j.o
#SBATCH -e logs/full_pipeline_%j.e
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 8:00:00
#SBATCH -p cca

# Full end-to-end pipeline for hydro replace analysis
#
# This runs:
#   1. Particle cache generation
#   2. Halo statistics computation
#   3. Density profile generation
#   4. 2D density map generation
#   5. Lens plane generation
#   6. Lux ray-tracing (convergence maps)
#
# Usage:
#   # Default: M > 10^12.5, 5×R200, snap 99
#   sbatch batch/run_pipeline_625.sh
#
#   # Custom mass range and radius
#   MASS_MIN=13.0 MASS_MAX=14.0 RADIUS_FACTOR=3.0 sbatch batch/run_pipeline_625.sh
#
#   # All ray-tracing snapshots
#   SNAP=rt sbatch batch/run_pipeline_625.sh
#
#   # Skip early steps (just lens planes + raytracing)
#   LENSPLANES_ONLY=1 SNAP=rt sbatch batch/run_pipeline_625.sh

set -e

echo "=========================================="
echo "FULL PIPELINE - L205n625TNG"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes:  $SLURM_JOB_NODELIST"
echo "Start:  $(date)"
echo ""

# Environment
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Configuration (can override with environment variables)
SIM_RES=${SIM_RES:-625}
SNAP=${SNAP:-99}
MASS_MIN=${MASS_MIN:-12.5}
MASS_MAX=${MASS_MAX:-""}
RADIUS_FACTOR=${RADIUS_FACTOR:-5.0}
SEED=${SEED:-2020}
LP_GRID=${LP_GRID:-4096}
RT_GRID=${RT_GRID:-1024}

# Skip flags
SKIP_CACHE=${SKIP_CACHE:-0}
SKIP_STATS=${SKIP_STATS:-0}
SKIP_PROFILES=${SKIP_PROFILES:-0}
SKIP_MAPS=${SKIP_MAPS:-0}
SKIP_LENSPLANES=${SKIP_LENSPLANES:-0}
SKIP_RAYTRACING=${SKIP_RAYTRACING:-0}
LENSPLANES_ONLY=${LENSPLANES_ONLY:-0}
RAYTRACING_ONLY=${RAYTRACING_ONLY:-0}

echo "Configuration:"
echo "  Resolution:    L205n${SIM_RES}TNG"
echo "  Snapshot:      ${SNAP}"
echo "  Mass min:      10^${MASS_MIN} Msun/h"
echo "  Mass max:      10^${MASS_MAX:-∞} Msun/h"
echo "  Radius factor: ${RADIUS_FACTOR}×R200"
echo "  Seed:          ${SEED}"
echo "  LP grid:       ${LP_GRID}"
echo "  RT grid:       ${RT_GRID}"
echo ""

# Build command
CMD="python scripts/run_pipeline.py \
    --sim-res ${SIM_RES} \
    --snap ${SNAP} \
    --mass-min ${MASS_MIN} \
    --radius-factor ${RADIUS_FACTOR} \
    --seed ${SEED} \
    --lp-grid ${LP_GRID} \
    --rt-grid ${RT_GRID}"

# Add mass-max if specified
if [ -n "${MASS_MAX}" ]; then
    CMD="${CMD} --mass-max ${MASS_MAX}"
fi

# Add skip flags
[ "${SKIP_CACHE}" == "1" ] && CMD="${CMD} --skip-cache"
[ "${SKIP_STATS}" == "1" ] && CMD="${CMD} --skip-stats"
[ "${SKIP_PROFILES}" == "1" ] && CMD="${CMD} --skip-profiles"
[ "${SKIP_MAPS}" == "1" ] && CMD="${CMD} --skip-maps"
[ "${SKIP_LENSPLANES}" == "1" ] && CMD="${CMD} --skip-lensplanes"
[ "${SKIP_RAYTRACING}" == "1" ] && CMD="${CMD} --skip-raytracing"
[ "${LENSPLANES_ONLY}" == "1" ] && CMD="${CMD} --lensplanes-only"
[ "${RAYTRACING_ONLY}" == "1" ] && CMD="${CMD} --raytracing-only"

echo "Command: ${CMD}"
echo ""

# Run
time ${CMD}

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "End: $(date)"
