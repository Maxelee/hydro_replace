#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J hydro_replace
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -o /mnt/home/mlee1/hydro_replace2/logs/generate_all.o%j
#SBATCH -e /mnt/home/mlee1/hydro_replace2/logs/generate_all.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH -t 12:00:00

# ==============================================================================
# Generate All Fields Pipeline
#
# Generates for a single snapshot:
#   - 2D projected maps (z-axis): DMO, Hydro, Replace, BCM (3 models)
#   - Density profiles for all matched halos
#
# Output structure:
#   /mnt/home/mlee1/ceph/hydro_replace_fields/
#   └── L205n{RES}TNG/
#       ├── matches/
#       │   └── matches_snap{NN}.npz
#       └── snap{NN}/
#           ├── projected/
#           │   ├── dmo.npz
#           │   ├── hydro.npz
#           │   ├── replace.npz
#           │   ├── bcm_arico20.npz
#           │   ├── bcm_schneider19.npz
#           │   └── bcm_schneider25.npz
#           └── profiles.h5
#
# Environment variables:
#   SNAP        - Snapshot number (default: 99)
#   SIM_RES     - Simulation resolution: 625, 1250, 2500 (default: 625)
#   GRID_RES    - Grid resolution for maps (default: 1024)
#   MASS_MIN    - log10(M_min) for halos (default: 12.5)
#   MASS_MAX    - log10(M_max) for halos (optional, for mass bins)
#   SKIP_EXISTING - Skip if output files exist (default: false)
#   ONLY_BCM    - Only generate BCM maps (default: false)
#   BCM_MODELS  - BCM models to run (default: "Arico20 Schneider19 Schneider25")
# ==============================================================================

set -e

# Configuration
SNAP=${SNAP:-99}
SIM_RES=${SIM_RES:-625}
GRID_RES=${GRID_RES:-1024}
MASS_MIN=${MASS_MIN:-12.5}
MASS_MAX=${MASS_MAX:-}
SKIP_EXISTING=${SKIP_EXISTING:-false}
ONLY_BCM=${ONLY_BCM:-false}
BCM_MODELS=${BCM_MODELS:-"Arico20 Schneider19 Schneider25"}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/home/mlee1/ceph/hydro_replace_fields}

# Setup environment
module load modules/2.4-20250724
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Change to scripts directory
cd /mnt/home/mlee1/hydro_replace2/scripts

echo "============================================================"
echo "Hydro Replace Pipeline"
echo "============================================================"
echo "Snapshot:       ${SNAP}"
echo "Resolution:     ${SIM_RES}"
echo "Grid:           ${GRID_RES}"
echo "Mass min:       log10(M) > ${MASS_MIN}"
echo "Mass max:       ${MASS_MAX:-none}"
echo "Skip existing:  ${SKIP_EXISTING}"
echo "Only BCM:       ${ONLY_BCM}"
echo "BCM models:     ${BCM_MODELS}"
echo "Output:         ${OUTPUT_DIR}"
echo "============================================================"

# Create output directories
mkdir -p "${OUTPUT_DIR}/L205n${SIM_RES}TNG/matches"
mkdir -p /mnt/home/mlee1/hydro_replace2/logs

# Step 1: Generate matches (if not exists) - uses fast hash-based method
MATCHES_FILE="${OUTPUT_DIR}/L205n${SIM_RES}TNG/matches/matches_snap$(printf '%03d' ${SNAP}).npz"
if [ ! -f "${MATCHES_FILE}" ]; then
    echo ""
    echo ">>> Step 1: Generating halo matches (fast hash-based)..."
    python -u generate_matches_fast.py \
        --snap ${SNAP} \
        --resolution ${SIM_RES} \
        --output-dir ${OUTPUT_DIR}
else
    echo ""
    echo ">>> Step 1: Matches already exist, skipping..."
fi

# Build command line arguments
EXTRA_ARGS=""
if [ -n "${MASS_MAX}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --mass-max ${MASS_MAX}"
fi
if [ "${SKIP_EXISTING}" = "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-existing"
fi
if [ "${ONLY_BCM}" = "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --only-bcm"
fi
if [ -n "${BCM_MODELS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --bcm-models ${BCM_MODELS}"
fi

# Step 2: Generate all fields and profiles
echo ""
echo ">>> Step 2: Generating 2D maps and profiles..."
echo "Extra args: ${EXTRA_ARGS}"
PYTHON_EXE="/mnt/home/mlee1/venvs/hydro_replace/bin/python"
srun --export=ALL -n ${SLURM_NTASKS} ${PYTHON_EXE} -u generate_all.py \
    --snap ${SNAP} \
    --sim-res ${SIM_RES} \
    --grid-res ${GRID_RES} \
    --mass-min ${MASS_MIN} \
    --output-dir ${OUTPUT_DIR} \
    ${EXTRA_ARGS}

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
