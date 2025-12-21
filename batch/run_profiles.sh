#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J profiles
#SBATCH -n 64
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH -o logs/profiles.o%j
#SBATCH -e logs/profiles.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH -t 06:00:00

# Usage:
#   SNAP=99 SIM_RES=2500 sbatch batch/run_profiles.sh
#   SNAP=99 SIM_RES=2500 MASS_MIN=13.0 sbatch batch/run_profiles.sh

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

# Defaults
SNAP=${SNAP:-99}
SIM_RES=${SIM_RES:-2500}
MASS_MIN=${MASS_MIN:-12.5}
MASS_MAX=${MASS_MAX:-}
SKIP_BCM=${SKIP_BCM:-}
BCM_MODELS=${BCM_MODELS:-"Arico20 Schneider19 Schneider25"}

echo "========================================"
echo "Generating Radial Profiles"
echo "========================================"
echo "Snapshot: ${SNAP}"
echo "Resolution: ${SIM_RES}"
echo "Mass range: ${MASS_MIN} - ${MASS_MAX:-inf}"
echo "BCM models: ${BCM_MODELS}"
echo "========================================"

CMD="srun -n64 python3 -u scripts/generate_profiles.py \
    --snap ${SNAP} \
    --sim-res ${SIM_RES} \
    --mass-min ${MASS_MIN}"

if [ -n "${MASS_MAX}" ]; then
    CMD="${CMD} --mass-max ${MASS_MAX}"
fi

if [ -n "${SKIP_BCM}" ]; then
    CMD="${CMD} --skip-bcm"
fi

if [ -n "${BCM_MODELS}" ]; then
    CMD="${CMD} --bcm-models ${BCM_MODELS}"
fi

echo "Running: ${CMD}"
echo ""

${CMD}
