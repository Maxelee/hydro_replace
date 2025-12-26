#!/bin/bash
#
# Run full lens plane generation for all models and mass cuts
#
# This script submits all necessary jobs to generate lens planes for:
#   - DMO, Hydro, Replace (3 mass cuts), BCM (3 models)
#
# Usage:
#   ./batch/run_all_lensplanes.sh 625    # Test with low-res
#   ./batch/run_all_lensplanes.sh 2500   # Production with high-res
#

SIM_RES=${1:-625}

echo "=============================================="
echo "Submitting lens plane jobs for L205n${SIM_RES}TNG"
echo "=============================================="

cd /mnt/home/mlee1/hydro_replace2
mkdir -p logs

# Job settings based on resolution
if [ "$SIM_RES" == "625" ]; then
    NODES=2
    TASKS=16
    TIME="02:00:00"
elif [ "$SIM_RES" == "1250" ]; then
    NODES=4
    TASKS=32
    TIME="04:00:00"
else
    NODES=8
    TASKS=64
    TIME="08:00:00"
fi

echo "Settings: ${NODES} nodes, ${TASKS} tasks, ${TIME} time limit"
echo ""

# Submit DMO
echo "Submitting DMO..."
SIM_RES=$SIM_RES MODEL=dmo SNAP=all sbatch -N $NODES -n $TASKS -t $TIME batch/run_lensplanes.sh
sleep 1

# Submit Hydro
echo "Submitting Hydro..."
SIM_RES=$SIM_RES MODEL=hydro SNAP=all sbatch -N $NODES -n $TASKS -t $TIME batch/run_lensplanes.sh
sleep 1

# Submit Replace with different mass cuts
for MASS_MIN in 12.0 12.5 13.0; do
    echo "Submitting Replace (M > 10^${MASS_MIN})..."
    SIM_RES=$SIM_RES MODEL=replace MASS_MIN=$MASS_MIN SNAP=all sbatch -N $NODES -n $TASKS -t $TIME batch/run_lensplanes.sh
    sleep 1
done

# Submit BCM models (all three in one job, since they share particle loading)
echo "Submitting BCM models (all three)..."
SIM_RES=$SIM_RES MODEL=bcm SNAP=all sbatch -N $NODES -n $TASKS -t $TIME batch/run_lensplanes.sh

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Check status with: squeue -u \$USER"
