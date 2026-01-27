#!/bin/bash
#SBATCH --job-name=assembly_bias
#SBATCH --partition=cca
#SBATCH --constraint=rome
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/home/mlee1/ceph/logs/assembly_bias_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/assembly_bias_%j.err

# Assembly Bias Detection Pipeline
# 
# This script runs the assembly bias analysis using halo replacement.
# It generates 2D projected maps and lensplanes for early-forming (high c)
# and late-forming (low c) halo populations.
#
# Usage:
#   sbatch run_assembly_bias.sh                    # Default: snap 99, mass 10^13
#   sbatch run_assembly_bias.sh --snap 96          # Specific snapshot
#   sbatch run_assembly_bias.sh --mass-center 12.5 # Different mass bin

set -e

# Load modules
module load python openmpi python-mpi hdf5

# Activate environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Go to project directory
cd /mnt/home/mlee1/hydro_replace2/

# Create log directory if needed
mkdir -p /mnt/home/mlee1/ceph/logs

# Run the assembly bias pipeline
# Default: snapshot 99, mass 10^13 ± 0.2 dex, 3×R200 radius, median split
srun python scripts/generate_assembly_bias.py \
    --snap ${SNAP:-99} \
    --sim-res ${SIM_RES:-2500} \
    --mass-center ${MASS_CENTER:-13.0} \
    --mass-width ${MASS_WIDTH:-0.2} \
    --radius-mult ${RADIUS_MULT:-3.0} \
    --grid ${GRID:-4096} \
    --split-method ${SPLIT_METHOD:-median} \
    ${EXTRA_ARGS:-}

echo "Assembly bias pipeline complete!"
