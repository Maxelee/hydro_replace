#!/bin/bash
#SBATCH --job-name=snap099_rerun
#SBATCH --output=/mnt/home/mlee1/hydro_replace2/logs/snap099_rerun_%j.o
#SBATCH --error=/mnt/home/mlee1/hydro_replace2/logs/snap099_rerun_%j.e
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --constraint=rome

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2/scripts

SIM_RES=625
SNAP=99
MASS_MIN=12.5
RADIUS_FACTOR=5.0

echo "=============================================="
echo "RERUN SNAP 99 PIPELINE"
echo "=============================================="
echo "Resolution: L205n${SIM_RES}TNG"
echo "Snapshot: ${SNAP}"
echo "Mass min: ${MASS_MIN}"
echo "Radius factor: ${RADIUS_FACTOR}"
echo "Start time: $(date)"
echo "=============================================="

# Step 0: Generate matches
echo -e "\n>>> Step 0: Generate matches..."
mpirun -np $SLURM_NTASKS python generate_matches_fast.py \
    --sim-res $SIM_RES \
    --snap $SNAP

# Step 1: Generate particle cache
echo -e "\n>>> Step 1: Generate particle cache..."
mpirun -np $SLURM_NTASKS python generate_particle_cache.py \
    --sim-res $SIM_RES \
    --snap $SNAP \
    --mass-min $MASS_MIN \
    --radius-factor $RADIUS_FACTOR

# Step 2: Compute statistics
echo -e "\n>>> Step 2: Compute halo statistics..."
mpirun -np $SLURM_NTASKS python compute_statistics_cached.py \
    --sim-res $SIM_RES \
    --snap $SNAP \
    --mass-min $MASS_MIN

# Step 3: Generate profiles
echo -e "\n>>> Step 3: Generate stacked profiles..."
mpirun -np $SLURM_NTASKS python generate_profiles_cached.py \
    --sim-res $SIM_RES \
    --snap $SNAP \
    --mass-min $MASS_MIN \
    --radius-factor $RADIUS_FACTOR

# Step 4: Generate maps
echo -e "\n>>> Step 4: Generate 2D maps..."
mpirun -np $SLURM_NTASKS python generate_maps_cached.py \
    --sim-res $SIM_RES \
    --snap $SNAP \
    --mass-min $MASS_MIN \
    --radius-factor $RADIUS_FACTOR

echo -e "\n=============================================="
echo "SNAP 99 PIPELINE COMPLETE"
echo "End time: $(date)"
echo "=============================================="
