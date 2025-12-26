#!/bin/bash
#SBATCH -J validate
#SBATCH -o logs/validate_%j.o
#SBATCH -e logs/validate_%j.e
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -p cca

# Validate that cache and direct methods produce consistent results

set -e

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

echo "=========================================="
echo "Validating Cache vs Direct Methods"
echo "=========================================="
echo "Start: $(date)"
echo ""

python << 'EOF'
import numpy as np
import h5py
from scipy.spatial import cKDTree
import time

print("=== Loading test data ===\n")

# Load halos from matches file
matches = np.load('/mnt/home/mlee1/ceph/hydro_replace_fields/L205n1250TNG/matches/matches_snap099.npz')
masses = matches['dmo_masses'] * 1e10  # Convert to Msun/h
positions = matches['dmo_positions'] / 1e3  # Convert to Mpc/h
radii = matches['dmo_radii'] / 1e3  # Convert to Mpc/h

# Select halos above mass threshold
mass_min = 12.5
log_masses = np.log10(masses)
mask = log_masses >= mass_min
positions = positions[mask]
radii = radii[mask]
n_halos = len(positions)
print(f"Selected {n_halos} halos above 10^{mass_min} Msun/h")

radius_mult = 5.0
radii_expanded = radii * radius_mult
max_radius = np.max(radii_expanded)
print(f"Max search radius: {max_radius:.3f} Mpc/h")

# Load a single DMO file (small test)
dmo_file = '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG_DM/output/snapdir_099/snap_099.0.hdf5'
with h5py.File(dmo_file, 'r') as f:
    test_coords = f['PartType1/Coordinates'][:].astype(np.float32) / 1e3
    test_ids = f['PartType1/ParticleIDs'][:]

n_particles = len(test_coords)
print(f"Test particles: {n_particles:,}")

# === Method 1: Direct spatial query (query_ball_point) ===
print("\n=== Method 1: Direct spatial query ===")
t0 = time.time()

# Build tree of halo centers
halo_tree = cKDTree(positions)

# For each particle, find all halos within max_radius
# Then check if within that halo's specific radius
near_any_halo = np.zeros(n_particles, dtype=bool)

# Process in chunks to manage memory
chunk_size = 100000
for start in range(0, n_particles, chunk_size):
    end = min(start + chunk_size, n_particles)
    chunk_coords = test_coords[start:end]
    
    # Find all halos within max_radius of each particle
    nearby_lists = halo_tree.query_ball_point(chunk_coords, max_radius)
    
    for i, nearby_halos in enumerate(nearby_lists):
        if len(nearby_halos) > 0:
            # Check if within ANY of the nearby halos' actual radii
            for halo_idx in nearby_halos:
                dist = np.linalg.norm(chunk_coords[i] - positions[halo_idx])
                if dist < radii_expanded[halo_idx]:
                    near_any_halo[start + i] = True
                    break

direct_in_halo = np.sum(near_any_halo)
direct_time = time.time() - t0
print(f"  Particles in halos: {direct_in_halo:,}")
print(f"  Time: {direct_time:.1f}s")

# === Method 2: Cache-based ID lookup ===
print("\n=== Method 2: Cache-based ID lookup ===")
t0 = time.time()

# Load cache and build ID set
cache_file = '/mnt/home/mlee1/ceph/hydro_replace_fields/L205n1250TNG/particle_cache/cache_snap099.h5'
with h5py.File(cache_file, 'r') as f:
    cache_masses = f['halo_info/masses'][:]
    cache_log_masses = np.log10(cache_masses)
    cache_mask = cache_log_masses >= mass_min
    selected_indices = np.where(cache_mask)[0]
    
    all_dmo_ids = []
    for idx in selected_indices:
        ids = f[f'dmo/halo_{idx}'][:]
        all_dmo_ids.append(ids)

all_dmo_ids_set = np.unique(np.concatenate(all_dmo_ids))
cache_load_time = time.time() - t0
print(f"  Cache IDs loaded: {len(all_dmo_ids_set):,}")
print(f"  Cache load time: {cache_load_time:.1f}s")

# Check which test particles are in the cache
t0 = time.time()
idx = np.searchsorted(all_dmo_ids_set, test_ids)
idx = np.clip(idx, 0, len(all_dmo_ids_set) - 1)
cache_in_halo = np.sum(all_dmo_ids_set[idx] == test_ids)
cache_lookup_time = time.time() - t0
print(f"  Particles in halos: {cache_in_halo:,}")
print(f"  Lookup time: {cache_lookup_time:.1f}s")

# === Comparison ===
print("\n=== Comparison ===")
print(f"Direct method: {direct_in_halo:,} particles in halos")
print(f"Cache method:  {cache_in_halo:,} particles in halos")
print(f"Difference:    {abs(direct_in_halo - cache_in_halo):,} ({100*abs(direct_in_halo - cache_in_halo)/max(direct_in_halo, cache_in_halo):.2f}%)")

# Check overlap
direct_mask = near_any_halo
cache_mask_ids = all_dmo_ids_set[idx] == test_ids

both = np.sum(direct_mask & cache_mask_ids)
direct_only = np.sum(direct_mask & ~cache_mask_ids)
cache_only = np.sum(~direct_mask & cache_mask_ids)

print(f"\nOverlap analysis:")
print(f"  Both methods agree (in halo):  {both:,}")
print(f"  Direct only (not in cache):    {direct_only:,}")
print(f"  Cache only (not in direct):    {cache_only:,}")

if direct_only > 0 or cache_only > 0:
    print("\n⚠️  Methods disagree on some particles!")
    print("    This may be due to floating point precision in radius comparisons")
else:
    print("\n✅ Methods produce identical results!")

EOF

echo ""
echo "=========================================="
echo "Validation complete: $(date)"
echo "=========================================="
