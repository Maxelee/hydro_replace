#!/usr/bin/env python
"""
Validate that the direct and cache methods select the same particles.

This compares:
1. Cache method: loads particle IDs from HDF5 cache
2. Direct method: uses spatial queries to find particles near halos

The particle selection should be IDENTICAL.
"""

import numpy as np
import h5py
from scipy.spatial import cKDTree
import sys
import glob

# Configuration
SIM_RES = 1250
SNAP = 99
MASS_MIN = 12.5
RADIUS_MULT = 5.0
BOX_SIZE = 205.0

# Paths
CACHE_FILE = f'/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{SIM_RES}TNG/particle_cache/cache_snap{SNAP:03d}.h5'
MATCHES_FILE = f'/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{SIM_RES}TNG/matches/matches_snap{SNAP:03d}.npz'
DMO_PATH = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L205n{SIM_RES}TNG_DM/output'
HYDRO_PATH = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L205n{SIM_RES}TNG/output'

print("=" * 70)
print("VALIDATION: Cache vs Direct Methods")
print("=" * 70)

# ============================================================================
# 1. Load Cache Method IDs
# ============================================================================
print("\n[1] Loading particle IDs from cache...")

with h5py.File(CACHE_FILE, 'r') as f:
    all_masses = f['halo_info/masses'][:]
    all_log_masses = np.log10(all_masses)
    
    mass_mask = all_log_masses >= MASS_MIN
    selected_indices = np.where(mass_mask)[0]
    n_halos = len(selected_indices)
    
    print(f"  Halos above 10^{MASS_MIN}: {n_halos}")
    
    # Collect all particle IDs
    dmo_ids_cache = []
    hydro_ids_cache = []
    
    for idx in selected_indices:
        dmo_ids_cache.append(f[f'dmo/halo_{idx}'][:])
        hydro_ids_cache.append(f[f'hydro_at_dmo/halo_{idx}'][:])

# Get unique IDs (like the cache method does with sets)
cache_dmo_ids = np.unique(np.concatenate(dmo_ids_cache))
cache_hydro_ids = np.unique(np.concatenate(hydro_ids_cache))

print(f"  Cache DMO IDs: {len(cache_dmo_ids):,}")
print(f"  Cache Hydro IDs: {len(cache_hydro_ids):,}")

# ============================================================================
# 2. Load Halo Catalog for Direct Method
# ============================================================================
print("\n[2] Loading halo catalog...")

matches = np.load(MATCHES_FILE)
dmo_masses = matches['dmo_masses'] * 1e10  # Convert to Msun/h
dmo_positions = matches['dmo_positions'] / 1e3  # Convert to Mpc
dmo_radii = matches['dmo_radii'] / 1e3  # Convert to Mpc

# Filter by mass
mass_mask = np.log10(dmo_masses) >= MASS_MIN
halo_positions = dmo_positions[mass_mask]
halo_radii = dmo_radii[mass_mask] * RADIUS_MULT  # Apply radius multiplier

print(f"  Halos for direct method: {len(halo_positions)}")
print(f"  Max halo radius: {np.max(halo_radii):.3f} Mpc/h")

# ============================================================================
# 3. Direct Method: Load ONE snapshot file and compare
# ============================================================================
print("\n[3] Testing on first snapshot file...")

# Build halo KDTree
halo_tree = cKDTree(halo_positions, boxsize=BOX_SIZE)
max_radius = np.max(halo_radii)

# Load one DMO file
dmo_files = sorted(glob.glob(f'{DMO_PATH}/snapdir_{SNAP:03d}/snap_{SNAP:03d}.*.hdf5'))
test_file = dmo_files[0]

print(f"  Test file: {test_file}")

with h5py.File(test_file, 'r') as f:
    coords = f['PartType1/Coordinates'][:].astype(np.float64) / 1e3
    pids = f['PartType1/ParticleIDs'][:]

print(f"  Particles in file: {len(coords):,}")

# Cache method: check which IDs are in the cache set
cache_set = set(cache_dmo_ids)
in_cache = np.array([pid in cache_set for pid in pids])

print(f"  Particles in cache: {np.sum(in_cache):,}")

# Direct method: spatial query
# Use query_ball_point to find ALL halos within max_radius of each particle
print("  Running spatial query (this may take a minute)...")

direct_mask = np.zeros(len(coords), dtype=bool)
nearby_lists = halo_tree.query_ball_point(coords, max_radius)

for i, nearby_halos in enumerate(nearby_lists):
    if len(nearby_halos) == 0:
        continue
    
    for halo_idx in nearby_halos:
        dx = coords[i] - halo_positions[halo_idx]
        dx = dx - BOX_SIZE * np.round(dx / BOX_SIZE)  # Periodic
        dist = np.sqrt(np.sum(dx**2))
        
        if dist <= halo_radii[halo_idx]:
            direct_mask[i] = True
            break

print(f"  Particles in direct: {np.sum(direct_mask):,}")

# ============================================================================
# 4. Compare
# ============================================================================
print("\n[4] Comparison:")

# IDs selected by each method
cache_selected = pids[in_cache]
direct_selected = pids[direct_mask]

cache_set_from_file = set(cache_selected)
direct_set_from_file = set(direct_selected)

in_both = cache_set_from_file & direct_set_from_file
only_in_cache = cache_set_from_file - direct_set_from_file
only_in_direct = direct_set_from_file - cache_set_from_file

print(f"  In both methods: {len(in_both):,}")
print(f"  Only in cache: {len(only_in_cache):,}")
print(f"  Only in direct: {len(only_in_direct):,}")

if len(only_in_cache) > 0 or len(only_in_direct) > 0:
    print("\n  ⚠️  MISMATCH DETECTED!")
    
    # Investigate a few mismatches
    if len(only_in_cache) > 0:
        print(f"\n  Investigating particles only in cache:")
        sample_ids = list(only_in_cache)[:5]
        for pid in sample_ids:
            idx = np.where(pids == pid)[0][0]
            pos = coords[idx]
            
            # Find nearest halos
            dists, nearest_idx = halo_tree.query(pos, k=3)
            print(f"    PID {pid}: pos={pos}")
            for d, hi in zip(dists, nearest_idx):
                print(f"      Halo {hi}: dist={d:.4f}, radius={halo_radii[hi]:.4f}, in_halo={d <= halo_radii[hi]}")
    
    if len(only_in_direct) > 0:
        print(f"\n  Investigating particles only in direct:")
        sample_ids = list(only_in_direct)[:5]
        for pid in sample_ids:
            idx = np.where(pids == pid)[0][0]
            pos = coords[idx]
            in_cache_global = pid in cache_set
            print(f"    PID {pid}: pos={pos}, in_global_cache={in_cache_global}")
else:
    print("\n  ✓ PERFECT MATCH!")

print("\n" + "=" * 70)
