#!/usr/bin/env python
"""
Quick validation of cache vs direct particle selection.
Tests on a small subset to verify both methods select the same particles.
"""

import numpy as np
import h5py
from scipy.spatial import cKDTree
import glob

# Configuration
SIM_RES = 1250
SNAP = 99
MASS_MIN = 12.5
RADIUS_MULT = 5.0
BOX_SIZE = 205.0

CACHE_FILE = f'/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{SIM_RES}TNG/particle_cache/cache_snap{SNAP:03d}.h5'
MATCHES_FILE = f'/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{SIM_RES}TNG/matches/matches_snap{SNAP:03d}.npz'
DMO_PATH = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L205n{SIM_RES}TNG_DM/output'

print("=" * 70)
print("VALIDATION: Cache vs Direct Methods")
print("=" * 70)

# 1. Load halo catalog
print("\n[1] Loading halo catalog...")
matches = np.load(MATCHES_FILE)
dmo_masses = matches['dmo_masses'] * 1e10
dmo_positions = matches['dmo_positions'] / 1e3
dmo_radii = matches['dmo_radii'] / 1e3

mass_mask = np.log10(dmo_masses) >= MASS_MIN
halo_positions = dmo_positions[mass_mask]
halo_radii = dmo_radii[mass_mask] * RADIUS_MULT

print(f"  Halos: {len(halo_positions)}")

# Build KDTree
halo_tree = cKDTree(halo_positions, boxsize=BOX_SIZE)
max_radius = np.max(halo_radii)
print(f"  Max radius: {max_radius:.3f} Mpc/h")

# 2. Load ONE snapshot file
print("\n[2] Loading test file...")
dmo_files = sorted(glob.glob(f'{DMO_PATH}/snapdir_{SNAP:03d}/snap_{SNAP:03d}.*.hdf5'))
test_file = dmo_files[0]

with h5py.File(test_file, 'r') as f:
    coords = f['PartType1/Coordinates'][:].astype(np.float64) / 1e3
    pids = f['PartType1/ParticleIDs'][:]

print(f"  Particles: {len(coords):,}")

# 3. Direct spatial query
print("\n[3] Direct spatial query...")
direct_mask = np.zeros(len(coords), dtype=bool)
nearby_lists = halo_tree.query_ball_point(coords, max_radius)

for i, nearby_halos in enumerate(nearby_lists):
    for halo_idx in nearby_halos:
        dx = coords[i] - halo_positions[halo_idx]
        dx = dx - BOX_SIZE * np.round(dx / BOX_SIZE)
        if np.sqrt(np.sum(dx**2)) <= halo_radii[halo_idx]:
            direct_mask[i] = True
            break

print(f"  Near halos (direct): {np.sum(direct_mask):,}")
direct_ids = set(pids[direct_mask])

# 4. Load cache (just for halos that might contain particles from this file)
print("\n[4] Loading cache IDs...")

# Get cache halo indices
with h5py.File(CACHE_FILE, 'r') as f:
    cache_masses = f['halo_info/masses'][:]
    cache_mask = np.log10(cache_masses) >= MASS_MIN
    cache_indices = np.where(cache_mask)[0]
    
    print(f"  Loading {len(cache_indices)} halos from cache...")
    
    all_ids = []
    for i, idx in enumerate(cache_indices):
        if i % 2000 == 0:
            print(f"    {i}/{len(cache_indices)}...", end='\r', flush=True)
        all_ids.append(f[f'dmo/halo_{idx}'][:])
    print(f"    Done loading {len(cache_indices)} halos    ")

cache_dmo_ids = set(np.concatenate(all_ids))
print(f"  Total cache IDs: {len(cache_dmo_ids):,}")

# Check which test file particles are in cache
in_cache = np.array([pid in cache_dmo_ids for pid in pids])
print(f"  Near halos (cache): {np.sum(in_cache):,}")
cache_ids_from_file = set(pids[in_cache])

# 5. Compare
print("\n[5] Comparison:")
in_both = direct_ids & cache_ids_from_file
only_cache = cache_ids_from_file - direct_ids
only_direct = direct_ids - cache_ids_from_file

print(f"  Both methods agree: {len(in_both):,}")
print(f"  Only in cache: {len(only_cache):,}")
print(f"  Only in direct: {len(only_direct):,}")

if only_cache or only_direct:
    print("\n⚠️  MISMATCH! Investigating...")
    
    if only_cache:
        print(f"\n  Sample particles only in cache:")
        for pid in list(only_cache)[:3]:
            idx = np.where(pids == pid)[0][0]
            pos = coords[idx]
            nearby = halo_tree.query_ball_point(pos, max_radius)
            print(f"    PID {pid}: pos={pos}")
            for hi in nearby[:3]:
                dx = pos - halo_positions[hi]
                dx = dx - BOX_SIZE * np.round(dx / BOX_SIZE)
                d = np.sqrt(np.sum(dx**2))
                print(f"      Halo {hi}: dist={d:.4f}, radius={halo_radii[hi]:.4f}, in={d<=halo_radii[hi]}")
else:
    print("\n✓ PERFECT MATCH!")

print("\n" + "=" * 70)
