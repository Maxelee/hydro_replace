#!/usr/bin/env python
"""
Verify cache and direct methods produce same results using small sample.
"""

import numpy as np
import h5py
from scipy.spatial import cKDTree
import glob

SIM_RES = 1250
SNAP = 99
MASS_MIN = 12.5
RADIUS_MULT = 5.0
BOX_SIZE = 205.0

CACHE_FILE = f'/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{SIM_RES}TNG/particle_cache/cache_snap{SNAP:03d}.h5'
MATCHES_FILE = f'/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{SIM_RES}TNG/matches/matches_snap{SNAP:03d}.npz'
DMO_PATH = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L205n{SIM_RES}TNG_DM/output'

print("=" * 70)
print("VALIDATION: Testing on 100k particle sample")
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

halo_tree = cKDTree(halo_positions, boxsize=BOX_SIZE)
max_radius = np.max(halo_radii)

# 2. Load SMALL sample of particles
print("\n[2] Loading particle sample...")
dmo_files = sorted(glob.glob(f'{DMO_PATH}/snapdir_{SNAP:03d}/snap_{SNAP:03d}.*.hdf5'))

with h5py.File(dmo_files[0], 'r') as f:
    all_coords = f['PartType1/Coordinates'][:].astype(np.float64) / 1e3
    all_pids = f['PartType1/ParticleIDs'][:]

# Take 100k random sample
np.random.seed(42)
sample_idx = np.random.choice(len(all_coords), size=100000, replace=False)
coords = all_coords[sample_idx]
pids = all_pids[sample_idx]

print(f"  Sample size: {len(coords):,}")

# 3. Direct method
print("\n[3] Direct spatial query on sample...")
direct_mask = np.zeros(len(coords), dtype=bool)

# Process in small batches
batch_size = 10000
for start in range(0, len(coords), batch_size):
    end = min(start + batch_size, len(coords))
    batch_coords = coords[start:end]
    
    nearby_lists = halo_tree.query_ball_point(batch_coords, max_radius)
    
    for i, nearby_halos in enumerate(nearby_lists):
        for halo_idx in nearby_halos:
            dx = batch_coords[i] - halo_positions[halo_idx]
            dx = dx - BOX_SIZE * np.round(dx / BOX_SIZE)
            if np.sqrt(np.sum(dx**2)) <= halo_radii[halo_idx]:
                direct_mask[start + i] = True
                break

print(f"  Near halos (direct): {np.sum(direct_mask):,}")

# 4. Load cache and check
print("\n[4] Loading cache...")
with h5py.File(CACHE_FILE, 'r') as f:
    cache_masses = f['halo_info/masses'][:]
    cache_mask = np.log10(cache_masses) >= MASS_MIN
    cache_indices = np.where(cache_mask)[0]
    
    all_ids = []
    for i, idx in enumerate(cache_indices):
        if i % 2000 == 0:
            print(f"    Loading halo {i}/{len(cache_indices)}...", end='\r', flush=True)
        all_ids.append(f[f'dmo/halo_{idx}'][:])
    print(f"    Loaded {len(cache_indices)} halos          ")

cache_dmo_ids = set(np.concatenate(all_ids))
print(f"  Cache total IDs: {len(cache_dmo_ids):,}")

# Check sample particles
in_cache = np.array([pid in cache_dmo_ids for pid in pids])
print(f"  Near halos (cache): {np.sum(in_cache):,}")

# 5. Compare
print("\n[5] Comparison:")
agree = (direct_mask == in_cache)
print(f"  Agreement: {np.sum(agree):,} / {len(agree):,} ({100*np.mean(agree):.2f}%)")

disagree_idx = np.where(~agree)[0]
if len(disagree_idx) > 0:
    print(f"\n⚠️  {len(disagree_idx)} disagreements!")
    
    # Analyze disagreements
    cache_not_direct = np.sum(in_cache & ~direct_mask)
    direct_not_cache = np.sum(direct_mask & ~in_cache)
    print(f"  In cache but not direct: {cache_not_direct}")
    print(f"  In direct but not cache: {direct_not_cache}")
    
    # Sample investigation
    if cache_not_direct > 0:
        print(f"\n  Sample particles in cache but NOT found by direct:")
        mask = in_cache & ~direct_mask
        sample = np.where(mask)[0][:3]
        for idx in sample:
            pos = coords[idx]
            pid = pids[idx]
            
            # Check against ALL halos (brute force)
            found = False
            for hi in range(len(halo_positions)):
                dx = pos - halo_positions[hi]
                dx = dx - BOX_SIZE * np.round(dx / BOX_SIZE)
                d = np.sqrt(np.sum(dx**2))
                if d <= halo_radii[hi]:
                    found = True
                    print(f"    PID {pid}: pos={pos}, actually in halo {hi} (d={d:.4f}, r={halo_radii[hi]:.4f})")
                    break
            if not found:
                print(f"    PID {pid}: pos={pos}, NOT in any halo by brute force!")
                
                # Check cache to see which halo it's in
                with h5py.File(CACHE_FILE, 'r') as f:
                    for i, cidx in enumerate(cache_indices):
                        ids = f[f'dmo/halo_{cidx}'][:]
                        if pid in ids:
                            # Get this halo's info
                            hpos = f['halo_info/positions'][cidx] / 1e3
                            hr = f['halo_info/radii'][cidx] / 1e3 * RADIUS_MULT
                            dx = pos - hpos
                            dx = dx - BOX_SIZE * np.round(dx / BOX_SIZE)
                            d = np.sqrt(np.sum(dx**2))
                            print(f"      Found in cache halo {cidx}: halo_pos={hpos}, d={d:.4f}, r={hr:.4f}")
                            break
else:
    print("\n✓ PERFECT MATCH!")

print("\n" + "=" * 70)
