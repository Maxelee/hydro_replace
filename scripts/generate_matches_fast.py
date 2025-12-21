#!/usr/bin/env python
"""
Fast bijective halo matching using particle ID hash maps.

Key insight: DMO and Hydro simulations share the same ICs, so particle IDs match.
Instead of spatial queries, we use hash lookups which are O(1).

Algorithm:
1. Build hash map: particle_ID -> halo_index for hydro (only massive halos)
2. For each DMO halo, look up its particles in the hash map
3. Count which hydro halo has most overlap -> that's the match

This is orders of magnitude faster than KD-tree approaches.

Usage:
    python generate_matches_fast.py --snap 99 --resolution 625
    python generate_matches_fast.py --snap 99 --resolution 2500
"""

import numpy as np
import argparse
import os
import time
import sys
sys.path.insert(0, '/mnt/home/mlee1/lux/illustris_python')
import illustris_python as il
from illustris_python import groupcat, snapshot
import h5py
import glob

# ============================================================================
# Configuration
# ============================================================================

SIM_PATHS = {
    2500: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output',
    },
    1250: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG/output',
    },
    625: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG/output',
    },
}

CONFIG = {
    'output_base': '/mnt/home/mlee1/ceph/hydro_replace_fields',
    'min_overlap_frac': 0.5,
    'min_particles': 100,
}


def load_halo_catalog(basePath, snapNum):
    """
    Load halo catalog with essential fields.
    """
    fields = ['GroupLenType', 'GroupFirstSub', 'Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
    halos = groupcat.loadHalos(basePath, snapNum, fields=fields)
    
    n_halos = halos['count']
    halo_len = halos['GroupLenType'][:, 1]  # DM particles per halo
    halo_mass = halos['Group_M_Crit200']
    halo_radius = halos['Group_R_Crit200']
    halo_pos = halos['GroupPos']
    
    return {
        'halo_len': halo_len,
        'halo_mass': halo_mass,
        'halo_radius': halo_radius,
        'halo_pos': halo_pos,
        'n_halos': n_halos,
        'basePath': basePath,
        'snapNum': snapNum,
    }


def build_particle_to_halo_map(basePath, snapNum, halo_indices, halo_len):
    """
    Build particle ID -> halo index map for specified halos.
    Uses illustris_python.snapshot.loadHalo for correct particle loading.
    
    Args:
        basePath: simulation path
        snapNum: snapshot number
        halo_indices: array of halo indices to include
        halo_len: array of particle counts per halo (for all halos)
    
    Returns: (sorted_pids, halo_indices) arrays for binary search lookup
    """
    print(f"  Building particle -> halo map for {len(halo_indices):,} halos...")
    t0 = time.time()
    
    all_pids = []
    all_halo_idx = []
    
    for i, halo_idx in enumerate(halo_indices):
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(halo_indices) - i) / rate
            print(f"    Loading {i:,}/{len(halo_indices):,} ({100*i/len(halo_indices):.1f}%) - ETA: {eta:.0f}s")
        
        # Load particle IDs using illustris_python (handles file boundaries correctly)
        pids = il.snapshot.loadHalo(basePath, snapNum, halo_idx, 'dm', fields=['ParticleIDs'])
        if pids is None or len(pids) == 0:
            continue
            
        all_pids.append(pids)
        all_halo_idx.append(np.full(len(pids), halo_idx, dtype=np.int32))
    
    if len(all_pids) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int32)
    
    # Concatenate and sort
    all_pids = np.concatenate(all_pids)
    all_halo_idx = np.concatenate(all_halo_idx)
    
    sort_idx = np.argsort(all_pids)
    sorted_pids = all_pids[sort_idx]
    sorted_halos = all_halo_idx[sort_idx]
    
    print(f"  Map built in {time.time() - t0:.1f}s ({len(sorted_pids):,} particles)")
    return sorted_pids, sorted_halos


def lookup_halos_batch(query_pids, sorted_pids, sorted_halos):
    """
    Look up halo indices for a batch of particle IDs using binary search.
    Returns array of halo indices (-1 if not found).
    """
    # Binary search
    idx = np.searchsorted(sorted_pids, query_pids)
    
    # Check if found (handle edge cases)
    valid = (idx < len(sorted_pids)) & (sorted_pids[np.minimum(idx, len(sorted_pids)-1)] == query_pids)
    
    result = np.full(len(query_pids), -1, dtype=np.int32)
    result[valid] = sorted_halos[idx[valid]]
    
    return result


def match_halos_fast(basePath_dmo, basePath_hydro, snapNum,
                     dmo_catalog, hydro_catalog, 
                     hydro_sorted_pids, hydro_sorted_halos,
                     min_overlap_frac=0.5, min_particles=100, min_mass=1e12):
    """
    Match DMO halos to Hydro halos using vectorized particle ID lookups.
    
    Now checks BOTH directions:
    - Forward: what fraction of DMO particles are in the Hydro halo
    - Reverse: what fraction of Hydro particles are in the DMO halo
    
    This prevents small DMO subhalos from matching to large Hydro clusters.
    """
    n_halos = dmo_catalog['n_halos']
    halo_len = dmo_catalog['halo_len']
    halo_mass = dmo_catalog['halo_mass']
    
    matches_dmo = []
    matches_hydro = []
    overlap_fracs_forward = []
    overlap_fracs_reverse = []
    
    # Mass cut in simulation units (1e10 Msun/h)
    min_mass_sim = min_mass / 1e10
    
    # Get indices of halos above mass cut
    candidate_indices = np.where((halo_mass >= min_mass_sim) & (halo_len >= min_particles))[0]
    print(f"  Matching {len(candidate_indices):,} DMO halos above cuts...")
    t0 = time.time()
    
    for idx, i in enumerate(candidate_indices):
        if idx % 500 == 0 and idx > 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            eta = (len(candidate_indices) - idx) / rate
            print(f"    {idx:,}/{len(candidate_indices):,} ({100*idx/len(candidate_indices):.1f}%) - ETA: {eta:.0f}s")
        
        n_part = halo_len[i]
        
        # Get particle IDs for this DMO halo (using illustris_python)
        halo_pids = il.snapshot.loadHalo(basePath_dmo, snapNum, i, 'dm', fields=['ParticleIDs'])
        if halo_pids is None or len(halo_pids) == 0:
            continue
        
        # Look up which hydro halos these particles belong to (vectorized)
        hydro_halo_idx = lookup_halos_batch(halo_pids, hydro_sorted_pids, hydro_sorted_halos)
        
        # Count occurrences (only valid matches)
        valid_mask = hydro_halo_idx >= 0
        if not np.any(valid_mask):
            continue
        
        valid_hydro = hydro_halo_idx[valid_mask]
        unique, counts = np.unique(valid_hydro, return_counts=True)
        
        # Find best match (forward direction: DMO -> Hydro)
        best_idx = np.argmax(counts)
        best_hydro = unique[best_idx]
        best_count = counts[best_idx]
        overlap_forward = best_count / n_part
        
        if overlap_forward < min_overlap_frac:
            continue
        
        # NOW CHECK REVERSE DIRECTION: what fraction of hydro halo is in this DMO halo?
        # Load hydro halo particles
        hydro_halo_pids = il.snapshot.loadHalo(basePath_hydro, snapNum, best_hydro, 'dm', fields=['ParticleIDs'])
        if hydro_halo_pids is None or len(hydro_halo_pids) == 0:
            continue
        
        # Count how many hydro particles are in the DMO halo
        dmo_set = set(halo_pids)
        n_in_this_dmo = sum(1 for p in hydro_halo_pids if p in dmo_set)
        n_hydro_part = len(hydro_halo_pids)
        overlap_reverse = n_in_this_dmo / n_hydro_part if n_hydro_part > 0 else 0
        
        # Require reasonable overlap in BOTH directions
        if overlap_reverse < min_overlap_frac:
            continue
        
        matches_dmo.append(i)
        matches_hydro.append(best_hydro)
        overlap_fracs_forward.append(overlap_forward)
        overlap_fracs_reverse.append(overlap_reverse)
    
    print(f"  Checked {len(candidate_indices):,} halos above mass cut")
    print(f"  Matching completed in {time.time() - t0:.1f}s")
    
    # Use geometric mean of forward and reverse overlap as final overlap
    overlap_fracs = np.sqrt(np.array(overlap_fracs_forward) * np.array(overlap_fracs_reverse))
    
    return np.array(matches_dmo), np.array(matches_hydro), overlap_fracs


def make_bijective(dmo_indices, hydro_indices, overlap_fracs):
    """
    Ensure bijective mapping: each hydro halo matches at most one DMO halo.
    Keep the match with highest overlap fraction.
    """
    print("  Making bijective...")
    
    # Group by hydro index
    hydro_to_matches = {}
    for i, (d, h, o) in enumerate(zip(dmo_indices, hydro_indices, overlap_fracs)):
        if h not in hydro_to_matches:
            hydro_to_matches[h] = []
        hydro_to_matches[h].append((d, o, i))
    
    # Keep best match for each hydro halo
    keep_indices = []
    for h, matches in hydro_to_matches.items():
        # Sort by overlap fraction (descending)
        matches.sort(key=lambda x: -x[1])
        # Keep best
        keep_indices.append(matches[0][2])
    
    keep_indices = sorted(keep_indices)
    
    final_dmo = dmo_indices[keep_indices]
    final_hydro = hydro_indices[keep_indices]
    final_overlap = overlap_fracs[keep_indices]
    
    print(f"  Bijective: {len(dmo_indices)} -> {len(final_dmo)} matches")
    
    return final_dmo, final_hydro, final_overlap


def generate_matches(snapNum, resolution, output_dir, min_mass=1e12):
    """Main matching function."""
    
    print("=" * 70)
    print("Fast Bijective Halo Matching (Hash-based)")
    print("=" * 70)
    print(f"Resolution: {resolution}^3")
    print(f"Snapshot: {snapNum}")
    print(f"Min mass: {min_mass:.0e} Msun/h")
    print("=" * 70)
    
    sim = SIM_PATHS[resolution]
    basePath_dmo = sim['dmo']
    basePath_hydro = sim['hydro']
    
    # Step 1: Load DMO halo catalog
    print("\n[1/4] Loading DMO halo catalog...")
    t0 = time.time()
    dmo_catalog = load_halo_catalog(basePath_dmo, snapNum)
    print(f"  {dmo_catalog['n_halos']:,} halos")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    
    # Step 2: Load Hydro halo catalog
    print("\n[2/4] Loading Hydro halo catalog...")
    t0 = time.time()
    hydro_catalog = load_halo_catalog(basePath_hydro, snapNum)
    print(f"  {hydro_catalog['n_halos']:,} halos")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    
    # Step 3: Build particle -> halo map for hydro (only massive halos)
    # This is for fast lookup: "which hydro halo does this particle belong to?"
    print("\n[3/4] Building Hydro particle map...")
    min_mass_sim = min_mass / 1e10
    hydro_massive = np.where(
        (hydro_catalog['halo_mass'] >= min_mass_sim * 0.1) &  # 10x lower to catch matches
        (hydro_catalog['halo_len'] >= CONFIG['min_particles'])
    )[0]
    print(f"  Including {len(hydro_massive):,} hydro halos above cuts")
    
    hydro_sorted_pids, hydro_sorted_halos = build_particle_to_halo_map(
        basePath_hydro, snapNum, hydro_massive, hydro_catalog['halo_len']
    )
    
    # Step 4: Match halos (bidirectional check)
    print("\n[4/4] Matching halos (bidirectional check)...")
    dmo_idx, hydro_idx, overlap = match_halos_fast(
        basePath_dmo, basePath_hydro, snapNum,
        dmo_catalog, hydro_catalog,
        hydro_sorted_pids, hydro_sorted_halos,
        min_overlap_frac=CONFIG['min_overlap_frac'],
        min_particles=CONFIG['min_particles'],
        min_mass=min_mass
    )
    print(f"  Found {len(dmo_idx):,} initial matches")
    
    # Make bijective (twice to handle both directions)
    print("\n  Making bijective...")
    dmo_idx, hydro_idx, overlap = make_bijective(dmo_idx, hydro_idx, overlap)
    # Run again with swapped roles
    hydro_idx, dmo_idx, overlap = make_bijective(hydro_idx, dmo_idx, overlap)
    
    # Save results
    out_dir = f"{output_dir}/L205n{resolution}TNG/matches"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/matches_snap{snapNum:03d}.npz"
    
    np.savez(out_file,
             dmo_indices=dmo_idx,
             hydro_indices=hydro_idx,
             overlap_fractions=overlap,
             dmo_masses=dmo_catalog['halo_mass'][dmo_idx],
             dmo_radii=dmo_catalog['halo_radius'][dmo_idx],
             dmo_positions=dmo_catalog['halo_pos'][dmo_idx],
             hydro_masses=hydro_catalog['halo_mass'][hydro_idx],
             hydro_radii=hydro_catalog['halo_radius'][hydro_idx],
             hydro_positions=hydro_catalog['halo_pos'][hydro_idx],
             snapshot=snapNum,
             resolution=resolution)
    
    print(f"\n{'=' * 70}")
    print(f"Saved {len(dmo_idx):,} bijective matches to:")
    print(f"  {out_file}")
    print(f"{'=' * 70}")
    
    return out_file


def main():
    parser = argparse.ArgumentParser(description='Fast bijective halo matching')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--resolution', type=int, required=True,
                        choices=[625, 1250, 2500], help='Simulation resolution')
    parser.add_argument('--output-dir', type=str, default=CONFIG['output_base'],
                        help='Output directory')
    parser.add_argument('--min-mass', type=float, default=1e12,
                        help='Minimum halo mass in Msun/h (default: 1e12)')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if matches exist')
    
    args = parser.parse_args()
    
    # Check if exists
    out_file = f"{args.output_dir}/L205n{args.resolution}TNG/matches/matches_snap{args.snap:03d}.npz"
    if os.path.exists(out_file) and not args.force:
        print(f"Matches already exist: {out_file}")
        print("Use --force to regenerate")
        return
    
    t_start = time.time()
    generate_matches(args.snap, args.resolution, args.output_dir, min_mass=args.min_mass)
    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
