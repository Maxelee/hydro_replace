#!/usr/bin/env python
"""
Memory-optimized 2D density map generation using pre-computed particle cache.

OPTIMIZED 2-PASS VERSION:
- Pass 1 (DMO): Load all DMO particles ONCE, generate both full DMO map AND 
  Replace-DMO component (excluding halo particles) in a single I/O pass.
- Pass 2 (Hydro): Load all Hydro particles ONCE, generate both full Hydro map AND
  Replace-Hydro component (only halo particles) in a single I/O pass.

This is 2x faster than the previous 4-pass approach which loaded each simulation twice.

Memory optimization strategy:
1. Load DMO + mask → generate DMO map + Replace DMO component → free DMO
2. Load Hydro + mask → generate Hydro map + Replace Hydro component → free Hydro
3. Combine Replace components → save

Usage:
    mpirun -np 16 python generate_maps_cached_lowmem.py --snap 99 --sim-res 2500 --mass-min 12.5

This generates:
    - DMO map (full simulation)
    - Hydro map (full simulation)  
    - Replace map (DMO background + Hydro halos)
"""

import numpy as np
import h5py
import argparse
import os
import sys
import time
import glob
import gc

from mpi4py import MPI
import Pk_library as PKL
import MAS_library as MASL

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ============================================================================
# Configuration
# ============================================================================

SIM_PATHS = {
    2500: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output',
        'dmo_dm_mass': 0.0047271638660809,
        'hydro_dm_mass': 0.00398342749867548,
    },
    1250: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG/output',
        'dmo_dm_mass': 0.0378173109,
        'hydro_dm_mass': 0.0318674199,
    },
    625: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG/output',
        'dmo_dm_mass': 0.3025384873,
        'hydro_dm_mass': 0.2549393594,
    },
}

CACHE_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields'
OUTPUT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields'
BOX_SIZE = 205.0  # Mpc/h
MASS_UNIT = 1e10  # Convert to Msun/h
GRID_RES = 4096   # Default grid resolution


# ============================================================================
# Map Generation
# ============================================================================

def project_to_2d(coords, masses, grid_res, axis=2):
    """Project particles to 2D density map."""
    if len(coords) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32)
    
    # Select 2D projection axes
    proj_axes = [0, 1, 2]
    proj_axes.pop(axis)
    
    pos_2d = coords[:, proj_axes].astype(np.float32).copy()
    pos_2d = np.mod(pos_2d, BOX_SIZE)
    
    field = np.zeros((grid_res, grid_res), dtype=np.float32)
    MASL.MA(pos_2d, field, np.float32(BOX_SIZE), MAS='TSC',
            W=masses.astype(np.float32), verbose=False)
    
    return field


def load_particles_with_mask(snapshot, sim_res, mode, id_set, exclude=True):
    """
    Load particles and generate masks in a SINGLE PASS.
    
    Args:
        snapshot: Snapshot number
        sim_res: Simulation resolution (625, 1250, 2500)
        mode: 'dmo' or 'hydro'
        id_set: Sorted numpy array of particle IDs
        exclude: If True, mask marks particles NOT in id_set (for DMO background)
                 If False, mask marks particles IN id_set (for Hydro halos)
    
    Returns:
        coords, masses, mask (all particles, with mask indicating filtered subset)
    """
    sim_config = SIM_PATHS[sim_res]
    
    if mode == 'dmo':
        basePath = sim_config['dmo']
        dm_mass = sim_config['dmo_dm_mass']
        particle_types = [1]
    else:
        basePath = sim_config['hydro']
        dm_mass = sim_config['hydro_dm_mass']
        particle_types = [0, 1, 4]
    
    snap_dir = f"{basePath}/snapdir_{snapshot:03d}/"
    all_files = sorted(glob.glob(f"{snap_dir}/snap_{snapshot:03d}.*.hdf5"))
    
    # Distribute files across ranks
    my_files = [f for i, f in enumerate(all_files) if i % size == rank]
    
    coords_list = []
    masses_list = []
    mask_list = []
    
    for filepath in my_files:
        with h5py.File(filepath, 'r') as f:
            for ptype in particle_types:
                pt_key = f'PartType{ptype}'
                if pt_key not in f:
                    continue
                
                n_part = f[pt_key]['Coordinates'].shape[0]
                if n_part == 0:
                    continue
                
                # Load coordinates
                coords = f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3
                coords_list.append(coords)
                
                # Load masses
                if 'Masses' in f[pt_key]:
                    m = f[pt_key]['Masses'][:].astype(np.float32) * MASS_UNIT
                else:
                    m = np.full(n_part, dm_mass * MASS_UNIT, dtype=np.float32)
                masses_list.append(m)
                
                # Load IDs and compute mask
                pids = f[pt_key]['ParticleIDs'][:]
                
                # Use searchsorted for O(N log M) lookup
                idx = np.searchsorted(id_set, pids)
                idx = np.clip(idx, 0, len(id_set) - 1)
                in_set = id_set[idx] == pids
                
                if exclude:
                    # Keep particles NOT in the set (DMO background)
                    mask_list.append(~in_set)
                else:
                    # Keep particles IN the set (Hydro halos)
                    mask_list.append(in_set)
    
    if coords_list:
        return np.concatenate(coords_list), np.concatenate(masses_list), np.concatenate(mask_list)
    else:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)


def generate_maps(args):
    """Generate DMO, Hydro, and Replace maps using cache with optimized 2-pass loading."""
    
    t_start = time.time()
    
    # Paths
    cache_file = os.path.join(
        CACHE_BASE, f'L205n{args.sim_res}TNG',
        'particle_cache', f'cache_snap{args.snap:03d}.h5'
    )
    output_dir = os.path.join(
        OUTPUT_BASE, f'L205n{args.sim_res}TNG',
        f'snap{args.snap:03d}', 'projected'
    )
    
    if rank == 0:
        print("=" * 70)
        print("CACHED MAP GENERATION (OPTIMIZED 2-PASS)")
        print("=" * 70)
        print(f"Snapshot: {args.snap}")
        print(f"Resolution: L205n{args.sim_res}TNG")
        print(f"Mass threshold: 10^{args.mass_min} Msun/h")
        print(f"Grid: {args.grid}²")
        print(f"Cache: {cache_file}")
        print("=" * 70)
        sys.stdout.flush()
        
        os.makedirs(output_dir, exist_ok=True)
    
    comm.Barrier()
    
    # ========================================================================
    # Load cache metadata and select halos
    # ========================================================================
    if rank == 0:
        print("\n[1/4] Loading cache and selecting halos...")
        sys.stdout.flush()
    
    with h5py.File(cache_file, 'r') as f:
        halo_info = f['halo_info']
        all_masses = halo_info['masses'][:]
        all_log_masses = np.log10(all_masses)
        cache_radius = f.attrs['radius_multiplier']
        
        # Select halos above mass threshold
        mass_mask = all_log_masses >= args.mass_min
        if args.mass_max:
            mass_mask &= all_log_masses < args.mass_max
        
        selected_indices = np.where(mass_mask)[0]
        n_halos = len(selected_indices)
        
        if rank == 0:
            print(f"  Cache radius: {cache_radius}×R200")
            print(f"  Total halos in cache: {len(all_masses)}")
            print(f"  Halos above 10^{args.mass_min}: {n_halos}")
            sys.stdout.flush()
        
        # Load particle IDs for selected halos - use numpy concatenation for speed
        if rank == 0:
            print(f"  Loading particle IDs for {n_halos} halos...", end=" ", flush=True)
        
        dmo_ids_list = []
        hydro_ids_list = []
        
        for idx in selected_indices:
            dmo_ids_list.append(f[f'dmo/halo_{idx}'][:])
            hydro_ids_list.append(f[f'hydro_at_dmo/halo_{idx}'][:])
    
    # Build ID arrays using numpy (much faster than Python sets)
    if rank == 0:
        print("done")
        print("  Building ID arrays...", end=" ", flush=True)
    
    # Concatenate all IDs and get unique values
    all_dmo_ids_to_remove = np.unique(np.concatenate(dmo_ids_list))
    all_hydro_ids_to_add = np.unique(np.concatenate(hydro_ids_list))
    
    # Free intermediate lists
    del dmo_ids_list, hydro_ids_list
    
    if rank == 0:
        print("done")
        print(f"  DMO particles to remove: {len(all_dmo_ids_to_remove):,}")
        print(f"  Hydro particles to add: {len(all_hydro_ids_to_add):,}")
        sys.stdout.flush()
    
    # Arrays are already sorted by np.unique
    
    gc.collect()
    
    # ========================================================================
    # PHASE 1: DMO processing (SINGLE PASS - both full map AND background)
    # ========================================================================
    if rank == 0:
        print("\n[2/4] Loading DMO particles (single pass for both maps)...")
        t0 = time.time()
        sys.stdout.flush()
    
    dmo_coords, dmo_masses, dmo_bg_mask = load_particles_with_mask(
        args.snap, args.sim_res, 'dmo', all_dmo_ids_to_remove, exclude=True
    )
    
    if rank == 0:
        print(f"  Loaded in {time.time()-t0:.1f}s")
        print(f"  Rank 0: {len(dmo_coords):,} particles, {np.sum(dmo_bg_mask):,} background")
        print("  Generating DMO map...", end=" ", flush=True)
    
    # Full DMO map (all particles)
    local_dmo_map = project_to_2d(dmo_coords, dmo_masses, args.grid)
    
    if rank == 0:
        global_dmo_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_dmo_map = None
    
    comm.Reduce(local_dmo_map, global_dmo_map, op=MPI.SUM, root=0)
    del local_dmo_map
    
    if rank == 0:
        dmo_file = os.path.join(output_dir, 'dmo.npz')
        np.savez_compressed(dmo_file, field=global_dmo_map, box_size=BOX_SIZE,
                           grid_resolution=args.grid, snapshot=args.snap)
        del global_dmo_map
        print("done")
        print("  Generating Replace DMO component...", end=" ", flush=True)
        sys.stdout.flush()
    
    # Replace DMO component (only background particles, exclude halo particles)
    local_replace_dmo = project_to_2d(dmo_coords[dmo_bg_mask], dmo_masses[dmo_bg_mask], args.grid)
    
    if rank == 0:
        global_replace_dmo = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_replace_dmo = None
    
    comm.Reduce(local_replace_dmo, global_replace_dmo, op=MPI.SUM, root=0)
    del local_replace_dmo
    
    if rank == 0:
        print("done")
        sys.stdout.flush()
    
    # Free DMO particles
    del dmo_coords, dmo_masses, dmo_bg_mask, all_dmo_ids_to_remove
    gc.collect()
    
    # ========================================================================
    # PHASE 2: Hydro processing (SINGLE PASS - both full map AND halo particles)
    # ========================================================================
    if rank == 0:
        print("\n[3/4] Loading Hydro particles (single pass for both maps)...")
        t0 = time.time()
        sys.stdout.flush()
    
    hydro_coords, hydro_masses, hydro_halo_mask = load_particles_with_mask(
        args.snap, args.sim_res, 'hydro', all_hydro_ids_to_add, exclude=False
    )
    
    if rank == 0:
        print(f"  Loaded in {time.time()-t0:.1f}s")
        print(f"  Rank 0: {len(hydro_coords):,} particles, {np.sum(hydro_halo_mask):,} in halos")
        print("  Generating Hydro map...", end=" ", flush=True)
    
    # Full Hydro map (all particles)
    local_hydro_map = project_to_2d(hydro_coords, hydro_masses, args.grid)
    
    if rank == 0:
        global_hydro_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_hydro_map = None
    
    comm.Reduce(local_hydro_map, global_hydro_map, op=MPI.SUM, root=0)
    del local_hydro_map
    
    if rank == 0:
        hydro_file = os.path.join(output_dir, 'hydro.npz')
        np.savez_compressed(hydro_file, field=global_hydro_map, box_size=BOX_SIZE,
                           grid_resolution=args.grid, snapshot=args.snap)
        del global_hydro_map
        print("done")
        print("  Generating Replace Hydro component...", end=" ", flush=True)
        sys.stdout.flush()
    
    # Replace Hydro component (only halo particles)
    local_replace_hydro = project_to_2d(hydro_coords[hydro_halo_mask], hydro_masses[hydro_halo_mask], args.grid)
    
    if rank == 0:
        global_replace_hydro = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_replace_hydro = None
    
    comm.Reduce(local_replace_hydro, global_replace_hydro, op=MPI.SUM, root=0)
    del local_replace_hydro
    
    if rank == 0:
        print("done")
    
    # Free Hydro particles
    del hydro_coords, hydro_masses, hydro_halo_mask, all_hydro_ids_to_add
    gc.collect()
    
    # ========================================================================
    # PHASE 3: Combine Replace map and save
    # ========================================================================
    mass_label = f"M{args.mass_min:.1f}".replace('.', 'p')
    if args.mass_max:
        mass_label += f"_M{args.mass_max:.1f}".replace('.', 'p')
    
    if rank == 0:
        print(f"\n[4/4] Saving Replace map ({mass_label})...")
        
        global_replace_map = global_replace_dmo + global_replace_hydro
        del global_replace_dmo, global_replace_hydro
        
        replace_file = os.path.join(output_dir, f'replace_{mass_label}.npz')
        np.savez_compressed(replace_file, field=global_replace_map, box_size=BOX_SIZE,
                           grid_resolution=args.grid, snapshot=args.snap,
                           log_mass_min=args.mass_min, log_mass_max=args.mass_max,
                           radius_multiplier=cache_radius)
        print("done")
    
    # ========================================================================
    # Summary
    # ========================================================================
    if rank == 0:
        print(f"\nComplete!")
        print("=" * 70)
        print(f"Total time: {time.time()-t_start:.1f}s")
        print(f"Output directory: {output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Generate maps using particle cache (low memory)')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, default=2500, choices=[625, 1250, 2500])
    parser.add_argument('--mass-min', type=float, default=12.5,
                        help='Minimum log10(M200c/Msun/h) for replacement')
    parser.add_argument('--mass-max', type=float, default=None,
                        help='Maximum log10(M200c/Msun/h) for replacement')
    parser.add_argument('--grid', type=int, default=GRID_RES,
                        help='Grid resolution')
    
    args = parser.parse_args()
    generate_maps(args)


if __name__ == '__main__':
    main()
