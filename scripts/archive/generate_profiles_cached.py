#!/usr/bin/env python
"""
MPI-parallel profile generation using particle ID cache.

This version uses pre-computed particle ID cache to avoid KDTree queries.
Much faster than the streaming approach since we:
1. Load particle cache (has particle IDs per halo)
2. Load all particles once, build ID→index mapping
3. For each halo, lookup cached IDs → get coords/masses → bin by radius

Usage:
    mpirun -n 32 python generate_profiles_cached.py --sim-res 2500 --snapshot 99
"""

import numpy as np
import h5py
import os
import sys
import time
import argparse
import glob
from mpi4py import MPI

# ============================================================================
# Configuration
# ============================================================================

SIM_PATHS = {
    2500: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output',
        'dmo_mass': 0.0047271638660809,
        'hydro_dm_mass': 0.00398342749867548,
    },
    1250: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG/output',
        'dmo_mass': 0.0378173109,
        'hydro_dm_mass': 0.0318674199,
    },
    625: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG/output',
        'dmo_mass': 0.3025384873,
        'hydro_dm_mass': 0.2549393594,
    },
}

CONFIG = {
    'box_size': 205.0,      # Mpc/h
    'mass_unit': 1e10,      # Convert to Msun/h
    'r_max_r200': 5.0,      # Maximum radius in units of R_200
    'r_min_r200': 0.01,     # Minimum radius in units of R_200
    'n_radial_bins': 25,    # Number of radial bins (log-spaced)
    'log_mass_min': 12.0,   # Minimum halo mass log10(M/Msun/h)
}

OUTPUT_DIR = '/mnt/home/mlee1/ceph/hydro_replace_fields'

# ============================================================================
# Helper Functions
# ============================================================================

def periodic_distance(coords, center, box_size):
    """Compute periodic distances from coords to center."""
    dx = coords - center
    dx = dx - np.round(dx / box_size) * box_size
    return np.linalg.norm(dx, axis=1)


def load_halo_catalog(basePath, snapNum, fields):
    """Load halo catalog fields."""
    from illustris_python import groupcat
    return groupcat.loadHalos(basePath, snapNum, fields=fields)


def load_particle_cache(snapNum, sim_res):
    """
    Load particle ID cache for a snapshot.
    
    Returns dict with:
        - halo_indices: array of halo indices in the cache
        - dmo_particle_ids: dict mapping halo_idx -> particle ID array
        - hydro_particle_ids: dict mapping halo_idx -> particle ID array
    """
    cache_file = f'{OUTPUT_DIR}/L205n{sim_res}TNG/particle_cache/cache_snap{snapNum:03d}.h5'
    
    if not os.path.exists(cache_file):
        return None
    
    cache_data = {
        'halo_indices': None,
        'dmo_particle_ids': {},
        'hydro_particle_ids': {},
    }
    
    with h5py.File(cache_file, 'r') as f:
        # Load halo info
        cache_data['halo_indices'] = f['halo_info/halo_indices'][:]
        cache_data['positions'] = f['halo_info/positions'][:]
        cache_data['radii'] = f['halo_info/radii'][:]
        cache_data['masses'] = f['halo_info/masses'][:]
        
        # Load DMO particle IDs
        if 'dmo' in f:
            for key in f['dmo'].keys():
                halo_idx = int(key.split('_')[1])
                cache_data['dmo_particle_ids'][halo_idx] = f['dmo'][key][:]
        
        # Load Hydro particle IDs
        if 'hydro' in f:
            for key in f['hydro'].keys():
                halo_idx = int(key.split('_')[1])
                cache_data['hydro_particle_ids'][halo_idx] = f['hydro'][key][:]
    
    return cache_data


def load_dmo_particles_distributed(basePath, snapNum, comm):
    """Load DMO particles distributed across MPI ranks."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    snap_dir = f"{basePath}/snapdir_{snapNum:03d}/"
    files = sorted(glob.glob(f"{snap_dir}/snap_{snapNum:03d}.*.hdf5"))
    
    if not files:
        files = sorted(glob.glob(f"{snap_dir}/snapshot_{snapNum:03d}.*.hdf5"))
    
    my_files = [f for i, f in enumerate(files) if i % size == rank]
    
    coords_list = []
    ids_list = []
    
    for fpath in my_files:
        with h5py.File(fpath, 'r') as f:
            if 'PartType1' not in f:
                continue
            coords_list.append(f['PartType1/Coordinates'][:].astype(np.float32) / 1e3)  # kpc -> Mpc
            ids_list.append(f['PartType1/ParticleIDs'][:])
    
    if coords_list:
        coords = np.concatenate(coords_list)
        ids = np.concatenate(ids_list)
    else:
        coords = np.zeros((0, 3), dtype=np.float32)
        ids = np.zeros(0, dtype=np.int64)
    
    return coords, ids


def load_hydro_particles_distributed(basePath, snapNum, comm):
    """Load Hydro particles (gas + DM + stars) distributed across MPI ranks."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    snap_dir = f"{basePath}/snapdir_{snapNum:03d}/"
    files = sorted(glob.glob(f"{snap_dir}/snap_{snapNum:03d}.*.hdf5"))
    
    if not files:
        files = sorted(glob.glob(f"{snap_dir}/snapshot_{snapNum:03d}.*.hdf5"))
    
    my_files = [f for i, f in enumerate(files) if i % size == rank]
    
    coords_list = []
    masses_list = []
    ids_list = []
    
    for fpath in my_files:
        with h5py.File(fpath, 'r') as f:
            # Gas (PartType0)
            if 'PartType0' in f:
                coords_list.append(f['PartType0/Coordinates'][:].astype(np.float32) / 1e3)
                masses_list.append(f['PartType0/Masses'][:].astype(np.float32) * CONFIG['mass_unit'])
                ids_list.append(f['PartType0/ParticleIDs'][:])
            
            # DM (PartType1)
            if 'PartType1' in f:
                c = f['PartType1/Coordinates'][:].astype(np.float32) / 1e3
                coords_list.append(c)
                masses_list.append(np.full(len(c), SIM_PATHS[2500]['hydro_dm_mass'] * CONFIG['mass_unit'], dtype=np.float32))
                ids_list.append(f['PartType1/ParticleIDs'][:])
            
            # Stars (PartType4)
            if 'PartType4' in f:
                coords_list.append(f['PartType4/Coordinates'][:].astype(np.float32) / 1e3)
                masses_list.append(f['PartType4/Masses'][:].astype(np.float32) * CONFIG['mass_unit'])
                ids_list.append(f['PartType4/ParticleIDs'][:])
    
    if coords_list:
        coords = np.concatenate(coords_list)
        masses = np.concatenate(masses_list)
        ids = np.concatenate(ids_list)
    else:
        coords = np.zeros((0, 3), dtype=np.float32)
        masses = np.zeros(0, dtype=np.float32)
        ids = np.zeros(0, dtype=np.int64)
    
    return coords, masses, ids


def compute_profile_for_halo(coords, masses, center, r200, r_bins, box_size):
    """
    Compute mass profile for a single halo from given particles.
    
    Returns:
        mass_in_bins: mass in each radial bin
        n_particles: number of particles in each radial bin
    """
    n_bins = len(r_bins) - 1
    
    if len(coords) == 0:
        return np.zeros(n_bins), np.zeros(n_bins, dtype=np.int64)
    
    # Compute distances with periodic BC
    r = periodic_distance(coords, center, box_size)
    
    # Scale by R_200
    r_scaled = r / r200
    
    # Bin particles
    bin_indices = np.searchsorted(r_bins, r_scaled) - 1
    valid = (bin_indices >= 0) & (bin_indices < n_bins)
    
    mass_in_bins = np.zeros(n_bins)
    n_particles = np.zeros(n_bins, dtype=np.int64)
    
    if np.any(valid):
        mass_in_bins = np.bincount(bin_indices[valid], weights=masses[valid], minlength=n_bins).astype(np.float64)
        n_particles = np.bincount(bin_indices[valid], minlength=n_bins).astype(np.int64)
    
    return mass_in_bins, n_particles


def mass_to_density(mass_in_bins, r200, r_bins):
    """Convert mass in bins to density profile."""
    n_bins = len(r_bins) - 1
    profile = np.zeros(n_bins, dtype=np.float64)
    
    for b in range(n_bins):
        r_inner = r_bins[b] * r200
        r_outer = r_bins[b + 1] * r200
        volume = 4/3 * np.pi * (r_outer**3 - r_inner**3)
        if volume > 0:
            profile[b] = mass_in_bins[b] / volume
    
    return profile


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Profile Generation with Cache')
    parser.add_argument('--sim-res', type=int, required=True, choices=[625, 1250, 2500])
    parser.add_argument('--snapshot', type=int, required=True)
    parser.add_argument('--mode', choices=['dmo', 'hydro', 'both'], default='both')
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=" * 70)
        print(f"PROFILE GENERATION WITH CACHE - L205n{args.sim_res}TNG snap {args.snapshot}")
        print("=" * 70)
        print(f"  Ranks: {size}")
        print(f"  Mode: {args.mode}")
    
    # Configuration
    sim_config = SIM_PATHS[args.sim_res]
    box_size = CONFIG['box_size']
    mass_unit = CONFIG['mass_unit']
    
    # Radial bins
    r_bins = np.logspace(
        np.log10(CONFIG['r_min_r200']),
        np.log10(CONFIG['r_max_r200']),
        CONFIG['n_radial_bins'] + 1
    )
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    n_bins = len(r_centers)
    
    # ========================================================================
    # Load particle cache
    # ========================================================================
    
    if rank == 0:
        print(f"\nLoading particle cache...")
        t0 = time.time()
    
    cache = load_particle_cache(args.snapshot, args.sim_res)
    
    if cache is None:
        if rank == 0:
            print("ERROR: Particle cache not found!")
            print(f"  Expected: {OUTPUT_DIR}/L205n{args.sim_res}TNG/particle_cache/cache_snap{args.snapshot:03d}.h5")
            print("  Run generate_particle_cache.py first.")
        return
    
    # Get halos from cache
    halo_indices = cache['halo_indices']
    halo_positions = cache['positions']  # Already in Mpc/h
    halo_radii = cache['radii']  # Already in Mpc/h
    halo_masses = cache['masses']  # Already in Msun/h
    n_halos = len(halo_indices)
    
    if rank == 0:
        print(f"  Loaded cache: {n_halos} halos")
        print(f"  DMO particle sets: {len(cache['dmo_particle_ids'])}")
        print(f"  Hydro particle sets: {len(cache['hydro_particle_ids'])}")
        print(f"  Time: {time.time()-t0:.1f}s")
    
    # ========================================================================
    # Distribute halos across ranks
    # ========================================================================
    
    halos_per_rank = n_halos // size
    remainder = n_halos % size
    
    if rank < remainder:
        my_start = rank * (halos_per_rank + 1)
        my_end = my_start + halos_per_rank + 1
    else:
        my_start = rank * halos_per_rank + remainder
        my_end = my_start + halos_per_rank
    
    my_local_indices = np.arange(my_start, my_end)  # Indices into cache arrays
    n_my_halos = len(my_local_indices)
    
    if rank == 0:
        print(f"\nHalo distribution: ~{halos_per_rank} halos per rank")
    
    results = {}
    
    # ========================================================================
    # Compute DMO profiles
    # ========================================================================
    
    if args.mode in ['dmo', 'both']:
        if rank == 0:
            print(f"\n{'='*50}")
            print("Computing DMO profiles...")
            print('='*50)
            t_dmo = time.time()
        
        # Load DMO particles (distributed)
        if rank == 0:
            print(f"  Loading DMO particles...")
            t_load = time.time()
        
        dmo_coords, dmo_ids = load_dmo_particles_distributed(
            sim_config['dmo'], args.snapshot, comm
        )
        dmo_masses_arr = np.full(len(dmo_coords), sim_config['dmo_mass'] * mass_unit, dtype=np.float32)
        
        if rank == 0:
            print(f"    Rank 0: {len(dmo_coords):,} particles")
            print(f"    Load time: {time.time()-t_load:.1f}s")
        
        # Build ID -> index mapping
        if rank == 0:
            print(f"  Building ID->index mapping...")
            t_map = time.time()
        
        dmo_id_to_idx = {pid: idx for idx, pid in enumerate(dmo_ids)}
        
        if rank == 0:
            print(f"    Map time: {time.time()-t_map:.1f}s")
        
        # Compute profiles for my halos
        if rank == 0:
            print(f"  Computing profiles for {n_my_halos} halos per rank...")
            t_prof = time.time()
        
        my_mass_dmo = np.zeros((n_my_halos, n_bins), dtype=np.float64)
        my_npart_dmo = np.zeros((n_my_halos, n_bins), dtype=np.int64)
        
        for i, local_idx in enumerate(my_local_indices):
            halo_idx = halo_indices[local_idx]  # Original halo index in catalog
            center = halo_positions[local_idx]
            r200 = halo_radii[local_idx]
            
            # Get cached particle IDs for this halo
            if halo_idx in cache['dmo_particle_ids']:
                cached_pids = cache['dmo_particle_ids'][halo_idx]
                
                # Find which particles we have locally
                local_particle_indices = [dmo_id_to_idx[pid] for pid in cached_pids if pid in dmo_id_to_idx]
                
                if len(local_particle_indices) > 0:
                    local_particle_indices = np.array(local_particle_indices)
                    p_coords = dmo_coords[local_particle_indices]
                    p_masses = dmo_masses_arr[local_particle_indices]
                    
                    mass_in_bins, n_particles = compute_profile_for_halo(
                        p_coords, p_masses, center, r200, r_bins, box_size
                    )
                    my_mass_dmo[i] = mass_in_bins
                    my_npart_dmo[i] = n_particles
            
            # Progress
            if rank == 0 and (i + 1) % max(1, n_my_halos // 5) == 0:
                print(f"    Halo {i+1}/{n_my_halos}")
        
        if rank == 0:
            print(f"    Profile time: {time.time()-t_prof:.1f}s")
        
        # Free DMO memory
        del dmo_coords, dmo_ids, dmo_masses_arr, dmo_id_to_idx
        
        # Gather to rank 0
        all_mass_dmo = comm.gather(my_mass_dmo, root=0)
        all_npart_dmo = comm.gather(my_npart_dmo, root=0)
        all_local_indices = comm.gather(my_local_indices, root=0)
        
        if rank == 0:
            full_mass_dmo = np.zeros((n_halos, n_bins), dtype=np.float64)
            full_npart_dmo = np.zeros((n_halos, n_bins), dtype=np.int64)
            
            for local_indices, mass, npart in zip(all_local_indices, all_mass_dmo, all_npart_dmo):
                for i, local_idx in enumerate(local_indices):
                    full_mass_dmo[local_idx] = mass[i]
                    full_npart_dmo[local_idx] = npart[i]
            
            # Convert to density
            profiles_dmo = np.zeros((n_halos, n_bins), dtype=np.float32)
            for h in range(n_halos):
                profiles_dmo[h] = mass_to_density(full_mass_dmo[h], halo_radii[h], r_bins)
            
            results['profiles_dmo'] = profiles_dmo
            results['n_particles_dmo'] = full_npart_dmo.astype(np.int32)
            
            print(f"  DMO done in {time.time()-t_dmo:.1f}s")
    
    # ========================================================================
    # Compute Hydro profiles
    # ========================================================================
    
    if args.mode in ['hydro', 'both']:
        if rank == 0:
            print(f"\n{'='*50}")
            print("Computing Hydro profiles...")
            print('='*50)
            t_hydro = time.time()
        
        # Load Hydro particles (distributed)
        if rank == 0:
            print(f"  Loading Hydro particles...")
            t_load = time.time()
        
        hydro_coords, hydro_masses_arr, hydro_ids = load_hydro_particles_distributed(
            sim_config['hydro'], args.snapshot, comm
        )
        
        if rank == 0:
            print(f"    Rank 0: {len(hydro_coords):,} particles")
            print(f"    Load time: {time.time()-t_load:.1f}s")
        
        # Build ID -> index mapping
        if rank == 0:
            print(f"  Building ID->index mapping...")
            t_map = time.time()
        
        hydro_id_to_idx = {pid: idx for idx, pid in enumerate(hydro_ids)}
        
        if rank == 0:
            print(f"    Map time: {time.time()-t_map:.1f}s")
        
        # Compute profiles for my halos
        if rank == 0:
            print(f"  Computing profiles for {n_my_halos} halos per rank...")
            t_prof = time.time()
        
        my_mass_hydro = np.zeros((n_my_halos, n_bins), dtype=np.float64)
        my_npart_hydro = np.zeros((n_my_halos, n_bins), dtype=np.int64)
        
        for i, local_idx in enumerate(my_local_indices):
            halo_idx = halo_indices[local_idx]
            center = halo_positions[local_idx]
            r200 = halo_radii[local_idx]
            
            # Get cached particle IDs for this halo
            if halo_idx in cache['hydro_particle_ids']:
                cached_pids = cache['hydro_particle_ids'][halo_idx]
                
                # Find which particles we have locally
                local_particle_indices = [hydro_id_to_idx[pid] for pid in cached_pids if pid in hydro_id_to_idx]
                
                if len(local_particle_indices) > 0:
                    local_particle_indices = np.array(local_particle_indices)
                    p_coords = hydro_coords[local_particle_indices]
                    p_masses = hydro_masses_arr[local_particle_indices]
                    
                    mass_in_bins, n_particles = compute_profile_for_halo(
                        p_coords, p_masses, center, r200, r_bins, box_size
                    )
                    my_mass_hydro[i] = mass_in_bins
                    my_npart_hydro[i] = n_particles
            
            # Progress
            if rank == 0 and (i + 1) % max(1, n_my_halos // 5) == 0:
                print(f"    Halo {i+1}/{n_my_halos}")
        
        if rank == 0:
            print(f"    Profile time: {time.time()-t_prof:.1f}s")
        
        # Free Hydro memory
        del hydro_coords, hydro_ids, hydro_masses_arr, hydro_id_to_idx
        
        # Gather to rank 0
        all_mass_hydro = comm.gather(my_mass_hydro, root=0)
        all_npart_hydro = comm.gather(my_npart_hydro, root=0)
        all_local_indices = comm.gather(my_local_indices, root=0)
        
        if rank == 0:
            full_mass_hydro = np.zeros((n_halos, n_bins), dtype=np.float64)
            full_npart_hydro = np.zeros((n_halos, n_bins), dtype=np.int64)
            
            for local_indices, mass, npart in zip(all_local_indices, all_mass_hydro, all_npart_hydro):
                for i, local_idx in enumerate(local_indices):
                    full_mass_hydro[local_idx] = mass[i]
                    full_npart_hydro[local_idx] = npart[i]
            
            # Convert to density
            profiles_hydro = np.zeros((n_halos, n_bins), dtype=np.float32)
            for h in range(n_halos):
                profiles_hydro[h] = mass_to_density(full_mass_hydro[h], halo_radii[h], r_bins)
            
            results['profiles_hydro'] = profiles_hydro
            results['n_particles_hydro'] = full_npart_hydro.astype(np.int32)
            
            print(f"  Hydro done in {time.time()-t_hydro:.1f}s")
    
    # ========================================================================
    # Save results
    # ========================================================================
    
    if rank == 0:
        out_dir = f"{OUTPUT_DIR}/L205n{args.sim_res}TNG"
        os.makedirs(out_dir, exist_ok=True)
        
        out_file = f"{out_dir}/profiles_cached_snap{args.snapshot:03d}.h5"
        
        print(f"\nSaving to {out_file}...")
        
        with h5py.File(out_file, 'w') as f:
            # Metadata
            f.attrs['snapshot'] = args.snapshot
            f.attrs['sim_resolution'] = args.sim_res
            f.attrs['n_halos'] = n_halos
            f.attrs['log_mass_min'] = CONFIG['log_mass_min']
            f.attrs['r_max_r200'] = CONFIG['r_max_r200']
            f.attrs['r_min_r200'] = CONFIG['r_min_r200']
            f.attrs['n_radial_bins'] = n_bins
            f.attrs['method'] = 'Particle cache lookup'
            f.attrs['n_ranks'] = size
            f.attrs['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Radial bins
            f.create_dataset('r_bins', data=r_bins)
            f.create_dataset('r_centers', data=r_centers)
            
            # Halo properties (from cache)
            f.create_dataset('halo_indices', data=halo_indices)
            f.create_dataset('halo_masses', data=halo_masses)
            f.create_dataset('halo_positions', data=halo_positions)
            f.create_dataset('halo_radii', data=halo_radii)
            
            # Profiles
            grp = f.create_group('profiles')
            if 'profiles_dmo' in results:
                grp.create_dataset('dmo', data=results['profiles_dmo'])
                grp.create_dataset('n_particles_dmo', data=results['n_particles_dmo'])
            if 'profiles_hydro' in results:
                grp.create_dataset('hydro', data=results['profiles_hydro'])
                grp.create_dataset('n_particles_hydro', data=results['n_particles_hydro'])
        
        print("Done!")
        print("=" * 70)


if __name__ == '__main__':
    main()
