#!/usr/bin/env python
"""
MPI-parallel profile generation using streaming approach.

Strategy:
- All ranks stream through ALL snapshot data together
- Each rank only accumulates profiles for its assigned halos
- Final gather to rank 0 for saving

This is O(data_size) regardless of N_halos, and parallelizes perfectly.

Usage:
    mpirun -n 64 python generate_profiles_mpi.py --sim-res 625 --snapshot 99
    mpirun -n 128 python generate_profiles_mpi.py --sim-res 2500 --snapshot 99 --mode both
"""

import numpy as np
import h5py
import os
import sys
import time
import argparse
import glob
from mpi4py import MPI
from scipy.spatial import cKDTree

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

FIELDS_DIR = '/mnt/home/mlee1/ceph/hydro_replace_fields'
OUTPUT_DIR = '/mnt/home/mlee1/ceph/hydro_replace_profiles'

# ============================================================================
# Helper Functions
# ============================================================================

def get_snapshot_path(basePath, snapNum, chunkNum=0):
    """Get path to snapshot chunk file."""
    snapPath = f"{basePath}/snapdir_{snapNum:03d}/"
    filePath1 = f"{snapPath}snap_{snapNum:03d}.{chunkNum}.hdf5"
    filePath2 = filePath1.replace('/snap_', '/snapshot_')
    
    if os.path.isfile(filePath1):
        return filePath1
    return filePath2


def count_snapshot_chunks(basePath, snapNum):
    """Count number of snapshot chunks."""
    search_pattern = f"{basePath}/snapdir_{snapNum:03d}/snap*.hdf5"
    files = glob.glob(search_pattern)
    if not files:
        search_pattern = f"{basePath}/snapdir_{snapNum:03d}/snapshot*.hdf5"
        files = glob.glob(search_pattern)
    return len(files)


def load_halo_catalog(basePath, snapNum, fields):
    """Load halo catalog fields."""
    from illustris_python import groupcat
    return groupcat.loadHalos(basePath, snapNum, fields=fields)


# ============================================================================
# MPI Streaming Profile Computation
# ============================================================================

def compute_profiles_mpi_streaming(comm, basePath, snapNum, centers, r200_arr, 
                                   my_halo_indices, box_size, r_bins,
                                   particle_types=['dm'], dm_mass=None, 
                                   mass_unit=1e10):
    """
    MPI-parallel streaming profile computation.
    
    All ranks stream through ALL data together.
    Each rank only accumulates for its assigned halos.
    
    Parameters:
    -----------
    comm : MPI communicator
    basePath : str
        Path to simulation
    snapNum : int
        Snapshot number
    centers : array (N_total_halos, 3)
        ALL halo centers (needed for KDTree)
    r200_arr : array (N_total_halos,)
        ALL R_200 values
    my_halo_indices : array
        Indices of halos THIS rank is responsible for
    box_size : float
    r_bins : array
    particle_types : list
    dm_mass : float or None
    mass_unit : float
    
    Returns:
    --------
    mass_in_bins : (N_my_halos, N_bins)
    n_particles : (N_my_halos, N_bins)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n_total_halos = len(centers)
    n_my_halos = len(my_halo_indices)
    n_bins = len(r_bins) - 1
    r_max_factor = r_bins[-1]
    
    # Pre-compute r_max for ALL halos (needed for KDTree queries)
    r_max_arr = r_max_factor * r200_arr
    max_r = np.max(r_max_arr)
    
    # Accumulators for MY halos only
    mass_in_bins = np.zeros((n_my_halos, n_bins), dtype=np.float64)
    n_particles = np.zeros((n_my_halos, n_bins), dtype=np.int64)
    
    # Create mapping from global halo index to local index
    global_to_local = {g: l for l, g in enumerate(my_halo_indices)}
    my_halo_set = set(my_halo_indices)
    
    # Build KDTree of ALL halo centers
    halo_tree = cKDTree(centers, boxsize=box_size)
    
    n_chunks = count_snapshot_chunks(basePath, snapNum)
    ptype_map = {'dm': 1, 'gas': 0, 'stars': 4}
    
    total_processed = 0
    total_assigned = 0
    
    for ptype_name in particle_types:
        ptNum = ptype_map[ptype_name]
        
        if rank == 0:
            print(f"  [{time.strftime('%H:%M:%S')}] Streaming {ptype_name} ({n_chunks} chunks)...")
        
        t0 = time.time()
        
        for chunk in range(n_chunks):
            try:
                fpath = get_snapshot_path(basePath, snapNum, chunk)
                with h5py.File(fpath, 'r') as f:
                    pgroup = f'PartType{ptNum}'
                    if pgroup not in f:
                        continue
                    
                    coords = f[pgroup]['Coordinates'][:].astype(np.float32) / 1e3
                    n_part = len(coords)
                    total_processed += n_part
                    
                    if n_part == 0:
                        continue
                    
                    if ptype_name == 'dm' and dm_mass is not None:
                        masses = np.full(n_part, dm_mass * mass_unit, dtype=np.float32)
                    else:
                        masses = f[pgroup]['Masses'][:].astype(np.float32) * mass_unit
                    
                    # Find particles close to ANY halo
                    distances, nearest = halo_tree.query(coords, k=1, workers=1)
                    close_mask = distances < max_r
                    
                    if not np.any(close_mask):
                        continue
                    
                    close_coords = coords[close_mask]
                    close_masses = masses[close_mask]
                    
                    # Find all nearby halos for close particles
                    nearby_lists = halo_tree.query_ball_point(close_coords, r=max_r, workers=1)
                    
                    # Process only MY halos
                    for local_idx, global_idx in enumerate(my_halo_indices):
                        # Find particles near this halo
                        particle_indices = [i for i, hlist in enumerate(nearby_lists) 
                                          if global_idx in hlist]
                        
                        if len(particle_indices) == 0:
                            continue
                        
                        p_idx = np.array(particle_indices)
                        p_coords = close_coords[p_idx]
                        p_masses = close_masses[p_idx]
                        
                        # Compute distances
                        dx = p_coords - centers[global_idx]
                        dx = dx - np.round(dx / box_size) * box_size
                        r = np.linalg.norm(dx, axis=1)
                        
                        # Filter by this halo's r_max
                        in_range = r < r_max_arr[global_idx]
                        if not np.any(in_range):
                            continue
                        
                        r = r[in_range]
                        p_masses = p_masses[in_range]
                        
                        # Bin particles
                        r_scaled = r / r200_arr[global_idx]
                        bin_indices = np.searchsorted(r_bins, r_scaled) - 1
                        valid = (bin_indices >= 0) & (bin_indices < n_bins)
                        
                        if np.any(valid):
                            mass_in_bins[local_idx] += np.bincount(
                                bin_indices[valid], weights=p_masses[valid], minlength=n_bins
                            )
                            n_particles[local_idx] += np.bincount(
                                bin_indices[valid], minlength=n_bins
                            )
                            total_assigned += valid.sum()
                            
            except Exception as e:
                if rank == 0:
                    print(f"    Chunk {chunk} error: {e}")
                continue
            
            # Progress update every 10 chunks
            if rank == 0 and (chunk + 1) % max(1, n_chunks // 10) == 0:
                print(f"    [{time.strftime('%H:%M:%S')}] Chunk {chunk+1}/{n_chunks}")
        
        if rank == 0:
            print(f"    {ptype_name} done in {time.time()-t0:.1f}s")
    
    # Sync before returning
    comm.Barrier()
    
    return mass_in_bins, n_particles, total_processed, total_assigned


def mass_to_density(mass_in_bins, r200_arr, r_bins):
    """Convert accumulated mass to density profiles."""
    n_halos = len(mass_in_bins)
    n_bins = len(r_bins) - 1
    
    profiles = np.zeros((n_halos, n_bins), dtype=np.float64)
    
    for h_idx in range(n_halos):
        r200 = r200_arr[h_idx]
        for b_idx in range(n_bins):
            r_inner = r_bins[b_idx] * r200
            r_outer = r_bins[b_idx + 1] * r200
            volume = 4/3 * np.pi * (r_outer**3 - r_inner**3)
            if volume > 0:
                profiles[h_idx, b_idx] = mass_in_bins[h_idx, b_idx] / volume
    
    return profiles.astype(np.float32)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MPI Profile Generation')
    parser.add_argument('--sim-res', type=int, default=625, choices=[625, 1250, 2500])
    parser.add_argument('--snapshot', type=int, default=99)
    parser.add_argument('--mode', choices=['dmo', 'hydro', 'both'], default='both')
    parser.add_argument('--max-halos', type=int, default=None, help='Limit halos for testing')
    args = parser.parse_args()
    
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
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
    n_bins = len(r_bins) - 1
    
    if rank == 0:
        print("=" * 70)
        print(f"MPI PROFILE GENERATION - L205n{args.sim_res}TNG snap {args.snapshot}")
        print("=" * 70)
        print(f"  Ranks: {size}")
        print(f"  Mode: {args.mode}")
        print(f"  Radial bins: {n_bins}")
        print()
    
    # ========================================================================
    # Load halo catalogs and matches (rank 0 only, then broadcast)
    # ========================================================================
    
    if rank == 0:
        print("Loading halo catalogs and matches...")
        t0 = time.time()
        
        # Load DMO halos
        halo_dmo = load_halo_catalog(
            sim_config['dmo'], args.snapshot,
            fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
        )
        
        # Load Hydro halos
        halo_hydro = load_halo_catalog(
            sim_config['hydro'], args.snapshot,
            fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
        )
        
        # Load matches
        matches_file = f"{FIELDS_DIR}/L205n{args.sim_res}TNG/matches/matches_snap{args.snapshot:03d}.npz"
        if not os.path.exists(matches_file):
            print(f"ERROR: No matches file at {matches_file}")
            comm.Abort(1)
        
        matches = np.load(matches_file)
        matched_dmo_idx = matches['dmo_indices']
        matched_hydro_idx = matches['hydro_indices']
        
        # Apply mass filter
        dmo_masses_all = halo_dmo['Group_M_Crit200'][matched_dmo_idx] * mass_unit
        log_masses = np.log10(dmo_masses_all + 1e-10)
        mass_mask = log_masses >= CONFIG['log_mass_min']
        
        selected_indices = np.where(mass_mask)[0]
        if args.max_halos is not None:
            selected_indices = selected_indices[:args.max_halos]
        
        n_halos = len(selected_indices)
        
        # Get selected halo properties
        sel_dmo_idx = matched_dmo_idx[selected_indices]
        sel_hydro_idx = matched_hydro_idx[selected_indices]
        
        dmo_positions = (halo_dmo['GroupPos'][sel_dmo_idx] / 1e3).astype(np.float32)
        dmo_radii = (halo_dmo['Group_R_Crit200'][sel_dmo_idx] / 1e3).astype(np.float32)
        dmo_masses = (halo_dmo['Group_M_Crit200'][sel_dmo_idx] * mass_unit).astype(np.float64)
        
        hydro_positions = (halo_hydro['GroupPos'][sel_hydro_idx] / 1e3).astype(np.float32)
        hydro_radii = (halo_hydro['Group_R_Crit200'][sel_hydro_idx] / 1e3).astype(np.float32)
        hydro_masses = (halo_hydro['Group_M_Crit200'][sel_hydro_idx] * mass_unit).astype(np.float64)
        
        print(f"  Loaded in {time.time()-t0:.1f}s")
        print(f"  Selected {n_halos} halos with log(M) >= {CONFIG['log_mass_min']}")
    else:
        n_halos = None
        dmo_positions = None
        dmo_radii = None
        dmo_masses = None
        hydro_positions = None
        hydro_radii = None
        hydro_masses = None
        sel_dmo_idx = None
        sel_hydro_idx = None
    
    # Broadcast metadata
    n_halos = comm.bcast(n_halos, root=0)
    
    # Allocate arrays on other ranks
    if rank != 0:
        dmo_positions = np.empty((n_halos, 3), dtype=np.float32)
        dmo_radii = np.empty(n_halos, dtype=np.float32)
        dmo_masses = np.empty(n_halos, dtype=np.float64)
        hydro_positions = np.empty((n_halos, 3), dtype=np.float32)
        hydro_radii = np.empty(n_halos, dtype=np.float32)
        hydro_masses = np.empty(n_halos, dtype=np.float64)
        sel_dmo_idx = np.empty(n_halos, dtype=np.int64)
        sel_hydro_idx = np.empty(n_halos, dtype=np.int64)
    
    # Broadcast arrays
    comm.Bcast(dmo_positions, root=0)
    comm.Bcast(dmo_radii, root=0)
    comm.Bcast(dmo_masses, root=0)
    comm.Bcast(hydro_positions, root=0)
    comm.Bcast(hydro_radii, root=0)
    comm.Bcast(hydro_masses, root=0)
    comm.Bcast(sel_dmo_idx, root=0)
    comm.Bcast(sel_hydro_idx, root=0)
    
    # ========================================================================
    # Distribute halos across ranks
    # ========================================================================
    
    halos_per_rank = n_halos // size
    remainder = n_halos % size
    
    if rank < remainder:
        my_start = rank * (halos_per_rank + 1)
        my_count = halos_per_rank + 1
    else:
        my_start = rank * halos_per_rank + remainder
        my_count = halos_per_rank
    
    my_halo_indices = np.arange(my_start, my_start + my_count)
    
    if rank == 0:
        print(f"\nHalo distribution: {halos_per_rank}-{halos_per_rank+1} halos per rank")
    
    # ========================================================================
    # Compute profiles
    # ========================================================================
    
    results = {}
    
    # DMO profiles
    if args.mode in ['dmo', 'both']:
        if rank == 0:
            print(f"\n{'='*50}")
            print("Computing DMO profiles...")
            print(f"{'='*50}")
        
        t_dmo = time.time()
        
        mass_dmo, npart_dmo, proc_dmo, assign_dmo = compute_profiles_mpi_streaming(
            comm, sim_config['dmo'], args.snapshot,
            dmo_positions, dmo_radii, my_halo_indices,
            box_size, r_bins,
            particle_types=['dm'],
            dm_mass=sim_config['dmo_mass'],
            mass_unit=mass_unit
        )
        
        # Gather results to rank 0
        all_mass_dmo = comm.gather(mass_dmo, root=0)
        all_npart_dmo = comm.gather(npart_dmo, root=0)
        all_indices = comm.gather(my_halo_indices, root=0)
        
        if rank == 0:
            # Reconstruct full arrays
            full_mass_dmo = np.zeros((n_halos, n_bins), dtype=np.float64)
            full_npart_dmo = np.zeros((n_halos, n_bins), dtype=np.int64)
            
            for indices, mass, npart in zip(all_indices, all_mass_dmo, all_npart_dmo):
                for i, global_idx in enumerate(indices):
                    full_mass_dmo[global_idx] = mass[i]
                    full_npart_dmo[global_idx] = npart[i]
            
            # Convert to density
            profiles_dmo = mass_to_density(full_mass_dmo, dmo_radii, r_bins)
            
            results['profiles_dmo'] = profiles_dmo
            results['n_particles_dmo'] = full_npart_dmo.astype(np.int32)
            
            print(f"  DMO done in {time.time()-t_dmo:.1f}s")
    
    # Hydro profiles
    if args.mode in ['hydro', 'both']:
        if rank == 0:
            print(f"\n{'='*50}")
            print("Computing Hydro profiles...")
            print(f"{'='*50}")
        
        t_hydro = time.time()
        
        mass_hydro, npart_hydro, proc_hydro, assign_hydro = compute_profiles_mpi_streaming(
            comm, sim_config['hydro'], args.snapshot,
            hydro_positions, hydro_radii, my_halo_indices,
            box_size, r_bins,
            particle_types=['dm', 'gas', 'stars'],
            dm_mass=sim_config['hydro_dm_mass'],
            mass_unit=mass_unit
        )
        
        # Gather results
        all_mass_hydro = comm.gather(mass_hydro, root=0)
        all_npart_hydro = comm.gather(npart_hydro, root=0)
        all_indices = comm.gather(my_halo_indices, root=0)
        
        if rank == 0:
            full_mass_hydro = np.zeros((n_halos, n_bins), dtype=np.float64)
            full_npart_hydro = np.zeros((n_halos, n_bins), dtype=np.int64)
            
            for indices, mass, npart in zip(all_indices, all_mass_hydro, all_npart_hydro):
                for i, global_idx in enumerate(indices):
                    full_mass_hydro[global_idx] = mass[i]
                    full_npart_hydro[global_idx] = npart[i]
            
            profiles_hydro = mass_to_density(full_mass_hydro, hydro_radii, r_bins)
            
            results['profiles_hydro'] = profiles_hydro
            results['n_particles_hydro'] = full_npart_hydro.astype(np.int32)
            
            print(f"  Hydro done in {time.time()-t_hydro:.1f}s")
    
    # ========================================================================
    # Save results (rank 0 only)
    # ========================================================================
    
    if rank == 0:
        out_dir = f"{OUTPUT_DIR}/L205n{args.sim_res}TNG"
        os.makedirs(out_dir, exist_ok=True)
        
        out_file = f"{out_dir}/profiles_spherical_snap{args.snapshot:03d}.h5"
        
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
            f.attrs['method'] = 'MPI streaming spherical aperture'
            f.attrs['n_ranks'] = size
            f.attrs['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Radial bins
            f.create_dataset('r_bins', data=r_bins)
            f.create_dataset('r_centers', data=r_centers)
            
            # Halo properties
            f.create_dataset('dmo_halo_indices', data=sel_dmo_idx)
            f.create_dataset('hydro_halo_indices', data=sel_hydro_idx)
            f.create_dataset('dmo_masses', data=dmo_masses)
            f.create_dataset('hydro_masses', data=hydro_masses)
            f.create_dataset('dmo_positions', data=dmo_positions)
            f.create_dataset('hydro_positions', data=hydro_positions)
            f.create_dataset('dmo_radii', data=dmo_radii)
            f.create_dataset('hydro_radii', data=hydro_radii)
            
            # Profiles
            grp = f.create_group('profiles')
            if 'profiles_dmo' in results:
                grp.create_dataset('dmo', data=results['profiles_dmo'])
                grp.create_dataset('n_particles_dmo', data=results['n_particles_dmo'])
            if 'profiles_hydro' in results:
                grp.create_dataset('hydro', data=results['profiles_hydro'])
                grp.create_dataset('n_particles_hydro', data=results['n_particles_hydro'])
        
        print(f"  Saved {n_halos} profiles with shape ({n_halos}, {n_bins})")
        print("\n" + "=" * 70)
        print("DONE!")
        print("=" * 70)


if __name__ == '__main__':
    main()
