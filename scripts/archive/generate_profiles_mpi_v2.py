#!/usr/bin/env python
"""
MPI-parallel profile generation - Memory-optimized version.

Strategy:
- Each rank processes its assigned halos independently
- For each halo, stream through ALL snapshot chunks
- Query particles near that ONE halo center
- No large intermediate arrays

This avoids the memory explosion from query_ball_point on millions of particles.

Usage:
    mpirun -n 32 python generate_profiles_mpi_v2.py --sim-res 2500 --snapshot 99
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

OUTPUT_DIR = '/mnt/home/mlee1/ceph/hydro_replace_fields'

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


def periodic_distance(coords, center, box_size):
    """Compute periodic distances from coords to center."""
    dx = coords - center
    dx = dx - np.round(dx / box_size) * box_size
    return np.linalg.norm(dx, axis=1)


# ============================================================================
# Memory-Efficient Profile Computation
# ============================================================================

def compute_profiles_for_halos(comm, basePath, snapNum, centers, r200_arr, masses_200,
                               my_halo_indices, box_size, r_bins,
                               particle_types=['dm'], dm_mass=None, mass_unit=1e10,
                               verbose=True):
    """
    Compute profiles for assigned halos by streaming through snapshot data.
    
    Key optimization: For each chunk, build a KDTree of particles and query
    each halo center separately. This avoids the memory explosion from
    building huge lists of nearby particles for all halos at once.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n_my_halos = len(my_halo_indices)
    n_bins = len(r_bins) - 1
    r_max_factor = r_bins[-1]
    
    # Accumulators for MY halos
    mass_in_bins = np.zeros((n_my_halos, n_bins), dtype=np.float64)
    n_particles = np.zeros((n_my_halos, n_bins), dtype=np.int64)
    
    # My halo properties
    my_centers = centers[my_halo_indices]
    my_r200 = r200_arr[my_halo_indices]
    my_r_max = r_max_factor * my_r200
    
    n_chunks = count_snapshot_chunks(basePath, snapNum)
    ptype_map = {'dm': 1, 'gas': 0, 'stars': 4}
    
    total_processed = 0
    
    for ptype_name in particle_types:
        ptNum = ptype_map[ptype_name]
        
        if rank == 0 and verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Processing {ptype_name} ({n_chunks} chunks)...")
        
        t0 = time.time()
        
        for chunk in range(n_chunks):
            try:
                fpath = get_snapshot_path(basePath, snapNum, chunk)
                with h5py.File(fpath, 'r') as f:
                    pgroup = f'PartType{ptNum}'
                    if pgroup not in f:
                        continue
                    
                    # Load particle data
                    coords = f[pgroup]['Coordinates'][:].astype(np.float32) / 1e3  # kpc -> Mpc
                    n_part = len(coords)
                    total_processed += n_part
                    
                    if n_part == 0:
                        continue
                    
                    # Get masses
                    if ptype_name == 'dm' and dm_mass is not None:
                        masses = np.full(n_part, dm_mass * mass_unit, dtype=np.float32)
                    else:
                        masses = f[pgroup]['Masses'][:].astype(np.float32) * mass_unit
                    
                    # Build KDTree of particles in this chunk
                    # Use boxsize for periodic queries
                    particle_tree = cKDTree(coords, boxsize=box_size)
                    
                    # Query each of my halos
                    for local_idx in range(n_my_halos):
                        center = my_centers[local_idx]
                        r_max = my_r_max[local_idx]
                        r200 = my_r200[local_idx]
                        
                        # Find particles within r_max of this halo
                        nearby_idx = particle_tree.query_ball_point(center, r=r_max, workers=1)
                        
                        if len(nearby_idx) == 0:
                            continue
                        
                        nearby_idx = np.array(nearby_idx)
                        p_coords = coords[nearby_idx]
                        p_masses = masses[nearby_idx]
                        
                        # Compute actual distances with periodic BC
                        r = periodic_distance(p_coords, center, box_size)
                        
                        # Bin particles by r/R_200
                        r_scaled = r / r200
                        bin_indices = np.searchsorted(r_bins, r_scaled) - 1
                        valid = (bin_indices >= 0) & (bin_indices < n_bins)
                        
                        if np.any(valid):
                            mass_in_bins[local_idx] += np.bincount(
                                bin_indices[valid], weights=p_masses[valid], minlength=n_bins
                            )
                            n_particles[local_idx] += np.bincount(
                                bin_indices[valid], minlength=n_bins
                            )
                            
            except Exception as e:
                if rank == 0:
                    print(f"    Chunk {chunk} error: {e}")
                continue
            
            # Progress update
            if rank == 0 and verbose and (chunk + 1) % max(1, n_chunks // 5) == 0:
                elapsed = time.time() - t0
                rate = (chunk + 1) / elapsed
                eta = (n_chunks - chunk - 1) / rate if rate > 0 else 0
                print(f"    [{time.strftime('%H:%M:%S')}] Chunk {chunk+1}/{n_chunks}, ETA: {eta:.0f}s")
        
        if rank == 0 and verbose:
            print(f"    {ptype_name} done in {time.time()-t0:.1f}s")
    
    comm.Barrier()
    return mass_in_bins, n_particles, total_processed


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
    parser = argparse.ArgumentParser(description='MPI Profile Generation v2')
    parser.add_argument('--sim-res', type=int, required=True, choices=[625, 1250, 2500])
    parser.add_argument('--snapshot', type=int, required=True)
    parser.add_argument('--mode', choices=['dmo', 'hydro', 'both'], default='both')
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=" * 70)
        print(f"MPI PROFILE GENERATION v2 - L205n{args.sim_res}TNG snap {args.snapshot}")
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
    # Load halo catalogs and matches
    # ========================================================================
    
    if rank == 0:
        print(f"\nLoading halo catalogs and matches...")
        t_load = time.time()
    
    # Load DMO halo catalog
    dmo_halos = load_halo_catalog(
        sim_config['dmo'], args.snapshot,
        fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
    )
    
    # Load Hydro halo catalog  
    hydro_halos = load_halo_catalog(
        sim_config['hydro'], args.snapshot,
        fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
    )
    
    # Load matches
    matches_file = f"{OUTPUT_DIR}/L205n{args.sim_res}TNG/matches/matches_snap{args.snapshot:03d}.npz"
    matches = np.load(matches_file)
    dmo_idx = matches['dmo_indices']
    hydro_idx = matches['hydro_indices']
    
    # Mass cut
    dmo_masses = dmo_halos['Group_M_Crit200'][dmo_idx] * mass_unit
    mass_mask = np.log10(dmo_masses) >= CONFIG['log_mass_min']
    
    sel_dmo_idx = dmo_idx[mass_mask]
    sel_hydro_idx = hydro_idx[mass_mask]
    n_halos = len(sel_dmo_idx)
    
    # Extract properties
    dmo_masses = dmo_halos['Group_M_Crit200'][sel_dmo_idx] * mass_unit
    dmo_positions = dmo_halos['GroupPos'][sel_dmo_idx] / 1e3  # kpc -> Mpc
    dmo_radii = dmo_halos['Group_R_Crit200'][sel_dmo_idx] / 1e3  # kpc -> Mpc
    
    hydro_masses = hydro_halos['Group_M_Crit200'][sel_hydro_idx] * mass_unit
    hydro_positions = hydro_halos['GroupPos'][sel_hydro_idx] / 1e3
    hydro_radii = hydro_halos['Group_R_Crit200'][sel_hydro_idx] / 1e3
    
    if rank == 0:
        print(f"  Loaded in {time.time()-t_load:.1f}s")
        print(f"  Selected {n_halos} halos with log(M) >= {CONFIG['log_mass_min']}")
    
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
    
    my_halo_indices = np.arange(my_start, my_end)
    
    if rank == 0:
        print(f"\nHalo distribution: {halos_per_rank}-{halos_per_rank + (1 if remainder else 0)} halos per rank")
    
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
        
        mass_dmo, npart_dmo, _ = compute_profiles_for_halos(
            comm, sim_config['dmo'], args.snapshot,
            dmo_positions, dmo_radii, dmo_masses, my_halo_indices,
            box_size, r_bins,
            particle_types=['dm'],
            dm_mass=sim_config['dmo_mass'],
            mass_unit=mass_unit
        )
        
        # Gather to rank 0
        all_mass_dmo = comm.gather(mass_dmo, root=0)
        all_npart_dmo = comm.gather(npart_dmo, root=0)
        all_indices = comm.gather(my_halo_indices, root=0)
        
        if rank == 0:
            full_mass_dmo = np.zeros((n_halos, n_bins), dtype=np.float64)
            full_npart_dmo = np.zeros((n_halos, n_bins), dtype=np.int64)
            
            for indices, mass, npart in zip(all_indices, all_mass_dmo, all_npart_dmo):
                for i, global_idx in enumerate(indices):
                    full_mass_dmo[global_idx] = mass[i]
                    full_npart_dmo[global_idx] = npart[i]
            
            profiles_dmo = mass_to_density(full_mass_dmo, dmo_radii, r_bins)
            
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
        
        mass_hydro, npart_hydro, _ = compute_profiles_for_halos(
            comm, sim_config['hydro'], args.snapshot,
            hydro_positions, hydro_radii, hydro_masses, my_halo_indices,
            box_size, r_bins,
            particle_types=['dm'],  # Can add 'gas', 'stars' later
            dm_mass=sim_config['hydro_dm_mass'],
            mass_unit=mass_unit
        )
        
        # Gather to rank 0
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
    # Save results
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
            f.attrs['method'] = 'MPI streaming v2 (per-halo KDTree query)'
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
        
        print("Done!")
        print("=" * 70)


if __name__ == '__main__':
    main()
