#!/usr/bin/env python
"""
Generate particle ID lookup tables for halos.

This creates a cache of particle IDs within 5×R200 for each matched halo,
enabling fast particle queries for profiles, lens planes, and BCM without
rebuilding KD-trees every time.

Output format: HDF5 with structure:
  /dmo/halo_{idx}/particle_ids  - DMO particle IDs
  /hydro/halo_{idx}/particle_ids - Hydro particle IDs
  /halo_info/positions, radii, masses - Halo properties

Usage:
    mpirun -np 64 python generate_particle_cache.py --sim-res 2500 --snap 99
    mpirun -np 64 python generate_particle_cache.py --sim-res 2500 --snap all
"""

import numpy as np
import h5py
import argparse
import os
import time
import glob
from mpi4py import MPI
from scipy.spatial import cKDTree

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
    'box_size': 205.0,  # Mpc/h
    'radius_multiplier': 5.0,  # Cache particles within 5×R200
    'output_base': '/mnt/home/mlee1/ceph/hydro_replace_fields',
    'min_mass': 12.0,  # log10(Msun/h) - cache halos above this mass
}


def load_matched_halos(matches_file):
    """Load matched halo information."""
    with np.load(matches_file) as data:
        result = {
            'dmo_indices': data['dmo_indices'],
            'hydro_indices': data['hydro_indices'],
            'dmo_masses': data['dmo_masses'],
            'dmo_radii': data['dmo_radii'],
            'dmo_positions': data['dmo_positions'],
        }
        # Also load hydro positions/radii if available (for proper hydro profiles)
        if 'hydro_positions' in data:
            result['hydro_positions'] = data['hydro_positions']
            result['hydro_radii'] = data['hydro_radii']
        return result


def load_particles_chunk(files, parttype=1):
    """Load particle coordinates and IDs from a set of files."""
    coords_list = []
    ids_list = []
    
    for filepath in files:
        with h5py.File(filepath, 'r') as f:
            pt_key = f'PartType{parttype}'
            if pt_key not in f:
                continue
            
            # Load coordinates (convert kpc -> Mpc)
            coords = f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3
            # Load particle IDs
            pids = f[pt_key]['ParticleIDs'][:]
            
            coords_list.append(coords)
            ids_list.append(pids)
    
    if len(coords_list) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int64)
    
    return np.concatenate(coords_list), np.concatenate(ids_list)


def load_hydro_particles_chunk(files):
    """Load all hydro particle types (gas, DM, stars)."""
    all_coords = []
    all_ids = []
    
    for filepath in files:
        with h5py.File(filepath, 'r') as f:
            for parttype in [0, 1, 4]:  # Gas, DM, Stars
                pt_key = f'PartType{parttype}'
                if pt_key not in f:
                    continue
                
                coords = f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3
                pids = f[pt_key]['ParticleIDs'][:]
                
                all_coords.append(coords)
                all_ids.append(pids)
    
    if len(all_coords) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int64)
    
    return np.concatenate(all_coords), np.concatenate(all_ids)


def query_particles_around_halos(coords, pids, halo_positions, halo_radii, 
                                  radius_mult, box_size):
    """
    Query particles within radius_mult × R200 for each halo.
    
    Handles periodic boundary conditions by querying with wrapped positions
    for halos near box edges.
    
    Returns dict: halo_idx -> particle_ids
    """
    if len(coords) == 0:
        return {}
    
    # Build KD-tree (this is the expensive operation we want to do once)
    tree = cKDTree(coords)
    
    halo_particles = {}
    
    for halo_idx, (pos, r200) in enumerate(zip(halo_positions, halo_radii)):
        r_query = radius_mult * r200
        
        # Check if halo is near a box boundary (within r_query of edge)
        near_edge = np.any(pos < r_query) or np.any(pos > box_size - r_query)
        
        if near_edge:
            # Query with wrapped positions (up to 8 copies for corner halos)
            all_indices = set()
            
            # Generate offsets: 0, +box_size, -box_size for each axis if near that edge
            offsets = []
            for axis in range(3):
                axis_offsets = [0.0]
                if pos[axis] < r_query:
                    axis_offsets.append(box_size)  # Wrap from other side
                if pos[axis] > box_size - r_query:
                    axis_offsets.append(-box_size)  # Wrap to other side
                offsets.append(axis_offsets)
            
            # Generate all combinations of offsets
            from itertools import product
            for ox, oy, oz in product(offsets[0], offsets[1], offsets[2]):
                wrapped_pos = pos + np.array([ox, oy, oz])
                indices = tree.query_ball_point(wrapped_pos, r_query)
                all_indices.update(indices)
            
            if len(all_indices) > 0:
                halo_particles[halo_idx] = pids[list(all_indices)]
        else:
            # Simple case: no boundary wrapping needed
            indices = tree.query_ball_point(pos, r_query)
            if len(indices) > 0:
                halo_particles[halo_idx] = pids[indices]
    
    return halo_particles


def process_snapshot(args, snapNum):
    """Process a single snapshot."""
    sim_config = SIM_PATHS[args.sim_res]
    dmo_base = sim_config['dmo']
    hydro_base = sim_config['hydro']
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Processing Snapshot {snapNum}")
        print(f"  Simulation: L205n{args.sim_res}TNG")
        print(f"  Radius: {CONFIG['radius_multiplier']}×R200")
        print(f"{'='*70}")
    
    # ========================================================================
    # Load matched halos
    # ========================================================================
    matches_file = os.path.join(
        CONFIG['output_base'],
        f'L205n{args.sim_res}TNG',
        'matches',
        f'matches_snap{snapNum:03d}.npz'
    )
    
    if not os.path.exists(matches_file):
        if rank == 0:
            print(f"ERROR: Matches file not found: {matches_file}")
        return
    
    if rank == 0:
        print(f"\n[1/5] Loading matched halos...")
        t0 = time.time()
    
    matches = load_matched_halos(matches_file)
    
    # Filter by mass (masses in matches file are in 10^10 Msun/h units)
    masses_msun_h = matches['dmo_masses'] * 1e10  # Convert to Msun/h
    min_mass_linear = 10**CONFIG['min_mass']
    mass_mask = masses_msun_h >= min_mass_linear
    
    halo_indices_dmo = matches['dmo_indices'][mass_mask]
    halo_positions_dmo = matches['dmo_positions'][mass_mask] / 1e3  # Convert kpc/h -> Mpc/h
    halo_radii_dmo = matches['dmo_radii'][mass_mask] / 1e3  # Convert kpc/h -> Mpc/h
    halo_masses = masses_msun_h[mass_mask]  # Now in Msun/h
    
    # Also get hydro positions for proper hydro profile comparison
    if 'hydro_positions' in matches:
        halo_positions_hydro = matches['hydro_positions'][mass_mask] / 1e3
        halo_radii_hydro = matches['hydro_radii'][mass_mask] / 1e3
        has_hydro_positions = True
    else:
        halo_positions_hydro = None
        halo_radii_hydro = None
        has_hydro_positions = False
    
    if rank == 0:
        print(f"  Total matched halos: {len(matches['dmo_indices'])}")
        print(f"  Halos above M > 10^{CONFIG['min_mass']}: {len(halo_indices_dmo)}")
        print(f"  Hydro positions available: {has_hydro_positions}")
        print(f"  Time: {time.time()-t0:.1f}s")
    
    # ========================================================================
    # Load DMO particles (distributed across ranks by file)
    # ========================================================================
    if rank == 0:
        print(f"\n[2/5] Loading DMO particles (distributed)...")
        t0 = time.time()
    
    dmo_dir = f"{dmo_base}/snapdir_{snapNum:03d}/"
    dmo_files = sorted(glob.glob(f"{dmo_dir}/snap_{snapNum:03d}.*.hdf5"))
    my_dmo_files = [f for i, f in enumerate(dmo_files) if i % size == rank]
    
    dmo_coords, dmo_pids = load_particles_chunk(my_dmo_files, parttype=1)
    
    if rank == 0:
        print(f"  Rank 0: {len(dmo_coords):,} DMO particles (from {len(my_dmo_files)} files)")
        print(f"  Time: {time.time()-t0:.1f}s")
    
    # ========================================================================
    # Query DMO particles around ALL halos (each rank contributes what it has)
    # ========================================================================
    if rank == 0:
        print(f"\n[3/5] Querying DMO particles around ALL halos...")
        t0 = time.time()
    
    # Each rank queries ALL halos with its subset of particles
    # This finds particles near halos that happen to be in this rank's files
    dmo_halo_particles = query_particles_around_halos(
        dmo_coords, dmo_pids,
        halo_positions_dmo,  # DMO halo centers
        halo_radii_dmo,       # DMO R200
        CONFIG['radius_multiplier'],
        CONFIG['box_size']
    )
    
    # Free DMO memory before loading hydro
    del dmo_coords, dmo_pids
    
    if rank == 0:
        n_with_particles = sum(1 for v in dmo_halo_particles.values() if len(v) > 0)
        print(f"  Rank 0 found particles for {n_with_particles} halos")
        print(f"  Time: {time.time()-t0:.1f}s")
    
    # ========================================================================
    # Load Hydro particles (distributed across ranks by file)
    # ========================================================================
    if rank == 0:
        print(f"\n[4/5] Loading Hydro particles (distributed)...")
        t0 = time.time()
    
    hydro_dir = f"{hydro_base}/snapdir_{snapNum:03d}/"
    hydro_files = sorted(glob.glob(f"{hydro_dir}/snap_{snapNum:03d}.*.hdf5"))
    my_hydro_files = [f for i, f in enumerate(hydro_files) if i % size == rank]
    
    hydro_coords, hydro_pids = load_hydro_particles_chunk(my_hydro_files)
    
    if rank == 0:
        print(f"  Rank 0: {len(hydro_coords):,} Hydro particles (from {len(my_hydro_files)} files)")
        print(f"  Time: {time.time()-t0:.1f}s")
    
    # ========================================================================
    # Query Hydro particles around DMO halo centers (for replacement)
    # ========================================================================
    if rank == 0:
        print(f"\n[5/6] Querying Hydro particles around DMO halo centers (for replacement)...")
        t0 = time.time()
    
    # Each rank queries ALL halos with its subset of particles
    # NOTE: This queries hydro particles at DMO positions - exactly what we need for replacement!
    hydro_at_dmo_particles = query_particles_around_halos(
        hydro_coords, hydro_pids,
        halo_positions_dmo,  # DMO halo centers (for replacement)
        halo_radii_dmo,       # DMO R200
        CONFIG['radius_multiplier'],
        CONFIG['box_size']
    )
    
    if rank == 0:
        n_with_particles = sum(1 for v in hydro_at_dmo_particles.values() if len(v) > 0)
        print(f"  Rank 0 found particles for {n_with_particles} halos")
        print(f"  Time: {time.time()-t0:.1f}s")
    
    # ========================================================================
    # Query Hydro particles around HYDRO halo centers (for true hydro profiles)
    # ========================================================================
    hydro_at_hydro_particles = {}
    if has_hydro_positions:
        if rank == 0:
            print(f"\n[6/7] Querying Hydro particles around Hydro halo centers (for profiles)...")
            t0 = time.time()
        
        hydro_at_hydro_particles = query_particles_around_halos(
            hydro_coords, hydro_pids,
            halo_positions_hydro,  # Hydro halo centers (true positions)
            halo_radii_hydro,       # Hydro R200
            CONFIG['radius_multiplier'],
            CONFIG['box_size']
        )
        
        if rank == 0:
            n_with_particles = sum(1 for v in hydro_at_hydro_particles.values() if len(v) > 0)
            print(f"  Rank 0 found particles for {n_with_particles} halos")
            print(f"  Time: {time.time()-t0:.1f}s")
    
    # Free memory
    del hydro_coords, hydro_pids
    
    # ========================================================================
    # Write each rank's data to temporary file, then merge (avoids MPI gather issues)
    # ========================================================================
    step = 7 if has_hydro_positions else 6
    if rank == 0:
        print(f"\\n[{step}/{step}] Writing temporary files and merging...")
        t0 = time.time()
    
    # Output directory
    output_dir = os.path.join(
        CONFIG['output_base'],
        f'L205n{args.sim_res}TNG',
        'particle_cache'
    )
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    comm.Barrier()
    
    # Each rank writes its own temporary file
    temp_file = os.path.join(output_dir, f'temp_snap{snapNum:03d}_rank{rank:04d}.h5')
    
    with h5py.File(temp_file, 'w') as f:
        # DMO particle IDs (DMO particles at DMO halo centers)
        dmo_grp = f.create_group('dmo')
        for halo_idx, pids in dmo_halo_particles.items():
            if len(pids) > 0:
                dmo_grp.create_dataset(f'halo_{halo_idx}', data=pids)
        
        # Hydro particle IDs at DMO centers (for replacement)
        hydro_at_dmo_grp = f.create_group('hydro_at_dmo')
        for halo_idx, pids in hydro_at_dmo_particles.items():
            if len(pids) > 0:
                hydro_at_dmo_grp.create_dataset(f'halo_{halo_idx}', data=pids)
        
        # Hydro particle IDs at Hydro centers (for true profiles)
        if has_hydro_positions:
            hydro_at_hydro_grp = f.create_group('hydro_at_hydro')
            for halo_idx, pids in hydro_at_hydro_particles.items():
                if len(pids) > 0:
                    hydro_at_hydro_grp.create_dataset(f'halo_{halo_idx}', data=pids)
    
    # Free local memory
    del dmo_halo_particles, hydro_at_dmo_particles, hydro_at_hydro_particles
    
    comm.Barrier()
    
    # Rank 0 merges all temporary files
    if rank == 0:
        print(f"  Merging data from {size} ranks...")
        
        # Initialize caches
        dmo_cache = {}
        hydro_at_dmo_cache = {}
        hydro_at_hydro_cache = {}
        
        # Read and merge all temporary files
        for r in range(size):
            temp_file_r = os.path.join(output_dir, f'temp_snap{snapNum:03d}_rank{r:04d}.h5')
            with h5py.File(temp_file_r, 'r') as f:
                # Merge DMO
                if 'dmo' in f:
                    for key in f['dmo'].keys():
                        halo_idx = int(key.split('_')[1])
                        pids = f['dmo'][key][:]
                        if halo_idx in dmo_cache:
                            dmo_cache[halo_idx] = np.concatenate([dmo_cache[halo_idx], pids])
                        else:
                            dmo_cache[halo_idx] = pids
                
                # Merge Hydro at DMO centers (for replacement)
                if 'hydro_at_dmo' in f:
                    for key in f['hydro_at_dmo'].keys():
                        halo_idx = int(key.split('_')[1])
                        pids = f['hydro_at_dmo'][key][:]
                        if halo_idx in hydro_at_dmo_cache:
                            hydro_at_dmo_cache[halo_idx] = np.concatenate([hydro_at_dmo_cache[halo_idx], pids])
                        else:
                            hydro_at_dmo_cache[halo_idx] = pids
                
                # Merge Hydro at Hydro centers (for true profiles)
                if 'hydro_at_hydro' in f:
                    for key in f['hydro_at_hydro'].keys():
                        halo_idx = int(key.split('_')[1])
                        pids = f['hydro_at_hydro'][key][:]
                        if halo_idx in hydro_at_hydro_cache:
                            hydro_at_hydro_cache[halo_idx] = np.concatenate([hydro_at_hydro_cache[halo_idx], pids])
                        else:
                            hydro_at_hydro_cache[halo_idx] = pids
            
            # Delete temp file after reading
            os.remove(temp_file_r)
        
        print(f"  DMO halos with particles: {len(dmo_cache)}")
        print(f"  Hydro@DMO halos with particles: {len(hydro_at_dmo_cache)}")
        print(f"  Hydro@Hydro halos with particles: {len(hydro_at_hydro_cache)}")
        
        # Write final cache file
        output_file = os.path.join(output_dir, f'cache_snap{snapNum:03d}.h5')
        
        with h5py.File(output_file, 'w') as f:
            # Store halo information (DMO positions/radii used for queries)
            halo_grp = f.create_group('halo_info')
            halo_grp.create_dataset('halo_indices', data=halo_indices_dmo)
            halo_grp.create_dataset('positions_dmo', data=halo_positions_dmo)
            halo_grp.create_dataset('radii_dmo', data=halo_radii_dmo)
            halo_grp.create_dataset('masses', data=halo_masses)
            
            # Also store hydro positions if available
            if has_hydro_positions:
                halo_grp.create_dataset('positions_hydro', data=halo_positions_hydro)
                halo_grp.create_dataset('radii_hydro', data=halo_radii_hydro)
            
            # Store DMO particle IDs (at DMO centers)
            dmo_grp = f.create_group('dmo')
            for halo_idx, pids in dmo_cache.items():
                dmo_grp.create_dataset(f'halo_{halo_idx}', data=pids, compression='gzip')
            
            # Store Hydro particle IDs at DMO centers (for replacement)
            hydro_at_dmo_grp = f.create_group('hydro_at_dmo')
            for halo_idx, pids in hydro_at_dmo_cache.items():
                hydro_at_dmo_grp.create_dataset(f'halo_{halo_idx}', data=pids, compression='gzip')
            
            # Store Hydro particle IDs at Hydro centers (for true profiles)
            if has_hydro_positions:
                hydro_at_hydro_grp = f.create_group('hydro_at_hydro')
                for halo_idx, pids in hydro_at_hydro_cache.items():
                    hydro_at_hydro_grp.create_dataset(f'halo_{halo_idx}', data=pids, compression='gzip')
            
            # Metadata
            f.attrs['snapshot'] = snapNum
            f.attrs['sim_res'] = args.sim_res
            f.attrs['radius_multiplier'] = CONFIG['radius_multiplier']
            f.attrs['min_mass_log'] = CONFIG['min_mass']
            f.attrs['n_halos'] = len(halo_indices_dmo)
            f.attrs['box_size'] = CONFIG['box_size']
            f.attrs['has_hydro_positions'] = has_hydro_positions
        
        print(f"  Wrote: {output_file}")
        print(f"  Halos cached: {len(halo_indices_dmo)}")
        print(f"  DMO halos with particles: {len(dmo_cache)}")
        print(f"  Hydro@DMO halos with particles: {len(hydro_at_dmo_cache)}")
        print(f"  Hydro@Hydro halos with particles: {len(hydro_at_hydro_cache)}")
        print(f"  Time: {time.time()-t0:.1f}s")
    
    comm.Barrier()
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Snapshot {snapNum} complete!")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Generate particle ID cache for halos')
    parser.add_argument('--sim-res', type=int, required=True, choices=[625, 1250, 2500])
    parser.add_argument('--snap', type=str, default='all',
                       help='Snapshot to process: "all", single number, or comma-separated')
    
    args = parser.parse_args()
    
    # Parse snapshot selection
    if args.snap == 'all':
        snapshots = [29, 31, 33, 35, 38, 41, 43, 46, 49, 52, 56, 59, 
                     63, 67, 71, 76, 80, 85, 90, 96, 99]
    else:
        snapshots = [int(s) for s in args.snap.split(',')]
    
    if rank == 0:
        print("=" * 70)
        print("PARTICLE ID CACHE GENERATION")
        print("=" * 70)
        print(f"Simulation: L205n{args.sim_res}TNG")
        print(f"Snapshots: {snapshots}")
        print(f"Radius: {CONFIG['radius_multiplier']}×R200")
        print(f"Min mass: 10^{CONFIG['min_mass']} Msun/h")
        print(f"MPI ranks: {size}")
        print("=" * 70)
    
    for snap in snapshots:
        process_snapshot(args, snap)
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("ALL SNAPSHOTS COMPLETE")
        print("=" * 70)


if __name__ == '__main__':
    main()
