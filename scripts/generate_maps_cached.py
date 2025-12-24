#!/usr/bin/env python
"""
Generate 2D density maps using pre-computed particle cache.

This is MUCH faster than generate_all.py because we skip KDTree queries.
The cache stores particle IDs for each halo, so we just need to:
1. Load snapshot particles (coords, masses, IDs)
2. Build ID→index lookup (fast dict)
3. For each halo, use cached IDs to directly index particles

Usage:
    mpirun -np 32 python generate_maps_cached.py --snap 99 --sim-res 2500 --mass-min 12.5

This generates:
    - DMO map (unchanged)
    - Hydro map (unchanged)  
    - Replace map (DMO with halos replaced by Hydro)
    
Different mass thresholds just change which halos are replaced - no re-querying!
"""

import numpy as np
import h5py
import argparse
import os
import sys
import time
import glob

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
# Distributed Particle Loader with ID Lookup
# ============================================================================

class DistributedParticleLoader:
    """
    Load particles distributed across MPI ranks with ID→index lookup.
    
    Each rank loads a subset of snapshot files and builds a local lookup table.
    """
    
    def __init__(self, snapshot: int, sim_res: int, mode: str):
        self.snapshot = snapshot
        self.sim_res = sim_res
        self.mode = mode
        self.sim_config = SIM_PATHS[sim_res]
        
        # Local data
        self.local_coords = None
        self.local_masses = None
        self.local_ids = None
        self.local_id_to_idx = None
        
        self._load()
    
    def _load(self):
        """Load snapshot files assigned to this rank."""
        if self.mode == 'dmo':
            basePath = self.sim_config['dmo']
            dm_mass = self.sim_config['dmo_dm_mass']
            particle_types = [1]
        else:
            basePath = self.sim_config['hydro']
            dm_mass = self.sim_config['hydro_dm_mass']
            particle_types = [0, 1, 4]
        
        snap_dir = f"{basePath}/snapdir_{self.snapshot:03d}/"
        all_files = sorted(glob.glob(f"{snap_dir}/snap_{self.snapshot:03d}.*.hdf5"))
        
        # Distribute files across ranks
        my_files = [f for i, f in enumerate(all_files) if i % size == rank]
        
        coords_list = []
        masses_list = []
        ids_list = []
        
        for filepath in my_files:
            with h5py.File(filepath, 'r') as f:
                for ptype in particle_types:
                    pt_key = f'PartType{ptype}'
                    if pt_key not in f:
                        continue
                    
                    n_part = f[pt_key]['Coordinates'].shape[0]
                    if n_part == 0:
                        continue
                    
                    coords = f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3
                    coords_list.append(coords)
                    
                    pids = f[pt_key]['ParticleIDs'][:]
                    ids_list.append(pids)
                    
                    if 'Masses' in f[pt_key]:
                        m = f[pt_key]['Masses'][:].astype(np.float32) * MASS_UNIT
                    else:
                        m = np.full(n_part, dm_mass * MASS_UNIT, dtype=np.float32)
                    masses_list.append(m)
        
        if coords_list:
            self.local_coords = np.concatenate(coords_list)
            self.local_masses = np.concatenate(masses_list)
            self.local_ids = np.concatenate(ids_list)
        else:
            self.local_coords = np.zeros((0, 3), dtype=np.float32)
            self.local_masses = np.zeros(0, dtype=np.float32)
            self.local_ids = np.zeros(0, dtype=np.int64)
        
        # Build local ID→index lookup
        self.local_id_to_idx = {int(pid): i for i, pid in enumerate(self.local_ids)}
    
    def get_particles_by_ids(self, particle_ids: np.ndarray):
        """
        Get coordinates and masses for a set of particle IDs.
        
        Returns local matches only - caller should gather across ranks.
        """
        local_idx = []
        for pid in particle_ids:
            if int(pid) in self.local_id_to_idx:
                local_idx.append(self.local_id_to_idx[int(pid)])
        
        if local_idx:
            local_idx = np.array(local_idx)
            return self.local_coords[local_idx], self.local_masses[local_idx]
        else:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
    
    def get_particles_excluding_ids(self, exclude_ids: set):
        """
        Get all local particles EXCEPT those in exclude_ids.
        
        Used for DMO background (everything except replaced halos).
        """
        if len(self.local_ids) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
        keep_mask = np.array([int(pid) not in exclude_ids for pid in self.local_ids], dtype=bool)
        return self.local_coords[keep_mask], self.local_masses[keep_mask]


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


def generate_maps(args):
    """Generate DMO, Hydro, and Replace maps using cache."""
    
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
        print("CACHED MAP GENERATION")
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
        print("\n[1/5] Loading cache and selecting halos...")
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
        
        # Load particle IDs for selected halos
        dmo_ids_per_halo = []
        hydro_ids_per_halo = []
        
        for idx in selected_indices:
            dmo_ids = f[f'dmo/halo_{idx}'][:]
            hydro_ids = f[f'hydro_at_dmo/halo_{idx}'][:]
            dmo_ids_per_halo.append(dmo_ids)
            hydro_ids_per_halo.append(hydro_ids)
    
    # Build sets of all IDs to exclude/include
    all_dmo_ids_to_remove = set()
    all_hydro_ids_to_add = set()
    
    for dmo_ids, hydro_ids in zip(dmo_ids_per_halo, hydro_ids_per_halo):
        all_dmo_ids_to_remove.update(int(x) for x in dmo_ids)
        all_hydro_ids_to_add.update(int(x) for x in hydro_ids)
    
    if rank == 0:
        print(f"  DMO particles to remove: {len(all_dmo_ids_to_remove):,}")
        print(f"  Hydro particles to add: {len(all_hydro_ids_to_add):,}")
        sys.stdout.flush()
    
    # ========================================================================
    # Load particles (distributed)
    # ========================================================================
    if rank == 0:
        print("\n[2/5] Loading DMO particles...")
        t0 = time.time()
        sys.stdout.flush()
    
    dmo_loader = DistributedParticleLoader(args.snap, args.sim_res, 'dmo')
    
    if rank == 0:
        print(f"  Loaded in {time.time()-t0:.1f}s")
        print(f"  Rank 0: {len(dmo_loader.local_ids):,} particles")
        print("\n[3/5] Loading Hydro particles...")
        t0 = time.time()
        sys.stdout.flush()
    
    hydro_loader = DistributedParticleLoader(args.snap, args.sim_res, 'hydro')
    
    if rank == 0:
        print(f"  Loaded in {time.time()-t0:.1f}s")
        print(f"  Rank 0: {len(hydro_loader.local_ids):,} particles")
        sys.stdout.flush()
    
    # ========================================================================
    # Generate DMO map (full, no replacement)
    # ========================================================================
    if rank == 0:
        print("\n[4/5] Generating maps...")
        print("  DMO map...", end=" ", flush=True)
    
    local_dmo_map = project_to_2d(dmo_loader.local_coords, dmo_loader.local_masses, args.grid)
    
    if rank == 0:
        global_dmo_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_dmo_map = None
    
    comm.Reduce(local_dmo_map, global_dmo_map, op=MPI.SUM, root=0)
    
    if rank == 0:
        dmo_file = os.path.join(output_dir, 'dmo.npz')
        np.savez_compressed(dmo_file, field=global_dmo_map, box_size=BOX_SIZE,
                           grid_resolution=args.grid, snapshot=args.snap)
        print("done")
    
    # ========================================================================
    # Generate Hydro map (full, no replacement)
    # ========================================================================
    if rank == 0:
        print("  Hydro map...", end=" ", flush=True)
    
    local_hydro_map = project_to_2d(hydro_loader.local_coords, hydro_loader.local_masses, args.grid)
    
    if rank == 0:
        global_hydro_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_hydro_map = None
    
    comm.Reduce(local_hydro_map, global_hydro_map, op=MPI.SUM, root=0)
    
    if rank == 0:
        hydro_file = os.path.join(output_dir, 'hydro.npz')
        np.savez_compressed(hydro_file, field=global_hydro_map, box_size=BOX_SIZE,
                           grid_resolution=args.grid, snapshot=args.snap)
        print("done")
    
    # ========================================================================
    # Generate Replace map (DMO background + Hydro in halos)
    # ========================================================================
    mass_label = f"M{args.mass_min:.1f}".replace('.', 'p')
    if args.mass_max:
        mass_label += f"_M{args.mass_max:.1f}".replace('.', 'p')
    
    if rank == 0:
        print(f"  Replace map ({mass_label})...", end=" ", flush=True)
    
    # DMO: exclude particles in replaced halos
    dmo_bg_coords, dmo_bg_masses = dmo_loader.get_particles_excluding_ids(all_dmo_ids_to_remove)
    local_replace_dmo = project_to_2d(dmo_bg_coords, dmo_bg_masses, args.grid)
    
    # Hydro: only particles in replaced halos
    hydro_halo_coords, hydro_halo_masses = [], []
    for pid in all_hydro_ids_to_add:
        if pid in hydro_loader.local_id_to_idx:
            idx = hydro_loader.local_id_to_idx[pid]
            hydro_halo_coords.append(hydro_loader.local_coords[idx])
            hydro_halo_masses.append(hydro_loader.local_masses[idx])
    
    if hydro_halo_coords:
        hydro_halo_coords = np.array(hydro_halo_coords)
        hydro_halo_masses = np.array(hydro_halo_masses)
    else:
        hydro_halo_coords = np.zeros((0, 3), dtype=np.float32)
        hydro_halo_masses = np.zeros(0, dtype=np.float32)
    
    local_replace_hydro = project_to_2d(hydro_halo_coords, hydro_halo_masses, args.grid)
    local_replace_map = local_replace_dmo + local_replace_hydro
    
    if rank == 0:
        global_replace_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_replace_map = None
    
    comm.Reduce(local_replace_map, global_replace_map, op=MPI.SUM, root=0)
    
    if rank == 0:
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
        print(f"\n[5/5] Complete!")
        print("=" * 70)
        print(f"Total time: {time.time()-t_start:.1f}s")
        print(f"Output directory: {output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Generate maps using particle cache')
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
