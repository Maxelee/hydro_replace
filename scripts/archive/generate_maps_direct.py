#!/usr/bin/env python
"""
Direct 2D density map generation using spatial queries (no cache).

This approach:
1. Loads halo catalog once (small - ~10k halos)
2. Builds a BallTree of halo centers for fast spatial queries
3. Streams through particle files, querying which particles are near halos
4. Generates all maps in a single pass through the data

This avoids the overhead of:
- Generating and storing particle ID caches (100s of GB)
- Loading cache files (slow HDF5 random access)
- ID matching via searchsorted on billions of particles

Usage:
    mpirun -np 32 python generate_maps_direct.py --snap 99 --sim-res 1250 --mass-min 12.5

Output:
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
from scipy.spatial import cKDTree
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

OUTPUT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields'
BOX_SIZE = 205.0  # Mpc/h
MASS_UNIT = 1e10  # Convert to Msun/h
GRID_RES = 4096   # Default grid resolution


# ============================================================================
# Halo Spatial Query Structure
# ============================================================================

class HaloQuery:
    """
    Efficient spatial query structure for checking if particles are near halos.
    
    Uses a KDTree of halo centers with variable radii per halo.
    Handles periodic boundary conditions.
    """
    
    def __init__(self, positions, radii, box_size=205.0):
        """
        Args:
            positions: (N, 3) array of halo center positions in Mpc/h
            radii: (N,) array of query radii (e.g., radius_mult * R200) in Mpc/h
            box_size: Simulation box size in Mpc/h
        """
        self.positions = np.asarray(positions, dtype=np.float64)
        self.radii = np.asarray(radii, dtype=np.float64)
        self.box_size = box_size
        self.n_halos = len(positions)
        
        # Build KDTree with periodic boundary handling via replication
        # For halos near edges, we need to consider periodic images
        self.tree = cKDTree(self.positions, boxsize=box_size)
        
        # Maximum query radius (for initial broad-phase query)
        self.max_radius = np.max(self.radii) if len(self.radii) > 0 else 0.0
    
    def query_near_any_halo(self, coords):
        """
        Check which particles are within their nearest halo's radius.
        
        Args:
            coords: (M, 3) array of particle positions
            
        Returns:
            mask: (M,) boolean array, True if particle is near any halo
        """
        if self.n_halos == 0:
            return np.zeros(len(coords), dtype=bool)
        
        coords = np.asarray(coords, dtype=np.float64)
        
        # Query all halos within max_radius of each particle
        # This is the broad phase - we'll refine below
        nearby_lists = self.tree.query_ball_point(coords, self.max_radius)
        
        # Refine: check if particle is within the specific radius of each nearby halo
        mask = np.zeros(len(coords), dtype=bool)
        
        for i, nearby_halos in enumerate(nearby_lists):
            if len(nearby_halos) == 0:
                continue
            
            # Check distance to each nearby halo against its specific radius
            for halo_idx in nearby_halos:
                # Compute periodic distance
                dx = coords[i] - self.positions[halo_idx]
                dx = dx - self.box_size * np.round(dx / self.box_size)
                dist = np.sqrt(np.sum(dx**2))
                
                if dist <= self.radii[halo_idx]:
                    mask[i] = True
                    break  # No need to check other halos
        
        return mask
    
    def query_near_any_halo_vectorized(self, coords, chunk_size=500000):
        """
        Vectorized check if particles are within any halo's radius.
        
        Uses query_ball_point to find ALL halos within max_radius,
        then checks if particle is within any of those halos' specific radii.
        
        Args:
            coords: (M, 3) array of particle positions
            chunk_size: Process in chunks to manage memory
            
        Returns:
            mask: (M,) boolean array
        """
        if self.n_halos == 0:
            return np.zeros(len(coords), dtype=bool)
        
        coords = np.asarray(coords, dtype=np.float64)
        n_particles = len(coords)
        mask = np.zeros(n_particles, dtype=bool)
        
        # Process in chunks
        for start in range(0, n_particles, chunk_size):
            end = min(start + chunk_size, n_particles)
            chunk_coords = coords[start:end]
            
            # Find all halos within max_radius of each particle
            nearby_lists = self.tree.query_ball_point(chunk_coords, self.max_radius)
            
            # Check each particle against its nearby halos
            for i, nearby_halos in enumerate(nearby_lists):
                if len(nearby_halos) == 0:
                    continue
                
                # Check if particle is within any nearby halo's radius
                for halo_idx in nearby_halos:
                    dx = chunk_coords[i] - self.positions[halo_idx]
                    # Periodic boundary
                    dx = dx - self.box_size * np.round(dx / self.box_size)
                    dist = np.sqrt(np.sum(dx**2))
                    
                    if dist <= self.radii[halo_idx]:
                        mask[start + i] = True
                        break  # Found one, no need to check more
        
        return mask


# ============================================================================
# Load Halo Catalog
# ============================================================================

def load_matched_halos(matches_file, mass_min=12.5, mass_max=None, radius_mult=5.0):
    """
    Load matched halo catalog and filter by mass.
    
    Returns:
        positions: (N, 3) halo positions in Mpc/h
        query_radii: (N,) query radii = radius_mult * R200 in Mpc/h
        masses: (N,) halo masses in Msun/h
    """
    with np.load(matches_file) as data:
        positions = data['dmo_positions'] / 1e3  # Convert kpc/h -> Mpc/h
        radii = data['dmo_radii'] / 1e3  # Convert kpc/h -> Mpc/h
        masses = data['dmo_masses'] * 1e10  # Convert from 10^10 Msun/h to Msun/h
    
    # Filter by mass
    log_masses = np.log10(masses)
    mask = log_masses >= mass_min
    if mass_max is not None:
        mask &= log_masses < mass_max
    
    positions = positions[mask]
    radii = radii[mask]
    masses = masses[mask]
    
    # Compute query radii
    query_radii = radius_mult * radii
    
    return positions, query_radii, masses


# ============================================================================
# Map Generation
# ============================================================================

def project_to_2d(coords, masses, grid_res, box_size=BOX_SIZE, axis=2):
    """Project particles to 2D density map using TSC."""
    if len(coords) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32)
    
    # Select 2D projection axes
    proj_axes = [0, 1, 2]
    proj_axes.pop(axis)
    
    pos_2d = coords[:, proj_axes].astype(np.float32).copy()
    pos_2d = np.mod(pos_2d, box_size)
    
    field = np.zeros((grid_res, grid_res), dtype=np.float32)
    MASL.MA(pos_2d, field, np.float32(box_size), MAS='TSC',
            W=masses.astype(np.float32), verbose=False)
    
    return field


def load_and_process_dmo(snapshot, sim_res, halo_query, grid_res):
    """
    Load DMO particles and generate both full map and background map.
    
    Returns:
        local_dmo_full: Full DMO density map (local contribution)
        local_dmo_bg: DMO background map (particles NOT near halos)
    """
    sim_config = SIM_PATHS[sim_res]
    basePath = sim_config['dmo']
    dm_mass = sim_config['dmo_dm_mass']
    
    snap_dir = f"{basePath}/snapdir_{snapshot:03d}/"
    all_files = sorted(glob.glob(f"{snap_dir}/snap_{snapshot:03d}.*.hdf5"))
    my_files = [f for i, f in enumerate(all_files) if i % size == rank]
    
    # Initialize local maps
    local_full = np.zeros((grid_res, grid_res), dtype=np.float32)
    local_bg = np.zeros((grid_res, grid_res), dtype=np.float32)
    
    total_particles = 0
    total_bg_particles = 0
    
    for filepath in my_files:
        with h5py.File(filepath, 'r') as f:
            if 'PartType1' not in f:
                continue
            
            n_part = f['PartType1']['Coordinates'].shape[0]
            if n_part == 0:
                continue
            
            # Load coordinates
            coords = f['PartType1']['Coordinates'][:].astype(np.float32) / 1e3  # kpc -> Mpc
            masses = np.full(n_part, dm_mass * MASS_UNIT, dtype=np.float32)
            
            total_particles += n_part
            
            # Query which particles are near halos
            near_halo_mask = halo_query.query_near_any_halo_vectorized(coords)
            bg_mask = ~near_halo_mask
            
            total_bg_particles += np.sum(bg_mask)
            
            # Accumulate to full map (all particles)
            local_full += project_to_2d(coords, masses, grid_res)
            
            # Accumulate to background map (only particles NOT near halos)
            if np.any(bg_mask):
                local_bg += project_to_2d(coords[bg_mask], masses[bg_mask], grid_res)
    
    return local_full, local_bg, total_particles, total_bg_particles


def load_and_process_hydro(snapshot, sim_res, halo_query, grid_res):
    """
    Load Hydro particles and generate both full map and halo-only map.
    
    Returns:
        local_hydro_full: Full Hydro density map (local contribution)
        local_hydro_halos: Hydro halo map (particles near DMO halo positions)
    """
    sim_config = SIM_PATHS[sim_res]
    basePath = sim_config['hydro']
    dm_mass = sim_config['hydro_dm_mass']
    
    snap_dir = f"{basePath}/snapdir_{snapshot:03d}/"
    all_files = sorted(glob.glob(f"{snap_dir}/snap_{snapshot:03d}.*.hdf5"))
    my_files = [f for i, f in enumerate(all_files) if i % size == rank]
    
    # Initialize local maps
    local_full = np.zeros((grid_res, grid_res), dtype=np.float32)
    local_halos = np.zeros((grid_res, grid_res), dtype=np.float32)
    
    total_particles = 0
    total_halo_particles = 0
    
    particle_types = [0, 1, 4]  # Gas, DM, Stars
    
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
                
                # Load masses
                if 'Masses' in f[pt_key]:
                    masses = f[pt_key]['Masses'][:].astype(np.float32) * MASS_UNIT
                else:
                    masses = np.full(n_part, dm_mass * MASS_UNIT, dtype=np.float32)
                
                total_particles += n_part
                
                # Query which particles are near DMO halo positions
                near_halo_mask = halo_query.query_near_any_halo_vectorized(coords)
                
                total_halo_particles += np.sum(near_halo_mask)
                
                # Accumulate to full map (all particles)
                local_full += project_to_2d(coords, masses, grid_res)
                
                # Accumulate to halo map (only particles near DMO halo positions)
                if np.any(near_halo_mask):
                    local_halos += project_to_2d(coords[near_halo_mask], masses[near_halo_mask], grid_res)
    
    return local_full, local_halos, total_particles, total_halo_particles


def generate_maps(args):
    """Main function to generate all maps."""
    
    t_start = time.time()
    
    # Paths
    matches_file = os.path.join(
        OUTPUT_BASE, f'L205n{args.sim_res}TNG',
        'matches', f'matches_snap{args.snap:03d}.npz'
    )
    output_dir = os.path.join(
        OUTPUT_BASE, f'L205n{args.sim_res}TNG',
        f'snap{args.snap:03d}', 'projected_direct'
    )
    
    if rank == 0:
        print("=" * 70)
        print("DIRECT MAP GENERATION (NO CACHE)")
        print("=" * 70)
        print(f"Snapshot: {args.snap}")
        print(f"Resolution: L205n{args.sim_res}TNG")
        print(f"Mass threshold: 10^{args.mass_min} Msun/h")
        print(f"Radius multiplier: {args.radius_mult}×R200")
        print(f"Grid: {args.grid}²")
        print(f"MPI ranks: {size}")
        print("=" * 70)
        sys.stdout.flush()
        
        os.makedirs(output_dir, exist_ok=True)
    
    comm.Barrier()
    
    # ========================================================================
    # Load halo catalog and build spatial query structure
    # ========================================================================
    if rank == 0:
        print("\n[1/4] Loading halo catalog and building spatial query...")
        t0 = time.time()
        sys.stdout.flush()
    
    positions, query_radii, masses = load_matched_halos(
        matches_file, 
        mass_min=args.mass_min, 
        mass_max=args.mass_max,
        radius_mult=args.radius_mult
    )
    
    n_halos = len(positions)
    
    # Build spatial query structure
    halo_query = HaloQuery(positions, query_radii, box_size=BOX_SIZE)
    
    if rank == 0:
        print(f"  Halos selected: {n_halos}")
        print(f"  Max query radius: {halo_query.max_radius:.3f} Mpc/h")
        print(f"  Setup time: {time.time()-t0:.1f}s")
        sys.stdout.flush()
    
    # ========================================================================
    # Process DMO particles
    # ========================================================================
    if rank == 0:
        print("\n[2/4] Processing DMO particles...")
        t0 = time.time()
        sys.stdout.flush()
    
    local_dmo_full, local_dmo_bg, dmo_total, dmo_bg_count = load_and_process_dmo(
        args.snap, args.sim_res, halo_query, args.grid
    )
    
    # Gather counts
    total_dmo = comm.reduce(dmo_total, op=MPI.SUM, root=0)
    total_dmo_bg = comm.reduce(dmo_bg_count, op=MPI.SUM, root=0)
    
    if rank == 0:
        print(f"  Total DMO particles: {total_dmo:,}")
        print(f"  Background particles: {total_dmo_bg:,} ({100*total_dmo_bg/total_dmo:.1f}%)")
        print(f"  Load+query time: {time.time()-t0:.1f}s")
        print("  Reducing maps...", end=" ", flush=True)
    
    # Reduce DMO maps
    if rank == 0:
        global_dmo_full = np.zeros((args.grid, args.grid), dtype=np.float32)
        global_dmo_bg = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_dmo_full = None
        global_dmo_bg = None
    
    comm.Reduce(local_dmo_full, global_dmo_full, op=MPI.SUM, root=0)
    comm.Reduce(local_dmo_bg, global_dmo_bg, op=MPI.SUM, root=0)
    
    del local_dmo_full, local_dmo_bg
    gc.collect()
    
    if rank == 0:
        print("done")
        # Save DMO map
        dmo_file = os.path.join(output_dir, 'dmo.npz')
        np.savez_compressed(dmo_file, field=global_dmo_full, box_size=BOX_SIZE,
                           grid_resolution=args.grid, snapshot=args.snap)
        del global_dmo_full
        print(f"  Saved: {dmo_file}")
        sys.stdout.flush()
    
    # ========================================================================
    # Process Hydro particles
    # ========================================================================
    if rank == 0:
        print("\n[3/4] Processing Hydro particles...")
        t0 = time.time()
        sys.stdout.flush()
    
    local_hydro_full, local_hydro_halos, hydro_total, hydro_halo_count = load_and_process_hydro(
        args.snap, args.sim_res, halo_query, args.grid
    )
    
    # Gather counts
    total_hydro = comm.reduce(hydro_total, op=MPI.SUM, root=0)
    total_hydro_halos = comm.reduce(hydro_halo_count, op=MPI.SUM, root=0)
    
    if rank == 0:
        print(f"  Total Hydro particles: {total_hydro:,}")
        print(f"  Halo particles: {total_hydro_halos:,} ({100*total_hydro_halos/total_hydro:.1f}%)")
        print(f"  Load+query time: {time.time()-t0:.1f}s")
        print("  Reducing maps...", end=" ", flush=True)
    
    # Reduce Hydro maps
    if rank == 0:
        global_hydro_full = np.zeros((args.grid, args.grid), dtype=np.float32)
        global_hydro_halos = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_hydro_full = None
        global_hydro_halos = None
    
    comm.Reduce(local_hydro_full, global_hydro_full, op=MPI.SUM, root=0)
    comm.Reduce(local_hydro_halos, global_hydro_halos, op=MPI.SUM, root=0)
    
    del local_hydro_full, local_hydro_halos
    gc.collect()
    
    if rank == 0:
        print("done")
        # Save Hydro map
        hydro_file = os.path.join(output_dir, 'hydro.npz')
        np.savez_compressed(hydro_file, field=global_hydro_full, box_size=BOX_SIZE,
                           grid_resolution=args.grid, snapshot=args.snap)
        del global_hydro_full
        print(f"  Saved: {hydro_file}")
        sys.stdout.flush()
    
    # ========================================================================
    # Combine and save Replace map
    # ========================================================================
    mass_label = f"M{args.mass_min:.1f}".replace('.', 'p')
    if args.mass_max:
        mass_label += f"_M{args.mass_max:.1f}".replace('.', 'p')
    mass_label += f"_R{args.radius_mult:.0f}"
    
    if rank == 0:
        print(f"\n[4/4] Saving Replace map ({mass_label})...")
        
        global_replace = global_dmo_bg + global_hydro_halos
        del global_dmo_bg, global_hydro_halos
        
        replace_file = os.path.join(output_dir, f'replace_{mass_label}.npz')
        np.savez_compressed(replace_file, field=global_replace, box_size=BOX_SIZE,
                           grid_resolution=args.grid, snapshot=args.snap,
                           log_mass_min=args.mass_min, log_mass_max=args.mass_max,
                           radius_multiplier=args.radius_mult)
        print(f"  Saved: {replace_file}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    if rank == 0:
        total_time = time.time() - t_start
        print(f"\n" + "=" * 70)
        print(f"COMPLETE - Total time: {total_time:.1f}s")
        print("=" * 70)
        print(f"Output directory: {output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Generate maps using direct spatial queries (no cache)')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, default=1250, choices=[625, 1250, 2500])
    parser.add_argument('--mass-min', type=float, default=12.5,
                        help='Minimum log10(M200c/Msun/h) for replacement')
    parser.add_argument('--mass-max', type=float, default=None,
                        help='Maximum log10(M200c/Msun/h) for replacement')
    parser.add_argument('--radius-mult', type=float, default=5.0,
                        help='Radius multiplier for halo region (× R200)')
    parser.add_argument('--grid', type=int, default=GRID_RES,
                        help='Grid resolution')
    
    args = parser.parse_args()
    generate_maps(args)


if __name__ == '__main__':
    main()
