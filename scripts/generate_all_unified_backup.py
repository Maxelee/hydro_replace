#!/usr/bin/env python
"""
Unified pipeline: Generate profiles, statistics, and maps in a single pass.

This script processes DMO and Hydro simulations sequentially, computing:
- Stacked density profiles (by mass bin)
- Halo statistics (masses, particle counts, etc.)
- 2D projected density maps (DMO, Hydro, Replace)

No caching required - everything computed in two passes (DMO + Hydro).

Algorithm:
  PASS 1 (DMO):
    - Load all DMO particles
    - Build local KDTree
    - For each halo: query particles, compute profiles/stats, mark for removal
    - Generate DMO map and DMO-background map
  
  PASS 2 (Hydro):
    - Load all Hydro particles
    - Build local KDTree
    - For each halo: query particles, compute profiles/stats, mark for inclusion
    - Generate Hydro map and Hydro-halo map
  
  COMBINE:
    - Replace = DMO-background + Hydro-halos

Usage:
    mpirun -np 32 python generate_all_unified.py --snap 99 --sim-res 1250 --mass-min 12.5
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

# Profile configuration
RADIAL_BINS = np.logspace(-2, np.log10(5), 31)  # 0.01 to 5 R200, 30 bins (matching generate_profiles)
MASS_BIN_EDGES = [12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 16.0]

# Statistics radii (matching compute_halo_statistics: 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
STATS_RADII_MULT = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])


# ============================================================================
# Data Loading
# ============================================================================

class DistributedParticles:
    """Load and manage particles distributed across MPI ranks."""
    
    def __init__(self, snapshot, sim_res, mode, radius_mult=5.0):
        self.snapshot = snapshot
        self.sim_res = sim_res
        self.mode = mode
        self.radius_mult = radius_mult
        self.sim_config = SIM_PATHS[sim_res]
        
        self.coords = None
        self.masses = None
        self.ids = None
        self.types = None  # Particle types (0=gas, 1=DM, 4=stars)
        self.tree = None
        
    def load(self):
        """Load particles for this rank."""
        t0 = time.time()
        
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
        my_files = [f for i, f in enumerate(all_files) if i % size == rank]
        
        if rank == 0:
            print(f"  Loading {self.mode.upper()} particles...")
            print(f"    Files: {len(all_files)} total, {len(my_files)} per rank")
        
        coords_list, masses_list, ids_list, types_list = [], [], [], []
        
        for filepath in my_files:
            with h5py.File(filepath, 'r') as f:
                for ptype in particle_types:
                    pt_key = f'PartType{ptype}'
                    if pt_key not in f or f[pt_key]['Coordinates'].shape[0] == 0:
                        continue
                    
                    n_part = f[pt_key]['Coordinates'].shape[0]
                    coords_list.append(f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3)
                    ids_list.append(f[pt_key]['ParticleIDs'][:])
                    types_list.append(np.full(n_part, ptype, dtype=np.int8))
                    
                    if 'Masses' in f[pt_key]:
                        masses_list.append(f[pt_key]['Masses'][:].astype(np.float32) * MASS_UNIT)
                    else:
                        masses_list.append(np.full(n_part, dm_mass * MASS_UNIT, dtype=np.float32))
        
        if coords_list:
            self.coords = np.concatenate(coords_list)
            self.masses = np.concatenate(masses_list)
            self.ids = np.concatenate(ids_list)
            self.types = np.concatenate(types_list)
        else:
            self.coords = np.zeros((0, 3), dtype=np.float32)
            self.masses = np.zeros(0, dtype=np.float32)
            self.ids = np.zeros(0, dtype=np.int64)
            self.types = np.zeros(0, dtype=np.int8)
        
        if rank == 0:
            print(f"    Rank 0: {len(self.coords):,} particles")
            print(f"    Load time: {time.time()-t0:.1f}s")
        
        return self
    
    def build_tree(self):
        """Build KDTree for spatial queries."""
        t0 = time.time()
        if rank == 0:
            print(f"  Building KDTree...", end=" ", flush=True)
        
        if len(self.coords) > 0:
            self.tree = cKDTree(self.coords)
        
        if rank == 0:
            print(f"done ({time.time()-t0:.1f}s)")
        
        return self
    
    def query_halo(self, center, radius):
        """
        Query particles within radius of halo center.
        Returns local indices of particles within radius.
        Handles periodic boundary conditions.
        """
        if self.tree is None or len(self.coords) == 0:
            return np.array([], dtype=int)
        
        search_radius = radius * self.radius_mult
        
        # Query tree (doesn't handle periodic BC directly)
        indices = self.tree.query_ball_point(center, search_radius)
        
        if len(indices) == 0:
            return np.array([], dtype=int)
        
        indices = np.array(indices)
        
        # Check periodic images if near box edge
        if np.any(center < search_radius) or np.any(center > BOX_SIZE - search_radius):
            # Check all 26 periodic images
            for dx in [-BOX_SIZE, 0, BOX_SIZE]:
                for dy in [-BOX_SIZE, 0, BOX_SIZE]:
                    for dz in [-BOX_SIZE, 0, BOX_SIZE]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        shifted_center = center + np.array([dx, dy, dz])
                        if np.all(shifted_center >= -search_radius) and np.all(shifted_center <= BOX_SIZE + search_radius):
                            more_indices = self.tree.query_ball_point(shifted_center, search_radius)
                            if len(more_indices) > 0:
                                indices = np.unique(np.concatenate([indices, more_indices]))
        
        return indices
    
    def free(self):
        """Free memory."""
        del self.coords, self.masses, self.ids, self.types, self.tree
        self.coords = self.masses = self.ids = self.types = self.tree = None
        gc.collect()


# ============================================================================
# Profile Computation
# ============================================================================

def compute_profile(coords, masses, types, center, r200, radial_bins):
    """
    Compute density profile for a single halo.
    
    Args:
        coords: Particle coordinates
        masses: Particle masses
        types: Particle types (0=gas, 1=DM, 4=stars)
        center: Halo center
        r200: Halo R200
        radial_bins: Radial bins in units of R200
    
    Returns:
        dict with density, mass, count profiles (total and by type)
    """
    if len(coords) == 0:
        n_bins = len(radial_bins) - 1
        return {
            'density': np.zeros(n_bins),
            'mass': np.zeros(n_bins),
            'count': np.zeros(n_bins, dtype=int),
            'density_dm': np.zeros(n_bins),
            'density_gas': np.zeros(n_bins),
            'density_stars': np.zeros(n_bins),
            'mass_dm': np.zeros(n_bins),
            'mass_gas': np.zeros(n_bins),
            'mass_stars': np.zeros(n_bins),
        }
    
    # Compute distances (with periodic BC)
    dx = coords - center
    dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
    dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
    r = np.linalg.norm(dx, axis=1) / r200  # In units of R200
    
    # Total profile
    mass_profile, _ = np.histogram(r, bins=radial_bins, weights=masses)
    count_profile, _ = np.histogram(r, bins=radial_bins)
    
    # Shell volumes (in physical units: (R200 * Mpc/h)^3)
    volumes = 4/3 * np.pi * r200**3 * (radial_bins[1:]**3 - radial_bins[:-1]**3)
    density = np.where(volumes > 0, mass_profile / volumes, 0)
    
    # By particle type
    dm_mask = types == 1
    gas_mask = types == 0
    star_mask = types == 4
    
    mass_dm, _ = np.histogram(r[dm_mask], bins=radial_bins, weights=masses[dm_mask])
    mass_gas, _ = np.histogram(r[gas_mask], bins=radial_bins, weights=masses[gas_mask])
    mass_stars, _ = np.histogram(r[star_mask], bins=radial_bins, weights=masses[star_mask])
    
    density_dm = np.where(volumes > 0, mass_dm / volumes, 0)
    density_gas = np.where(volumes > 0, mass_gas / volumes, 0)
    density_stars = np.where(volumes > 0, mass_stars / volumes, 0)
    
    return {
        'density': density,
        'mass': mass_profile,
        'count': count_profile,
        'density_dm': density_dm,
        'density_gas': density_gas,
        'density_stars': density_stars,
        'mass_dm': mass_dm,
        'mass_gas': mass_gas,
        'mass_stars': mass_stars,
    }


def compute_statistics_at_radii(coords, masses, types, center, r200, radii_mult):
    """
    Compute statistics at specific radii (matching compute_halo_statistics).
    
    Returns:
        dict with baryon fractions and masses at each radius
    """
    n_radii = len(radii_mult)
    
    if len(coords) == 0:
        return {
            'f_baryon': np.zeros(n_radii, dtype=np.float32),
            'f_gas': np.zeros(n_radii, dtype=np.float32),
            'f_stellar': np.zeros(n_radii, dtype=np.float32),
            'm_total': np.zeros(n_radii, dtype=np.float64),
            'm_gas': np.zeros(n_radii, dtype=np.float64),
            'm_stellar': np.zeros(n_radii, dtype=np.float64),
            'm_dm': np.zeros(n_radii, dtype=np.float64),
        }
    
    # Compute distances
    dx = coords - center
    dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
    dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
    r = np.linalg.norm(dx, axis=1) / r200  # In units of R200
    
    results = {
        'f_baryon': np.zeros(n_radii, dtype=np.float32),
        'f_gas': np.zeros(n_radii, dtype=np.float32),
        'f_stellar': np.zeros(n_radii, dtype=np.float32),
        'm_total': np.zeros(n_radii, dtype=np.float64),
        'm_gas': np.zeros(n_radii, dtype=np.float64),
        'm_stellar': np.zeros(n_radii, dtype=np.float64),
        'm_dm': np.zeros(n_radii, dtype=np.float64),
    }
    
    for i, r_mult in enumerate(radii_mult):
        mask = r <= r_mult
        
        if np.sum(mask) == 0:
            continue
        
        m_tot = np.sum(masses[mask])
        m_g = np.sum(masses[mask & (types == 0)])  # Gas
        m_s = np.sum(masses[mask & (types == 4)])  # Stars
        m_d = np.sum(masses[mask & (types == 1)])  # DM
        
        results['m_total'][i] = m_tot
        results['m_gas'][i] = m_g
        results['m_stellar'][i] = m_s
        results['m_dm'][i] = m_d
        
        if m_tot > 0:
            results['f_baryon'][i] = (m_g + m_s) / m_tot
            results['f_gas'][i] = m_g / m_tot
            results['f_stellar'][i] = m_s / m_tot
    
    return results


# ============================================================================
# Map Generation
# ============================================================================

def project_to_2d(coords, masses, grid_res, axis=2):
    """Project particles to 2D density map using TSC."""
    if len(coords) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32)
    
    proj_axes = [0, 1, 2]
    proj_axes.pop(axis)
    
    pos_2d = np.ascontiguousarray(coords[:, proj_axes].astype(np.float32))
    pos_2d = np.mod(pos_2d, BOX_SIZE)
    
    field = np.zeros((grid_res, grid_res), dtype=np.float32)
    MASL.MA(pos_2d, field, np.float32(BOX_SIZE), MAS='TSC',
            W=masses.astype(np.float32), verbose=False)
    
    return field


# ============================================================================
# Main Pipeline
# ============================================================================

def run_unified_pipeline(args):
    """Run the unified pipeline for profiles, statistics, and maps."""
    
    t_start = time.time()
    
    # Output paths
    # Use suffix for test runs to avoid overwriting existing data
    output_suffix = getattr(args, 'output_suffix', '')
    output_dir_base = os.path.join(OUTPUT_BASE, f'L205n{args.sim_res}TNG')
    output_dir = os.path.join(OUTPUT_BASE, f'L205n{args.sim_res}TNG{output_suffix}')
    snap_dir = os.path.join(output_dir, f'snap{args.snap:03d}')
    
    if rank == 0:
        print("=" * 70)
        print("UNIFIED PIPELINE (Profiles + Statistics + Maps)")
        print("=" * 70)
        print(f"Snapshot: {args.snap}")
        print(f"Resolution: L205n{args.sim_res}TNG")
        print(f"Mass threshold: 10^{args.mass_min} Msun/h")
        print(f"Radius: {args.radius_mult}×R200")
        print(f"Grid: {args.grid}²")
        if output_suffix:
            print(f"Output suffix: {output_suffix}")
        print(f"Output directory: {output_dir}")
        print("=" * 70)
        sys.stdout.flush()
        
        os.makedirs(os.path.join(output_dir, 'profiles'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'analysis'), exist_ok=True)
        os.makedirs(os.path.join(snap_dir, 'projected'), exist_ok=True)
    
    comm.Barrier()
    
    # ========================================================================
    # Load halo catalog
    # ========================================================================
    if rank == 0:
        print("\n[1/7] Loading halo catalog...")
        sys.stdout.flush()
    
    # Matches file is always in the base directory (not in test output)
    matches_file = os.path.join(output_dir_base, 'matches', f'matches_snap{args.snap:03d}.npz')
    
    with np.load(matches_file) as data:
        all_masses = data['dmo_masses'] * MASS_UNIT
        all_positions = data['dmo_positions'] / 1e3  # kpc -> Mpc
        all_radii = data['dmo_radii'] / 1e3  # kpc -> Mpc
        
        if 'hydro_positions' in data:
            hydro_positions = data['hydro_positions'] / 1e3
            hydro_radii = data['hydro_radii'] / 1e3
        else:
            hydro_positions = all_positions
            hydro_radii = all_radii
    
    # Select halos above mass threshold
    log_masses = np.log10(all_masses)
    mass_mask = log_masses >= args.mass_min
    
    halo_masses = all_masses[mass_mask]
    halo_positions = all_positions[mass_mask]
    halo_radii = all_radii[mass_mask]
    halo_hydro_positions = hydro_positions[mass_mask]
    halo_hydro_radii = hydro_radii[mass_mask]
    halo_log_masses = log_masses[mass_mask]
    
    n_halos = len(halo_masses)
    
    if rank == 0:
        print(f"  Total halos: {len(all_masses)}")
        print(f"  Halos above 10^{args.mass_min}: {n_halos}")
    
    # Assign halos to mass bins
    mass_bin_indices = np.digitize(halo_log_masses, MASS_BIN_EDGES) - 1
    n_mass_bins = len(MASS_BIN_EDGES) - 1
    
    # Initialize profile accumulators (per mass bin)
    n_radial_bins = len(RADIAL_BINS) - 1
    
    # ========================================================================
    # PASS 1: DMO
    # ========================================================================
    if rank == 0:
        print("\n[2/7] Processing DMO simulation...")
        sys.stdout.flush()
    
    dmo = DistributedParticles(args.snap, args.sim_res, 'dmo', args.radius_mult)
    dmo.load().build_tree()
    
    # Initialize accumulators
    local_dmo_profiles = np.zeros((n_mass_bins, n_radial_bins), dtype=np.float64)
    local_dmo_counts = np.zeros((n_mass_bins, n_radial_bins), dtype=np.int64)
    
    # Statistics at multiple radii
    n_stats_radii = len(STATS_RADII_MULT)
    local_dmo_stats = np.zeros((n_halos, n_stats_radii), dtype=np.float64)  # m_total at each radius
    
    # Mask for particles to KEEP (background, not in any halo)
    dmo_keep_mask = np.ones(len(dmo.coords), dtype=bool)
    
    if rank == 0:
        print(f"  Computing profiles and building mask for {n_halos} halos...")
        t0 = time.time()
    
    for i in range(n_halos):
        center = halo_positions[i]
        r200 = halo_radii[i]
        mass_bin = mass_bin_indices[i]
        
        # Query local particles
        local_idx = dmo.query_halo(center, r200)
        
        if len(local_idx) > 0:
            local_coords = dmo.coords[local_idx]
            local_masses = dmo.masses[local_idx]
            local_types = np.ones(len(local_idx), dtype=int)  # All DM for DMO
            
            # Compute profile contribution
            profile = compute_profile(
                local_coords, local_masses, local_types, center, r200, RADIAL_BINS
            )
            
            if 0 <= mass_bin < n_mass_bins:
                local_dmo_profiles[mass_bin] += profile['density']
                local_dmo_counts[mass_bin] += profile['count']
            
            # Statistics at multiple radii
            dx = local_coords - center
            dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
            dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
            r_norm = np.linalg.norm(dx, axis=1) / r200
            
            for j, r_mult in enumerate(STATS_RADII_MULT):
                mask = r_norm <= r_mult
                local_dmo_stats[i, j] = np.sum(local_masses[mask])
            
            # Mark particles for removal from background
            dmo_keep_mask[local_idx] = False
        
        if rank == 0 and (i + 1) % 500 == 0:
            print(f"    Halo {i+1}/{n_halos}...")
            sys.stdout.flush()
    
    if rank == 0:
        print(f"  Halo processing time: {time.time()-t0:.1f}s")
    
    # Reduce profiles and stats across ranks
    global_dmo_profiles = np.zeros_like(local_dmo_profiles)
    global_dmo_counts = np.zeros_like(local_dmo_counts)
    global_dmo_stats = np.zeros_like(local_dmo_stats)
    
    comm.Reduce(local_dmo_profiles, global_dmo_profiles, op=MPI.SUM, root=0)
    comm.Reduce(local_dmo_counts, global_dmo_counts, op=MPI.SUM, root=0)
    comm.Reduce(local_dmo_stats, global_dmo_stats, op=MPI.SUM, root=0)
    
    # Generate DMO maps
    if rank == 0:
        print("  Generating DMO maps...")
        sys.stdout.flush()
    
    # Full DMO map
    local_dmo_map = project_to_2d(dmo.coords, dmo.masses, args.grid)
    
    if rank == 0:
        global_dmo_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_dmo_map = None
    comm.Reduce(local_dmo_map, global_dmo_map, op=MPI.SUM, root=0)
    del local_dmo_map
    
    # DMO background map (excluding halos)
    local_dmo_bg_map = project_to_2d(dmo.coords[dmo_keep_mask], dmo.masses[dmo_keep_mask], args.grid)
    
    if rank == 0:
        global_dmo_bg_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_dmo_bg_map = None
    comm.Reduce(local_dmo_bg_map, global_dmo_bg_map, op=MPI.SUM, root=0)
    del local_dmo_bg_map
    
    # Free DMO data
    dmo.free()
    del dmo_keep_mask
    gc.collect()
    
    if rank == 0:
        # Save DMO map
        dmo_file = os.path.join(snap_dir, 'projected', 'dmo.npz')
        np.savez_compressed(dmo_file, field=global_dmo_map, box_size=BOX_SIZE,
                           grid_resolution=args.grid, snapshot=args.snap)
        del global_dmo_map
        print("    DMO map saved")
    
    # ========================================================================
    # PASS 2: Hydro
    # ========================================================================
    if rank == 0:
        print("\n[3/7] Processing Hydro simulation...")
        sys.stdout.flush()
    
    hydro = DistributedParticles(args.snap, args.sim_res, 'hydro', args.radius_mult)
    hydro.load().build_tree()
    
    # Initialize accumulators
    local_hydro_profiles = np.zeros((n_mass_bins, n_radial_bins), dtype=np.float64)
    local_hydro_counts = np.zeros((n_mass_bins, n_radial_bins), dtype=np.int64)
    
    # Separated by particle type
    local_hydro_profiles_dm = np.zeros((n_mass_bins, n_radial_bins), dtype=np.float64)
    local_hydro_profiles_gas = np.zeros((n_mass_bins, n_radial_bins), dtype=np.float64)
    local_hydro_profiles_stars = np.zeros((n_mass_bins, n_radial_bins), dtype=np.float64)
    
    # Statistics - only store masses, compute fractions after MPI reduce
    local_hydro_stats = {
        'm_total': np.zeros((n_halos, n_stats_radii), dtype=np.float64),
        'm_gas': np.zeros((n_halos, n_stats_radii), dtype=np.float64),
        'm_stellar': np.zeros((n_halos, n_stats_radii), dtype=np.float64),
        'm_dm': np.zeros((n_halos, n_stats_radii), dtype=np.float64),
    }
    
    # Mask for particles to INCLUDE (in halos, for replace map)
    hydro_halo_mask = np.zeros(len(hydro.coords), dtype=bool)
    
    if rank == 0:
        print(f"  Computing profiles and building mask for {n_halos} halos...")
        t0 = time.time()
    
    for i in range(n_halos):
        # For REPLACE MAP: use DMO position and radius
        center_dmo = halo_positions[i]  # DMO position
        r200_dmo = halo_radii[i]  # DMO radius
        
        # For STATISTICS: use Hydro position and radius (matching original pipeline)
        center_hydro = halo_hydro_positions[i]  # Hydro position
        r200_hydro = halo_hydro_radii[i]  # Hydro radius
        
        mass_bin = mass_bin_indices[i]
        
        # Query local particles for REPLACE MAP (DMO center)
        local_idx_replace = hydro.query_halo(center_dmo, r200_dmo)
        
        # Query local particles for STATISTICS (Hydro center)
        local_idx_stats = hydro.query_halo(center_hydro, r200_hydro)
        
        # Compute profiles and stats
        if len(local_idx_stats) > 0:
            local_coords = hydro.coords[local_idx_stats]
            local_masses = hydro.masses[local_idx_stats]
            local_types = hydro.types[local_idx_stats]
            
            # Compute profile contribution (with type separation) - using HYDRO center/radius
            profile = compute_profile(
                local_coords, local_masses, local_types, center_hydro, r200_hydro, RADIAL_BINS
            )
            
            if 0 <= mass_bin < n_mass_bins:
                local_hydro_profiles[mass_bin] += profile['density']
                local_hydro_counts[mass_bin] += profile['count']
                local_hydro_profiles_dm[mass_bin] += profile['density_dm']
                local_hydro_profiles_gas[mass_bin] += profile['density_gas']
                local_hydro_profiles_stars[mass_bin] += profile['density_stars']
            
            # Statistics at multiple radii - using HYDRO center/radius
            stats = compute_statistics_at_radii(
                local_coords, local_masses, local_types, center_hydro, r200_hydro, STATS_RADII_MULT
            )
            # Only copy the mass fields (fractions will be computed after MPI reduce)
            for key in ['m_total', 'm_gas', 'm_stellar', 'm_dm']:
                local_hydro_stats[key][i] = stats[key]
        
        # Mark particles for inclusion in REPLACE map (DMO center)
        if len(local_idx_replace) > 0:
            hydro_halo_mask[local_idx_replace] = True
        
        if rank == 0 and (i + 1) % 500 == 0:
            print(f"    Halo {i+1}/{n_halos}...")
            sys.stdout.flush()
    
    if rank == 0:
        print(f"  Halo processing time: {time.time()-t0:.1f}s")
    
    # Reduce profiles and stats
    global_hydro_profiles = np.zeros_like(local_hydro_profiles)
    global_hydro_counts = np.zeros_like(local_hydro_counts)
    global_hydro_profiles_dm = np.zeros_like(local_hydro_profiles_dm)
    global_hydro_profiles_gas = np.zeros_like(local_hydro_profiles_gas)
    global_hydro_profiles_stars = np.zeros_like(local_hydro_profiles_stars)
    
    comm.Reduce(local_hydro_profiles, global_hydro_profiles, op=MPI.SUM, root=0)
    comm.Reduce(local_hydro_counts, global_hydro_counts, op=MPI.SUM, root=0)
    comm.Reduce(local_hydro_profiles_dm, global_hydro_profiles_dm, op=MPI.SUM, root=0)
    comm.Reduce(local_hydro_profiles_gas, global_hydro_profiles_gas, op=MPI.SUM, root=0)
    comm.Reduce(local_hydro_profiles_stars, global_hydro_profiles_stars, op=MPI.SUM, root=0)
    
    global_hydro_stats = {}
    for key in local_hydro_stats:
        global_hydro_stats[key] = np.zeros_like(local_hydro_stats[key])
        comm.Reduce(local_hydro_stats[key], global_hydro_stats[key], op=MPI.SUM, root=0)
    
    # Generate Hydro maps
    if rank == 0:
        print("  Generating Hydro maps...")
        sys.stdout.flush()
    
    # Full Hydro map
    local_hydro_map = project_to_2d(hydro.coords, hydro.masses, args.grid)
    
    if rank == 0:
        global_hydro_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_hydro_map = None
    comm.Reduce(local_hydro_map, global_hydro_map, op=MPI.SUM, root=0)
    del local_hydro_map
    
    # Hydro halo map (only particles in halos)
    local_hydro_halo_map = project_to_2d(hydro.coords[hydro_halo_mask], hydro.masses[hydro_halo_mask], args.grid)
    
    if rank == 0:
        global_hydro_halo_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_hydro_halo_map = None
    comm.Reduce(local_hydro_halo_map, global_hydro_halo_map, op=MPI.SUM, root=0)
    del local_hydro_halo_map
    
    # Free Hydro data
    hydro.free()
    del hydro_halo_mask
    gc.collect()
    
    if rank == 0:
        # Save Hydro map
        hydro_file = os.path.join(snap_dir, 'projected', 'hydro.npz')
        np.savez_compressed(hydro_file, field=global_hydro_map, box_size=BOX_SIZE,
                           grid_resolution=args.grid, snapshot=args.snap)
        del global_hydro_map
        print("    Hydro map saved")
    
    # ========================================================================
    # Generate Replace map
    # ========================================================================
    if rank == 0:
        print("\n[4/7] Generating Replace map...")
        
        global_replace_map = global_dmo_bg_map + global_hydro_halo_map
        del global_dmo_bg_map, global_hydro_halo_map
        
        mass_label = f"M{args.mass_min:.1f}".replace('.', 'p')
        replace_file = os.path.join(snap_dir, 'projected', f'replace_{mass_label}.npz')
        np.savez_compressed(replace_file, field=global_replace_map, box_size=BOX_SIZE,
                           grid_resolution=args.grid, snapshot=args.snap,
                           log_mass_min=args.mass_min, radius_multiplier=args.radius_mult)
        del global_replace_map
        print(f"    Replace map saved: {replace_file}")
    
    # ========================================================================
    # Save profiles
    # ========================================================================
    if rank == 0:
        print("\n[5/7] Saving profiles...")
        
        profile_file = os.path.join(output_dir, 'profiles', f'profiles_snap{args.snap:03d}.h5')
        with h5py.File(profile_file, 'w') as f:
            f.attrs['snapshot'] = args.snap
            f.attrs['mass_min'] = args.mass_min
            f.attrs['radius_multiplier'] = args.radius_mult
            f.attrs['radial_bins'] = RADIAL_BINS
            f.attrs['mass_bin_edges'] = MASS_BIN_EDGES
            f.attrs['box_size'] = BOX_SIZE
            
            # DMO profiles
            f.create_dataset('stacked_dmo', data=global_dmo_profiles)
            f.create_dataset('counts_dmo', data=global_dmo_counts)
            
            # Hydro profiles (total)
            f.create_dataset('stacked_hydro', data=global_hydro_profiles)
            f.create_dataset('counts_hydro', data=global_hydro_counts)
            
            # Hydro profiles (by particle type)
            f.create_dataset('stacked_hydro_dm', data=global_hydro_profiles_dm)
            f.create_dataset('stacked_hydro_gas', data=global_hydro_profiles_gas)
            f.create_dataset('stacked_hydro_stars', data=global_hydro_profiles_stars)
        
        print(f"    Saved: {profile_file}")
    
    # ========================================================================
    # Save statistics (matching original format)
    # ========================================================================
    if rank == 0:
        print("\n[6/7] Saving statistics...")
        
        # Compute baryon fractions from reduced masses (NOT from summed fractions!)
        m_total = global_hydro_stats['m_total']
        m_gas = global_hydro_stats['m_gas']
        m_stellar = global_hydro_stats['m_stellar']
        m_dm = global_hydro_stats['m_dm']
        
        # Compute fractions safely (avoid divide by zero)
        f_baryon = np.where(m_total > 0, (m_gas + m_stellar) / m_total, 0).astype(np.float32)
        f_gas = np.where(m_total > 0, m_gas / m_total, 0).astype(np.float32)
        f_stellar = np.where(m_total > 0, m_stellar / m_total, 0).astype(np.float32)
        
        # Compute mass conservation ratios
        ratio_total = np.where(global_dmo_stats > 0, m_total / global_dmo_stats, 0).astype(np.float32)
        ratio_dm = np.where(global_dmo_stats > 0, m_dm / global_dmo_stats, 0).astype(np.float32)
        
        stats_file = os.path.join(output_dir, 'analysis', f'halo_statistics_snap{args.snap:03d}.h5')
        with h5py.File(stats_file, 'w') as f:
            # Attributes (matching original format)
            f.attrs['snapshot'] = args.snap
            f.attrs['mass_min'] = args.mass_min
            f.attrs['n_halos'] = n_halos
            f.attrs['radii_r200'] = np.array(STATS_RADII_MULT)
            f.attrs['sim_res'] = args.sim_res
            
            # Halo properties (matching original format)
            f.create_dataset('log_masses', data=halo_log_masses.astype(np.float32))
            f.create_dataset('positions_dmo', data=halo_positions.astype(np.float32))
            f.create_dataset('positions_hydro', data=halo_hydro_positions.astype(np.float32))
            f.create_dataset('radii_dmo', data=halo_radii.astype(np.float32))
            f.create_dataset('radii_hydro', data=halo_hydro_radii.astype(np.float32))
            
            # Baryon fractions - shape (n_halos, n_radii)
            f.create_dataset('f_baryon', data=f_baryon)
            f.create_dataset('f_gas', data=f_gas)
            f.create_dataset('f_stellar', data=f_stellar)
            
            # Masses - shape (n_halos, n_radii)
            f.create_dataset('m_total', data=m_total)
            f.create_dataset('m_gas', data=m_gas)
            f.create_dataset('m_stellar', data=m_stellar)
            f.create_dataset('m_dm_hydro', data=m_dm)
            f.create_dataset('m_dmo', data=global_dmo_stats)
            
            # Mass conservation ratios - shape (n_halos, n_radii)
            f.create_dataset('ratio_total', data=ratio_total)
            f.create_dataset('ratio_dm', data=ratio_dm)
        
        print(f"    Saved: {stats_file}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    if rank == 0:
        print("\n[7/7] Complete!")
        print("=" * 70)
        print(f"Total time: {time.time()-t_start:.1f}s")
        print(f"Output directory: {output_dir}")
        print("=" * 70)
        
        # Print profile summary
        print("\nProfile summary (halos per mass bin):")
        for i in range(n_mass_bins):
            n_in_bin = np.sum(mass_bin_indices == i)
            print(f"  [{MASS_BIN_EDGES[i]:.1f}, {MASS_BIN_EDGES[i+1]:.1f}): {n_in_bin}")


def main():
    parser = argparse.ArgumentParser(description='Unified pipeline for profiles, stats, and maps')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, default=2500, choices=[625, 1250, 2500])
    parser.add_argument('--mass-min', type=float, default=12.5,
                        help='Minimum log10(M200c/Msun/h) for halo selection')
    parser.add_argument('--radius-mult', type=float, default=5.0,
                        help='Radius multiplier (×R200) for particle queries')
    parser.add_argument('--grid', type=int, default=GRID_RES,
                        help='Grid resolution for maps')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Suffix for output directory (e.g., "_test" to avoid overwriting)')
    
    args = parser.parse_args()
    run_unified_pipeline(args)


if __name__ == '__main__':
    main()
