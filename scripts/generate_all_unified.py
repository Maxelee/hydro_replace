#!/usr/bin/env python
"""
Unified pipeline: Generate profiles, statistics, maps, and lensplanes.

This script processes DMO and Hydro simulations sequentially, computing:
- Stacked density profiles (by mass bin)
- Halo statistics (masses, particle counts, etc.)
- 2D projected density maps (DMO, Hydro, Replace)
- Lensplanes for ray-tracing with multiple (mass_bin, R_factor) configurations

No caching required - KDTree is built once and reused for multiple configs.

Algorithm:
  PHASE 1 (Load & Build):
    - Load all DMO and Hydro particles
    - Build KDTrees (kept in memory for lensplane configs)
  
  PHASE 2 (Profiles & Statistics):
    - For each halo: query particles, compute profiles/stats
  
  PHASE 3 (2D Maps):
    - Generate DMO, Hydro, and Replace maps
    - Track particle masks for lensplane generation
  
  PHASE 4 (DMO & Hydro Lensplanes):
    - Generate N×pps lensplanes for DMO and Hydro
    - These are computed once (independent of mass bin/R factor)
  
  PHASE 5 (Replace Lensplanes):
    - Loop over (mass_bin, R_factor) configurations
    - Query KDTrees (fast - already built)
    - Generate Replace lensplanes for each config

Usage:
    # Full pipeline
    mpirun -np 32 python generate_all_unified.py --snap 99 --sim-res 625 --enable-lensplanes
    
    # Phase 5 only (add new configurations)
    mpirun -np 32 python generate_all_unified.py --snap 99 --sim-res 625 --phase 5 --incremental
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
# Lensplane Configuration
# ============================================================================

LENSPLANE_CONFIG = {
    'enabled': False,  # Set via --enable-lensplanes
    'n_realizations': 10,  # N random rotation/translation realizations (LP directories)
    'planes_per_snapshot': 2,  # pps - depth slices per snapshot
    'grid_res': 4096,  # Grid resolution for lensplanes
    'seed': 2020,  # Random seed for reproducibility
    'output_base': '/mnt/home/mlee1/ceph/hydro_replace_LP',
}

# Snapshot order for ray-tracing (from z≈0 to z≈2)
# Maps snapshot number -> index in lightcone (0-19)
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
SNAPSHOT_TO_INDEX = {snap: idx for idx, snap in enumerate(SNAPSHOT_ORDER)}
N_SNAPSHOTS = len(SNAPSHOT_ORDER)

# Mass bins for Replace configurations
# Format: (M_lo, M_hi, label)
MASS_BINS = [
    # Exclusive bins
    (10**12.0, 10**12.5, 'M12.0-12.5'),
    (10**12.5, 10**13.0, 'M12.5-13.0'),
    (10**13.0, 10**13.5, 'M13.0-13.5'),
    (10**13.5, 10**15.0, 'M13.5-15.0'),
    # Cumulative bins
    (10**12.0, np.inf, 'M12.0+'),
    (10**12.5, np.inf, 'M12.5+'),
    (10**13.0, np.inf, 'M13.0+'),
    (10**13.5, np.inf, 'M13.5+'),
]

# Excision radius factors
R_FACTORS = [0.5, 1.0, 3.0, 5.0]


# ============================================================================
# Lensplane Transform Generator
# ============================================================================

class TransformGenerator:
    """Generate reproducible random transforms for lensplanes.
    
    Structure for lux compatibility:
    - N_realizations LP directories (LP_00, LP_01, ..., LP_09)
    - Each LP has unique transforms for each snapshot
    - Same transform used for both pps slices of same snapshot
    - SAME transforms used across all models (DMO, Hydro, Replace)
    
    Transform array shape: (n_realizations, n_snapshots)
    """
    
    def __init__(self, n_realizations=10, n_snapshots=20, pps=2, seed=2020, box_size=205.0):
        self.n_realizations = n_realizations
        self.n_snapshots = n_snapshots
        self.pps = pps
        self.seed = seed
        self.box_size = box_size
        
        # Pre-generate transforms: shape (n_realizations, n_snapshots)
        rng = np.random.RandomState(seed)
        self.proj_dirs = rng.randint(0, 3, (n_realizations, n_snapshots))
        self.displacements = rng.uniform(0, box_size, (n_realizations, n_snapshots, 3))
        self.flips = rng.choice([True, False], (n_realizations, n_snapshots))
    
    def get_transform(self, realization_idx, snapshot_idx):
        """Get transform parameters for a specific (realization, snapshot).
        
        Args:
            realization_idx: Which LP directory (0-9)
            snapshot_idx: Which snapshot in the sequence (0-19, where 0=snap96)
        
        Returns:
            dict with proj_dir, displacement, flip
        """
        return {
            'proj_dir': self.proj_dirs[realization_idx, snapshot_idx],
            'displacement': self.displacements[realization_idx, snapshot_idx],
            'flip': self.flips[realization_idx, snapshot_idx],
        }
    
    def save(self, filepath):
        """Save transforms for reproducibility."""
        with h5py.File(filepath, 'w') as f:
            f.attrs['n_realizations'] = self.n_realizations
            f.attrs['n_snapshots'] = self.n_snapshots
            f.attrs['pps'] = self.pps
            f.attrs['seed'] = self.seed
            f.attrs['box_size'] = self.box_size
            f.create_dataset('proj_dirs', data=self.proj_dirs)
            f.create_dataset('displacements', data=self.displacements)
            f.create_dataset('flips', data=self.flips)
            f.create_dataset('snapshot_order', data=SNAPSHOT_ORDER)
    
    @classmethod
    def load(cls, filepath):
        """Load transforms from file."""
        with h5py.File(filepath, 'r') as f:
            gen = cls(
                n_realizations=f.attrs['n_realizations'],
                n_snapshots=f.attrs['n_snapshots'],
                pps=f.attrs['pps'],
                seed=f.attrs['seed'],
                box_size=f.attrs['box_size']
            )
            # Overwrite with saved values (in case seed changed)
            gen.proj_dirs = f['proj_dirs'][:]
            gen.displacements = f['displacements'][:]
            gen.flips = f['flips'][:]
        return gen


def apply_transform(pos, transform, box_size):
    """Apply rotation/translation/flip to positions.
    
    Args:
        pos: (N, 3) original positions
        transform: dict with 'displacement', 'flip'
        box_size: simulation box size
    
    Returns:
        (N, 3) transformed positions (periodic BC applied)
    """
    # Apply displacement
    pos_t = pos + transform['displacement']
    
    # Apply flip (negate coordinates if True)
    if transform['flip']:
        pos_t = box_size - pos_t
    
    # Periodic boundary conditions
    pos_t = pos_t % box_size
    
    return pos_t


def project_lensplane(pos, mass, transform, grid_res, box_size, pps_slice, pps=2):
    """Transform positions and project to 2D lensplane.
    
    Args:
        pos: (N, 3) particle positions in original coordinates
        mass: (N,) particle masses
        transform: dict with 'proj_dir', 'displacement', 'flip'
        grid_res: output grid resolution
        box_size: simulation box size (Mpc/h)
        pps_slice: which depth slice (0 or 1 for pps=2)
        pps: planes per snapshot (for depth slicing)
    
    Returns:
        (grid_res, grid_res) density field
    """
    if len(pos) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float64)
    
    # Apply transformation
    pos_t = apply_transform(pos, transform, box_size)
    
    # Get projection axes based on projection direction
    proj_dir = transform['proj_dir']
    depth_axis = proj_dir
    plane_axes = [i for i in range(3) if i != depth_axis]
    
    # Slice by depth (for pps=2, each slice is half the box)
    depth = pos_t[:, depth_axis]
    depth_min = pps_slice * box_size / pps
    depth_max = (pps_slice + 1) * box_size / pps
    in_slice = (depth >= depth_min) & (depth < depth_max)
    
    if np.sum(in_slice) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32)
    
    # Extract 2D coordinates
    pos_2d = np.ascontiguousarray(pos_t[in_slice][:, plane_axes].astype(np.float32))
    mass_slice = mass[in_slice].astype(np.float32)
    
    # TSC mass assignment - MASL requires float32 arrays
    delta = np.zeros((grid_res, grid_res), dtype=np.float32)
    MASL.MA(pos_2d, delta, np.float32(box_size), MAS='TSC', W=mass_slice, verbose=False)
    
    return delta


def write_lensplane(filepath, delta, grid_res):
    """Write lensplane in lux binary format.
    
    Format: [int32: grid_size] [float64[grid²]: data] [int32: grid_size]
    
    Args:
        filepath: output file path
        delta: (grid_res, grid_res) density field
        grid_res: grid size
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        # Header: grid size as int32
        f.write(np.array([grid_res], dtype=np.int32).tobytes())
        # Data: density field as float64, row-major
        f.write(delta.astype(np.float64).tobytes())
        # Footer: grid size as int32
        f.write(np.array([grid_res], dtype=np.int32).tobytes())


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
    
    # Free DMO data (unless lensplanes enabled - need to keep for Phase 4-5)
    if not args.enable_lensplanes:
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
    
    # Free Hydro data (unless lensplanes enabled)
    if not args.enable_lensplanes:
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
    # Lensplane Generation (Phase 4-5)
    # ========================================================================
    if args.enable_lensplanes:
        # Prepare halos dict for lensplane generation
        halos = {
            'masses': halo_masses,
            'positions': halo_positions,
            'radii': halo_radii,
        }
        
        run_lensplane_generation(args, dmo, hydro, halos, comm)
        
        # Now free the particles
        dmo.free()
        hydro.free()
        gc.collect()
    
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


# ============================================================================
# Lensplane Generation Functions
# ============================================================================

def generate_model_lensplanes(model_name, pos, mass, transforms, lp_grid, box_size, 
                               output_dir, snap, comm):
    """Generate lensplanes for a single model (DMO or Hydro).
    
    Args:
        model_name: 'dmo' or 'hydro'
        pos: particle positions (local to this rank)
        mass: particle masses (local to this rank)
        transforms: TransformGenerator instance
        lp_grid: lensplane grid resolution
        box_size: simulation box size
        output_dir: base output directory for lensplanes (WITHOUT snap subdirectory)
        snap: snapshot number
        comm: MPI communicator
    """
    rank = comm.Get_rank()
    
    # Get snapshot index in the lightcone ordering
    if snap not in SNAPSHOT_TO_INDEX:
        if rank == 0:
            print(f"  Warning: snapshot {snap} not in SNAPSHOT_ORDER, skipping lensplanes")
        return
    
    snapshot_idx = SNAPSHOT_TO_INDEX[snap]
    pps = transforms.pps
    n_realizations = transforms.n_realizations
    
    if rank == 0:
        print(f"  Generating {model_name.upper()} lensplanes for snap {snap} (idx={snapshot_idx})...")
        print(f"    {n_realizations} realizations × {pps} pps slices")
        sys.stdout.flush()
    
    # Loop over realizations and pps slices
    for real_idx in range(n_realizations):
        # Get transform for this (realization, snapshot) - same for both pps slices
        t = transforms.get_transform(real_idx, snapshot_idx)
        
        for pps_slice in range(pps):
            # File index: snapshot_idx * pps + pps_slice
            # e.g., snap96 (idx=0): lenspot00.dat, lenspot01.dat
            # e.g., snap90 (idx=1): lenspot02.dat, lenspot03.dat
            file_idx = snapshot_idx * pps + pps_slice
            
            # Each rank projects its local particles (returns float32)
            local_delta = project_lensplane(pos, mass, t, lp_grid, box_size, pps_slice, pps)
            
            # Reduce across MPI ranks - use float64 for accumulation precision
            if rank == 0:
                global_delta = np.zeros((lp_grid, lp_grid), dtype=np.float64)
            else:
                global_delta = None
            
            # Convert to float64 for MPI reduce (better precision for large sums)
            local_delta_contig = np.ascontiguousarray(local_delta.astype(np.float64))
            comm.Reduce(local_delta_contig, global_delta, op=MPI.SUM, root=0)
            
            # Write (rank 0 only) - lux expects float64
            if rank == 0:
                plane_dir = os.path.join(output_dir, model_name, f'LP_{real_idx:02d}')
                os.makedirs(plane_dir, exist_ok=True)
                filepath = os.path.join(plane_dir, f'lenspot{file_idx:02d}.dat')
                write_lensplane(filepath, global_delta, lp_grid)
            
            del local_delta, local_delta_contig
            if rank == 0:
                del global_delta
        
        if rank == 0 and (real_idx + 1) % 5 == 0:
            print(f"    Realization {real_idx + 1}/{n_realizations} done")
            sys.stdout.flush()
    
    comm.Barrier()


def generate_replace_lensplanes_for_config(config, transforms, dmo_particles, hydro_particles,
                                            halos, lp_grid, box_size, output_dir, snap, comm):
    """Generate Replace lensplanes for one (mass_bin, R_factor) configuration.
    
    Args:
        config: tuple of (M_lo, M_hi, R_factor, label)
        transforms: TransformGenerator instance
        dmo_particles: DistributedParticles for DMO
        hydro_particles: DistributedParticles for Hydro
        halos: dict with 'masses', 'positions', 'radii'
        lp_grid: lensplane grid resolution
        box_size: simulation box size
        output_dir: base output directory for lensplanes (WITHOUT snap subdirectory)
        snap: snapshot number
        comm: MPI communicator
    """
    rank = comm.Get_rank()
    M_lo, M_hi, R_factor, label = config
    
    # Get snapshot index in the lightcone ordering
    if snap not in SNAPSHOT_TO_INDEX:
        if rank == 0:
            print(f"  Warning: snapshot {snap} not in SNAPSHOT_ORDER, skipping lensplanes")
        return
    
    snapshot_idx = SNAPSHOT_TO_INDEX[snap]
    pps = transforms.pps
    n_realizations = transforms.n_realizations
    
    # Select halos in mass bin
    halo_mask = (halos['masses'] >= M_lo) & (halos['masses'] < M_hi)
    selected_positions = halos['positions'][halo_mask]
    selected_radii = halos['radii'][halo_mask]
    n_selected = len(selected_positions)
    
    if rank == 0:
        print(f"  Config {label}, R={R_factor}: {n_selected} halos")
        sys.stdout.flush()
    
    if n_selected == 0:
        if rank == 0:
            print(f"    No halos in this bin, skipping")
        return
    
    # Query which LOCAL DMO particles are in halos (to EXCLUDE)
    excision_radii = selected_radii * R_factor
    local_dmo_in_halo = np.zeros(len(dmo_particles.coords), dtype=bool)
    
    for center, radius in zip(selected_positions, excision_radii):
        idx = dmo_particles.query_halo(center, radius / dmo_particles.radius_mult)
        if len(idx) > 0:
            local_dmo_in_halo[idx] = True
    
    # Query which LOCAL Hydro particles are in halos (to INCLUDE)
    local_hydro_in_halo = np.zeros(len(hydro_particles.coords), dtype=bool)
    
    for center, radius in zip(selected_positions, excision_radii):
        idx = hydro_particles.query_halo(center, radius / hydro_particles.radius_mult)
        if len(idx) > 0:
            local_hydro_in_halo[idx] = True
    
    # Build local Replace arrays
    local_pos_replace = np.concatenate([
        dmo_particles.coords[~local_dmo_in_halo],      # DMO background
        hydro_particles.coords[local_hydro_in_halo]    # Hydro halos
    ])
    local_mass_replace = np.concatenate([
        dmo_particles.masses[~local_dmo_in_halo],
        hydro_particles.masses[local_hydro_in_halo]
    ])
    
    n_dmo_bg = np.sum(~local_dmo_in_halo)
    n_hydro_halo = np.sum(local_hydro_in_halo)
    
    if rank == 0:
        print(f"    Rank 0: {n_dmo_bg:,} DMO bg + {n_hydro_halo:,} Hydro halo particles")
        sys.stdout.flush()
    
    # Output directory for this config
    config_label = f"hydro_replace_Ml_{M_lo:.2e}_Mu_{M_hi:.2e}_R_{R_factor}".replace('+', '')
    
    # Generate lensplanes for all realizations and pps slices
    for real_idx in range(n_realizations):
        # Get transform for this (realization, snapshot) - same for both pps slices
        t = transforms.get_transform(real_idx, snapshot_idx)
        
        for pps_slice in range(pps):
            # File index: snapshot_idx * pps + pps_slice
            file_idx = snapshot_idx * pps + pps_slice
            
            # Transform and project local particles (returns float32)
            local_delta = project_lensplane(local_pos_replace, local_mass_replace, t, 
                                             lp_grid, box_size, pps_slice, pps)
            
            # Reduce across MPI ranks - use float64 for accumulation precision
            if rank == 0:
                global_delta = np.zeros((lp_grid, lp_grid), dtype=np.float64)
            else:
                global_delta = None
            
            # Convert to float64 for MPI reduce
            local_delta_contig = np.ascontiguousarray(local_delta.astype(np.float64))
            comm.Reduce(local_delta_contig, global_delta, op=MPI.SUM, root=0)
            
            # Write (rank 0 only) - lux expects float64
            if rank == 0:
                plane_dir = os.path.join(output_dir, config_label, f'LP_{real_idx:02d}')
                os.makedirs(plane_dir, exist_ok=True)
                filepath = os.path.join(plane_dir, f'lenspot{file_idx:02d}.dat')
                write_lensplane(filepath, global_delta, lp_grid)
            
            del local_delta, local_delta_contig
            if rank == 0:
                del global_delta
    
    if rank == 0:
        print(f"    Done: {n_realizations} realizations × {pps} pps slices for snap {snap}")
        sys.stdout.flush()
    
    comm.Barrier()


def run_lensplane_generation(args, dmo_particles, hydro_particles, halos, comm):
    """Run lensplane generation (Phases 4 and 5).
    
    Args:
        args: command-line arguments
        dmo_particles: DistributedParticles for DMO (with KDTree built)
        hydro_particles: DistributedParticles for Hydro (with KDTree built)
        halos: dict with 'masses', 'positions', 'radii'
        comm: MPI communicator
    """
    rank = comm.Get_rank()
    
    lp_config = LENSPLANE_CONFIG.copy()
    lp_config['grid_res'] = args.lensplane_grid
    
    # Output directory: NO snap subdirectory - files accumulate across snapshots
    # Structure: {output_base}/L205n{res}TNG/{model}/LP_{real:02d}/lenspot{idx:02d}.dat
    output_dir = os.path.join(lp_config['output_base'], f'L205n{args.sim_res}TNG')
    
    # Check if this snapshot is in the ray-tracing list
    if args.snap not in SNAPSHOT_TO_INDEX:
        if rank == 0:
            print(f"\nWarning: Snapshot {args.snap} not in SNAPSHOT_ORDER, skipping lensplanes")
            print(f"  Valid snapshots: {SNAPSHOT_ORDER}")
        return
    
    snapshot_idx = SNAPSHOT_TO_INDEX[args.snap]
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("LENSPLANE GENERATION")
        print("=" * 70)
        print(f"Snapshot: {args.snap} (index {snapshot_idx} of {N_SNAPSHOTS})")
        print(f"N realizations: {lp_config['n_realizations']}")
        print(f"Planes per snapshot: {lp_config['planes_per_snapshot']}")
        print(f"Grid resolution: {lp_config['grid_res']}")
        print(f"Output: {output_dir}")
        print(f"Files: lenspot{snapshot_idx * lp_config['planes_per_snapshot']:02d}.dat - lenspot{(snapshot_idx + 1) * lp_config['planes_per_snapshot'] - 1:02d}.dat")
        print("=" * 70)
        sys.stdout.flush()
        
        os.makedirs(output_dir, exist_ok=True)
    
    comm.Barrier()
    
    # Create transform generator
    transforms = TransformGenerator(
        n_realizations=lp_config['n_realizations'],
        pps=lp_config['planes_per_snapshot'],
        seed=lp_config['seed'],
        box_size=BOX_SIZE
    )
    
    # Save transforms for reproducibility (only once, first snapshot)
    if rank == 0:
        transform_file = os.path.join(output_dir, 'transforms.h5')
        if not os.path.exists(transform_file):
            transforms.save(transform_file)
            print(f"  Saved transforms to {transform_file}")
        else:
            print(f"  Transforms already exist at {transform_file}")
    
    # ========================================================================
    # Phase 4: DMO and Hydro lensplanes
    # ========================================================================
    if not args.phase5_only:
        if rank == 0:
            print("\n[Phase 4] Generating DMO and Hydro lensplanes...")
            t0 = time.time()
        
        # DMO lensplanes
        generate_model_lensplanes('dmo', dmo_particles.coords, dmo_particles.masses,
                                   transforms, lp_config['grid_res'], BOX_SIZE,
                                   output_dir, args.snap, comm)
        
        # Hydro lensplanes
        generate_model_lensplanes('hydro', hydro_particles.coords, hydro_particles.masses,
                                   transforms, lp_config['grid_res'], BOX_SIZE,
                                   output_dir, args.snap, comm)
        
        if rank == 0:
            print(f"  Phase 4 time: {time.time()-t0:.1f}s")
    
    # ========================================================================
    # Phase 5: Replace lensplanes for each configuration
    # ========================================================================
    if rank == 0:
        print("\n[Phase 5] Generating Replace lensplanes...")
        print(f"  Mass bins: {len(MASS_BINS)}")
        print(f"  R factors: {R_FACTORS}")
        print(f"  Total configs: {len(MASS_BINS) * len(R_FACTORS)}")
        t0 = time.time()
    
    config_count = 0
    total_configs = len(MASS_BINS) * len(R_FACTORS)
    
    for M_lo, M_hi, label in MASS_BINS:
        for R_factor in R_FACTORS:
            config_count += 1
            
            # Check if this config already exists (incremental mode)
            # Now checks ALL LPs and ALL pps slices for this snapshot
            if args.incremental:
                config_label = f"hydro_replace_Ml_{M_lo:.2e}_Mu_{M_hi:.2e}_R_{R_factor}".replace('+', '')
                config_dir = os.path.join(output_dir, config_label)
                
                # Check all LP directories and all pps slices for this snapshot
                pps = lp_config['planes_per_snapshot']
                all_exist = True
                missing_count = 0
                for lp_idx in range(lp_config['n_realizations']):
                    for pps_slice in range(pps):
                        file_idx = snapshot_idx * pps + pps_slice
                        check_file = os.path.join(config_dir, f'LP_{lp_idx:02d}', f'lenspot{file_idx:02d}.dat')
                        if not os.path.exists(check_file):
                            all_exist = False
                            missing_count += 1
                
                if all_exist:
                    if rank == 0:
                        print(f"\n[{config_count}/{total_configs}] Skipping {label}, R={R_factor} (all LPs complete)")
                    continue
                else:
                    if rank == 0:
                        print(f"\n[{config_count}/{total_configs}] Processing {label}, R={R_factor} ({missing_count} missing files)...")
            else:
                if rank == 0:
                    print(f"\n[{config_count}/{total_configs}] Processing {label}, R={R_factor}...")
            
            config = (M_lo, M_hi, R_factor, label)
            generate_replace_lensplanes_for_config(
                config, transforms, dmo_particles, hydro_particles,
                halos, lp_config['grid_res'], BOX_SIZE, output_dir, args.snap, comm
            )
    
    if rank == 0:
        print(f"\n  Phase 5 time: {time.time()-t0:.1f}s")
        print(f"  Total configs processed: {config_count}")


def main():
    parser = argparse.ArgumentParser(description='Unified pipeline for profiles, stats, maps, and lensplanes')
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
    
    # Lensplane arguments
    parser.add_argument('--enable-lensplanes', action='store_true',
                        help='Enable lensplane generation (Phases 4-5)')
    parser.add_argument('--lensplane-grid', type=int, default=LENSPLANE_CONFIG['grid_res'],
                        help='Grid resolution for lensplanes')
    parser.add_argument('--phase5-only', action='store_true',
                        help='Skip Phases 1-4, only run Phase 5 (Replace lensplanes)')
    parser.add_argument('--incremental', action='store_true',
                        help='Skip configs that already exist')
    
    args = parser.parse_args()
    
    if args.phase5_only:
        # Phase 5 only: need to load particles and build trees
        run_phase5_only(args)
    else:
        run_unified_pipeline(args)


def run_phase5_only(args):
    """Run only Phase 5 (Replace lensplanes) with existing data."""
    t_start = time.time()
    
    if rank == 0:
        print("=" * 70)
        print("PHASE 5 ONLY MODE (Replace Lensplanes)")
        print("=" * 70)
        print(f"Snapshot: {args.snap}")
        print(f"Resolution: L205n{args.sim_res}TNG")
        print("=" * 70)
        sys.stdout.flush()
    
    # Load halo catalog
    output_dir_base = os.path.join(OUTPUT_BASE, f'L205n{args.sim_res}TNG')
    matches_file = os.path.join(output_dir_base, 'matches', f'matches_snap{args.snap:03d}.npz')
    
    if rank == 0:
        print("\n[1/3] Loading halo catalog...")
    
    with np.load(matches_file) as data:
        all_masses = data['dmo_masses'] * MASS_UNIT
        all_positions = data['dmo_positions'] / 1e3
        all_radii = data['dmo_radii'] / 1e3
    
    # Select halos above minimum threshold (use lowest mass bin)
    min_mass = min(mb[0] for mb in MASS_BINS)
    mass_mask = all_masses >= min_mass
    
    halos = {
        'masses': all_masses[mass_mask],
        'positions': all_positions[mass_mask],
        'radii': all_radii[mass_mask],
    }
    
    if rank == 0:
        print(f"  Halos above {min_mass:.1e}: {len(halos['masses'])}")
    
    # Load particles
    if rank == 0:
        print("\n[2/3] Loading particles and building KDTrees...")
    
    dmo = DistributedParticles(args.snap, args.sim_res, 'dmo', args.radius_mult)
    dmo.load().build_tree()
    
    hydro = DistributedParticles(args.snap, args.sim_res, 'hydro', args.radius_mult)
    hydro.load().build_tree()
    
    # Run lensplane generation
    if rank == 0:
        print("\n[3/3] Running lensplane generation...")
    
    # Force phase5_only mode for this call
    args.phase5_only = True
    run_lensplane_generation(args, dmo, hydro, halos, comm)
    
    # Cleanup
    dmo.free()
    hydro.free()
    
    if rank == 0:
        print("\n" + "=" * 70)
        print(f"Total time: {time.time()-t_start:.1f}s")
        print("=" * 70)


if __name__ == '__main__':
    main()
