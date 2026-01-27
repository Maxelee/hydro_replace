#!/usr/bin/env python
"""
Assembly Bias Detection via Halo Replacement.

This script implements a method to detect assembly bias signatures in weak lensing
by replacing halos based on their concentration (formation history proxy).

Methodology:
1. Select halos in a narrow mass bin (e.g., 10^13 M⊙/h)
2. Compute concentration c200c from R200c and Rmax (from SubhaloVmaxRad)
3. Split halos at median concentration:
   - Early-forming (high c): halos with c > median(c)
   - Late-forming (low c): halos with c <= median(c)
4. Generate Replace density fields:
   - Replace_early: DMO background + early-forming DMO halos
   - Replace_late: DMO background + late-forming DMO halos
5. Compare lensing statistics (power spectrum, peaks) between the two

The key insight: In DMO simulations, both populations should trace the same LSS
since baryonic effects are absent. Any difference in clustering/lensing between
early vs late forming halos would be a signature of assembly bias.

Usage:
    mpirun -np 32 python generate_assembly_bias.py --snap 99 --sim-res 2500 --enable-lensplanes
    
    # Quick test with single snapshot
    mpirun -np 8 python generate_assembly_bias.py --snap 99 --sim-res 2500 --test
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
import illustris_python as il

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

OUTPUT_BASE = '/mnt/home/mlee1/ceph/assembly_bias'
BOX_SIZE = 205.0  # Mpc/h
MASS_UNIT = 1e10  # Convert to Msun/h
GRID_RES = 4096   # Default grid resolution

# Target mass bin for assembly bias study (log10 M⊙/h)
DEFAULT_MASS_CENTER = 13.0
DEFAULT_MASS_WIDTH = 0.2  # ± 0.2 dex -> 12.8 to 13.2

# Excision radius for Replace fields (in units of R200)
DEFAULT_RADIUS_MULT = 3.0

# Snapshot order for ray-tracing (from z≈0 to z≈2)
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
SNAPSHOT_TO_INDEX = {snap: idx for idx, snap in enumerate(SNAPSHOT_ORDER)}


# ============================================================================
# Lensplane Configuration
# ============================================================================

LENSPLANE_CONFIG = {
    'n_realizations': 20,
    'planes_per_snapshot': 2,
    'grid_res': 4096,
    'seed': 2020,
}


# ============================================================================
# Concentration Computation
# ============================================================================

def compute_concentrations(basePath, snap, halo_indices):
    """Compute c200c for selected halos using Rmax from central subhalos.
    
    The concentration is computed as:
        c200c = 2.163 * R200c / Rmax
    
    where Rmax is SubhaloVmaxRad (radius of maximum circular velocity) and
    2.163 is the factor relating Rmax to Rs for an NFW profile.
    
    Args:
        basePath: path to TNG simulation output
        snap: snapshot number
        halo_indices: array of halo indices to compute concentrations for
    
    Returns:
        c200c: array of concentrations (NaN for halos without valid central subhalo)
        valid_mask: boolean mask of halos with valid concentration
    """
    # Load halo properties
    halos = il.groupcat.loadHalos(basePath, snap, 
        fields=['Group_R_Crit200', 'GroupFirstSub'])
    
    # Load SubhaloVmaxRad for all subhalos
    vmax_rad_all = il.groupcat.loadSubhalos(basePath, snap, fields=['SubhaloVmaxRad'])
    
    # Get R200 and first subhalo index for selected halos
    r200 = halos['Group_R_Crit200'][halo_indices]  # kpc/h
    first_sub = halos['GroupFirstSub'][halo_indices]
    
    # Initialize output
    c200c = np.full(len(halo_indices), np.nan)
    valid_mask = first_sub >= 0
    
    # Get Rmax for valid halos (halos with central subhalo)
    valid_first_sub = first_sub[valid_mask].astype(np.int64)
    rmax = vmax_rad_all[valid_first_sub]  # kpc/h
    r200_valid = r200[valid_mask]
    
    # Compute concentration: c = 2.163 * R200 / Rmax
    c200c[valid_mask] = 2.163 * r200_valid / rmax
    
    return c200c, valid_mask


def split_by_concentration(c200c, method='median'):
    """Split halos into early-forming (high c) and late-forming (low c) populations.
    
    Args:
        c200c: array of concentrations
        method: 'median' for median split, 'quartile' for top/bottom quartiles
    
    Returns:
        early_mask: boolean mask for early-forming halos (high c)
        late_mask: boolean mask for late-forming halos (low c)
        threshold: concentration value used for splitting
    """
    # Ignore NaN values
    valid = np.isfinite(c200c)
    
    if method == 'median':
        threshold = np.median(c200c[valid])
        early_mask = valid & (c200c > threshold)
        late_mask = valid & (c200c <= threshold)
    elif method == 'quartile':
        q25, q75 = np.percentile(c200c[valid], [25, 75])
        early_mask = valid & (c200c >= q75)  # Top 25%
        late_mask = valid & (c200c <= q25)   # Bottom 25%
        threshold = (q25, q75)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return early_mask, late_mask, threshold


# ============================================================================
# Data Loading (adapted from generate_all_unified.py)
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
        
        coords_list, masses_list = [], []
        
        for filepath in my_files:
            with h5py.File(filepath, 'r') as f:
                for ptype in particle_types:
                    pt_key = f'PartType{ptype}'
                    if pt_key not in f or f[pt_key]['Coordinates'].shape[0] == 0:
                        continue
                    
                    n_part = f[pt_key]['Coordinates'].shape[0]
                    coords_list.append(f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3)
                    
                    if 'Masses' in f[pt_key]:
                        masses_list.append(f[pt_key]['Masses'][:].astype(np.float32) * MASS_UNIT)
                    else:
                        masses_list.append(np.full(n_part, dm_mass * MASS_UNIT, dtype=np.float32))
        
        if coords_list:
            self.coords = np.concatenate(coords_list)
            self.masses = np.concatenate(masses_list)
        else:
            self.coords = np.zeros((0, 3), dtype=np.float32)
            self.masses = np.zeros(0, dtype=np.float32)
        
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
        """Query particles within radius of halo center."""
        if self.tree is None or len(self.coords) == 0:
            return np.array([], dtype=int)
        
        search_radius = radius * self.radius_mult
        indices = self.tree.query_ball_point(center, search_radius)
        
        if len(indices) == 0:
            return np.array([], dtype=int)
        
        indices = np.array(indices)
        
        # Handle periodic boundary conditions
        if np.any(center < search_radius) or np.any(center > BOX_SIZE - search_radius):
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
        del self.coords, self.masses, self.tree
        self.coords = self.masses = self.tree = None
        gc.collect()


# ============================================================================
# Lensplane Generation
# ============================================================================

class TransformGenerator:
    """Generate reproducible random transforms for lensplanes."""
    
    def __init__(self, n_realizations=20, n_snapshots=20, pps=2, seed=2020, box_size=205.0):
        self.n_realizations = n_realizations
        self.n_snapshots = n_snapshots
        self.pps = pps
        self.seed = seed
        self.box_size = box_size
        
        rng = np.random.RandomState(seed)
        self.proj_dirs = rng.randint(0, 3, (n_realizations, n_snapshots))
        self.displacements = rng.uniform(0, box_size, (n_realizations, n_snapshots, 3))
        self.flips = rng.choice([True, False], (n_realizations, n_snapshots))
    
    def get_transform(self, realization_idx, snapshot_idx):
        """Get transform parameters for a specific (realization, snapshot)."""
        return {
            'proj_dir': self.proj_dirs[realization_idx, snapshot_idx],
            'displacement': self.displacements[realization_idx, snapshot_idx],
            'flip': self.flips[realization_idx, snapshot_idx],
        }


def apply_transform(pos, transform, box_size):
    """Apply rotation/translation/flip to positions."""
    pos_t = pos + transform['displacement']
    if transform['flip']:
        pos_t = box_size - pos_t
    pos_t = pos_t % box_size
    return pos_t


def project_lensplane(pos, mass, transform, grid_res, box_size, pps_slice, pps=2):
    """Transform positions and project to 2D lensplane."""
    if len(pos) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float64)
    
    pos_t = apply_transform(pos, transform, box_size)
    
    proj_dir = transform['proj_dir']
    depth_axis = proj_dir
    plane_axes = [i for i in range(3) if i != depth_axis]
    
    depth = pos_t[:, depth_axis]
    depth_min = pps_slice * box_size / pps
    depth_max = (pps_slice + 1) * box_size / pps
    in_slice = (depth >= depth_min) & (depth < depth_max)
    
    if np.sum(in_slice) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32)
    
    pos_2d = np.ascontiguousarray(pos_t[in_slice][:, plane_axes].astype(np.float32))
    mass_slice = mass[in_slice].astype(np.float32)
    
    delta = np.zeros((grid_res, grid_res), dtype=np.float32)
    MASL.MA(pos_2d, delta, np.float32(box_size), MAS='TSC', W=mass_slice, verbose=False)
    
    return delta


def write_lensplane(filepath, delta, grid_res):
    """Write lensplane in lux binary format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        f.write(np.array([grid_res], dtype=np.int32).tobytes())
        f.write(delta.astype(np.float64).tobytes())
        f.write(np.array([grid_res], dtype=np.int32).tobytes())


# ============================================================================
# Main Pipeline
# ============================================================================

def run_assembly_bias_pipeline(args):
    """Run the assembly bias detection pipeline."""
    
    t_start = time.time()
    
    # Output paths
    output_dir = os.path.join(OUTPUT_BASE, f'L205n{args.sim_res}TNG')
    snap_dir = os.path.join(output_dir, f'snap{args.snap:03d}')
    
    if rank == 0:
        print("=" * 70)
        print("ASSEMBLY BIAS DETECTION PIPELINE")
        print("=" * 70)
        print(f"Snapshot: {args.snap}")
        print(f"Resolution: L205n{args.sim_res}TNG")
        print(f"Mass bin: 10^{args.mass_center - args.mass_width:.1f} - 10^{args.mass_center + args.mass_width:.1f} Msun/h")
        print(f"Radius: {args.radius_mult}×R200")
        print(f"Grid: {args.grid}²")
        print(f"Split method: {args.split_method}")
        print(f"Output directory: {output_dir}")
        print("=" * 70)
        sys.stdout.flush()
        
        os.makedirs(snap_dir, exist_ok=True)
    
    comm.Barrier()
    
    # ========================================================================
    # Load halo catalog and compute concentrations
    # ========================================================================
    if rank == 0:
        print("\n[1/5] Loading halo catalog and computing concentrations...")
        sys.stdout.flush()
    
    sim_config = SIM_PATHS[args.sim_res]
    basePath = sim_config['dmo']
    
    # Load halo masses and positions
    halos = il.groupcat.loadHalos(basePath, args.snap, 
        fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos'])
    
    all_masses = halos['Group_M_Crit200'] * MASS_UNIT
    all_positions = halos['GroupPos'] / 1e3  # kpc -> Mpc
    all_radii = halos['Group_R_Crit200'] / 1e3  # kpc -> Mpc
    
    # Select halos in target mass bin
    log_masses = np.log10(all_masses)
    mass_min = args.mass_center - args.mass_width
    mass_max = args.mass_center + args.mass_width
    mass_mask = (log_masses >= mass_min) & (log_masses < mass_max)
    
    halo_indices = np.where(mass_mask)[0]
    n_halos = len(halo_indices)
    
    if rank == 0:
        print(f"  Total halos: {len(all_masses):,}")
        print(f"  Halos in mass bin: {n_halos}")
    
    # Compute concentrations
    c200c, valid_mask = compute_concentrations(basePath, args.snap, halo_indices)
    
    if rank == 0:
        print(f"  Halos with valid concentration: {np.sum(valid_mask)}")
        print(f"  Concentration stats:")
        print(f"    Median: {np.nanmedian(c200c):.2f}")
        print(f"    Mean: {np.nanmean(c200c):.2f}")
        print(f"    Std: {np.nanstd(c200c):.2f}")
        print(f"    10th percentile: {np.nanpercentile(c200c, 10):.2f}")
        print(f"    90th percentile: {np.nanpercentile(c200c, 90):.2f}")
    
    # Split into early and late forming
    early_mask, late_mask, threshold = split_by_concentration(c200c, method=args.split_method)
    
    # Convert to halo catalog indices
    early_halo_idx = halo_indices[early_mask]
    late_halo_idx = halo_indices[late_mask]
    
    if rank == 0:
        print(f"\n  Assembly bias split ({args.split_method}):")
        print(f"    Threshold: {threshold}")
        print(f"    Early-forming (high c): {len(early_halo_idx)} halos")
        print(f"    Late-forming (low c): {len(late_halo_idx)} halos")
        if len(early_halo_idx) > 0:
            print(f"    Early median c: {np.median(c200c[early_mask]):.2f}")
        if len(late_halo_idx) > 0:
            print(f"    Late median c: {np.median(c200c[late_mask]):.2f}")
    
    # Save halo selection and concentrations
    if rank == 0:
        selection_file = os.path.join(snap_dir, 'halo_selection.npz')
        np.savez_compressed(selection_file,
            halo_indices=halo_indices,
            c200c=c200c,
            early_mask=early_mask,
            late_mask=late_mask,
            early_halo_idx=early_halo_idx,
            late_halo_idx=late_halo_idx,
            threshold=threshold,
            mass_center=args.mass_center,
            mass_width=args.mass_width,
            split_method=args.split_method,
        )
        print(f"\n  Saved halo selection: {selection_file}")
    
    # Prepare halo data for Replace field generation
    halos_early = {
        'indices': early_halo_idx,
        'masses': all_masses[early_halo_idx],
        'positions': all_positions[early_halo_idx],
        'radii': all_radii[early_halo_idx],
    }
    halos_late = {
        'indices': late_halo_idx,
        'masses': all_masses[late_halo_idx],
        'positions': all_positions[late_halo_idx],
        'radii': all_radii[late_halo_idx],
    }
    
    # ========================================================================
    # Load DMO particles
    # ========================================================================
    if rank == 0:
        print("\n[2/5] Loading DMO particles...")
        sys.stdout.flush()
    
    dmo = DistributedParticles(args.snap, args.sim_res, 'dmo', radius_mult=args.radius_mult)
    dmo.load()
    dmo.build_tree()
    
    # ========================================================================
    # Generate particle masks for early and late forming halos
    # ========================================================================
    if rank == 0:
        print("\n[3/5] Computing particle masks for early/late halos...")
        sys.stdout.flush()
    
    def compute_halo_particle_mask(particles, halos_dict, radius_mult):
        """Compute mask of particles within halos."""
        mask = np.zeros(len(particles.coords), dtype=bool)
        
        for i, (center, r200) in enumerate(zip(halos_dict['positions'], halos_dict['radii'])):
            r_excise = r200 * radius_mult
            idx = particles.query_halo(center, r_excise)
            if len(idx) > 0:
                mask[idx] = True
            
            if rank == 0 and (i + 1) % 200 == 0:
                print(f"    Processed {i+1}/{len(halos_dict['positions'])} halos...")
                sys.stdout.flush()
        
        return mask
    
    # Mask particles in early-forming halos
    if rank == 0:
        print(f"  Computing mask for {len(halos_early['positions'])} early-forming halos...")
        sys.stdout.flush()
    early_mask_particles = compute_halo_particle_mask(dmo, halos_early, args.radius_mult)
    n_early_local = np.sum(early_mask_particles)
    
    # Mask particles in late-forming halos
    if rank == 0:
        print(f"  Computing mask for {len(halos_late['positions'])} late-forming halos...")
        sys.stdout.flush()
    late_mask_particles = compute_halo_particle_mask(dmo, halos_late, args.radius_mult)
    n_late_local = np.sum(late_mask_particles)
    
    if rank == 0:
        print(f"  Rank 0 particles in early halos: {n_early_local:,}")
        print(f"  Rank 0 particles in late halos: {n_late_local:,}")
    
    # ========================================================================
    # Generate 2D projected maps
    # ========================================================================
    if rank == 0:
        print("\n[4/5] Generating 2D projected maps...")
        sys.stdout.flush()
    
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
    
    # Full DMO map
    local_dmo_map = project_to_2d(dmo.coords, dmo.masses, args.grid)
    if rank == 0:
        global_dmo_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_dmo_map = None
    comm.Reduce(local_dmo_map, global_dmo_map, op=MPI.SUM, root=0)
    del local_dmo_map
    
    # Replace_early: DMO background + early-forming halos (still DMO particles)
    # This is equivalent to keeping only early-forming halos and removing late-forming halos
    # Actually, for assembly bias, we want:
    # - Replace_early_only: Only particles in early-forming halos
    # - Replace_late_only: Only particles in late-forming halos
    
    # Map with only early-forming halo particles
    local_early_map = project_to_2d(dmo.coords[early_mask_particles], 
                                     dmo.masses[early_mask_particles], args.grid)
    if rank == 0:
        global_early_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_early_map = None
    comm.Reduce(local_early_map, global_early_map, op=MPI.SUM, root=0)
    del local_early_map
    
    # Map with only late-forming halo particles
    local_late_map = project_to_2d(dmo.coords[late_mask_particles], 
                                    dmo.masses[late_mask_particles], args.grid)
    if rank == 0:
        global_late_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_late_map = None
    comm.Reduce(local_late_map, global_late_map, op=MPI.SUM, root=0)
    del local_late_map
    
    # Background (particles not in any selected halo)
    combined_mask = early_mask_particles | late_mask_particles
    local_bg_map = project_to_2d(dmo.coords[~combined_mask], 
                                  dmo.masses[~combined_mask], args.grid)
    if rank == 0:
        global_bg_map = np.zeros((args.grid, args.grid), dtype=np.float32)
    else:
        global_bg_map = None
    comm.Reduce(local_bg_map, global_bg_map, op=MPI.SUM, root=0)
    del local_bg_map
    
    # Save maps
    if rank == 0:
        os.makedirs(os.path.join(snap_dir, 'projected'), exist_ok=True)
        
        # Full DMO map
        np.savez_compressed(os.path.join(snap_dir, 'projected', 'dmo.npz'),
            field=global_dmo_map, box_size=BOX_SIZE, grid_resolution=args.grid)
        print(f"  Saved DMO map")
        
        # Early-forming halos only
        np.savez_compressed(os.path.join(snap_dir, 'projected', 'early_halos.npz'),
            field=global_early_map, box_size=BOX_SIZE, grid_resolution=args.grid,
            n_halos=len(halos_early['indices']))
        print(f"  Saved early-forming halos map")
        
        # Late-forming halos only
        np.savez_compressed(os.path.join(snap_dir, 'projected', 'late_halos.npz'),
            field=global_late_map, box_size=BOX_SIZE, grid_resolution=args.grid,
            n_halos=len(halos_late['indices']))
        print(f"  Saved late-forming halos map")
        
        # Background (useful for Replace field construction)
        np.savez_compressed(os.path.join(snap_dir, 'projected', 'background.npz'),
            field=global_bg_map, box_size=BOX_SIZE, grid_resolution=args.grid)
        print(f"  Saved background map")
        
        # Replace fields: background + one population
        replace_early = global_bg_map + global_early_map
        replace_late = global_bg_map + global_late_map
        
        np.savez_compressed(os.path.join(snap_dir, 'projected', 'replace_early.npz'),
            field=replace_early, box_size=BOX_SIZE, grid_resolution=args.grid,
            n_halos=len(halos_early['indices']))
        print(f"  Saved Replace_early map (bg + early halos)")
        
        np.savez_compressed(os.path.join(snap_dir, 'projected', 'replace_late.npz'),
            field=replace_late, box_size=BOX_SIZE, grid_resolution=args.grid,
            n_halos=len(halos_late['indices']))
        print(f"  Saved Replace_late map (bg + late halos)")
        
        del global_dmo_map, global_early_map, global_late_map, global_bg_map
        del replace_early, replace_late
    
    # ========================================================================
    # Generate lensplanes (if enabled)
    # ========================================================================
    if args.enable_lensplanes:
        if rank == 0:
            print("\n[5/5] Generating lensplanes...")
            sys.stdout.flush()
        
        # Check if snapshot is in the ray-tracing sequence
        if args.snap not in SNAPSHOT_TO_INDEX:
            if rank == 0:
                print(f"  Warning: snapshot {args.snap} not in SNAPSHOT_ORDER, skipping lensplanes")
        else:
            snapshot_idx = SNAPSHOT_TO_INDEX[args.snap]
            transforms = TransformGenerator(
                n_realizations=LENSPLANE_CONFIG['n_realizations'],
                n_snapshots=len(SNAPSHOT_ORDER),
                pps=LENSPLANE_CONFIG['planes_per_snapshot'],
                seed=LENSPLANE_CONFIG['seed'],
                box_size=BOX_SIZE
            )
            
            lp_grid = LENSPLANE_CONFIG['grid_res']
            pps = LENSPLANE_CONFIG['planes_per_snapshot']
            n_real = LENSPLANE_CONFIG['n_realizations']
            
            lp_output = os.path.join(OUTPUT_BASE, f'L205n{args.sim_res}TNG_LP')
            
            # Generate lensplanes for each model
            for model_name, mask in [('dmo', None), 
                                      ('early', early_mask_particles),
                                      ('late', late_mask_particles),
                                      ('replace_early', ~late_mask_particles),  # bg + early
                                      ('replace_late', ~early_mask_particles)]:  # bg + late
                
                if rank == 0:
                    print(f"  Generating {model_name} lensplanes...")
                    sys.stdout.flush()
                
                # Select particles
                if mask is None:
                    pos = dmo.coords
                    mass = dmo.masses
                else:
                    pos = dmo.coords[mask]
                    mass = dmo.masses[mask]
                
                # Generate lensplanes for all realizations
                for real_idx in range(n_real):
                    t = transforms.get_transform(real_idx, snapshot_idx)
                    
                    for pps_slice in range(pps):
                        file_idx = snapshot_idx * pps + pps_slice
                        
                        local_delta = project_lensplane(pos, mass, t, lp_grid, BOX_SIZE, 
                                                        pps_slice, pps)
                        
                        if rank == 0:
                            global_delta = np.zeros((lp_grid, lp_grid), dtype=np.float64)
                        else:
                            global_delta = None
                        
                        local_delta_contig = np.ascontiguousarray(local_delta.astype(np.float64))
                        comm.Reduce(local_delta_contig, global_delta, op=MPI.SUM, root=0)
                        
                        if rank == 0:
                            plane_dir = os.path.join(lp_output, f'snap_{args.snap:03d}', 
                                                     model_name, f'LP_{real_idx:02d}')
                            os.makedirs(plane_dir, exist_ok=True)
                            filepath = os.path.join(plane_dir, f'lenspot{file_idx:02d}.dat')
                            write_lensplane(filepath, global_delta, lp_grid)
                        
                        del local_delta, local_delta_contig
                        if rank == 0:
                            del global_delta
                    
                    if rank == 0 and (real_idx + 1) % 5 == 0:
                        print(f"    Realization {real_idx + 1}/{n_real} done")
                        sys.stdout.flush()
                
                comm.Barrier()
    else:
        if rank == 0:
            print("\n[5/5] Lensplane generation skipped (use --enable-lensplanes)")
    
    # Clean up
    dmo.free()
    gc.collect()
    
    # ========================================================================
    # Summary
    # ========================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("COMPLETE!")
        print("=" * 70)
        print(f"Total time: {time.time()-t_start:.1f}s")
        print(f"\nOutput files:")
        print(f"  Halo selection: {snap_dir}/halo_selection.npz")
        print(f"  2D maps: {snap_dir}/projected/")
        if args.enable_lensplanes:
            print(f"  Lensplanes: {lp_output}/")
        print("\nNext steps:")
        print("  1. Compute power spectra from 2D maps")
        print("  2. Compare P(k) for early vs late forming halos")
        print("  3. Look for scale-dependent differences (assembly bias signature)")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Assembly bias detection via halo replacement')
    parser.add_argument('--snap', type=int, default=99, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, default=2500, choices=[625, 1250, 2500],
                        help='Simulation resolution (particle count per dimension)')
    parser.add_argument('--mass-center', type=float, default=DEFAULT_MASS_CENTER,
                        help='Center of mass bin (log10 Msun/h)')
    parser.add_argument('--mass-width', type=float, default=DEFAULT_MASS_WIDTH,
                        help='Width of mass bin (± dex)')
    parser.add_argument('--radius-mult', type=float, default=DEFAULT_RADIUS_MULT,
                        help='Excision radius in units of R200')
    parser.add_argument('--grid', type=int, default=GRID_RES, help='Grid resolution for maps')
    parser.add_argument('--split-method', choices=['median', 'quartile'], default='median',
                        help='Method for splitting early/late forming halos')
    parser.add_argument('--enable-lensplanes', action='store_true',
                        help='Generate lensplanes for ray-tracing')
    parser.add_argument('--test', action='store_true',
                        help='Test mode with reduced grid resolution')
    
    args = parser.parse_args()
    
    if args.test:
        args.grid = 1024
        if rank == 0:
            print("TEST MODE: Using 1024² grid")
    
    run_assembly_bias_pipeline(args)


if __name__ == '__main__':
    main()
