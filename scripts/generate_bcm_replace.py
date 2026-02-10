#!/usr/bin/env python
"""
Generate BCM Replace lensplanes for all (M, α) configurations.

This script implements the BCM analogue of the Hydro Replace experiment:
1. Load DMO particles
2. Apply BCM displacement to ALL particles (ONCE)
3. For each (M_lo, M_hi, R_inner, R_outer) config:
   - Mask particles in selected halo shells
   - Build Replace field: DMO[~mask] + BCM[mask]
   - Generate lensplanes

This is efficient because BCM is applied only once, then reused for 100+ configs.

Output structure:
    /mnt/home/mlee1/ceph/hydro_replace_LP_bcm/L205n2500TNG/
    ├── bcm_{model}_full/LP_XX/lenspotNN.dat         # Full BCM reference
    └── bcm_{model}_replace_Ml_..._Ro_.../LP_XX/     # Replace configs

Usage:
    # Generate all 100 configs for one snapshot
    mpirun -np 64 python generate_bcm_replace.py --snap 96 --bcm-model arico20_tng
    
    # Generate specific config range (for job splitting)
    mpirun -np 64 python generate_bcm_replace.py --snap 96 --bcm-model arico20_tng \
        --config-start 0 --config-end 25
    
    # List all configs
    python generate_bcm_replace.py --list-configs

Requirements:
    - BaryonForge: pip install BaryonForge
    - pyccl: pip install pyccl
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

# BaryonForge imports
try:
    import BaryonForge as bfg
    import pyccl as ccl
    HAS_BARYONFORGE = True
except ImportError:
    HAS_BARYONFORGE = False
    print("Warning: BaryonForge not installed. Install with: pip install BaryonForge")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ============================================================================
# Configuration (matching generate_all_unified.py)
# ============================================================================

SIM_PATHS = {
    2500: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output',
        'dmo_dm_mass': 0.0047271638660809,
    },
    1250: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG_DM/output',
        'dmo_dm_mass': 0.0378173109,
    },
    625: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG_DM/output',
        'dmo_dm_mass': 0.3025384873,
    },
}

# TNG cosmology
TNG_COSMOLOGY = {
    'Omega_m': 0.3089,
    'Omega_b': 0.0486,
    'h': 0.6774,
    'sigma8': 0.8159,
    'n_s': 0.9667,
    'w0': -1.0,
}

# Output paths
OUTPUT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_LP_bcm'
FIELDS_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields'
BOX_SIZE = 205.0
MASS_UNIT = 1e10
GRID_RES = 4096

# Lensplane configuration
LENSPLANE_CONFIG = {
    'n_realizations': 20,
    'planes_per_snapshot': 2,
    'grid_res': 4096,
    'seed': 2020,  # Same as Hydro pipeline for identical transforms
}

# Snapshot ordering for ray-tracing
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
SNAPSHOT_TO_INDEX = {snap: idx for idx, snap in enumerate(SNAPSHOT_ORDER)}

# Binned configuration (same as generate_all_unified.py)
from itertools import combinations

MASS_EDGES = [12.0, 12.5, 13.0, 13.5, 15.0]
RADIUS_EDGES = [0.0, 0.5, 1.0, 3.0, 5.0]


def generate_binned_configs():
    """Generate all (mass_bin, radius_shell) combinations.
    
    Returns:
        list of (M_lo, M_hi, R_inner, R_outer) tuples
        Total: C(5,2) × C(5,2) = 10 × 10 = 100 configurations
    """
    configs = []
    
    mass_bins = [(10**m_lo, 10**m_hi) for m_lo, m_hi in combinations(MASS_EDGES, 2)]
    radius_shells = list(combinations(RADIUS_EDGES, 2))
    
    for M_lo, M_hi in mass_bins:
        for R_inner, R_outer in radius_shells:
            configs.append((M_lo, M_hi, R_inner, R_outer))
    
    return configs


def get_binned_config_label(M_lo, M_hi, R_inner, R_outer, prefix=''):
    """Generate config label.
    
    Format: {prefix}hydro_replace_Ml_{M_lo}_Mu_{M_hi}_Ri_{R_inner}_Ro_{R_outer}
    """
    base = f"hydro_replace_Ml_{M_lo:.2e}_Mu_{M_hi:.2e}_Ri_{R_inner}_Ro_{R_outer}"
    base = base.replace('+', '')
    if prefix:
        return f"{prefix}_{base}"
    return base


# ============================================================================
# BCM Model Setup (from generate_all_unified_bcm.py)
# ============================================================================

def get_bcm_params_arico20_tng(h=0.6774):
    """BCM parameters for Arico20 model - TNG profile-fitted."""
    return dict(
        M_c=10**11.5463, mu=0.05, beta=3.9326, M_inn=10**9.1835,
        theta_inn=0.0617, theta_out=10.6409, epsilon_hydro=np.sqrt(5),
        theta_rg=0.3, sigma_rg=0.1, M_r=10**18.0, beta_r=2,
        eta=0.2610, eta_delta=0.1,
        M1_0=10**9.6509, alpha_g=2, epsilon_h=0.015,
        M1_fsat=3.98, eps_fsat=1.0, alpha_fsat=1.0, delta_fsat=0.99, gamma_fsat=1.67,
        a=0.3, n=2, p=0.3, q=0.707,
        A_nt=0.495, T_w=10**6.5, alpha_nt=0.1,
        mean_molecular_weight=0.59,
    )


def get_bcm_params_schneider19(h=0.6774):
    """BCM parameters for Schneider19 model."""
    return dict(
        theta_ej=4.0, theta_co=0.1, M_c=1e14/h, mu_beta=0.4,
        gamma=2.0, delta=7.0,
        eta=0.3, eta_delta=0.3, tau=-1.5, tau_delta=0.0,
        A=0.09/2, M1=2.5e11/h, epsilon_h=0.015,
        a=0.3, n=2.0, epsilon=4.0, p=0.3, q=0.707,
        cutoff=100.0, proj_cutoff=100.0,
    )


def get_bcm_params_arico20(h=0.6774):
    """BCM parameters for Arico20 model - default."""
    return dict(
        M_c=10**13.0, mu=0.15, beta=0.35, M_inn=10**12.0,
        theta_inn=0.3, theta_out=1.0, epsilon_hydro=np.sqrt(5),
        theta_rg=0.3, sigma_rg=0.1, M_r=10**18.0, beta_r=2, eta=0.5,
        M1_0=10**12.0, alpha_g=2, epsilon_h=0.015,
        M1_fsat=3.98, eps_fsat=1.0, alpha_fsat=1.0, delta_fsat=0.99, gamma_fsat=1.67,
        a=0.3, n=2, p=0.3, q=0.707,
        A_nt=0.495, T_w=10**6.5, alpha_nt=0.1,
        mean_molecular_weight=0.59,
    )


BCM_MODELS = {}
if HAS_BARYONFORGE:
    BCM_MODELS = {
        'arico20_tng': {
            'DMO': bfg.Profiles.Arico20.DarkMatterOnly,
            'DMB': bfg.Profiles.Arico20.DarkMatterBaryon,
            'params': get_bcm_params_arico20_tng,
            'Rdelta_sampling': True,
        },
        'arico20': {
            'DMO': bfg.Profiles.Arico20.DarkMatterOnly,
            'DMB': bfg.Profiles.Arico20.DarkMatterBaryon,
            'params': get_bcm_params_arico20,
            'Rdelta_sampling': True,
        },
        'schneider19': {
            'DMO': bfg.Profiles.Schneider19.DarkMatterOnly,
            'DMB': bfg.Profiles.Schneider19.DarkMatterBaryon,
            'params': get_bcm_params_schneider19,
            'Rdelta_sampling': False,
        },
    }


def create_baryonification_model(model_name, cosmo_params, epsilon_max=20.0):
    """Create BaryonForge baryonification model."""
    if model_name not in BCM_MODELS:
        raise ValueError(f"Unknown BCM model: {model_name}. Available: {list(BCM_MODELS.keys())}")
    
    model_config = BCM_MODELS[model_name]
    h = cosmo_params['h']
    bcm_params = model_config['params'](h=h)
    
    cosmo = ccl.Cosmology(
        Omega_c=cosmo_params['Omega_m'] - cosmo_params['Omega_b'],
        Omega_b=cosmo_params['Omega_b'],
        h=h,
        sigma8=cosmo_params['sigma8'],
        n_s=cosmo_params['n_s'],
        w0=cosmo_params.get('w0', -1.0),
        transfer_function='boltzmann_camb',
        matter_power_spectrum='linear'
    )
    
    DMO = model_config['DMO'](**bcm_params)
    DMB = model_config['DMB'](**bcm_params)
    
    baryons = bfg.Baryonification3D(DMO, DMB, cosmo, N_int=50_000)
    
    return baryons, cosmo, bcm_params


# ============================================================================
# Particle Loading (from generate_all_unified_bcm.py)
# ============================================================================

class DistributedParticles:
    """Load and manage particles distributed across MPI ranks."""
    
    def __init__(self, snapshot, sim_res, radius_mult=5.0):
        self.snapshot = snapshot
        self.sim_res = sim_res
        self.radius_mult = radius_mult
        self.sim_config = SIM_PATHS[sim_res]
        
        self.coords = None
        self.masses = None
        self.tree = None
        
    def load(self):
        """Load DMO particles for this rank."""
        t0 = time.time()
        
        basePath = self.sim_config['dmo']
        dm_mass = self.sim_config['dmo_dm_mass']
        
        snap_dir = f"{basePath}/snapdir_{self.snapshot:03d}/"
        all_files = sorted(glob.glob(f"{snap_dir}/snap_{self.snapshot:03d}.*.hdf5"))
        my_files = [f for i, f in enumerate(all_files) if i % size == rank]
        
        if rank == 0:
            print(f"  Loading DMO particles...")
            print(f"    Files: {len(all_files)} total, {len(my_files)} per rank")
        
        coords_list, masses_list = [], []
        
        for filepath in my_files:
            with h5py.File(filepath, 'r') as f:
                pt_key = 'PartType1'
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
        
        indices = self.tree.query_ball_point(center, radius)
        
        if len(indices) == 0:
            return np.array([], dtype=int)
        
        indices = np.array(indices)
        
        # Handle periodic images if near box edge
        if np.any(center < radius) or np.any(center > BOX_SIZE - radius):
            for dx in [-BOX_SIZE, 0, BOX_SIZE]:
                for dy in [-BOX_SIZE, 0, BOX_SIZE]:
                    for dz in [-BOX_SIZE, 0, BOX_SIZE]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        shifted = center + np.array([dx, dy, dz])
                        if np.all(shifted >= -radius) and np.all(shifted <= BOX_SIZE + radius):
                            more = self.tree.query_ball_point(shifted, radius)
                            if len(more) > 0:
                                indices = np.unique(np.concatenate([indices, more]))
        
        return indices
    
    def free(self):
        """Free memory."""
        del self.coords, self.masses, self.tree
        self.coords = self.masses = self.tree = None
        gc.collect()


# ============================================================================
# BCM Displacement (from generate_all_unified_bcm.py)
# ============================================================================

def apply_bcm_displacements(baryons, particles, halos, z_snap, cosmo_params, epsilon_max, comm):
    """
    Apply BCM displacements to ALL particles.
    
    Returns:
        bcm_coords: (N, 3) BCM-displaced coordinates
    """
    r = comm.Get_rank()
    h = cosmo_params['h']
    a_snap = 1.0 / (1.0 + z_snap)
    
    n_particles = len(particles.coords)
    n_halos = len(halos['masses'])
    
    if r == 0:
        print(f"  Setting up BCM interpolator...")
        t0 = time.time()
    
    baryons.setup_interpolator(
        z_min=0, z_max=3,
        z_linear_sampling=True,
        N_samples_R=10000,
        verbose=(r == 0)
    )
    
    if r == 0:
        print(f"    Interpolator setup: {time.time()-t0:.1f}s")
        print(f"  Computing BCM displacements for {n_particles:,} particles...")
        t0 = time.time()
    
    # Initialize displaced coords
    displaced = np.mod(particles.coords.copy(), BOX_SIZE).astype(np.float64)
    
    # Build tree with periodic BC
    if n_particles > 0:
        tree = cKDTree(displaced, boxsize=BOX_SIZE)
    else:
        tree = None
    
    n_displaced = 0
    for j in range(n_halos):
        halo_pos = halos['positions'][j]
        halo_r200 = halos['radii'][j]
        halo_mass = halos['masses'][j]
        
        R_query = min(epsilon_max * halo_r200, BOX_SIZE / 2)
        
        if tree is None:
            continue
        
        inds = tree.query_ball_point(halo_pos, R_query)
        
        if len(inds) == 0:
            continue
        
        inds = np.array(inds)
        region_coords = displaced[inds]
        
        try:
            # Compute periodic distance
            dx = region_coords - halo_pos
            dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
            dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
            
            r_dist = np.sqrt(np.sum(dx**2, axis=1))
            r_mpc = r_dist / h
            
            # Unit vectors
            r_safe = np.maximum(r_dist, 1e-10)
            unit = dx / r_safe[:, None]
            
            # Get BCM displacement
            offset = baryons.displacement(r_mpc, halo_mass, a_snap)
            offset = np.where(np.isfinite(offset), offset, 0.0)
            offset_mpch = offset * h
            
            # Apply radial displacement
            displaced[inds] += offset_mpch[:, None] * unit
            n_displaced += len(inds)
            
        except Exception as e:
            if r == 0 and j < 10:
                print(f"    Warning: Halo {j} failed: {e}")
            continue
        
        if r == 0 and (j + 1) % 1000 == 0:
            print(f"    Processed {j+1}/{n_halos} halos...")
            sys.stdout.flush()
    
    bcm_coords = np.mod(displaced, BOX_SIZE).astype(np.float32)
    
    if r == 0:
        print(f"    Displaced {n_displaced:,} particle instances in {time.time()-t0:.1f}s")
    
    return bcm_coords


# ============================================================================
# Halo Data Precomputation (for fast per-config masking)
# ============================================================================

def precompute_halo_particle_data(particles, halos, max_radius_mult, comm):
    """
    Precompute particle indices and distances for all halos.
    
    Query KDTree ONCE per halo at max_radius, store indices and r/R200.
    
    Returns:
        halo_data: list of (indices, r_over_r200) per halo
    """
    r = comm.Get_rank()
    n_halos = len(halos['masses'])
    
    if r == 0:
        print(f"  Precomputing halo particle data ({n_halos} halos)...")
        t0 = time.time()
    
    halo_data = []
    
    for i in range(n_halos):
        center = halos['positions'][i]
        r200 = halos['radii'][i]
        r_max = r200 * max_radius_mult
        
        idx = particles.query_halo(center, r_max)
        
        if len(idx) == 0:
            halo_data.append((np.array([], dtype=np.int64), np.array([], dtype=np.float32)))
            continue
        
        # Compute distances in units of R200
        coords = particles.coords[idx]
        dx = coords - center
        dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
        dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
        dist_r200 = np.linalg.norm(dx, axis=1) / r200
        
        halo_data.append((np.array(idx, dtype=np.int64), dist_r200.astype(np.float32)))
        
        if r == 0 and (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n_halos} halos ({elapsed:.1f}s)")
    
    if r == 0:
        print(f"    Precomputation done: {time.time()-t0:.1f}s")
    
    return halo_data


def get_shell_mask(halo_data, halo_indices, R_inner, R_outer, n_particles):
    """
    Get particle mask for shell using precomputed distances.
    
    Returns:
        mask: (n_particles,) bool array where True = particle in shell
    """
    mask = np.zeros(n_particles, dtype=bool)
    
    for i in halo_indices:
        idx, r_r200 = halo_data[i]
        if len(idx) == 0:
            continue
        
        shell = (r_r200 >= R_inner) & (r_r200 < R_outer)
        mask[idx[shell]] = True
    
    return mask


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
        return {
            'proj_dir': self.proj_dirs[realization_idx, snapshot_idx],
            'displacement': self.displacements[realization_idx, snapshot_idx],
            'flip': self.flips[realization_idx, snapshot_idx],
        }
    
    def save(self, filepath):
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


def apply_transform(pos, transform, box_size):
    """Apply rotation/translation/flip to positions."""
    pos_t = pos + transform['displacement']
    if transform['flip']:
        pos_t = box_size - pos_t
    return pos_t % box_size


def project_lensplane(pos, mass, transform, grid_res, box_size, pps_slice, pps=2):
    """Transform and project particles to 2D lensplane."""
    if len(pos) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32)
    
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


def generate_lensplanes_for_field(label, coords, masses, transforms, lp_grid, box_size,
                                   output_dir, snap, comm):
    """Generate lensplanes for a given field (DMO, BCM, or Replace)."""
    r = comm.Get_rank()
    
    if snap not in SNAPSHOT_TO_INDEX:
        if r == 0:
            print(f"    Warning: snap {snap} not in SNAPSHOT_ORDER")
        return
    
    snapshot_idx = SNAPSHOT_TO_INDEX[snap]
    pps = transforms.pps
    n_real = transforms.n_realizations
    
    for real_idx in range(n_real):
        t = transforms.get_transform(real_idx, snapshot_idx)
        
        for pps_slice in range(pps):
            file_idx = snapshot_idx * pps + pps_slice
            
            local_delta = project_lensplane(coords, masses, t, lp_grid, box_size, pps_slice, pps)
            
            if r == 0:
                global_delta = np.zeros((lp_grid, lp_grid), dtype=np.float64)
            else:
                global_delta = None
            
            local_contig = np.ascontiguousarray(local_delta.astype(np.float64))
            comm.Reduce(local_contig, global_delta, op=MPI.SUM, root=0)
            
            if r == 0:
                plane_dir = os.path.join(output_dir, label, f'LP_{real_idx:02d}')
                os.makedirs(plane_dir, exist_ok=True)
                filepath = os.path.join(plane_dir, f'lenspot{file_idx:02d}.dat')
                write_lensplane(filepath, global_delta, lp_grid)
            
            del local_delta, local_contig
            if r == 0:
                del global_delta
    
    comm.Barrier()


# ============================================================================
# Main Pipeline
# ============================================================================

def get_snapshot_redshift(snapshot, sim_res):
    """Get redshift for a given snapshot."""
    basePath = SIM_PATHS[sim_res]['dmo']
    snap_file = f"{basePath}/snapdir_{snapshot:03d}/snap_{snapshot:03d}.0.hdf5"
    
    with h5py.File(snap_file, 'r') as f:
        z = f['Header'].attrs['Redshift']
    
    return z


def run_bcm_replace_pipeline(args):
    """
    Main pipeline: Generate BCM Replace lensplanes for all configs.
    
    1. Load DMO particles
    2. Apply BCM to ALL particles (once)
    3. Precompute halo particle data
    4. For each config: mask + combine + generate lensplanes
    """
    if not HAS_BARYONFORGE:
        if rank == 0:
            print("ERROR: BaryonForge not installed")
        return
    
    t_start = time.time()
    
    # Output directory
    output_dir = os.path.join(OUTPUT_BASE, f'L205n{args.sim_res}TNG')
    
    # Get snapshot redshift
    z_snap = get_snapshot_redshift(args.snap, args.sim_res)
    
    if rank == 0:
        print("=" * 70)
        print("BCM REPLACE LENSPLANE PIPELINE")
        print("=" * 70)
        print(f"Snapshot: {args.snap} (z = {z_snap:.4f})")
        print(f"BCM model: {args.bcm_model}")
        print(f"Resolution: L205n{args.sim_res}TNG")
        print(f"Output: {output_dir}")
        print("=" * 70)
        sys.stdout.flush()
        
        os.makedirs(output_dir, exist_ok=True)
    
    comm.Barrier()
    
    # ========================================================================
    # PHASE 1: Load halos
    # ========================================================================
    if rank == 0:
        print("\n[1/6] Loading halo catalog...")
    
    matches_file = os.path.join(FIELDS_BASE, f'L205n{args.sim_res}TNG', 'matches',
                                 f'matches_snap{args.snap:03d}.npz')
    
    with np.load(matches_file) as data:
        all_masses = data['dmo_masses'] * MASS_UNIT
        all_positions = data['dmo_positions'] / 1e3
        all_radii = data['dmo_radii'] / 1e3
    
    # Select halos above minimum mass (10^12)
    mass_mask = np.log10(all_masses) >= 12.0
    
    halos = {
        'masses': all_masses[mass_mask],
        'positions': all_positions[mass_mask],
        'radii': all_radii[mass_mask],
    }
    
    if rank == 0:
        print(f"  Halos above 10^12: {len(halos['masses'])}")
    
    # ========================================================================
    # PHASE 2: Load DMO particles
    # ========================================================================
    if rank == 0:
        print("\n[2/6] Loading DMO particles...")
    
    dmo = DistributedParticles(args.snap, args.sim_res, radius_mult=args.radius_mult)
    dmo.load().build_tree()
    
    # ========================================================================
    # PHASE 3: Apply BCM to ALL particles (ONCE)
    # ========================================================================
    if rank == 0:
        print("\n[3/6] Applying BCM displacements to ALL particles...")
        t0 = time.time()
    
    baryons, cosmo, bcm_params = create_baryonification_model(
        args.bcm_model, TNG_COSMOLOGY, args.epsilon_max
    )
    
    bcm_coords = apply_bcm_displacements(
        baryons, dmo, halos, z_snap, TNG_COSMOLOGY, args.epsilon_max, comm
    )
    
    if rank == 0:
        print(f"  BCM application time: {time.time()-t0:.1f}s")
    
    # ========================================================================
    # PHASE 4: Precompute halo particle data
    # ========================================================================
    if rank == 0:
        print("\n[4/6] Precomputing halo particle data...")
    
    halo_data = precompute_halo_particle_data(dmo, halos, max_radius_mult=5.0, comm=comm)
    
    # ========================================================================
    # PHASE 5: Generate lensplanes
    # ========================================================================
    if rank == 0:
        print("\n[5/6] Generating lensplanes...")
    
    # Setup transforms
    transforms = TransformGenerator(
        n_realizations=LENSPLANE_CONFIG['n_realizations'],
        pps=LENSPLANE_CONFIG['planes_per_snapshot'],
        seed=LENSPLANE_CONFIG['seed'],
        box_size=BOX_SIZE
    )
    
    if rank == 0:
        transform_file = os.path.join(output_dir, 'transforms.h5')
        if not os.path.exists(transform_file):
            transforms.save(transform_file)
    
    lp_grid = args.lensplane_grid
    
    # 5a. Generate BCM-full reference (if not skipping)
    if not args.skip_bcm_full:
        if rank == 0:
            print(f"\n  Generating BCM-full reference...")
            t0 = time.time()
        
        bcm_full_label = f"bcm_{args.bcm_model}_full"
        generate_lensplanes_for_field(
            bcm_full_label, bcm_coords, dmo.masses,
            transforms, lp_grid, BOX_SIZE, output_dir, args.snap, comm
        )
        
        if rank == 0:
            print(f"    BCM-full time: {time.time()-t0:.1f}s")
    
    # 5b. Generate DMO reference (if requested)
    if args.generate_dmo:
        if rank == 0:
            print(f"\n  Generating DMO reference...")
            t0 = time.time()
        
        generate_lensplanes_for_field(
            'dmo', dmo.coords, dmo.masses,
            transforms, lp_grid, BOX_SIZE, output_dir, args.snap, comm
        )
        
        if rank == 0:
            print(f"    DMO time: {time.time()-t0:.1f}s")
    
    # 5c. Generate Replace configs
    all_configs = generate_binned_configs()
    
    config_start = args.config_start
    config_end = args.config_end if args.config_end else len(all_configs)
    configs_to_process = all_configs[config_start:config_end]
    
    if rank == 0:
        print(f"\n  Processing {len(configs_to_process)} Replace configs [{config_start}:{config_end}]...")
    
    for config_idx, (M_lo, M_hi, R_inner, R_outer) in enumerate(configs_to_process):
        global_idx = config_start + config_idx
        
        # Get config label
        config_label = f"bcm_{args.bcm_model}_" + get_binned_config_label(M_lo, M_hi, R_inner, R_outer)
        
        # Check if already exists
        if args.skip_existing:
            snapshot_idx = SNAPSHOT_TO_INDEX.get(args.snap, -1)
            if snapshot_idx >= 0:
                pps = transforms.pps
                check_dir = os.path.join(output_dir, config_label, 'LP_19')
                check_file = os.path.join(check_dir, f'lenspot{snapshot_idx * pps + pps - 1:02d}.dat')
                if os.path.exists(check_file):
                    if rank == 0:
                        print(f"    [{global_idx+1}/{len(all_configs)}] Skipping {config_label} (exists)")
                    continue
        
        if rank == 0:
            print(f"    [{global_idx+1}/{len(all_configs)}] {config_label}...")
            t0 = time.time()
        
        # Select halos in mass bin
        halo_mask = (halos['masses'] >= M_lo) & (halos['masses'] < M_hi)
        halo_indices = np.where(halo_mask)[0]
        
        if len(halo_indices) == 0:
            if rank == 0:
                print(f"      No halos in mass bin, skipping")
            continue
        
        # Get shell mask
        shell_mask = get_shell_mask(halo_data, halo_indices, R_inner, R_outer, len(dmo.coords))
        
        n_in_shell = np.sum(shell_mask)
        if rank == 0:
            print(f"      {len(halo_indices)} halos, {n_in_shell:,} particles in shells")
        
        # Build Replace field: DMO outside shells + BCM inside shells
        replace_coords = np.concatenate([
            dmo.coords[~shell_mask],
            bcm_coords[shell_mask]
        ])
        replace_masses = np.concatenate([
            dmo.masses[~shell_mask],
            dmo.masses[shell_mask]
        ])
        
        # Generate lensplanes
        generate_lensplanes_for_field(
            config_label, replace_coords, replace_masses,
            transforms, lp_grid, BOX_SIZE, output_dir, args.snap, comm
        )
        
        del replace_coords, replace_masses
        
        if rank == 0:
            print(f"      Done: {time.time()-t0:.1f}s")
    
    # ========================================================================
    # PHASE 6: Cleanup
    # ========================================================================
    dmo.free()
    del bcm_coords, halo_data
    gc.collect()
    
    if rank == 0:
        print("\n" + "=" * 70)
        print(f"Complete! Total time: {time.time()-t_start:.1f}s")
        print(f"Output: {output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Generate BCM Replace lensplanes')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, default=2500, choices=[625, 1250, 2500])
    parser.add_argument('--bcm-model', type=str, default='arico20_tng',
                        choices=list(BCM_MODELS.keys()) if BCM_MODELS else ['arico20_tng', 'arico20', 'schneider19'],
                        help='BCM model to use')
    parser.add_argument('--epsilon-max', type=float, default=20.0,
                        help='Maximum BCM displacement radius (R200 units)')
    parser.add_argument('--radius-mult', type=float, default=5.0,
                        help='KDTree query radius multiplier')
    parser.add_argument('--lensplane-grid', type=int, default=GRID_RES)
    
    # Config range (for job splitting)
    parser.add_argument('--config-start', type=int, default=0)
    parser.add_argument('--config-end', type=int, default=None)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--skip-bcm-full', action='store_true',
                        help='Skip BCM-full reference generation')
    parser.add_argument('--generate-dmo', action='store_true',
                        help='Also generate DMO reference lensplanes')
    
    # Utility
    parser.add_argument('--list-configs', action='store_true',
                        help='List all configs and exit')
    
    args = parser.parse_args()
    
    if args.list_configs:
        configs = generate_binned_configs()
        print(f"Total configs: {len(configs)}")
        print("=" * 70)
        for i, (M_lo, M_hi, R_inner, R_outer) in enumerate(configs):
            label = f"bcm_{args.bcm_model}_" + get_binned_config_label(M_lo, M_hi, R_inner, R_outer)
            print(f"[{i:3d}] {label}")
        return
    
    run_bcm_replace_pipeline(args)


if __name__ == '__main__':
    main()
