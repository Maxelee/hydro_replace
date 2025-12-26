#!/usr/bin/env python
"""
Unified pipeline with Baryonic Correction Model (BCM): Generate profiles, statistics, maps, and lensplanes.

This script loads DMO simulations and applies baryonic corrections using BaryonForge
to displace particles according to the BCM prescription. Supports three BCM models:
- Schneider19: Original baryonification model
- Schneider25: Updated model with improved gas profiles
- Arico20: Alternative model with sharper features

For each model, the pipeline:
- Computes stacked density profiles (by mass bin)
- Computes halo statistics (masses at various radii)
- Generates 2D projected density maps
- Generates lensplanes for ray-tracing (optional)

Uses BaryonForge's BaryonifySnapshot class for efficient particle displacement.
Only outputs BCM results (no DMO output).

Algorithm:
  PHASE 1 (Load & Build):
    - Load halo catalog and DMO particles
    - Build KDTree for particle queries
  
  PHASE 2 (Loop over BCM models):
    For each model (schneider19, schneider25, arico20):
    - Setup BaryonForge model and displacement interpolator
    - Apply BCM displacements using BaryonifySnapshot.process()
    - Compute profiles and statistics
    - Generate 2D maps
    - Generate lensplanes (optional)
  
  PHASE 3 (Save Results):
    - Save profiles (one file per model)
    - Save statistics (one file with all models)

Usage:
    # Full pipeline with all models
    mpirun -np 32 python generate_all_unified_bcm.py --snap 99 --sim-res 625 --enable-lensplanes
    
    # Single model only
    mpirun -np 32 python generate_all_unified_bcm.py --snap 99 --sim-res 625 --models schneider19
    
    # Maps only (no lensplanes)
    mpirun -np 32 python generate_all_unified_bcm.py --snap 99 --sim-res 625

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
# Configuration
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

# TNG cosmology parameters
TNG_COSMOLOGY = {
    'Omega_m': 0.3089,
    'Omega_b': 0.0486,
    'h': 0.6774,
    'sigma8': 0.8159,
    'n_s': 0.9667,
    'w0': -1.0,
}

OUTPUT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields_bcm'
BOX_SIZE = 205.0  # Mpc/h
MASS_UNIT = 1e10  # Convert to Msun/h
GRID_RES = 4096   # Default grid resolution

# Profile configuration
RADIAL_BINS = np.logspace(-2, np.log10(5), 31)  # 0.01 to 5 R200, 30 bins
MASS_BIN_EDGES = [12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 16.0]

# Statistics radii (0.5, 1.0, 2.0, 3.0, 4.0, 5.0 R200)
STATS_RADII_MULT = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])


# ============================================================================
# BCM Configuration (Schneider+19 model parameters)
# ============================================================================

def get_bcm_params(h=0.6774):
    """
    Get BCM parameters for the Schneider+19 model.
    
    These are typical parameters that reproduce TNG-like baryon effects.
    Can be modified to explore different BCM configurations.
    """
    return dict(
        # Gas profile parameters
        theta_ej=4.0,           # Ejection radius in units of R200
        theta_co=0.1,           # Core radius in units of R200
        M_c=1e14/h,             # Characteristic mass for beta transition
        mu_beta=0.4,            # Mass slope for beta
        gamma=2.0,              # Outer gas slope
        delta=7.0,              # Inner gas slope
        
        # Stellar profile parameters
        eta=0.3,                # Stellar fraction normalization
        eta_delta=0.3,          # Central stellar fraction offset
        tau=-1.5,               # Low-mass stellar slope
        tau_delta=0.0,          # Central stellar slope offset
        A=0.09/2,               # Stellar amplitude
        M1=2.5e11/h,            # Stellar characteristic mass
        epsilon_h=0.015,        # Half-light radius factor
        
        # NFW/adiabatic contraction parameters
        a=0.3,                  # Adiabatic contraction parameter
        n=2.0,                  # NFW inner slope modification
        epsilon=4.0,            # Truncation radius in units of R200
        p=0.3,                  # Two-halo term parameter
        q=0.707,                # Two-halo term parameter
        
        # Numerical parameters
        cutoff=100.0,           # Real-space cutoff radius (Mpc)
        proj_cutoff=100.0,      # Projected cutoff radius (Mpc)
    )


def get_bcm_params_schneider19(h=0.6774):
    """
    BCM parameters for Schneider19 model - TNG calibrated.
    
    Values from Schneider+19 (arXiv:1810.08629) Figure 14,
    calibrated to match IllustrisTNG-100 gas fractions and power spectrum.
    Uses θ_ej=4 fit (middle value between 3 and 6).
    """
    return dict(
        # Gas profile parameters - TNG calibrated
        theta_ej=4.0,              # Gas ejection radius [R200]
        theta_co=0.1,              # Gas core radius [R200]
        M_c=1.1e13 / h,            # TNG fit: characteristic mass for beta transition
        mu_beta=0.55,              # TNG fit: mass slope for beta (0.55 for θ_ej=4)
        gamma=2.0,                 # Outer gas slope
        delta=7.0,                 # Inner gas slope
        
        # Stellar parameters - TNG calibrated
        eta=0.3,                   # Total stellar fraction normalization
        eta_delta=0.3,             # Central stellar fraction offset (η_cga = η + η_delta = 0.6)
        tau=-1.5,                  # Low-mass stellar slope
        tau_delta=0.0,             # Central stellar slope offset
        A=0.09 / 2,                # Stellar amplitude (0.045)
        M1=2.5e11 / h,             # Stellar characteristic mass
        epsilon_h=0.015,           # Half-light radius factor
        
        # Adiabatic relaxation parameters
        a=0.3,                     # Relaxation parameter
        n=2,                       # NFW inner slope modification
        
        # NFW truncation and 2-halo parameters
        epsilon=4.0,               # Truncation radius [R200]
        p=0.3,                     # Two-halo term parameter
        q=0.707,                   # Two-halo term parameter
    )


def build_cosmodict(cosmo_params):
    """
    Build cosmology dictionary for BaryonForge ParticleSnapshot/HaloNDCatalog.
    
    Args:
        cosmo_params: dict with Omega_m, Omega_b, h, sigma8, n_s, w0
    
    Returns:
        dict compatible with BaryonForge
    """
    return {
        'Omega_m': cosmo_params['Omega_m'],
        'Omega_b': cosmo_params['Omega_b'],
        'sigma8': cosmo_params['sigma8'],
        'h': cosmo_params['h'],
        'n_s': cosmo_params['n_s'],
        'w0': cosmo_params.get('w0', -1.0),
        'wa': cosmo_params.get('wa', 0.0),
    }


def get_bcm_params_schneider25(h=0.6774):
    """
    BCM parameters for Schneider25 model.
    
    Values from BaryonForge test defaults, which are based on
    Schneider+25 model calibrations. This is an updated model
    with improved gas profile parametrization.
    """
    return dict(
        # DM profile params
        epsilon0=4,                # Base truncation radius
        epsilon1=0.5,              # Peak-height dependence of truncation
        alpha_excl=0.4,            # Exclusion radius parameter
        p=0.3,                     # Two-halo term parameter
        q=0.707,                   # Two-halo term parameter
        
        # Gas profile params
        M_c=1e15,                  # Characteristic mass for gas slope
        mu=0.8,                    # Mass dependence of gas slope (note: 'mu' not 'mu_beta')
        theta_c=0.3,               # Gas core radius [R200]
        nu_theta_c=1/2,            # Redshift evolution of theta_c
        alpha=1,                   # Core slope
        gamma=3/2,                 # Intermediate-scale slope
        delta=7,                   # Large-scale slope
        
        # Inner gas fraction params (REQUIRED for Schneider25)
        c_iga=0.1,                 # Inner gas amplitude
        nu_c_iga=3/2,              # Inner gas redshift evolution
        r_min_iga=1e-3,            # Minimum radius for inner gas profile [Mpc]
        
        # Relaxation params
        q0=0.075,                  # Base relaxation amplitude
        q1=0.25,                   # Central galaxy relaxation
        q2=0.7,                    # Hot gas relaxation
        nu_q0=0,                   # Redshift evolution of q0
        nu_q1=1,                   # Redshift evolution of q1
        nu_q2=0,                   # Redshift evolution of q2
        nstep=3/2,                 # Step function exponent for relaxation (REQUIRED)
        
        # Star params
        tau=-1.376,                # Low-mass stellar slope
        tau_delta=0.0,             # Central stellar slope offset
        Mstar=3e11,                # Stellar characteristic mass
        Nstar=0.03,                # Stellar normalization
        eta=0.1,                   # Stellar fraction normalization
        eta_delta=0.22,            # Central stellar fraction offset
        epsilon_cga=0.03,          # Central galaxy size [R200]
        
        # Stellar profile params
        M1_0=1e11 / h,             # Stellar mass scale
        epsilon_h=0.015,           # Half-light radius factor
        
        # Non-thermal pressure params
        alpha_nt=0.1,              # Non-thermal pressure amplitude
        nu_nt=0.5,                 # Non-thermal pressure redshift evolution
        gamma_nt=0.8,              # Non-thermal pressure slope
        mean_molecular_weight=0.6125,  # Gas mean molecular weight
    )


def get_bcm_params_arico20(h=0.6774):
    """
    BCM parameters for Arico20 model - TNG calibrated.
    
    Values from Arico+20 (arXiv:2009.14225) calibrated to fit
    IllustrisTNG-300 power spectrum and bispectrum.
    Note: Arico20 has sharper features, so Rdelta_sampling should be True.
    """
    return dict(
        # Gas profile parameters - TNG calibrated
        M_c=1.2e14 / h,            # TNG fit: characteristic mass for gas slope
        mu=0.31,                   # TNG fit: gas slope mass dependence
        beta=0.6,                  # TNG fit: gas slope parameter
        M_inn=3.3e13 / h,          # Inner gas characteristic mass
        theta_inn=0.1,             # Inner gas radius [R200]
        theta_out=3.0,             # Outer gas radius [R200]
        epsilon_hydro=np.sqrt(5),  # Hydrostatic equilibrium factor
        theta_rg=0.3,              # Reaccreted gas radius
        sigma_rg=0.1,              # Reaccreted gas width
        M_r=1e16,                  # Reaccreted gas mass scale
        beta_r=2,                  # Reaccreted gas slope
        
        # Stellar parameters - TNG calibrated
        eta=0.6,                   # TNG fit: stellar fraction normalization
        M1_0=2.2e11 / h,           # TNG fit: stellar characteristic mass
        alpha_g=2,                 # Stellar profile slope
        epsilon_h=0.015,           # Half-light radius factor
        
        # Satellite galaxy parameters
        alpha_fsat=1,              # Satellite fraction mass slope
        M1_fsat=1,                 # Satellite characteristic mass
        delta_fsat=1,              # Satellite delta parameter
        gamma_fsat=1,              # Satellite gamma parameter
        eps_fsat=1,                # Satellite epsilon parameter
        
        # Adiabatic relaxation parameters
        a=0.3,                     # Relaxation parameter
        n=2,                       # NFW inner slope modification
        
        # Two-halo term parameters
        p=0.3,                     # Two-halo p parameter
        q=0.707,                   # Two-halo q parameter
        
        # Non-thermal pressure parameters
        A_nt=0.495,                # Non-thermal pressure amplitude
        alpha_nt=0.1,              # Non-thermal pressure slope
        
        # Gas thermodynamics
        mean_molecular_weight=0.59,  # Gas mean molecular weight
    )


# Model registry mapping model names to their classes and parameter functions
BCM_MODELS = {
    'schneider19': {
        'DMO': bfg.Profiles.Schneider19.DarkMatterOnly if HAS_BARYONFORGE else None,
        'DMB': bfg.Profiles.Schneider19.DarkMatterBaryon if HAS_BARYONFORGE else None,
        'params': get_bcm_params_schneider19,
        'Rdelta_sampling': False,
    },
    'schneider25': {
        'DMO': bfg.Profiles.Schneider25.DarkMatterOnly if HAS_BARYONFORGE else None,
        'DMB': bfg.Profiles.Schneider25.DarkMatterBaryon if HAS_BARYONFORGE else None,
        'params': get_bcm_params_schneider25,
        'Rdelta_sampling': False,
    },
    'arico20': {
        'DMO': bfg.Profiles.Arico20.DarkMatterOnly if HAS_BARYONFORGE else None,
        'DMB': bfg.Profiles.Arico20.DarkMatterBaryon if HAS_BARYONFORGE else None,
        'params': get_bcm_params_arico20,
        'Rdelta_sampling': True,  # Important for Arico20 due to sharp features
    },
}


# ============================================================================
# Lensplane Configuration
# ============================================================================

LENSPLANE_CONFIG = {
    'enabled': False,
    'n_realizations': 10,
    'planes_per_snapshot': 2,
    'grid_res': 4096,
    'seed': 2020,
    'output_base': '/mnt/home/mlee1/ceph/hydro_replace_LP_bcm',
}

# Snapshot order for ray-tracing (from z≈0 to z≈2)
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
SNAPSHOT_TO_INDEX = {snap: idx for idx, snap in enumerate(SNAPSHOT_ORDER)}
N_SNAPSHOTS = len(SNAPSHOT_ORDER)


# ============================================================================
# Lensplane Transform Generator (same as in generate_all_unified.py)
# ============================================================================

class TransformGenerator:
    """Generate reproducible random transforms for lensplanes."""
    
    def __init__(self, n_realizations=10, n_snapshots=20, pps=2, seed=2020, box_size=205.0):
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
    
    @classmethod
    def load(cls, filepath):
        with h5py.File(filepath, 'r') as f:
            gen = cls(
                n_realizations=f.attrs['n_realizations'],
                n_snapshots=f.attrs['n_snapshots'],
                pps=f.attrs['pps'],
                seed=f.attrs['seed'],
                box_size=f.attrs['box_size']
            )
            gen.proj_dirs = f['proj_dirs'][:]
            gen.displacements = f['displacements'][:]
            gen.flips = f['flips'][:]
        return gen


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
# Data Loading
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
        self.ids = None
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
        
        coords_list, masses_list, ids_list = [], [], []
        
        for filepath in my_files:
            with h5py.File(filepath, 'r') as f:
                pt_key = 'PartType1'  # DM particles only
                if pt_key not in f or f[pt_key]['Coordinates'].shape[0] == 0:
                    continue
                
                n_part = f[pt_key]['Coordinates'].shape[0]
                coords_list.append(f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3)  # kpc -> Mpc
                ids_list.append(f[pt_key]['ParticleIDs'][:])
                
                if 'Masses' in f[pt_key]:
                    masses_list.append(f[pt_key]['Masses'][:].astype(np.float32) * MASS_UNIT)
                else:
                    masses_list.append(np.full(n_part, dm_mass * MASS_UNIT, dtype=np.float32))
        
        if coords_list:
            self.coords = np.concatenate(coords_list)
            self.masses = np.concatenate(masses_list)
            self.ids = np.concatenate(ids_list)
        else:
            self.coords = np.zeros((0, 3), dtype=np.float32)
            self.masses = np.zeros(0, dtype=np.float32)
            self.ids = np.zeros(0, dtype=np.int64)
        
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
        """Query particles within radius of halo center (handles periodic BC)."""
        if self.tree is None or len(self.coords) == 0:
            return np.array([], dtype=int)
        
        search_radius = radius * self.radius_mult
        indices = self.tree.query_ball_point(center, search_radius)
        
        if len(indices) == 0:
            return np.array([], dtype=int)
        
        indices = np.array(indices)
        
        # Check periodic images if near box edge
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
        del self.coords, self.masses, self.ids, self.tree
        self.coords = self.masses = self.ids = self.tree = None
        gc.collect()


# ============================================================================
# BCM Displacement using BaryonifySnapshot
# ============================================================================

def create_baryonification_model(model_name, cosmo_params, epsilon_max=20.0):
    """
    Create a Baryonification3D model for a given BCM model.
    
    Args:
        model_name: One of 'schneider19', 'schneider25', 'arico20'
        cosmo_params: dict with cosmology parameters
        epsilon_max: maximum displacement radius in units of R200
    
    Returns:
        baryons: Baryonification3D model
        cosmo: CCL cosmology object
        bcm_params: BCM parameter dict
    """
    if model_name not in BCM_MODELS:
        raise ValueError(f"Unknown BCM model: {model_name}. Available: {list(BCM_MODELS.keys())}")
    
    model_config = BCM_MODELS[model_name]
    h = cosmo_params['h']
    bcm_params = model_config['params'](h=h)
    
    # Create CCL cosmology
    # Note: BaryonForge requires matter_power_spectrum='linear' for the 2-halo term
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
    
    # Create profiles
    DMO = model_config['DMO'](**bcm_params)
    DMB = model_config['DMB'](**bcm_params)
    
    # Create baryonification model with N_int for integration accuracy
    baryons = bfg.Baryonification3D(
        DMO, DMB,
        cosmo,
        N_int=50_000,
    )
    
    return baryons, cosmo, bcm_params


def create_baryonify_snapshot(baryons, halo_catalog, particle_snapshot, epsilon_max=20.0, verbose=True):
    """
    Create a BaryonifySnapshot runner with all required inputs.
    
    The BaryonForge API requires HaloNDCatalog and ParticleSnapshot at init time.
    
    Args:
        baryons: Baryonification3D model
        halo_catalog: bfg.utils.HaloNDCatalog object
        particle_snapshot: bfg.utils.ParticleSnapshot object
        epsilon_max: maximum displacement radius in units of R200
        verbose: whether to print progress
    
    Returns:
        BaryonifySnapshot runner object
    """
    runner = bfg.Runners.BaryonifySnapshot(
        HaloNDCatalog=halo_catalog,
        ParticleSnapshot=particle_snapshot,
        epsilon_max=epsilon_max,
        model=baryons,
        verbose=verbose
    )
    
    return runner


def compute_periodic_distance(dx, box_size):
    """Compute distance with periodic boundary conditions."""
    dx = np.where(dx > box_size/2, dx - box_size, dx)
    dx = np.where(dx < -box_size/2, dx + box_size, dx)
    return dx


def apply_bcm_displacements_baryonforge(baryons, particles, halos, z_snap, M_min, M_max, cosmo_params, epsilon_max, comm):
    """
    Apply BCM displacements using BaryonForge's displacement() method.
    
    This function handles MPI-distributed particles by computing displacements
    on each rank's local particles. Uses the Baryonification3D.displacement()
    method to compute radial displacements for particles near each halo.
    
    Args:
        baryons: Baryonification3D model object
        particles: DistributedParticles object with DMO particles
        halos: dict with 'masses', 'positions', 'radii'
        z_snap: snapshot redshift
        M_min: minimum halo mass for interpolator
        M_max: maximum halo mass for interpolator
        cosmo_params: dict with cosmology parameters
        epsilon_max: maximum displacement radius in units of R200
        comm: MPI communicator
    
    Returns:
        bcm_coords: (N, 3) array of BCM-displaced coordinates
    """
    if rank == 0:
        print(f"  Setting up Baryonification interpolator...")
        t0 = time.time()
    
    # Setup the interpolator for this snapshot's redshift range
    baryons.setup_interpolator(
        z_min=0, z_max=3,
        z_linear_sampling=True,
        N_samples_R=10000,
        verbose=(rank == 0)
    )
    
    if rank == 0:
        print(f"    Interpolator setup time: {time.time()-t0:.1f}s")
    
    h = cosmo_params['h']
    a_snap = 1.0 / (1.0 + z_snap)
    
    n_particles = len(particles.coords)
    n_halos = len(halos['masses'])
    
    if rank == 0:
        print(f"  Computing BCM displacements...")
        print(f"    N particles (this rank): {n_particles:,}")
        print(f"    N halos: {n_halos}")
        t0 = time.time()
    
    # Initialize displaced coordinates as copy of original
    # Apply periodic BC first to ensure all coords are within [0, BOX_SIZE)
    displaced = np.mod(particles.coords.copy(), BOX_SIZE).astype(np.float64)
    
    # Build KDTree for this rank's particles (with periodic BC)
    if n_particles > 0:
        tree = cKDTree(displaced, boxsize=BOX_SIZE)
    else:
        tree = None
    
    # Loop over halos and compute displacements
    n_displaced = 0
    for j in range(n_halos):
        halo_pos = halos['positions'][j]
        halo_r200 = halos['radii'][j]
        halo_mass = halos['masses'][j]
        
        # Query radius - in comoving Mpc/h
        R_query = epsilon_max * halo_r200
        R_query = min(R_query, BOX_SIZE / 2)  # Can't query more than half box
        
        if tree is None:
            continue
        
        # Query particles within R_query of this halo
        inds = tree.query_ball_point(halo_pos, R_query)
        
        if len(inds) == 0:
            continue
        
        inds = np.array(inds)
        region_coords = displaced[inds]  # Use wrapped coordinates
        
        try:
            # Compute distance from halo center (with periodic BC)
            dx = region_coords[:, 0] - halo_pos[0]
            dy = region_coords[:, 1] - halo_pos[1]
            dz = region_coords[:, 2] - halo_pos[2]
            
            # Apply periodic boundary conditions to distance vectors
            dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
            dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
            dy = np.where(dy > BOX_SIZE/2, dy - BOX_SIZE, dy)
            dy = np.where(dy < -BOX_SIZE/2, dy + BOX_SIZE, dy)
            dz = np.where(dz > BOX_SIZE/2, dz - BOX_SIZE, dz)
            dz = np.where(dz < -BOX_SIZE/2, dz + BOX_SIZE, dz)
            
            # Distance in comoving Mpc/h
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Convert to comoving Mpc (BaryonForge uses Mpc, not Mpc/h)
            r_mpc = r / h
            
            # Unit vectors (direction from halo to particle)
            r_safe = np.maximum(r, 1e-10)  # Avoid division by zero
            x_hat = dx / r_safe
            y_hat = dy / r_safe
            z_hat = dz / r_safe
            
            # Get displacement from BaryonForge (in comoving Mpc)
            # displacement(r, M, a) returns radial displacement
            offset = baryons.displacement(r_mpc, halo_mass, a_snap)
            
            # Handle NaN/Inf values
            offset = np.where(np.isfinite(offset), offset, 0.0)
            
            # Convert displacement back to Mpc/h
            offset_mpch = offset * h
            
            # Apply radial displacement
            displaced[inds, 0] += offset_mpch * x_hat
            displaced[inds, 1] += offset_mpch * y_hat
            displaced[inds, 2] += offset_mpch * z_hat
            
            n_displaced += len(inds)
            
        except Exception as e:
            # Skip halos that cause errors
            if rank == 0 and j < 10:
                print(f"    Warning: Halo {j} displacement failed: {e}")
            continue
        
        if rank == 0 and (j + 1) % 500 == 0:
            print(f"    Processed {j+1}/{n_halos} halos...")
            sys.stdout.flush()
    
    # Enforce periodic boundary conditions
    bcm_coords = np.mod(displaced, BOX_SIZE)
    
    if rank == 0:
        print(f"    Displaced {n_displaced:,} particle instances")
        print(f"    Displacement calculation time: {time.time()-t0:.1f}s")
    
    return bcm_coords.astype(np.float32)


# ============================================================================
# Profile Computation
# ============================================================================

def compute_profile(coords, masses, center, r200, radial_bins):
    """Compute density profile for a single halo."""
    if len(coords) == 0:
        n_bins = len(radial_bins) - 1
        return {
            'density': np.zeros(n_bins),
            'mass': np.zeros(n_bins),
            'count': np.zeros(n_bins, dtype=int),
        }
    
    # Compute distances (with periodic BC)
    dx = coords - center
    dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
    dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
    r = np.linalg.norm(dx, axis=1) / r200  # In units of R200
    
    # Profile
    mass_profile, _ = np.histogram(r, bins=radial_bins, weights=masses)
    count_profile, _ = np.histogram(r, bins=radial_bins)
    
    # Shell volumes (in physical units: (R200 * Mpc/h)^3)
    volumes = 4/3 * np.pi * r200**3 * (radial_bins[1:]**3 - radial_bins[:-1]**3)
    density = np.where(volumes > 0, mass_profile / volumes, 0)
    
    return {
        'density': density,
        'mass': mass_profile,
        'count': count_profile,
    }


def compute_statistics_at_radii(coords, masses, center, r200, radii_mult):
    """Compute statistics at specific radii."""
    n_radii = len(radii_mult)
    
    if len(coords) == 0:
        return {
            'm_total': np.zeros(n_radii, dtype=np.float64),
        }
    
    # Compute distances
    dx = coords - center
    dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
    dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
    r = np.linalg.norm(dx, axis=1) / r200
    
    results = {
        'm_total': np.zeros(n_radii, dtype=np.float64),
    }
    
    for i, r_mult in enumerate(radii_mult):
        mask = r <= r_mult
        results['m_total'][i] = np.sum(masses[mask])
    
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

def get_snapshot_redshift(snapshot, sim_res):
    """Get redshift for a given snapshot number."""
    basePath = SIM_PATHS[sim_res]['dmo']
    snap_file = f"{basePath}/snapdir_{snapshot:03d}/snap_{snapshot:03d}.0.hdf5"
    
    with h5py.File(snap_file, 'r') as f:
        z = f['Header'].attrs['Redshift']
    
    return z


def run_bcm_pipeline(args):
    """
    Run the BCM pipeline for profiles, statistics, and maps.
    
    Supports multiple BCM models: Schneider19, Schneider25, Arico20.
    Uses BaryonForge's BaryonifySnapshot for displacements.
    Only outputs BCM results (no DMO).
    """
    
    if not HAS_BARYONFORGE:
        if rank == 0:
            print("ERROR: BaryonForge not installed. Cannot run BCM pipeline.")
        return
    
    t_start = time.time()
    
    # Parse models to run
    models_to_run = [m.strip().lower() for m in args.models.split(',')]
    for model in models_to_run:
        if model not in BCM_MODELS:
            if rank == 0:
                print(f"ERROR: Unknown model '{model}'. Available: {list(BCM_MODELS.keys())}")
            return
    
    # Output paths
    output_suffix = getattr(args, 'output_suffix', '')
    output_dir_base = os.path.join(OUTPUT_BASE.replace('_bcm', ''), f'L205n{args.sim_res}TNG')
    output_dir = os.path.join(OUTPUT_BASE, f'L205n{args.sim_res}TNG{output_suffix}')
    snap_dir = os.path.join(output_dir, f'snap{args.snap:03d}')
    
    # Get snapshot redshift
    z_snap = get_snapshot_redshift(args.snap, args.sim_res)
    a_snap = 1.0 / (1.0 + z_snap)
    
    if rank == 0:
        print("=" * 70)
        print("BCM PIPELINE (Baryonic Correction Model)")
        print("=" * 70)
        print(f"Snapshot: {args.snap} (z = {z_snap:.4f})")
        print(f"Resolution: L205n{args.sim_res}TNG")
        print(f"Mass threshold: 10^{args.mass_min} Msun/h")
        print(f"Radius: {args.radius_mult}×R200")
        print(f"Grid: {args.grid}²")
        print(f"BCM epsilon_max: {args.epsilon_max}")
        print(f"BCM models: {', '.join(models_to_run)}")
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
        print("\n[1/5] Loading halo catalog...")
        sys.stdout.flush()
    
    matches_file = os.path.join(output_dir_base, 'matches', f'matches_snap{args.snap:03d}.npz')
    
    with np.load(matches_file) as data:
        all_masses = data['dmo_masses'] * MASS_UNIT
        all_positions = data['dmo_positions'] / 1e3  # kpc -> Mpc
        all_radii = data['dmo_radii'] / 1e3  # kpc -> Mpc
    
    # Select halos above mass threshold
    log_masses = np.log10(all_masses)
    mass_mask = log_masses >= args.mass_min
    
    halo_masses = all_masses[mass_mask]
    halo_positions = all_positions[mass_mask]
    halo_radii = all_radii[mass_mask]
    halo_log_masses = log_masses[mass_mask]
    
    n_halos = len(halo_masses)
    
    if rank == 0:
        print(f"  Total halos: {len(all_masses)}")
        print(f"  Halos above 10^{args.mass_min}: {n_halos}")
    
    # Assign halos to mass bins
    mass_bin_indices = np.digitize(halo_log_masses, MASS_BIN_EDGES) - 1
    n_mass_bins = len(MASS_BIN_EDGES) - 1
    n_radial_bins = len(RADIAL_BINS) - 1
    n_stats_radii = len(STATS_RADII_MULT)
    
    # Count halos per mass bin for proper profile normalization
    n_halos_per_bin = np.zeros(n_mass_bins, dtype=np.int64)
    for mb in mass_bin_indices:
        if 0 <= mb < n_mass_bins:
            n_halos_per_bin[mb] += 1
    
    if rank == 0:
        print(f"  Halos per mass bin: {dict(zip([f'{MASS_BIN_EDGES[i]:.1f}-{MASS_BIN_EDGES[i+1]:.1f}' for i in range(n_mass_bins)], n_halos_per_bin))}")
    
    halos = {
        'masses': halo_masses,
        'positions': halo_positions,
        'radii': halo_radii,
    }
    
    # ========================================================================
    # Load DMO particles
    # ========================================================================
    if rank == 0:
        print("\n[2/5] Loading DMO particles...")
        sys.stdout.flush()
    
    dmo = DistributedParticles(args.snap, args.sim_res, args.radius_mult)
    dmo.load().build_tree()
    
    # Storage for all BCM model results
    all_bcm_results = {}
    
    # ========================================================================
    # Loop over BCM models
    # ========================================================================
    for model_idx, model_name in enumerate(models_to_run):
        if rank == 0:
            print(f"\n[3/5] Processing BCM model: {model_name.upper()} ({model_idx+1}/{len(models_to_run)})")
            print("=" * 50)
            sys.stdout.flush()
        
        # --------------------------------------------------------------------
        # Setup BCM model and apply displacements
        # --------------------------------------------------------------------
        if rank == 0:
            print(f"\n  Setting up {model_name} BCM model...")
            sys.stdout.flush()
        
        baryons, cosmo, bcm_params = create_baryonification_model(
            model_name, TNG_COSMOLOGY,
            epsilon_max=args.epsilon_max
        )
        
        if rank == 0:
            print(f"\n  Applying {model_name} displacements...")
            sys.stdout.flush()
        
        bcm_coords = apply_bcm_displacements_baryonforge(
            baryons, dmo, halos, z_snap,
            M_min=10**args.mass_min, M_max=1e16,
            cosmo_params=TNG_COSMOLOGY,
            epsilon_max=args.epsilon_max,
            comm=comm
        )
        
        # Sanity check: verify BCM coords have same shape as DMO coords
        if rank == 0:
            print(f"    BCM coords shape: {bcm_coords.shape}, DMO coords shape: {dmo.coords.shape}")
        
        # --------------------------------------------------------------------
        # Compute BCM profiles and statistics
        # --------------------------------------------------------------------
        if rank == 0:
            print(f"\n  Computing {model_name} profiles...")
            sys.stdout.flush()
        
        # Build tree with BCM coordinates
        bcm_tree = cKDTree(bcm_coords) if len(bcm_coords) > 0 else None
        
        local_bcm_profiles = np.zeros((n_mass_bins, n_radial_bins), dtype=np.float64)
        local_bcm_counts = np.zeros((n_mass_bins, n_radial_bins), dtype=np.int64)
        local_bcm_stats = np.zeros((n_halos, n_stats_radii), dtype=np.float64)
        
        # Individual profiles for each halo (density and mass)
        local_individual_profiles = np.zeros((n_halos, n_radial_bins), dtype=np.float64)
        local_individual_mass_profiles = np.zeros((n_halos, n_radial_bins), dtype=np.float64)
        
        for i in range(n_halos):
            center = halo_positions[i]
            r200 = halo_radii[i]
            mass_bin = mass_bin_indices[i]
            
            # Query BCM particles
            search_radius = r200 * args.radius_mult
            if bcm_tree is not None:
                local_idx = bcm_tree.query_ball_point(center, search_radius)
            else:
                local_idx = []
            
            if len(local_idx) > 0:
                local_idx = np.array(local_idx)
                local_coords = bcm_coords[local_idx]
                local_masses = dmo.masses[local_idx]  # Masses unchanged
                
                profile = compute_profile(local_coords, local_masses, center, r200, RADIAL_BINS)
                
                # Store individual profile for this halo
                local_individual_profiles[i] = profile['density']
                local_individual_mass_profiles[i] = profile['mass']
                
                if 0 <= mass_bin < n_mass_bins:
                    local_bcm_profiles[mass_bin] += profile['density']
                    local_bcm_counts[mass_bin] += profile['count']
                
                # Statistics at multiple radii
                dx = local_coords - center
                dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
                dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
                r_norm = np.linalg.norm(dx, axis=1) / r200
                
                for j, r_mult in enumerate(STATS_RADII_MULT):
                    mask = r_norm <= r_mult
                    local_bcm_stats[i, j] = np.sum(local_masses[mask])
            
            if rank == 0 and (i + 1) % 500 == 0:
                print(f"    {model_name} Halo {i+1}/{n_halos}...")
                sys.stdout.flush()
        
        # Reduce BCM profiles and stats
        global_bcm_profiles = np.zeros_like(local_bcm_profiles)
        global_bcm_counts = np.zeros_like(local_bcm_counts)
        global_bcm_stats = np.zeros_like(local_bcm_stats)
        
        comm.Reduce(local_bcm_profiles, global_bcm_profiles, op=MPI.SUM, root=0)
        comm.Reduce(local_bcm_counts, global_bcm_counts, op=MPI.SUM, root=0)
        comm.Reduce(local_bcm_stats, global_bcm_stats, op=MPI.SUM, root=0)
        
        # Reduce individual profiles (each halo's profile is summed across ranks)
        global_individual_profiles = np.zeros_like(local_individual_profiles)
        global_individual_mass_profiles = np.zeros_like(local_individual_mass_profiles)
        comm.Reduce(local_individual_profiles, global_individual_profiles, op=MPI.SUM, root=0)
        comm.Reduce(local_individual_mass_profiles, global_individual_mass_profiles, op=MPI.SUM, root=0)
        
        # Print summary statistics for verification
        if rank == 0:
            total_mass_in_halos = np.sum(global_bcm_stats[:, 1])  # at R200
            print(f"    Total mass within R200 across all halos: {total_mass_in_halos:.3e} Msun/h")
            print(f"    Mean mass within R200 per halo: {total_mass_in_halos/n_halos:.3e} Msun/h")
        
        del bcm_tree
        gc.collect()
        
        # --------------------------------------------------------------------
        # Generate BCM map
        # --------------------------------------------------------------------
        if rank == 0:
            print(f"\n  Generating {model_name} map...")
            sys.stdout.flush()
        
        local_bcm_map = project_to_2d(bcm_coords, dmo.masses, args.grid)
        if rank == 0:
            global_bcm_map = np.zeros((args.grid, args.grid), dtype=np.float32)
        else:
            global_bcm_map = None
        comm.Reduce(local_bcm_map, global_bcm_map, op=MPI.SUM, root=0)
        del local_bcm_map
        
        if rank == 0:
            # Save BCM map
            bcm_file = os.path.join(snap_dir, 'projected', f'{model_name}.npz')
            np.savez_compressed(bcm_file, field=global_bcm_map, box_size=BOX_SIZE,
                               grid_resolution=args.grid, snapshot=args.snap,
                               model=model_name)
            print(f"    {model_name} map saved: {bcm_file}")
            del global_bcm_map
        
        # Store results for this model
        all_bcm_results[model_name] = {
            'profiles': global_bcm_profiles if rank == 0 else None,
            'counts': global_bcm_counts if rank == 0 else None,
            'stats': global_bcm_stats if rank == 0 else None,
            'individual_profiles': global_individual_profiles if rank == 0 else None,
            'individual_mass_profiles': global_individual_mass_profiles if rank == 0 else None,
            'params': bcm_params,
            'coords': bcm_coords,  # Keep for lensplanes
        }
        
        # Run lensplanes for this model if enabled
        if args.enable_lensplanes:
            run_lensplane_generation_single_model(
                args, model_name, bcm_coords, dmo.masses, comm
            )
        
        # Clean up coordinates if not needed for lensplanes
        if not args.enable_lensplanes:
            del bcm_coords
            all_bcm_results[model_name]['coords'] = None
        
        gc.collect()
    
    # ========================================================================
    # Save profiles and statistics for all models
    # ========================================================================
    if rank == 0:
        print("\n[4/5] Saving results...")
        
        # Save profiles (one file per model)
        for model_name in models_to_run:
            result = all_bcm_results[model_name]
            
            profile_file = os.path.join(output_dir, 'profiles', f'profiles_{model_name}_snap{args.snap:03d}.h5')
            with h5py.File(profile_file, 'w') as f:
                f.attrs['snapshot'] = args.snap
                f.attrs['redshift'] = z_snap
                f.attrs['mass_min'] = args.mass_min
                f.attrs['radius_multiplier'] = args.radius_mult
                f.attrs['epsilon_max'] = args.epsilon_max
                f.attrs['radial_bins'] = RADIAL_BINS
                f.attrs['mass_bin_edges'] = MASS_BIN_EDGES
                f.attrs['box_size'] = BOX_SIZE
                f.attrs['bcm_model'] = model_name
                f.attrs['n_halos'] = n_halos
                
                # BCM parameters
                bcm_grp = f.create_group('bcm_params')
                for k, v in result['params'].items():
                    try:
                        bcm_grp.attrs[k] = v
                    except TypeError:
                        bcm_grp.attrs[k] = str(v)
                
                # Individual profiles for each halo (n_halos, n_radial_bins)
                # density profiles in units of Msun/h / (Mpc/h)^3
                f.create_dataset('individual_density_profiles', data=result['individual_profiles'].astype(np.float32),
                                compression='gzip', compression_opts=4)
                # mass profiles in units of Msun/h per radial bin
                f.create_dataset('individual_mass_profiles', data=result['individual_mass_profiles'].astype(np.float32),
                                compression='gzip', compression_opts=4)
                
                # Halo properties (for reference with individual profiles)
                f.create_dataset('halo_log_masses', data=halo_log_masses.astype(np.float32))
                f.create_dataset('halo_radii', data=halo_radii.astype(np.float32))
                f.create_dataset('halo_positions', data=halo_positions.astype(np.float32))
                f.create_dataset('mass_bin_indices', data=mass_bin_indices.astype(np.int32))
                
                # Stacked profiles (summed across all halos in each mass bin)
                f.create_dataset('stacked_bcm', data=result['profiles'])
                f.create_dataset('counts_bcm', data=result['counts'])
                
                # Number of halos per mass bin (for computing average profiles)
                f.create_dataset('n_halos_per_bin', data=n_halos_per_bin)
                
                # Also save averaged profiles for convenience
                avg_profiles = np.zeros_like(result['profiles'])
                for mb in range(n_mass_bins):
                    if n_halos_per_bin[mb] > 0:
                        avg_profiles[mb] = result['profiles'][mb] / n_halos_per_bin[mb]
                f.create_dataset('averaged_bcm', data=avg_profiles)
            
            print(f"    {model_name} profiles saved: {profile_file}")
            print(f"      - Individual profiles: {n_halos} halos × {n_radial_bins} radial bins")
        
        # Save statistics (one file with all models)
        stats_file = os.path.join(output_dir, 'analysis', f'bcm_halo_statistics_snap{args.snap:03d}.h5')
        with h5py.File(stats_file, 'w') as f:
            f.attrs['snapshot'] = args.snap
            f.attrs['redshift'] = z_snap
            f.attrs['mass_min'] = args.mass_min
            f.attrs['n_halos'] = n_halos
            f.attrs['radii_r200'] = np.array(STATS_RADII_MULT)
            f.attrs['sim_res'] = args.sim_res
            f.attrs['epsilon_max'] = args.epsilon_max
            f.attrs['bcm_models'] = ','.join(models_to_run)
            
            # Halo properties
            f.create_dataset('log_masses', data=halo_log_masses.astype(np.float32))
            f.create_dataset('positions', data=halo_positions.astype(np.float32))
            f.create_dataset('radii', data=halo_radii.astype(np.float32))
            
            # BCM statistics for each model
            for model_name in models_to_run:
                result = all_bcm_results[model_name]
                grp = f.create_group(model_name)
                grp.create_dataset('m_bcm', data=result['stats'])
                
                # Store params
                param_grp = grp.create_group('params')
                for k, v in result['params'].items():
                    try:
                        param_grp.attrs[k] = v
                    except TypeError:
                        param_grp.attrs[k] = str(v)
        
        print(f"    Statistics saved: {stats_file}")
    
    # Cleanup
    dmo.free()
    for model_name in models_to_run:
        if all_bcm_results[model_name]['coords'] is not None:
            del all_bcm_results[model_name]['coords']
    del all_bcm_results
    gc.collect()
    
    # ========================================================================
    # Summary
    # ========================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("Complete!")
        print(f"Total time: {time.time()-t_start:.1f}s")
        print(f"Output directory: {output_dir}")
        print(f"BCM models processed: {', '.join(models_to_run)}")
        print("=" * 70)


# ============================================================================
# Lensplane Generation for BCM (single model)
# ============================================================================

def run_lensplane_generation_single_model(args, model_name, bcm_coords, particle_masses, comm):
    """Run lensplane generation for a single BCM model."""
    rank = comm.Get_rank()
    
    lp_config = LENSPLANE_CONFIG.copy()
    lp_config['grid_res'] = args.lensplane_grid
    
    output_dir = os.path.join(lp_config['output_base'], f'L205n{args.sim_res}TNG')
    
    if args.snap not in SNAPSHOT_TO_INDEX:
        if rank == 0:
            print(f"\nWarning: Snapshot {args.snap} not in SNAPSHOT_ORDER, skipping lensplanes")
        return
    
    snapshot_idx = SNAPSHOT_TO_INDEX[args.snap]
    
    if rank == 0:
        print(f"\n  Generating {model_name} lensplanes...")
        print(f"    Snapshot: {args.snap} (index {snapshot_idx})")
        print(f"    N realizations: {lp_config['n_realizations']}")
        print(f"    Grid resolution: {lp_config['grid_res']}")
        t0 = time.time()
        sys.stdout.flush()
        
        os.makedirs(output_dir, exist_ok=True)
    
    comm.Barrier()
    
    transforms = TransformGenerator(
        n_realizations=lp_config['n_realizations'],
        pps=lp_config['planes_per_snapshot'],
        seed=lp_config['seed'],
        box_size=BOX_SIZE
    )
    
    # Save transforms (only once)
    if rank == 0:
        transform_file = os.path.join(output_dir, 'transforms.h5')
        if not os.path.exists(transform_file):
            transforms.save(transform_file)
    
    # Generate lensplanes for this model
    generate_model_lensplanes_bcm(model_name, bcm_coords, particle_masses,
                                   transforms, lp_config['grid_res'], BOX_SIZE,
                                   output_dir, args.snap, comm)
    
    if rank == 0:
        print(f"    {model_name} lensplane generation time: {time.time()-t0:.1f}s")
    
    comm.Barrier()


def generate_model_lensplanes_bcm(model_name, pos, mass, transforms, lp_grid, box_size,
                                   output_dir, snap, comm):
    """Generate lensplanes for a single model (DMO or BCM)."""
    rank = comm.Get_rank()
    
    if snap not in SNAPSHOT_TO_INDEX:
        return
    
    snapshot_idx = SNAPSHOT_TO_INDEX[snap]
    pps = transforms.pps
    n_realizations = transforms.n_realizations
    
    for real_idx in range(n_realizations):
        t = transforms.get_transform(real_idx, snapshot_idx)
        
        for pps_slice in range(pps):
            file_idx = snapshot_idx * pps + pps_slice
            
            local_delta = project_lensplane(pos, mass, t, lp_grid, box_size, pps_slice, pps)
            
            if rank == 0:
                global_delta = np.zeros((lp_grid, lp_grid), dtype=np.float64)
            else:
                global_delta = None
            
            local_delta_contig = np.ascontiguousarray(local_delta.astype(np.float64))
            comm.Reduce(local_delta_contig, global_delta, op=MPI.SUM, root=0)
            
            if rank == 0:
                plane_dir = os.path.join(output_dir, model_name, f'LP_{real_idx:02d}')
                os.makedirs(plane_dir, exist_ok=True)
                filepath = os.path.join(plane_dir, f'lenspot{file_idx:02d}.dat')
                write_lensplane(filepath, global_delta, lp_grid)
            
            del local_delta, local_delta_contig
            if rank == 0:
                del global_delta
        
        if rank == 0 and (real_idx + 1) % 5 == 0:
            print(f"    {model_name.upper()}: Realization {real_idx + 1}/{n_realizations} done")
            sys.stdout.flush()
    
    comm.Barrier()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='BCM pipeline for profiles, stats, maps, and lensplanes')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, default=2500, choices=[625, 1250, 2500])
    parser.add_argument('--mass-min', type=float, default=12.5,
                        help='Minimum log10(M200c/Msun/h) for halo selection')
    parser.add_argument('--radius-mult', type=float, default=5.0,
                        help='Radius multiplier (×R200) for particle queries')
    parser.add_argument('--grid', type=int, default=GRID_RES,
                        help='Grid resolution for maps')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Suffix for output directory')
    
    # BCM arguments
    parser.add_argument('--epsilon-max', type=float, default=20.0,
                        help='Maximum displacement radius in units of R200')
    parser.add_argument('--models', type=str, default='schneider19,schneider25,arico20',
                        help='Comma-separated list of BCM models to run. '
                             'Available: schneider19, schneider25, arico20. '
                             'Default: all three models.')
    
    # Lensplane arguments
    parser.add_argument('--enable-lensplanes', action='store_true',
                        help='Enable lensplane generation')
    parser.add_argument('--lensplane-grid', type=int, default=LENSPLANE_CONFIG['grid_res'],
                        help='Grid resolution for lensplanes')
    
    args = parser.parse_args()
    
    run_bcm_pipeline(args)


if __name__ == '__main__':
    main()

