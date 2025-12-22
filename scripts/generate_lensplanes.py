#!/usr/bin/env python
"""
Generate lens planes for ray-tracing with lux.

This script generates 2D projected density fields (δ × dz) for:
  - DMO (dark matter only)
  - Hydro (full hydrodynamic simulation)
  - BCM models (Arico20, Schneider19, Schneider25)
  - Replace (DMO + hydro in halo regions)

Output format: Binary files compatible with lux PreProjected mode.

Usage:
    # Test with low-res (L205n625TNG)
    mpirun -np 16 python generate_lensplanes.py --sim-res 625 --model dmo --snap 99
    
    # Production with high-res (L205n2500TNG)
    mpirun -np 64 python generate_lensplanes.py --sim-res 2500 --model all --snap all

    # Generate all lens planes for a single model
    mpirun -np 64 python generate_lensplanes.py --sim-res 2500 --model replace --mass-min 12.5 --snap all
"""

import numpy as np
import h5py
import glob
import argparse
import os
import struct
import time
from mpi4py import MPI
from scipy.spatial import cKDTree
from illustris_python import groupcat
import MAS_library as MASL
import pyccl as ccl
import BaryonForge as bfg

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

# Snapshot configuration for ray-tracing (from lux.ini)
# Format: (snapshot, redshift, stack_flag)
# stack_flag=True means use doubled transverse box (for high-z)
SNAPSHOT_CONFIG = [
    (96, 0.04, False),
    (90, 0.15, False),
    (85, 0.27, False),
    (80, 0.40, False),
    (76, 0.50, False),
    (71, 0.64, False),
    (67, 0.78, False),
    (63, 0.93, False),
    (59, 1.07, False),
    (56, 1.18, False),
    (52, 1.36, True),
    (49, 1.50, True),
    (46, 1.65, True),
    (43, 1.82, True),
    (41, 1.93, True),
    (38, 2.12, True),
    (35, 2.32, True),
    (33, 2.49, True),
    (31, 2.68, True),
    (29, 2.87, True),
]

CONFIG = {
    'box_size': 205.0,  # Mpc/h
    'mass_unit': 1e10,  # Convert to Msun/h
    'grid_res': 4096,   # Lens plane resolution (default)
    'planes_per_snapshot': 2,  # Lens planes per snapshot
    'output_base': '/mnt/home/mlee1/ceph/hydro_replace_lensplanes',
    'radius_multiplier': 5,  # Replace within 5 × R_200
}

# TNG300 cosmology
COSMO = ccl.Cosmology(
    Omega_c=0.3089 - 0.0486, Omega_b=0.0486, h=0.6774,
    sigma8=0.8158, n_s=0.9649, matter_power_spectrum='linear'
)
h = COSMO.cosmo.params.h
Omega_m = COSMO.cosmo.params.Omega_m
rho_c0 = 27.7536627  # Critical density in 10^10 Msun/h / (Mpc/h)^3

# BCM Parameters (from original publications)
BCM_PARAMS = {
    # Arico+20 fiducial values from Table 1 of the paper (Lu et al. calibration)
    'Arico20': dict(
        M_c=3.3e13,       # h^-1 Msun (fiducial from paper)
        M1_0=8.63e11,     # h^-1 Msun (fiducial from paper)
        eta=0.54,         # star formation efficiency (fiducial)
        beta=0.12,        # gas profile slope (fiducial)
        mu=0.31, M_inn=3.3e13, theta_inn=0.1, theta_out=3,
        epsilon_h=0.015, alpha_g=2,
        epsilon_hydro=np.sqrt(5), theta_rg=0.3, sigma_rg=0.1,
        a=0.3, n=2, p=0.3, q=0.707,
        alpha_fsat=1, M1_fsat=1, delta_fsat=1, gamma_fsat=1, eps_fsat=1,
        M_r=1e16, beta_r=2,
        A_nt=0.495, alpha_nt=0.1,  # non-thermal pressure
    ),
    'Schneider19': dict(
        theta_ej=4, theta_co=0.1, M_c=1e14/h, mu_beta=0.4,
        gamma=2, delta=7,
        eta=0.3, eta_delta=0.3, tau=-1.5, tau_delta=0,
        A=0.09/2, M1=2.5e11/h, epsilon_h=0.015,
        a=0.3, n=2, epsilon=4, p=0.3, q=0.707,
    ),
    'Schneider25': dict(
        M_c=1e15, mu=0.8,
        q0=0.075, q1=0.25, q2=0.7, nu_q0=0, nu_q1=1, nu_q2=0, nstep=3/2,
        theta_c=0.3, nu_theta_c=1/2, c_iga=0.1, nu_c_iga=3/2, r_min_iga=1e-3,
        alpha=1, gamma=3/2, delta=7,
        tau=-1.376, tau_delta=0, Mstar=3e11, Nstar=0.03,
        eta=0.1, eta_delta=0.22, epsilon_cga=0.03,
        alpha_nt=0.1, nu_nt=0.5, gamma_nt=0.8, mean_molecular_weight=0.6125,
        epsilon0=4, epsilon1=0.5, alpha_excl=0.4, p=0.3, q=0.707,
    ),
}


# ============================================================================
# Randomization (consistent across all models)
# ============================================================================

class RandomizationState:
    """
    Generate consistent randomization for all models.
    Uses the same seed as lux for reproducibility.
    """
    
    def __init__(self, seed=2020):
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.n_snapshots = len(SNAPSHOT_CONFIG)
        
        # Pre-generate all randomization parameters
        self.proj_dirs = self.rng.integers(0, 3, size=self.n_snapshots)
        self.displacements = self.rng.uniform(0, CONFIG['box_size'], 
                                               size=(self.n_snapshots, 3))
        self.flips = self.rng.integers(0, 2, size=(self.n_snapshots, 3)).astype(bool)
    
    def get_params(self, snap_idx):
        """Get randomization parameters for a snapshot index."""
        return {
            'proj_dir': self.proj_dirs[snap_idx],
            'displacement': self.displacements[snap_idx],
            'flip': self.flips[snap_idx],
        }


def apply_randomization(coords, params, box_size):
    """
    Apply translation and flip transformations to coordinates.
    
    Matches lux behavior exactly:
      1. Add displacement: x = x0 + disp
      2. Apply flip (negate): if flip: x = -x
      3. Apply periodic boundary conditions
    
    Parameters
    ----------
    coords : ndarray (N, 3)
        Particle coordinates in Mpc/h
    params : dict
        Randomization parameters from RandomizationState
    box_size : float
        Box size in Mpc/h
        
    Returns
    -------
    coords_transformed : ndarray (N, 3)
        Transformed coordinates
    """
    coords_out = coords.copy()
    
    # Step 1: Apply displacement
    coords_out = coords_out + params['displacement']
    
    # Step 2: Apply flips (negate coordinates)
    # This is what lux does: if(flip[axis]) x = -x
    for axis in range(3):
        if params['flip'][axis]:
            coords_out[:, axis] = -coords_out[:, axis]
    
    # Step 3: Apply periodic boundary conditions
    coords_out = np.mod(coords_out, box_size)
    
    return coords_out


def get_projection_axes(proj_dir):
    """
    Get the two axes to keep for 2D projection.
    
    proj_dir: 0 -> project along x, keep (y, z)
    proj_dir: 1 -> project along y, keep (z, x)
    proj_dir: 2 -> project along z, keep (x, y)
    """
    if proj_dir == 0:
        return 1, 2  # y, z
    elif proj_dir == 1:
        return 2, 0  # z, x
    else:
        return 0, 1  # x, y


# ============================================================================
# Particle Loading
# ============================================================================

def load_dmo_particles(basePath, snapNum, my_files, dmo_mass, mass_unit):
    """Load DMO particle coordinates and masses."""
    coords_list = []
    
    for filepath in my_files:
        with h5py.File(filepath, 'r') as f:
            if 'PartType1' not in f:
                continue
            coords = f['PartType1']['Coordinates'][:].astype(np.float32) / 1e3
            coords_list.append(coords)
    
    if len(coords_list) == 0:
        coords = np.zeros((0, 3), dtype=np.float32)
    else:
        coords = np.concatenate(coords_list)
    
    masses = np.ones(len(coords), dtype=np.float32) * dmo_mass * mass_unit
    return coords, masses


def load_hydro_particles(basePath, snapNum, my_files, hydro_dm_mass, mass_unit):
    """Load hydro particle coordinates and masses (gas + DM + stars)."""
    coords_list = []
    masses_list = []
    
    for filepath in my_files:
        with h5py.File(filepath, 'r') as f:
            # Gas
            if 'PartType0' in f:
                coords_list.append(f['PartType0']['Coordinates'][:].astype(np.float32) / 1e3)
                masses_list.append(f['PartType0']['Masses'][:].astype(np.float32) * mass_unit)
            
            # DM
            if 'PartType1' in f:
                c = f['PartType1']['Coordinates'][:].astype(np.float32) / 1e3
                m = np.ones(len(c), dtype=np.float32) * hydro_dm_mass * mass_unit
                coords_list.append(c)
                masses_list.append(m)
            
            # Stars
            if 'PartType4' in f:
                coords_list.append(f['PartType4']['Coordinates'][:].astype(np.float32) / 1e3)
                masses_list.append(f['PartType4']['Masses'][:].astype(np.float32) * mass_unit)
    
    if len(coords_list) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
    
    return np.concatenate(coords_list), np.concatenate(masses_list)


# ============================================================================
# 2D Projection
# ============================================================================

def project_to_2d(coords, masses, box_size, grid_res, proj_dir, plane_idx, pps):
    """
    Project particles to 2D density field for a specific lens plane.
    
    Parameters
    ----------
    coords : ndarray (N, 3)
        Particle coordinates in Mpc/h
    masses : ndarray (N,)
        Particle masses in Msun/h
    box_size : float
        Box size in Mpc/h
    grid_res : int
        Grid resolution
    proj_dir : int
        Projection direction (0=x, 1=y, 2=z)
    plane_idx : int
        Plane index within snapshot (0 or 1 for pps=2)
    pps : int
        Planes per snapshot
        
    Returns
    -------
    field : ndarray (grid_res, grid_res)
        2D surface density field
    """
    if len(coords) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32)
    
    # Get projection axes
    ax1, ax2 = get_projection_axes(proj_dir)
    proj_axis = proj_dir
    
    # Slice thickness
    slice_thickness = box_size / pps
    z_min = plane_idx * slice_thickness
    z_max = (plane_idx + 1) * slice_thickness
    
    # Select particles in this slice
    z_coords = coords[:, proj_axis]
    mask = (z_coords >= z_min) & (z_coords < z_max)
    
    if mask.sum() == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32)
    
    # Get 2D positions
    pos_2d = np.column_stack([coords[mask, ax1], coords[mask, ax2]]).astype(np.float32)
    pos_2d = np.ascontiguousarray(pos_2d)
    
    # Ensure within bounds
    pos_2d = np.mod(pos_2d, box_size)
    
    # Project to grid using TSC (Triangular Shaped Cloud) to match lux
    field = np.zeros((grid_res, grid_res), dtype=np.float32)
    MASL.MA(pos_2d, field, np.float32(box_size), MAS='TSC',
            W=masses[mask].astype(np.float32), verbose=False)
    
    return field


# ============================================================================
# Binary Output (lux format)
# ============================================================================

def write_density_plane(filename, delta_dz, grid_size):
    """
    Write density plane in lux binary format.
    
    Format:
      int32: grid_size
      float64[grid_size*grid_size]: delta values (row-major)
      int32: grid_size (footer)
    """
    with open(filename, 'wb') as f:
        # Header
        f.write(struct.pack('i', grid_size))
        # Data (ensure row-major, float64)
        f.write(delta_dz.astype(np.float64).tobytes())
        # Footer
        f.write(struct.pack('i', grid_size))


def write_config(output_dir, snap_config, random_state, grid_res, pps):
    """
    Write configuration file for lux.
    
    This contains:
      - Number of planes and snapshots
      - Scale factors at each plane
      - Comoving distances
      - Box sizes
      - Randomization parameters
    """
    Ns = len(snap_config)
    Np = Ns * pps
    
    # Compute comoving distances and scale factors
    a = np.zeros(Np + 1)
    chi = np.zeros(Np + 1)
    chi_out = np.zeros(Np)
    Ll = np.zeros(Ns)  # Longitudinal box size
    Lt = np.zeros(Ns)  # Transverse box size
    
    a[0] = 1.0  # Observer at z=0
    chi[0] = 0.0
    
    L = CONFIG['box_size']
    
    for s, (snap, z, stack) in enumerate(snap_config):
        Ll[s] = L
        Lt[s] = 2 * L if stack else L
    
    # Compute comoving distances at plane boundaries
    cumulative_dist = 0.0
    for p in range(1, Np + 1):
        s = (p - 1) // pps  # Snapshot index
        plane_in_snap = (p - 1) % pps
        
        # Distance to plane center
        chi[p] = cumulative_dist + Ll[s] / pps * (plane_in_snap + 0.5)
        
        if plane_in_snap == pps - 1:
            cumulative_dist += Ll[s]
    
    # Output distances (plane far edges)
    cumulative_dist = 0.0
    for p in range(Np):
        s = p // pps
        plane_in_snap = p % pps
        chi_out[p] = cumulative_dist + Ll[s] / pps * (plane_in_snap + 1)
        if plane_in_snap == pps - 1:
            cumulative_dist += Ll[s]
    
    # Compute scale factors from comoving distances
    for p in range(1, Np + 1):
        # Use CCL to convert comoving distance to scale factor
        z_at_chi = ccl.comoving_radial_distance(COSMO, 1.0) - chi[p]
        # Simple approximation: a ≈ 1 / (1 + z) where chi ∝ z for small z
        # For more accuracy, solve numerically
        a[p] = ccl.scale_factor_of_chi(COSMO, chi[p])
    
    # Get randomization parameters
    proj_dirs = random_state.proj_dirs.astype(np.int32)
    disp = random_state.displacements.flatten()  # (Ns, 3) -> (3*Ns,)
    flip = random_state.flips.flatten().astype(np.uint8)
    
    # Write binary config file
    config_file = os.path.join(output_dir, 'config.dat')
    with open(config_file, 'wb') as f:
        f.write(struct.pack('i', Np))
        f.write(struct.pack('i', Ns))
        f.write(a.astype(np.float64).tobytes())
        f.write(chi.astype(np.float64).tobytes())
        f.write(chi_out.astype(np.float64).tobytes())
        f.write(Ll.astype(np.float64).tobytes())
        f.write(Lt.astype(np.float64).tobytes())
        f.write(proj_dirs.tobytes())
        f.write(disp.astype(np.float64).tobytes())
        f.write(flip.tobytes())
    
    if rank == 0:
        print(f"  Wrote config: {config_file}")
        print(f"    Np={Np}, Ns={Ns}")


# ============================================================================
# BCM Model Setup
# ============================================================================

def setup_bcm_model(model_name):
    """Setup BaryonForge displacement model."""
    import warnings
    warnings.filterwarnings('ignore')
    
    params = BCM_PARAMS[model_name]
    
    if model_name == 'Arico20':
        DMB = bfg.Profiles.Arico20.DarkMatterBaryon(**params)
        DMO = bfg.Profiles.Arico20.DarkMatterOnly(**params)
    elif model_name == 'Schneider19':
        DMB = bfg.Profiles.Schneider19.DarkMatterBaryon(**params)
        DMO = bfg.Profiles.Schneider19.DarkMatterOnly(**params)
    elif model_name == 'Schneider25':
        DMB = bfg.Profiles.Schneider25.DarkMatterBaryon(**params)
        DMO = bfg.Profiles.Schneider25.DarkMatterOnly(**params)
    else:
        raise ValueError(f"Unknown BCM model: {model_name}")
    
    Displacement = bfg.Baryonification3D(DMO, DMB, COSMO, N_int=50_000)
    Displacement.setup_interpolator(
        z_min=0, z_max=3.0, z_linear_sampling=True,
        N_samples_R=10000, Rdelta_sampling=True
    )
    return Displacement


def apply_bcm_displacement(coords, masses, halo_positions, halo_radii, halo_masses,
                            displacement_model, scale_factor, box_size):
    """
    Apply BCM displacement to particles around halos.
    
    Only displaces particles within radius_multiplier × R_200 of halo centers.
    """
    if len(coords) == 0 or len(halo_positions) == 0:
        return coords.copy()
    
    coords_out = coords.copy()
    tree = cKDTree(coords)
    radius_mult = CONFIG['radius_multiplier']
    
    for i, (pos, r200, M200) in enumerate(zip(halo_positions, halo_radii, halo_masses)):
        # Find particles within radius_mult × R_200
        r_max = radius_mult * r200
        idx = tree.query_ball_point(pos, r_max)
        
        if len(idx) == 0:
            continue
        
        # Compute radial distance from halo center
        dx = coords[idx] - pos
        # Periodic boundary
        dx = dx - np.round(dx / box_size) * box_size
        r = np.linalg.norm(dx, axis=1)
        
        # Get displacement (in Mpc/h) for each particle
        r_safe = np.maximum(r, 1e-6)  # Avoid r=0
        dr = displacement_model.displacement(r_safe, M200, a=scale_factor)
        
        # Handle NaN displacements (can happen at very small radii)
        nan_mask = np.isnan(dr)
        if nan_mask.any():
            dr[nan_mask] = 0.0
        
        # Apply radial displacement
        unit_vec = dx / r_safe[:, np.newaxis]
        coords_out[idx] += dr[:, np.newaxis] * unit_vec
        
        # Apply periodic boundary
        coords_out[idx] = coords_out[idx] % box_size
    
    return coords_out


# ============================================================================
# Main Processing
# ============================================================================

def process_snapshot_multi_seed(args, snap_idx, snap_info, seeds, bcm_models=None):
    """
    Process a single snapshot to generate lens planes for multiple seeds.
    
    Key optimization: Load particle data ONCE, then loop over seeds for
    randomization and projection. This is much faster than loading data
    separately for each seed.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    snap_idx : int
        Index in SNAPSHOT_CONFIG
    snap_info : tuple
        (snapshot_number, redshift, stack_flag)
    seeds : list of int
        Random seeds to generate (e.g., [2020, 2021, ..., 2039])
    bcm_models : dict, optional
        Pre-loaded BCM displacement models
    """
    snapNum, redshift, stack = snap_info
    sim_config = SIM_PATHS[args.sim_res]
    dmo_basePath = sim_config['dmo']
    hydro_basePath = sim_config['hydro']
    
    box_size = CONFIG['box_size']
    grid_res = args.grid_res
    pps = CONFIG['planes_per_snapshot']
    scale_factor = 1.0 / (1.0 + redshift)
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Snapshot {snapNum} (z={redshift:.2f}, a={scale_factor:.3f})")
        print(f"  Stack: {stack}")
        print(f"  Seeds: {seeds[0]} to {seeds[-1]} ({len(seeds)} realizations)")
        print(f"{'='*70}")
    
    # ========================================================================
    # Load particles (distributed) - THIS IS THE BOTTLENECK, DONE ONCE
    # ========================================================================
    if rank == 0:
        print("\n[1/4] Loading particles (done once for all seeds)...")
        t_start = time.time()
    
    dmo_dir = f"{dmo_basePath}/snapdir_{snapNum:03d}/"
    hydro_dir = f"{hydro_basePath}/snapdir_{snapNum:03d}/"
    
    dmo_files = sorted(glob.glob(f"{dmo_dir}/snap_{snapNum:03d}.*.hdf5"))
    hydro_files = sorted(glob.glob(f"{hydro_dir}/snap_{snapNum:03d}.*.hdf5"))
    
    my_dmo_files = [f for i, f in enumerate(dmo_files) if i % size == rank]
    my_hydro_files = [f for i, f in enumerate(hydro_files) if i % size == rank]
    
    # Load DMO particles (original coordinates - will transform per seed)
    dmo_coords_orig, dmo_masses = load_dmo_particles(
        dmo_basePath, snapNum, my_dmo_files,
        sim_config['dmo_mass'], CONFIG['mass_unit']
    )
    
    # Load Hydro particles (only if needed)
    if args.model in ['hydro', 'replace', 'all']:
        hydro_coords_orig, hydro_masses = load_hydro_particles(
            hydro_basePath, snapNum, my_hydro_files,
            sim_config['hydro_dm_mass'], CONFIG['mass_unit']
        )
    else:
        hydro_coords_orig = np.zeros((0, 3), dtype=np.float32)
        hydro_masses = np.zeros(0, dtype=np.float32)
    
    if rank == 0:
        t_load = time.time() - t_start
        print(f"  Rank 0: {len(dmo_coords_orig):,} DMO, {len(hydro_coords_orig):,} hydro particles")
        print(f"  Data loading time: {t_load:.1f}s")
    
    # ========================================================================
    # Load halo catalog (for replace and BCM) - ALSO DONE ONCE
    # ========================================================================
    halo_info = None
    if args.model in ['replace', 'bcm', 'all'] or (bcm_models is not None):
        if rank == 0:
            print("\n[2/4] Loading halo catalog...")
        
        halo_dmo = groupcat.loadHalos(
            dmo_basePath, snapNum,
            fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
        )
        
        # Apply mass cut
        all_masses = halo_dmo['Group_M_Crit200'] * CONFIG['mass_unit']
        log_masses = np.log10(all_masses)
        
        mass_min = args.mass_min
        mass_max = getattr(args, 'mass_max', None)
        
        if mass_max is not None:
            mask = (log_masses >= mass_min) & (log_masses < mass_max)
        else:
            mask = log_masses >= mass_min
        
        halo_indices = np.where(mask)[0]
        
        # Store original positions (will transform per seed)
        halo_info = {
            'positions_orig': halo_dmo['GroupPos'][halo_indices] / 1e3,
            'radii': halo_dmo['Group_R_Crit200'][halo_indices] / 1e3,
            'masses': halo_dmo['Group_M_Crit200'][halo_indices] * CONFIG['mass_unit'],
        }
        
        if rank == 0:
            print(f"  Halos above log10(M) >= {mass_min}: {len(halo_indices)}")
    
    # ========================================================================
    # Loop over seeds - Apply transformations and generate lens planes
    # ========================================================================
    if rank == 0:
        print(f"\n[3/4] Generating lens planes for {len(seeds)} seeds...")
    
    # Compute mean surface density for normalization (constant across seeds)
    slice_thickness = box_size / pps
    sigma_mean = Omega_m * rho_c0 * CONFIG['mass_unit'] * slice_thickness
    
    # Determine which models to process
    models_to_process = []
    if args.model == 'all':
        models_to_process = ['dmo', 'hydro', 'replace']
        if bcm_models:
            models_to_process.extend([f'bcm_{m.lower()}' for m in bcm_models.keys()])
    elif args.model == 'bcm':
        models_to_process = [f'bcm_{m.lower()}' for m in bcm_models.keys()]
    else:
        models_to_process = [args.model]
    
    for seed in seeds:
        if rank == 0:
            print(f"\n  ---- Seed {seed} ----")
            t_seed_start = time.time()
        
        # Initialize randomization for this seed
        random_state = RandomizationState(seed=seed)
        rand_params = random_state.get_params(snap_idx)
        proj_dir = rand_params['proj_dir']
        
        if rank == 0:
            print(f"    Projection axis: {proj_dir} ({'xyz'[proj_dir]})")
        
        # Apply randomization to coordinates (translation + flip)
        dmo_coords = apply_randomization(dmo_coords_orig, rand_params, box_size)
        if len(hydro_coords_orig) > 0:
            hydro_coords = apply_randomization(hydro_coords_orig, rand_params, box_size)
        else:
            hydro_coords = hydro_coords_orig
        
        # Transform halo positions if needed
        if halo_info is not None:
            halo_positions = apply_randomization(
                halo_info['positions_orig'], rand_params, box_size
            )
            halo_radii = halo_info['radii']
            halo_masses = halo_info['masses']
        else:
            halo_positions = np.array([])
            halo_radii = np.array([])
            halo_masses = np.array([])
        
        # Process each plane
        for plane_idx in range(pps):
            plane_num = snap_idx * pps + plane_idx + 1  # 1-indexed for lux
            
            for model_name in models_to_process:
                # Build output directory name with seed
                if model_name == 'replace':
                    dir_name = f"replace_Mgt{args.mass_min:.1f}"
                elif model_name.startswith('bcm_'):
                    dir_name = f"{model_name}_Mgt{args.mass_min:.1f}"
                else:
                    dir_name = model_name
                
                # Output path includes seed subdirectory
                output_dir = os.path.join(
                    args.output_dir, f'L205n{args.sim_res}TNG', 
                    f'seed{seed}', dir_name
                )
                if rank == 0:
                    os.makedirs(output_dir, exist_ok=True)
                comm.Barrier()
                
                output_file = os.path.join(output_dir, f'density{plane_num:02d}.dat')
                
                # Skip if exists
                if args.skip_existing and os.path.exists(output_file):
                    continue
                
                # Choose coordinates based on model
                if model_name == 'dmo':
                    coords = dmo_coords
                    masses = dmo_masses
                elif model_name == 'hydro':
                    coords = hydro_coords
                    masses = hydro_masses
                elif model_name == 'replace':
                    # Replace: DMO outside halos + Hydro inside halos
                    coords, masses = build_replace_field(
                        dmo_coords, dmo_masses, hydro_coords, hydro_masses,
                        halo_positions, halo_radii, box_size
                    )
                elif model_name.startswith('bcm_'):
                    bcm_key = model_name.replace('bcm_', '').title()
                    if bcm_key.endswith('19'):
                        bcm_key = 'Schneider19'
                    elif bcm_key.endswith('20'):
                        bcm_key = 'Arico20'
                    elif bcm_key.endswith('25'):
                        bcm_key = 'Schneider25'
                    
                    disp_model = bcm_models.get(bcm_key)
                    if disp_model is None:
                        continue
                    
                    coords = apply_bcm_displacement(
                        dmo_coords, dmo_masses, halo_positions, halo_radii, halo_masses,
                        disp_model, scale_factor, box_size
                    )
                    masses = dmo_masses
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                # Project to 2D
                local_field = project_to_2d(
                    coords, masses, box_size, grid_res, proj_dir, plane_idx, pps
                )
                
                # Ensure contiguous array for MPI
                local_field = np.ascontiguousarray(local_field, dtype=np.float64)
                
                # Reduce across all ranks
                if rank == 0:
                    global_field = np.zeros((grid_res, grid_res), dtype=np.float64)
                else:
                    global_field = np.zeros((grid_res, grid_res), dtype=np.float64)
                
                comm.Reduce([local_field, MPI.DOUBLE], [global_field, MPI.DOUBLE], 
                           op=MPI.SUM, root=0)
                
                # Convert to overdensity × dz and write
                if rank == 0:
                    # δ × dz = (Σ / Σ_mean - 1) × dz
                    delta_dz = (global_field / sigma_mean - 1.0) * slice_thickness
                    
                    # Handle NaN values (BCM can produce NaNs at halo centers)
                    nan_count = np.isnan(delta_dz).sum()
                    if nan_count > 0:
                        delta_dz = np.nan_to_num(delta_dz, nan=0.0)
                    
                    write_density_plane(output_file, delta_dz, grid_res)
        
        if rank == 0:
            t_seed = time.time() - t_seed_start
            print(f"    Seed {seed} done in {t_seed:.1f}s")
        
        # Write config file for this seed (at last snapshot)
        if rank == 0 and snap_idx == len(SNAPSHOT_CONFIG) - 1:
            for model_name in models_to_process:
                if model_name == 'replace':
                    dir_name = f"replace_Mgt{args.mass_min:.1f}"
                elif model_name.startswith('bcm_'):
                    dir_name = f"{model_name}_Mgt{args.mass_min:.1f}"
                else:
                    dir_name = model_name
                output_dir = os.path.join(
                    args.output_dir, f'L205n{args.sim_res}TNG',
                    f'seed{seed}', dir_name
                )
                write_config(output_dir, SNAPSHOT_CONFIG, random_state, grid_res, pps)
    
    if rank == 0:
        print(f"\n  All {len(seeds)} seeds complete for snapshot {snapNum}!")


# Backwards compatibility wrapper
def process_snapshot(args, snap_idx, snap_info, random_state, bcm_models=None):
    """Wrapper for backwards compatibility - uses single seed."""
    process_snapshot_multi_seed(args, snap_idx, snap_info, [args.seed], bcm_models)


def build_replace_field(dmo_coords, dmo_masses, hydro_coords, hydro_masses,
                        halo_positions, halo_radii, box_size):
    """
    Build replace field: DMO outside halos + Hydro inside halos.
    """
    if len(halo_positions) == 0:
        return dmo_coords, dmo_masses
    
    radius_mult = CONFIG['radius_multiplier']
    
    # Build KD-trees
    dmo_tree = cKDTree(dmo_coords) if len(dmo_coords) > 0 else None
    hydro_tree = cKDTree(hydro_coords) if len(hydro_coords) > 0 else None
    
    # DMO: keep everything EXCEPT within halo regions
    dmo_keep_mask = np.ones(len(dmo_coords), dtype=bool)
    if dmo_tree is not None:
        for pos, r200 in zip(halo_positions, halo_radii):
            idx = dmo_tree.query_ball_point(pos, radius_mult * r200)
            dmo_keep_mask[idx] = False
    
    # Hydro: keep ONLY within halo regions
    hydro_keep_mask = np.zeros(len(hydro_coords), dtype=bool)
    if hydro_tree is not None:
        for pos, r200 in zip(halo_positions, halo_radii):
            idx = hydro_tree.query_ball_point(pos, radius_mult * r200)
            hydro_keep_mask[idx] = True
    
    # Combine
    coords = np.concatenate([
        dmo_coords[dmo_keep_mask],
        hydro_coords[hydro_keep_mask]
    ])
    masses = np.concatenate([
        dmo_masses[dmo_keep_mask],
        hydro_masses[hydro_keep_mask]
    ])
    
    return coords, masses


def main():
    parser = argparse.ArgumentParser(description='Generate lens planes for ray-tracing')
    parser.add_argument('--sim-res', type=int, default=625, choices=[625, 1250, 2500],
                       help='Simulation resolution (625, 1250, 2500)')
    parser.add_argument('--model', type=str, default='all',
                       choices=['dmo', 'hydro', 'replace', 'bcm', 'all'],
                       help='Which model(s) to generate')
    parser.add_argument('--snap', type=str, default='all',
                       help='Snapshot(s) to process: "all", single number, or comma-separated')
    parser.add_argument('--mass-min', type=float, default=12.5,
                       help='Minimum halo mass (log10 Msun/h) for replace/BCM')
    parser.add_argument('--mass-max', type=float, default=None,
                       help='Maximum halo mass (log10 Msun/h)')
    parser.add_argument('--grid-res', type=int, default=CONFIG['grid_res'],
                       help='Grid resolution (default: 4096, use 1024 for testing)')
    parser.add_argument('--output-dir', type=str, default=CONFIG['output_base'],
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=2020,
                       help='Random seed for consistent randomization (or starting seed if --num-seeds > 1)')
    parser.add_argument('--num-seeds', type=int, default=1,
                       help='Number of random seed realizations to generate (seeds will be: seed, seed+1, ...)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip if output file exists')
    parser.add_argument('--bcm-models', type=str, nargs='+',
                       default=['Arico20', 'Schneider19', 'Schneider25'],
                       help='BCM models to use')
    
    args = parser.parse_args()
    
    if rank == 0:
        print("=" * 70)
        print("LENS PLANE GENERATION FOR RAY-TRACING")
        print("=" * 70)
        print(f"Simulation: L205n{args.sim_res}TNG")
        print(f"Model: {args.model}")
        print(f"Mass cut: log10(M) >= {args.mass_min}")
        print(f"Random seeds: {args.seed} to {args.seed + args.num_seeds - 1} ({args.num_seeds} realizations)")
        print(f"Output: {args.output_dir}")
        print(f"MPI ranks: {size}")
        print("=" * 70)
    
    # Parse snapshot selection
    if args.snap == 'all':
        snap_indices = list(range(len(SNAPSHOT_CONFIG)))
    else:
        snap_nums = [int(s) for s in args.snap.split(',')]
        snap_indices = []
        for sn in snap_nums:
            found = False
            for i, (snap, _, _) in enumerate(SNAPSHOT_CONFIG):
                if snap == sn:
                    snap_indices.append(i)
                    found = True
                    break
            if not found and rank == 0:
                print(f"WARNING: Snapshot {sn} not in SNAPSHOT_CONFIG!")
                print(f"  Valid snapshots: {[s[0] for s in SNAPSHOT_CONFIG]}")
    
    if len(snap_indices) == 0:
        if rank == 0:
            print("ERROR: No valid snapshots to process!")
            print(f"  Requested: {args.snap}")
            print(f"  Valid: {[s[0] for s in SNAPSHOT_CONFIG]}")
        return
    
    # Load BCM models if needed
    bcm_models = None
    if args.model in ['bcm', 'all']:
        if rank == 0:
            print("\nLoading BCM models...")
        bcm_models = {}
        for bcm_name in args.bcm_models:
            if rank == 0:
                print(f"  Setting up {bcm_name}...")
            bcm_models[bcm_name] = setup_bcm_model(bcm_name)
        if rank == 0:
            print("  BCM models loaded.")
    
    # Generate list of seeds
    seeds = list(range(args.seed, args.seed + args.num_seeds))
    
    # Process snapshots - data loaded once, then loop over seeds internally
    for snap_idx in snap_indices:
        snap_info = SNAPSHOT_CONFIG[snap_idx]
        process_snapshot_multi_seed(args, snap_idx, snap_info, seeds, bcm_models)
    
    comm.Barrier()
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("LENS PLANE GENERATION COMPLETE")
        print("=" * 70)


if __name__ == '__main__':
    main()
