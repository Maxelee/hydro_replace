#!/usr/bin/env python
"""
Generate 2D projected density fields and halo profiles for all modes.

Outputs per snapshot:
  - 2D projected maps (z-axis): DMO, Hydro, Replace, BCM (3 models)
  - Density profiles for all matched halos

Usage:
    mpirun -np 64 python generate_all.py --snap 99 --sim-res 2500
    mpirun -np 32 python generate_all.py --snap 99 --sim-res 625 --mass-min 12.5

Default parameters:
  - radius_multiplier: 5 (replace within 5 * R_200)
  - mass-min: 12.5 (log10 M_sun/h)
"""

import numpy as np
import h5py
import glob
import argparse
import os
import time
from mpi4py import MPI
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic
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

CONFIG = {
    'box_size': 205.0,  # Mpc/h
    'mass_unit': 1e10,  # Convert to Msun/h
    'output_dir': '/mnt/home/mlee1/ceph/hydro_replace_fields',
    'radius_multiplier': 5,  # Replace within 5 * R_200
    'profile_r_min': 0.01,  # Minimum radius in units of R_200
    'profile_r_max': 5.0,   # Maximum radius in units of R_200
    'profile_n_bins': 50,
}

# TNG300 cosmology
COSMO = ccl.Cosmology(
    Omega_c=0.3089 - 0.0486, Omega_b=0.0486, h=0.6774,
    sigma8=0.8158, n_s=0.9649, matter_power_spectrum='linear'
)
h = COSMO.cosmo.params.h

# BCM Parameters - from BaryonForge test defaults
BCM_PARAMS = {
    'Arico20': dict(
        M_c=0.23e14/h, eta=0.14, mu=0.31, beta=4.09,
        M_inn=3.3e13/h, theta_inn=0.1, theta_out=5,
        M1_0=0.22e11/h, epsilon_h=0.015, alpha_g=2,
        epsilon_hydro=np.sqrt(5), theta_rg=0.3, sigma_rg=0.1,
        a=0.3, n=2, p=0.3, q=0.707,
        alpha_fsat=1, M1_fsat=1, delta_fsat=1, gamma_fsat=1, eps_fsat=1,
        M_r=1e30, beta_r=2,
    ),
    'Schneider19': dict(
        # Gas parameters
        theta_ej=4, theta_co=0.1, M_c=1e14/h, mu_beta=0.4,
        gamma=2, delta=7,
        # Stellar parameters (tau, eta for mass fraction)
        eta=0.3, eta_delta=0.3, tau=-1.5, tau_delta=0,
        A=0.09/2, M1=2.5e11/h, epsilon_h=0.015,
        # DM profile and 2-halo params
        a=0.3, n=2, epsilon=4, p=0.3, q=0.707,
    ),
    'Schneider25': dict(
        # Gas parameters
        M_c=1e15, mu=0.8,
        q0=0.075, q1=0.25, q2=0.7, nu_q0=0, nu_q1=1, nu_q2=0, nstep=3/2,
        theta_c=0.3, nu_theta_c=1/2, c_iga=0.1, nu_c_iga=3/2, r_min_iga=1e-3,
        alpha=1, gamma=3/2, delta=7,
        # Stellar parameters
        tau=-1.376, tau_delta=0, Mstar=3e11, Nstar=0.03,
        eta=0.1, eta_delta=0.22, epsilon_cga=0.03,
        # Non-thermal
        alpha_nt=0.1, nu_nt=0.5, gamma_nt=0.8, mean_molecular_weight=0.6125,
        # DM profile and 2-halo params
        epsilon0=4, epsilon1=0.5, alpha_excl=0.4, p=0.3, q=0.707,
    ),
}


def build_cosmodict(cosmo):
    """Extract cosmological parameters from pyccl Cosmology."""
    return {
        'Omega_m': cosmo.cosmo.params.Omega_m,
        'Omega_b': cosmo.cosmo.params.Omega_b,
        'sigma8': cosmo.cosmo.params.sigma8,
        'h': cosmo.cosmo.params.h,
        'n_s': cosmo.cosmo.params.n_s,
        'w0': cosmo.cosmo.params.w0,
        'wa': cosmo.cosmo.params.wa,
    }


def setup_bcm_model(model_name):
    """Setup BaryonForge displacement model."""
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
        z_min=0, z_linear_sampling=True,
        N_samples_R=10000, Rdelta_sampling=True
    )
    return Displacement


def apply_periodic_boundary(dx, box_size):
    """Apply minimum image convention."""
    return dx - np.round(dx / box_size) * box_size


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
# 2D Projection (CIC)
# ============================================================================

def project_to_2d(coords, masses, box_size, grid_res):
    """Project particles to 2D map along z-axis using CIC."""
    if len(coords) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32)
    
    # Use only x, y coordinates
    pos_2d = coords[:, :2].astype(np.float32)
    
    # Create 2D field
    field = np.zeros((grid_res, grid_res), dtype=np.float32)
    
    # CIC assignment in 2D
    cell_size = box_size / grid_res
    
    for i in range(len(pos_2d)):
        x, y = pos_2d[i]
        m = masses[i]
        
        # Cell indices
        ix = x / cell_size
        iy = y / cell_size
        
        ix0 = int(np.floor(ix - 0.5)) % grid_res
        iy0 = int(np.floor(iy - 0.5)) % grid_res
        ix1 = (ix0 + 1) % grid_res
        iy1 = (iy0 + 1) % grid_res
        
        # Weights
        dx = ix - 0.5 - np.floor(ix - 0.5)
        dy = iy - 0.5 - np.floor(iy - 0.5)
        
        # Distribute mass
        field[ix0, iy0] += m * (1 - dx) * (1 - dy)
        field[ix1, iy0] += m * dx * (1 - dy)
        field[ix0, iy1] += m * (1 - dx) * dy
        field[ix1, iy1] += m * dx * dy
    
    return field


def project_to_2d_fast(coords, masses, box_size, grid_res, axis=2):
    """Project particles to 2D map by summing along one axis.
    
    This is the correct approach used in the original Hydro_replacement scripts:
    - Drop the projection axis coordinate
    - Use only the remaining 2 coordinates for 2D pixelization
    """
    if len(coords) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32)
    
    # Select the two axes to keep (drop the projection axis)
    proj_axes = [0, 1, 2]
    proj_axes.pop(axis)  # Remove projection axis
    
    # Get 2D positions (only x,y if axis=2)
    pos_2d = coords[:, proj_axes].astype(np.float32).copy()
    
    # Ensure coordinates are within bounds
    pos_2d = np.mod(pos_2d, box_size)
    
    # Create 2D field and pixelize
    field = np.zeros((grid_res, grid_res), dtype=np.float32)
    MASL.MA(pos_2d, field, np.float32(box_size), MAS='CIC',
            W=masses.astype(np.float32), verbose=False)
    
    return field


# ============================================================================
# Density Profiles
# ============================================================================

def compute_density_profile(coords, masses, center, radius, box_size, config):
    """Compute spherically-averaged density profile."""
    r_min = config['profile_r_min']
    r_max = config['profile_r_max']
    n_bins = config['profile_n_bins']
    
    bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    
    dx = coords - center
    dx = apply_periodic_boundary(dx, box_size)
    r = np.linalg.norm(dx, axis=1) / radius  # Normalize by R_200
    
    # Bin masses
    prof, _, _ = binned_statistic(r, masses, statistic='sum', bins=bins)
    
    # Shell volumes (in physical units, then normalized)
    volumes = 4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3) * radius**3
    density = prof / volumes
    
    # Bin centers (geometric mean)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    
    return bin_centers, density


# ============================================================================
# Main Processing
# ============================================================================

def process_snapshot(args):
    """Process a single snapshot: generate 2D maps and profiles."""
    
    sim_res = args.sim_res
    snapNum = args.snap
    grid_res = args.grid_res
    log_mass_min = args.mass_min
    log_mass_max = getattr(args, 'mass_max', None)
    skip_existing = getattr(args, 'skip_existing', False)
    only_bcm = getattr(args, 'only_bcm', False)
    bcm_models = getattr(args, 'bcm_models', ['Arico20', 'Schneider19', 'Schneider25'])
    
    sim_config = SIM_PATHS[sim_res]
    dmo_basePath = sim_config['dmo']
    hydro_basePath = sim_config['hydro']
    dmo_mass = sim_config['dmo_mass']
    hydro_dm_mass = sim_config['hydro_dm_mass']
    
    output_dir = args.output_dir or CONFIG['output_dir']
    
    # Build mass label for output directory
    if log_mass_max is not None:
        mass_label = f"M{log_mass_min:.1f}-{log_mass_max:.1f}"
    else:
        mass_label = f"Mgt{log_mass_min:.1f}"
    
    snap_dir = f"{output_dir}/L205n{sim_res}TNG/snap{snapNum:03d}"
    
    if rank == 0:
        os.makedirs(f"{snap_dir}/projected", exist_ok=True)
        print("=" * 70)
        print(f"Processing snapshot {snapNum}")
        print(f"Simulation: L205n{sim_res}TNG")
        print(f"Grid resolution: {grid_res}")
        if log_mass_max is not None:
            print(f"Mass range: {log_mass_min} < log10(M) < {log_mass_max}")
        else:
            print(f"Mass cut: log10(M) > {log_mass_min}")
        print(f"Skip existing: {skip_existing}")
        print(f"Only BCM: {only_bcm}")
        print(f"BCM models: {bcm_models}")
        print(f"Output: {snap_dir}")
        print("=" * 70)
    
    comm.Barrier()
    
    # ========================================================================
    # Load halo catalogs and matches
    # ========================================================================
    if rank == 0:
        print("\n[1/6] Loading halo catalogs...")
    
    halo_dmo = groupcat.loadHalos(
        dmo_basePath, snapNum,
        fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
    )
    halo_hydro = groupcat.loadHalos(
        hydro_basePath, snapNum,
        fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
    )
    
    # Check for existing matches (still load for profile comparisons, but not for replacement)
    matches_file = f"{output_dir}/L205n{sim_res}TNG/matches/matches_snap{snapNum:03d}.npz"
    
    if os.path.exists(matches_file):
        if rank == 0:
            print(f"  Loading matches from {matches_file}")
        matches = np.load(matches_file)
        matched_dmo_indices = matches['dmo_indices']
        matched_hydro_indices = matches['hydro_indices']
    else:
        if rank == 0:
            print(f"  WARNING: No matches file found at {matches_file}")
        matched_dmo_indices = np.array([], dtype=int)
        matched_hydro_indices = np.array([], dtype=int)
    
    # For REPLACEMENT: Use ALL DMO halos above mass cut (like original script)
    # This is the correct approach - we replace at DMO halo positions regardless of matching
    all_dmo_masses = halo_dmo['Group_M_Crit200'] * CONFIG['mass_unit']
    all_dmo_log_masses = np.log10(all_dmo_masses)
    
    # Apply mass range filter
    if log_mass_max is not None:
        replace_mask = (all_dmo_log_masses >= log_mass_min) & (all_dmo_log_masses < log_mass_max)
    else:
        replace_mask = all_dmo_log_masses >= log_mass_min
    replace_halo_indices = np.where(replace_mask)[0]
    
    if rank == 0:
        print(f"  Total DMO halos: {halo_dmo['count']}")
        if log_mass_max is not None:
            print(f"  DMO halos in range {log_mass_min} <= log10(M) < {log_mass_max}: {len(replace_halo_indices)}")
        else:
            print(f"  DMO halos above log10(M) >= {log_mass_min}: {len(replace_halo_indices)}")
        print(f"  Total matched halos (all masses): {len(matched_dmo_indices)}")
    
    # Get halo positions/radii/masses for replacement (ALL DMO halos above mass cut)
    halo_positions = halo_dmo['GroupPos'][replace_halo_indices] / 1e3
    halo_radii = halo_dmo['Group_R_Crit200'][replace_halo_indices] / 1e3
    halo_masses = halo_dmo['Group_M_Crit200'][replace_halo_indices] * CONFIG['mass_unit']
    
    # ========================================================================
    # Load particles (distributed)
    # ========================================================================
    if rank == 0:
        print("\n[2/6] Loading particles...")
    
    # Get file lists
    dmo_dir = f"{dmo_basePath}/snapdir_{snapNum:03d}/"
    hydro_dir = f"{hydro_basePath}/snapdir_{snapNum:03d}/"
    
    dmo_files = sorted(glob.glob(f"{dmo_dir}/snap_{snapNum:03d}.*.hdf5"))
    hydro_files = sorted(glob.glob(f"{hydro_dir}/snap_{snapNum:03d}.*.hdf5"))
    
    my_dmo_files = [f for i, f in enumerate(dmo_files) if i % size == rank]
    my_hydro_files = [f for i, f in enumerate(hydro_files) if i % size == rank]
    
    # Load DMO particles
    dmo_coords, dmo_masses = load_dmo_particles(
        dmo_basePath, snapNum, my_dmo_files, dmo_mass, CONFIG['mass_unit']
    )
    
    # Load Hydro particles
    hydro_coords, hydro_masses = load_hydro_particles(
        hydro_basePath, snapNum, my_hydro_files, hydro_dm_mass, CONFIG['mass_unit']
    )
    
    if rank == 0:
        print(f"  Rank 0: {len(dmo_coords):,} DMO, {len(hydro_coords):,} hydro particles")
    
    # Build KD-trees
    dmo_tree = cKDTree(dmo_coords) if len(dmo_coords) > 0 else None
    hydro_tree = cKDTree(hydro_coords) if len(hydro_coords) > 0 else None
    
    # For BCM-only mode or skip-existing, we need global_dmo_map for fallback
    global_dmo_map = None
    global_hydro_map = None
    global_replace_map = None
    
    # ========================================================================
    # Generate DMO 2D map
    # ========================================================================
    dmo_file = f"{snap_dir}/projected/dmo.npz"
    skip_dmo = only_bcm or (skip_existing and os.path.exists(dmo_file))
    
    if skip_dmo:
        if rank == 0:
            print(f"\n[3/6] Skipping DMO map (exists or only_bcm mode)")
            if os.path.exists(dmo_file):
                global_dmo_map = np.load(dmo_file)['field']
    else:
        if rank == 0:
            print("\n[3/6] Generating DMO 2D map...")
        
        local_dmo_map = project_to_2d_fast(dmo_coords, dmo_masses, CONFIG['box_size'], grid_res)
        
        if rank == 0:
            global_dmo_map = np.zeros_like(local_dmo_map)
        else:
            global_dmo_map = None
        
        comm.Reduce(local_dmo_map, global_dmo_map, op=MPI.SUM, root=0)
        
        if rank == 0:
            np.savez_compressed(dmo_file,
                               field=global_dmo_map, box_size=CONFIG['box_size'],
                               grid_resolution=grid_res, snapshot=snapNum)
            print(f"  Saved: {dmo_file}")
    
    # ========================================================================
    # Generate Hydro 2D map
    # ========================================================================
    hydro_file = f"{snap_dir}/projected/hydro.npz"
    skip_hydro = only_bcm or (skip_existing and os.path.exists(hydro_file))
    
    if skip_hydro:
        if rank == 0:
            print(f"\n[4/6] Skipping Hydro map (exists or only_bcm mode)")
            if os.path.exists(hydro_file):
                global_hydro_map = np.load(hydro_file)['field']
    else:
        if rank == 0:
            print("\n[4/6] Generating Hydro 2D map...")
        
        local_hydro_map = project_to_2d_fast(hydro_coords, hydro_masses, CONFIG['box_size'], grid_res)
        
        if rank == 0:
            global_hydro_map = np.zeros_like(local_hydro_map)
        else:
            global_hydro_map = None
        
        comm.Reduce(local_hydro_map, global_hydro_map, op=MPI.SUM, root=0)
        
        if rank == 0:
            np.savez_compressed(hydro_file,
                               field=global_hydro_map, box_size=CONFIG['box_size'],
                               grid_resolution=grid_res, snapshot=snapNum)
            print(f"  Saved: {hydro_file}")
    
    # ========================================================================
    # Generate Replace 2D map
    # ========================================================================
    replace_file = f"{snap_dir}/projected/replace.npz"
    skip_replace = only_bcm or (skip_existing and os.path.exists(replace_file))
    
    if skip_replace:
        if rank == 0:
            print(f"\n[5/6] Skipping Replace map (exists or only_bcm mode)")
    else:
        if rank == 0:
            print("\n[5/6] Generating Replace 2D map...")
        
        if len(halo_positions) > 0:
            radius_mult = CONFIG['radius_multiplier']
            
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
            
            # Project both components
            local_replace_dmo = project_to_2d_fast(
                dmo_coords[dmo_keep_mask], dmo_masses[dmo_keep_mask],
                CONFIG['box_size'], grid_res
            )
            local_replace_hydro = project_to_2d_fast(
                hydro_coords[hydro_keep_mask], hydro_masses[hydro_keep_mask],
                CONFIG['box_size'], grid_res
            )
            local_replace_map = local_replace_dmo + local_replace_hydro
        else:
            local_replace_map = project_to_2d_fast(dmo_coords, dmo_masses, CONFIG['box_size'], grid_res)
        
        if rank == 0:
            global_replace_map = np.zeros((grid_res, grid_res), dtype=np.float32)
        else:
            global_replace_map = None
        
        comm.Reduce(local_replace_map, global_replace_map, op=MPI.SUM, root=0)
        
        if rank == 0:
            np.savez_compressed(replace_file,
                               field=global_replace_map, box_size=CONFIG['box_size'],
                               grid_resolution=grid_res, snapshot=snapNum,
                               log_mass_min=log_mass_min, 
                               log_mass_max=log_mass_max if log_mass_max else 'None',
                               radius_multiplier=CONFIG['radius_multiplier'])
            print(f"  Saved: {replace_file}")
    
    # ========================================================================
    # Generate BCM maps and profiles
    # ========================================================================
    if rank == 0:
        print("\n[6/6] Generating BCM maps...")
    
    cdict = build_cosmodict(COSMO)
    
    # Each rank processes its own particles locally, then we reduce the maps
    # This avoids the memory bottleneck of gathering all particles to rank 0
    
    # Process each BCM model SEQUENTIALLY to manage memory
    # Free memory between models using gc.collect()
    import gc
    
    for bcm_name in bcm_models:
        bcm_file = f"{snap_dir}/projected/bcm_{bcm_name.lower()}.npz"
        
        # Skip if already exists
        if skip_existing and os.path.exists(bcm_file):
            if rank == 0:
                print(f"\n  Skipping BCM {bcm_name} (exists)")
            continue
        
        if rank == 0:
            print(f"\n  Processing BCM: {bcm_name}")
            t_bcm_start = time.time()
        
        try:
            # Setup BCM model on all ranks
            bcm_model = setup_bcm_model(bcm_name)
            
            # Each rank creates a snapshot with its LOCAL particles
            if len(dmo_coords) > 0:
                # Wrap coordinates to [0, box_size) to avoid KDTree periodic boundary errors
                box = CONFIG['box_size']
                wrapped_coords = dmo_coords % box
                
                Snap = bfg.ParticleSnapshot(
                    x=wrapped_coords[:, 0].astype(np.float64),
                    y=wrapped_coords[:, 1].astype(np.float64),
                    z=wrapped_coords[:, 2].astype(np.float64),
                    L=box,
                    redshift=0,
                    cosmo=cdict,
                    M=dmo_masses[0] if len(dmo_masses) > 0 else dmo_mass * CONFIG['mass_unit']
                )
                
                # Use ALL halos (BaryonForge handles periodic boundaries)
                HCat = bfg.HaloNDCatalog(
                    x=halo_positions[:, 0],
                    y=halo_positions[:, 1],
                    z=halo_positions[:, 2],
                    M=halo_masses,
                    redshift=0,
                    cosmo=cdict
                )
                
                # Create runner and process
                # epsilon_max=5 ensures we capture the full BCM effect out to ~5 R_200
                # (Schneider models have extended profiles)
                Runner = bfg.Runners.BaryonifySnapshot(
                    HCat, Snap, epsilon_max=5.0, model=bcm_model,
                    KDTree_kwargs={'leafsize': 1000, 'balanced_tree': False},
                    verbose=(rank == 0)
                )
                
                Baryonified = Runner.process()
                
                bcm_coords_local = np.array([
                    Baryonified['x'][:],
                    Baryonified['y'][:],
                    Baryonified['z'][:]
                ]).T.astype(np.float32)
                
                # Project local BCM particles to 2D
                local_bcm_map = project_to_2d_fast(bcm_coords_local, dmo_masses,
                                                   CONFIG['box_size'], grid_res)
                
                # Free BCM-specific memory immediately
                del bcm_coords_local, Baryonified, Runner, Snap, HCat, wrapped_coords
            else:
                local_bcm_map = np.zeros((grid_res, grid_res), dtype=np.float32)
            
            # Reduce all local maps to rank 0
            if rank == 0:
                global_bcm_map = np.zeros((grid_res, grid_res), dtype=np.float32)
            else:
                global_bcm_map = None
            
            comm.Reduce(local_bcm_map, global_bcm_map, op=MPI.SUM, root=0)
            
            if rank == 0:
                np.savez_compressed(f"{snap_dir}/projected/bcm_{bcm_name.lower()}.npz",
                                   field=global_bcm_map, box_size=CONFIG['box_size'],
                                   grid_resolution=grid_res, snapshot=snapNum,
                                   bcm_model=bcm_name)
                print(f"    Saved: bcm_{bcm_name.lower()}.npz ({time.time() - t_bcm_start:.1f}s)")
            
            # Free memory and force garbage collection between BCM models
            del bcm_model, local_bcm_map
            if global_bcm_map is not None:
                del global_bcm_map
            gc.collect()
            comm.Barrier()  # Ensure all ranks sync before next model
                
        except Exception as e:
            if rank == 0:
                print(f"    ERROR in {bcm_name}: {e}")
                import traceback
                traceback.print_exc()
                # Save DMO map as fallback
                np.savez_compressed(f"{snap_dir}/projected/bcm_{bcm_name.lower()}.npz",
                                   field=global_dmo_map, box_size=CONFIG['box_size'],
                                   grid_resolution=grid_res, snapshot=snapNum,
                                   bcm_model=bcm_name, error=str(e))
            gc.collect()  # Clean up even on error
    
    comm.Barrier()  # Ensure all ranks wait for BCM to finish
    
    # ========================================================================
    # Save basic halo info (profiles can be computed separately if needed)
    # ========================================================================
    if rank == 0:
        print("\n  Saving halo info...")
        
        with h5py.File(f"{snap_dir}/profiles.h5", 'w') as f:
            f.attrs['snapshot'] = snapNum
            f.attrs['sim_resolution'] = sim_res
            f.attrs['n_halos'] = len(replace_halo_indices)
            f.attrs['log_mass_min'] = log_mass_min
            
            f.create_dataset('dmo_halo_indices', data=replace_halo_indices)
            f.create_dataset('matched_dmo_indices', data=matched_dmo_indices)
            f.create_dataset('matched_hydro_indices', data=matched_hydro_indices)
            f.create_dataset('halo_masses', data=halo_masses)
            f.create_dataset('halo_positions', data=halo_positions)
            f.create_dataset('halo_radii', data=halo_radii)
        
        print(f"  Saved: {snap_dir}/profiles.h5")
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Done!")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Generate 2D maps and profiles')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, required=True,
                        choices=[625, 1250, 2500], help='Simulation resolution')
    parser.add_argument('--grid-res', type=int, default=4096,
                        help='Grid resolution for 2D maps')
    parser.add_argument('--mass-min', type=float, default=12.5,
                        help='log10(M_min) for halos')
    parser.add_argument('--mass-max', type=float, default=None,
                        help='log10(M_max) for halos (optional, for mass bins)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip generation if output files already exist')
    parser.add_argument('--only-bcm', action='store_true',
                        help='Only generate BCM maps (skip DMO, Hydro, Replace)')
    parser.add_argument('--bcm-models', type=str, nargs='+', 
                        default=['Arico20', 'Schneider19', 'Schneider25'],
                        help='BCM models to run')
    
    args = parser.parse_args()
    
    if args.sim_res not in SIM_PATHS:
        raise ValueError(f"Unknown resolution: {args.sim_res}")
    
    t_start = time.time()
    process_snapshot(args)
    
    if rank == 0:
        print(f"\nTotal time: {(time.time() - t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
