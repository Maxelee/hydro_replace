#!/usr/bin/env python
"""
MPI-parallel BCM profile generation using streaming approach.

Strategy:
- Load DMO particles and apply BCM displacement using BaryonForge
- Compute density profiles around DMO halo centers
- All ranks stream through data, each accumulates its assigned halos

This creates profiles for BCM-displaced particles, which can be compared
to DMO and Hydro profiles to validate BCM models.

Usage:
    mpirun -n 64 python generate_profiles_bcm.py --sim-res 625 --snapshot 99
    mpirun -n 128 python generate_profiles_bcm.py --sim-res 2500 --snapshot 99 --bcm-models Arico20 Schneider19
"""

import numpy as np
import h5py
import os
import sys
import time
import argparse
import glob
from mpi4py import MPI
from scipy.spatial import cKDTree

import pyccl as ccl
import BaryonForge as bfg

# ============================================================================
# Configuration
# ============================================================================

# TNG300-1 cosmology
h = 0.6774
COSMO = ccl.Cosmology(
    Omega_c=0.2589, Omega_b=0.0486, h=h,
    sigma8=0.8159, n_s=0.9667,
    transfer_function='boltzmann_camb',
    matter_power_spectrum='halofit'
)

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
    'box_size': 205.0,      # Mpc/h
    'mass_unit': 1e10,      # Convert to Msun/h
    'r_max_r200': 5.0,      # Maximum radius in units of R_200
    'r_min_r200': 0.01,     # Minimum radius in units of R_200
    'n_radial_bins': 25,    # Number of radial bins (log-spaced)
    'log_mass_min': 12.0,   # Minimum halo mass log10(M/Msun/h)
}

# BCM parameters (same as generate_all.py)
BCM_PARAMS = {
    'Arico20': dict(
        M_c=3.3e13, M1_0=8.63e11, eta=0.54, beta=0.12,
        mu=0.31, M_inn=3.3e13, theta_inn=0.1, theta_out=3,
        epsilon_h=0.015, alpha_g=2,
        epsilon_hydro=np.sqrt(5), theta_rg=0.3, sigma_rg=0.1,
        a=0.3, n=2, p=0.3, q=0.707,
        alpha_fsat=1, M1_fsat=1, delta_fsat=1, gamma_fsat=1, eps_fsat=1,
        M_r=1e16, beta_r=2,
        A_nt=0.495, alpha_nt=0.1,
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

FIELDS_DIR = '/mnt/home/mlee1/ceph/hydro_replace_fields'
OUTPUT_DIR = '/mnt/home/mlee1/ceph/hydro_replace_profiles'

# ============================================================================
# Helper Functions
# ============================================================================

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


def get_snapshot_path(basePath, snapNum, chunkNum=0):
    """Get path to snapshot chunk file."""
    snapPath = f"{basePath}/snapdir_{snapNum:03d}/"
    filePath1 = f"{snapPath}snap_{snapNum:03d}.{chunkNum}.hdf5"
    filePath2 = filePath1.replace('/snap_', '/snapshot_')
    
    if os.path.isfile(filePath1):
        return filePath1
    return filePath2


def count_snapshot_chunks(basePath, snapNum):
    """Count number of snapshot chunks."""
    search_pattern = f"{basePath}/snapdir_{snapNum:03d}/snap*.hdf5"
    files = glob.glob(search_pattern)
    if not files:
        search_pattern = f"{basePath}/snapdir_{snapNum:03d}/snapshot*.hdf5"
        files = glob.glob(search_pattern)
    return len(files)


def load_halo_catalog(basePath, snapNum, fields):
    """Load halo catalog fields."""
    from illustris_python import groupcat
    return groupcat.loadHalos(basePath, snapNum, fields=fields)


# ============================================================================
# BCM Application using BaryonForge
# ============================================================================

def apply_bcm_to_chunk(coords, masses, halo_positions, halo_masses, 
                       bcm_model, box_size, cdict, rank=0):
    """
    Apply BCM displacement to a chunk of particles using BaryonForge.
    
    Parameters:
    -----------
    coords : array (N, 3)
        Particle coordinates in Mpc/h
    masses : array (N,) or float
        Particle masses in Msun/h
    halo_positions : array (M, 3)
        Halo centers in Mpc/h
    halo_masses : array (M,)
        Halo M_200 in Msun/h
    bcm_model : bfg.Baryonification3D
        BCM displacement model
    box_size : float
    cdict : dict
        Cosmology dictionary
        
    Returns:
    --------
    displaced_coords : array (N, 3)
    """
    if len(coords) == 0:
        return coords.copy()
    
    # Wrap coordinates to [0, box_size)
    wrapped_coords = coords % box_size
    
    # Get particle mass (assume all same)
    if isinstance(masses, np.ndarray):
        M_particle = masses[0] if len(masses) > 0 else 1e10
    else:
        M_particle = masses
    
    # Create BaryonForge snapshot
    Snap = bfg.ParticleSnapshot(
        x=wrapped_coords[:, 0].astype(np.float64),
        y=wrapped_coords[:, 1].astype(np.float64),
        z=wrapped_coords[:, 2].astype(np.float64),
        L=box_size,
        redshift=0,
        cosmo=cdict,
        M=M_particle
    )
    
    # Create halo catalog
    HCat = bfg.HaloNDCatalog(
        x=halo_positions[:, 0],
        y=halo_positions[:, 1],
        z=halo_positions[:, 2],
        M=halo_masses,
        redshift=0,
        cosmo=cdict
    )
    
    # Apply BCM displacement
    Runner = bfg.Runners.BaryonifySnapshot(
        HCat, Snap, epsilon_max=5.0, model=bcm_model,
        KDTree_kwargs={'leafsize': 1000, 'balanced_tree': False},
        verbose=False
    )
    
    Baryonified = Runner.process()
    
    displaced_coords = np.array([
        Baryonified['x'][:],
        Baryonified['y'][:],
        Baryonified['z'][:]
    ]).T.astype(np.float32)
    
    # Clean up
    del Runner, Snap, HCat, Baryonified
    
    return displaced_coords


# ============================================================================
# MPI Streaming Profile Computation (with BCM)
# ============================================================================

def compute_bcm_profiles_mpi_streaming(comm, basePath, snapNum, centers, r200_arr, 
                                       halo_masses_full, my_halo_indices, 
                                       box_size, r_bins, bcm_model,
                                       dm_mass=None, mass_unit=1e10):
    """
    MPI-parallel streaming BCM profile computation.
    
    Process:
    1. Load DMO particles chunk by chunk
    2. Apply BCM displacement to each chunk
    3. Accumulate profiles for assigned halos
    
    Parameters:
    -----------
    comm : MPI communicator
    basePath : str
        Path to DMO simulation
    snapNum : int
        Snapshot number
    centers : array (N_total_halos, 3)
        ALL halo centers (needed for BCM and KDTree)
    r200_arr : array (N_total_halos,)
        ALL R_200 values
    halo_masses_full : array (N_total_halos,)
        ALL halo masses (needed for BCM)
    my_halo_indices : array
        Indices of halos THIS rank is responsible for
    box_size : float
    r_bins : array
    bcm_model : BaryonForge displacement model
    dm_mass : float or None
    mass_unit : float
    
    Returns:
    --------
    mass_in_bins : (N_my_halos, N_bins)
    n_particles : (N_my_halos, N_bins)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    cdict = build_cosmodict(COSMO)
    
    n_total_halos = len(centers)
    n_my_halos = len(my_halo_indices)
    n_bins = len(r_bins) - 1
    r_max_factor = r_bins[-1]
    
    # Pre-compute r_max for ALL halos
    r_max_arr = r_max_factor * r200_arr
    max_r = np.max(r_max_arr)
    
    # Accumulators for MY halos only
    mass_in_bins = np.zeros((n_my_halos, n_bins), dtype=np.float64)
    n_particles = np.zeros((n_my_halos, n_bins), dtype=np.int64)
    
    # Create mapping from global halo index to local index
    global_to_local = {g: l for l, g in enumerate(my_halo_indices)}
    
    # Build KDTree of ALL halo centers (for particle assignment)
    halo_tree = cKDTree(centers, boxsize=box_size)
    
    n_chunks = count_snapshot_chunks(basePath, snapNum)
    
    total_processed = 0
    
    if rank == 0:
        print(f"  [{time.strftime('%H:%M:%S')}] Streaming DMO + BCM ({n_chunks} chunks)...")
    
    t0 = time.time()
    
    for chunk in range(n_chunks):
        try:
            fpath = get_snapshot_path(basePath, snapNum, chunk)
            with h5py.File(fpath, 'r') as f:
                if 'PartType1' not in f:
                    continue
                
                # Load DMO particles
                coords = f['PartType1']['Coordinates'][:].astype(np.float32) / 1e3  # kpc -> Mpc
                n_part = len(coords)
                total_processed += n_part
                
                if n_part == 0:
                    continue
                
                if dm_mass is not None:
                    masses = np.full(n_part, dm_mass * mass_unit, dtype=np.float32)
                else:
                    masses = f['PartType1']['Masses'][:].astype(np.float32) * mass_unit
                
                # Apply BCM displacement
                bcm_coords = apply_bcm_to_chunk(
                    coords, masses[0], centers, halo_masses_full,
                    bcm_model, box_size, cdict, rank
                )
                
                # Find particles close to ANY halo (using BCM-displaced coordinates)
                distances, nearest = halo_tree.query(bcm_coords, k=1, workers=1)
                close_mask = distances < max_r
                
                if not np.any(close_mask):
                    del bcm_coords
                    continue
                
                close_coords = bcm_coords[close_mask]
                close_masses = masses[close_mask]
                
                # Find all nearby halos for close particles
                nearby_lists = halo_tree.query_ball_point(close_coords, r=max_r, workers=1)
                
                # Process only MY halos
                for local_idx, global_idx in enumerate(my_halo_indices):
                    # Find particles near this halo
                    particle_indices = [i for i, hlist in enumerate(nearby_lists) 
                                      if global_idx in hlist]
                    
                    if len(particle_indices) == 0:
                        continue
                    
                    p_idx = np.array(particle_indices)
                    p_coords = close_coords[p_idx]
                    p_masses = close_masses[p_idx]
                    
                    # Compute distances from halo center
                    dx = p_coords - centers[global_idx]
                    dx = dx - np.round(dx / box_size) * box_size
                    r = np.linalg.norm(dx, axis=1)
                    
                    # Filter by this halo's r_max
                    in_range = r < r_max_arr[global_idx]
                    if not np.any(in_range):
                        continue
                    
                    r = r[in_range]
                    p_masses = p_masses[in_range]
                    
                    # Bin particles
                    r_scaled = r / r200_arr[global_idx]
                    bin_indices = np.searchsorted(r_bins, r_scaled) - 1
                    valid = (bin_indices >= 0) & (bin_indices < n_bins)
                    
                    if np.any(valid):
                        mass_in_bins[local_idx] += np.bincount(
                            bin_indices[valid], weights=p_masses[valid], minlength=n_bins
                        )
                        n_particles[local_idx] += np.bincount(
                            bin_indices[valid], minlength=n_bins
                        )
                
                del bcm_coords, close_coords, close_masses
                        
        except Exception as e:
            if rank == 0:
                print(f"    Chunk {chunk} error: {e}")
            continue
        
        # Progress update every 10 chunks
        if rank == 0 and (chunk + 1) % max(1, n_chunks // 10) == 0:
            print(f"    [{time.strftime('%H:%M:%S')}] Chunk {chunk+1}/{n_chunks}")
    
    if rank == 0:
        print(f"    Done in {time.time()-t0:.1f}s, processed {total_processed:,} particles")
    
    # Sync before returning
    comm.Barrier()
    
    return mass_in_bins, n_particles


def mass_to_density(mass_in_bins, r200_arr, r_bins):
    """Convert accumulated mass to density profiles."""
    n_halos = len(mass_in_bins)
    n_bins = len(r_bins) - 1
    
    profiles = np.zeros((n_halos, n_bins), dtype=np.float64)
    
    for h_idx in range(n_halos):
        r200 = r200_arr[h_idx]
        for b_idx in range(n_bins):
            r_inner = r_bins[b_idx] * r200
            r_outer = r_bins[b_idx + 1] * r200
            volume = 4/3 * np.pi * (r_outer**3 - r_inner**3)
            if volume > 0:
                profiles[h_idx, b_idx] = mass_in_bins[h_idx, b_idx] / volume
    
    return profiles.astype(np.float32)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MPI BCM Profile Generation')
    parser.add_argument('--sim-res', type=int, default=625, choices=[625, 1250, 2500])
    parser.add_argument('--snapshot', type=int, default=99)
    parser.add_argument('--bcm-models', nargs='+', 
                        default=['Arico20', 'Schneider19', 'Schneider25'],
                        help='BCM models to compute profiles for')
    parser.add_argument('--max-halos', type=int, default=None, help='Limit halos for testing')
    args = parser.parse_args()
    
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    sim_config = SIM_PATHS[args.sim_res]
    box_size = CONFIG['box_size']
    mass_unit = CONFIG['mass_unit']
    
    # Radial bins
    r_bins = np.logspace(
        np.log10(CONFIG['r_min_r200']), 
        np.log10(CONFIG['r_max_r200']), 
        CONFIG['n_radial_bins'] + 1
    )
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    n_bins = len(r_bins) - 1
    
    if rank == 0:
        print("=" * 70)
        print(f"MPI BCM PROFILE GENERATION - L205n{args.sim_res}TNG snap {args.snapshot}")
        print("=" * 70)
        print(f"  Ranks: {size}")
        print(f"  BCM models: {args.bcm_models}")
        print(f"  Radial bins: {n_bins}")
        print()
    
    # ========================================================================
    # Load halo catalogs and matches (rank 0 only, then broadcast)
    # ========================================================================
    
    if rank == 0:
        print("Loading halo catalogs and matches...")
        t0 = time.time()
        
        # Load DMO halos
        halo_dmo = load_halo_catalog(
            sim_config['dmo'], args.snapshot,
            fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
        )
        
        # Load matches
        matches_file = f"{FIELDS_DIR}/L205n{args.sim_res}TNG/matches/matches_snap{args.snapshot:03d}.npz"
        if not os.path.exists(matches_file):
            print(f"ERROR: No matches file at {matches_file}")
            comm.Abort(1)
        
        matches = np.load(matches_file)
        matched_dmo_idx = matches['dmo_indices']
        
        # Apply mass filter
        dmo_masses_all = halo_dmo['Group_M_Crit200'][matched_dmo_idx] * mass_unit
        log_masses = np.log10(dmo_masses_all + 1e-10)
        mass_mask = log_masses >= CONFIG['log_mass_min']
        
        selected_indices = np.where(mass_mask)[0]
        if args.max_halos is not None:
            selected_indices = selected_indices[:args.max_halos]
        
        n_halos = len(selected_indices)
        
        # Get selected halo properties (use DMO halo positions/masses)
        sel_dmo_idx = matched_dmo_idx[selected_indices]
        
        dmo_positions = (halo_dmo['GroupPos'][sel_dmo_idx] / 1e3).astype(np.float32)
        dmo_radii = (halo_dmo['Group_R_Crit200'][sel_dmo_idx] / 1e3).astype(np.float32)
        dmo_masses = (halo_dmo['Group_M_Crit200'][sel_dmo_idx] * mass_unit).astype(np.float64)
        
        print(f"  Loaded in {time.time()-t0:.1f}s")
        print(f"  Selected {n_halos} halos with log(M) >= {CONFIG['log_mass_min']}")
    else:
        n_halos = None
        dmo_positions = None
        dmo_radii = None
        dmo_masses = None
        sel_dmo_idx = None
    
    # Broadcast metadata
    n_halos = comm.bcast(n_halos, root=0)
    
    # Allocate arrays on other ranks
    if rank != 0:
        dmo_positions = np.empty((n_halos, 3), dtype=np.float32)
        dmo_radii = np.empty(n_halos, dtype=np.float32)
        dmo_masses = np.empty(n_halos, dtype=np.float64)
        sel_dmo_idx = np.empty(n_halos, dtype=np.int64)
    
    # Broadcast arrays
    comm.Bcast(dmo_positions, root=0)
    comm.Bcast(dmo_radii, root=0)
    comm.Bcast(dmo_masses, root=0)
    comm.Bcast(sel_dmo_idx, root=0)
    
    # ========================================================================
    # Distribute halos across ranks
    # ========================================================================
    
    halos_per_rank = n_halos // size
    remainder = n_halos % size
    
    if rank < remainder:
        my_start = rank * (halos_per_rank + 1)
        my_count = halos_per_rank + 1
    else:
        my_start = rank * halos_per_rank + remainder
        my_count = halos_per_rank
    
    my_halo_indices = np.arange(my_start, my_start + my_count)
    
    if rank == 0:
        print(f"\nHalo distribution: {halos_per_rank}-{halos_per_rank+1} halos per rank")
    
    # ========================================================================
    # Compute profiles for each BCM model
    # ========================================================================
    
    results = {}
    
    for bcm_name in args.bcm_models:
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Computing {bcm_name} profiles...")
            print(f"{'='*50}")
            print("  Setting up BCM model...")
        
        # Setup BCM model (all ranks need this)
        bcm_model = setup_bcm_model(bcm_name)
        
        if rank == 0:
            print("  BCM model ready")
        
        t_bcm = time.time()
        
        mass_bcm, npart_bcm = compute_bcm_profiles_mpi_streaming(
            comm, sim_config['dmo'], args.snapshot,
            dmo_positions, dmo_radii, dmo_masses, my_halo_indices,
            box_size, r_bins, bcm_model,
            dm_mass=sim_config['dmo_mass'],
            mass_unit=mass_unit
        )
        
        # Gather results to rank 0
        all_mass_bcm = comm.gather(mass_bcm, root=0)
        all_npart_bcm = comm.gather(npart_bcm, root=0)
        all_indices = comm.gather(my_halo_indices, root=0)
        
        if rank == 0:
            # Reconstruct full arrays
            full_mass_bcm = np.zeros((n_halos, n_bins), dtype=np.float64)
            full_npart_bcm = np.zeros((n_halos, n_bins), dtype=np.int64)
            
            for indices, mass, npart in zip(all_indices, all_mass_bcm, all_npart_bcm):
                for i, global_idx in enumerate(indices):
                    full_mass_bcm[global_idx] = mass[i]
                    full_npart_bcm[global_idx] = npart[i]
            
            # Convert to density
            profiles_bcm = mass_to_density(full_mass_bcm, dmo_radii, r_bins)
            
            results[f'profiles_{bcm_name}'] = profiles_bcm
            results[f'n_particles_{bcm_name}'] = full_npart_bcm.astype(np.int32)
            
            print(f"  {bcm_name} done in {time.time()-t_bcm:.1f}s")
        
        # Clean up BCM model
        del bcm_model
    
    # ========================================================================
    # Save results (rank 0 only)
    # ========================================================================
    
    if rank == 0:
        out_dir = f"{OUTPUT_DIR}/L205n{args.sim_res}TNG"
        os.makedirs(out_dir, exist_ok=True)
        
        out_file = f"{out_dir}/profiles_bcm_snap{args.snapshot:03d}.h5"
        
        print(f"\nSaving to {out_file}...")
        
        with h5py.File(out_file, 'w') as f:
            # Metadata
            f.attrs['snapshot'] = args.snapshot
            f.attrs['sim_resolution'] = args.sim_res
            f.attrs['n_halos'] = n_halos
            f.attrs['log_mass_min'] = CONFIG['log_mass_min']
            f.attrs['r_max_r200'] = CONFIG['r_max_r200']
            f.attrs['r_min_r200'] = CONFIG['r_min_r200']
            f.attrs['n_radial_bins'] = n_bins
            f.attrs['method'] = 'MPI streaming BCM spherical aperture'
            f.attrs['n_ranks'] = size
            f.attrs['bcm_models'] = args.bcm_models
            f.attrs['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Radial bins
            f.create_dataset('r_bins', data=r_bins)
            f.create_dataset('r_centers', data=r_centers)
            
            # Halo properties (DMO)
            f.create_dataset('dmo_halo_indices', data=sel_dmo_idx)
            f.create_dataset('dmo_masses', data=dmo_masses)
            f.create_dataset('dmo_positions', data=dmo_positions)
            f.create_dataset('dmo_radii', data=dmo_radii)
            
            # BCM parameters
            params_grp = f.create_group('bcm_parameters')
            for bcm_name in args.bcm_models:
                bcm_grp = params_grp.create_group(bcm_name)
                for key, val in BCM_PARAMS[bcm_name].items():
                    bcm_grp.attrs[key] = val
            
            # Profiles for each BCM model
            grp = f.create_group('profiles')
            for bcm_name in args.bcm_models:
                if f'profiles_{bcm_name}' in results:
                    grp.create_dataset(bcm_name, data=results[f'profiles_{bcm_name}'])
                    grp.create_dataset(f'n_particles_{bcm_name}', 
                                      data=results[f'n_particles_{bcm_name}'])
        
        print(f"  Saved {n_halos} profiles for {len(args.bcm_models)} BCM models")
        print("\n" + "=" * 70)
        print("DONE!")
        print("=" * 70)


if __name__ == '__main__':
    main()
