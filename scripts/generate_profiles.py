#!/usr/bin/env python
"""
Generate 3D radial density profiles for matched halos.

For each matched halo, computes profiles from:
  - DMO particles (around DMO halo position)
  - Hydro particles (around Hydro halo position)
  - BCM-displaced DMO particles (3 models, around DMO halo position)

Output: profiles.h5 with radial density profiles for all matched halos.

Usage:
    mpirun -np 64 python generate_profiles.py --snap 99 --sim-res 2500
"""

import numpy as np
import h5py
import glob
import argparse
import os
import time
from mpi4py import MPI
from scipy.spatial import cKDTree
import pyccl as ccl
import BaryonForge as bfg
from illustris_python import groupcat

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
    'r_max_r200': 5.0,  # Maximum radius in units of R_200
    'n_radial_bins': 25,  # Number of radial bins (log-spaced)
    'r_min_r200': 0.001,  # Minimum radius in units of R_200 (matching BCM.py reference)
}

# TNG300 cosmology
COSMO = ccl.Cosmology(
    Omega_c=0.3089 - 0.0486, Omega_b=0.0486, h=0.6774,
    sigma8=0.8158, n_s=0.9649, matter_power_spectrum='linear'
)
h = COSMO.cosmo.params.h

# BCM Parameters
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


def compute_radial_profile(coords, masses, center, r200, box_size, r_bins):
    """
    Compute spherically-averaged density profile.
    
    Parameters:
    -----------
    coords : array (N, 3)
        Particle coordinates in Mpc/h
    masses : array (N,)
        Particle masses in Msun/h
    center : array (3,)
        Halo center in Mpc/h
    r200 : float
        R_200c in Mpc/h
    box_size : float
        Box size in Mpc/h
    r_bins : array
        Radial bin edges in units of R_200
    
    Returns:
    --------
    density : array
        Density profile in Msun/h / (Mpc/h)^3
    mass_enclosed : array
        Cumulative mass enclosed in each bin
    n_particles : array
        Number of particles in each bin
    """
    # Distance from center with periodic boundaries
    dx = coords - center
    dx = apply_periodic_boundary(dx, box_size)
    r = np.linalg.norm(dx, axis=1) / r200  # in units of R_200
    
    n_bins = len(r_bins) - 1
    density = np.zeros(n_bins)
    mass_enclosed = np.zeros(n_bins)
    n_particles = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        n_particles[i] = mask.sum()
        
        if n_particles[i] > 0:
            mass_in_shell = masses[mask].sum()
            mass_enclosed[i] = mass_in_shell
            
            # Shell volume in physical units
            r_inner = r_bins[i] * r200
            r_outer = r_bins[i+1] * r200
            volume = 4/3 * np.pi * (r_outer**3 - r_inner**3)
            
            density[i] = mass_in_shell / volume
    
    return density, mass_enclosed, n_particles


def load_particles_around_halos(basePath, snapNum, halo_positions, halo_radii, 
                                 r_max_factor, box_size, particle_types, 
                                 dm_mass=None, mass_unit=1e10):
    """
    Load particles within r_max_factor * R200 of each halo.
    
    Returns dict with 'coords' and 'masses' for each halo index.
    """
    # Get file list
    snap_dir = f"{basePath}/snapdir_{snapNum:03d}/"
    files = sorted(glob.glob(f"{snap_dir}/snap_{snapNum:03d}.*.hdf5"))
    
    # Distribute files across ranks
    my_files = [f for i, f in enumerate(files) if i % size == rank]
    
    # Load all particles from my files
    coords_list = []
    masses_list = []
    
    for filepath in my_files:
        with h5py.File(filepath, 'r') as f:
            for ptype in particle_types:
                if ptype not in f:
                    continue
                    
                c = f[ptype]['Coordinates'][:].astype(np.float32) / 1e3  # kpc -> Mpc
                
                if 'Masses' in f[ptype]:
                    m = f[ptype]['Masses'][:].astype(np.float32) * mass_unit
                else:
                    # DM particles have fixed mass
                    m = np.ones(len(c), dtype=np.float32) * dm_mass * mass_unit
                
                coords_list.append(c)
                masses_list.append(m)
    
    if len(coords_list) == 0:
        all_coords = np.zeros((0, 3), dtype=np.float32)
        all_masses = np.zeros(0, dtype=np.float32)
    else:
        all_coords = np.concatenate(coords_list)
        all_masses = np.concatenate(masses_list)
    
    # Build KD-tree for fast spatial queries
    if len(all_coords) > 0:
        tree = cKDTree(all_coords)
    else:
        tree = None
    
    # For each halo, find particles within r_max
    halo_particles = {}
    
    for i, (pos, r200) in enumerate(zip(halo_positions, halo_radii)):
        r_max = r_max_factor * r200
        
        if tree is not None:
            idx = tree.query_ball_point(pos, r_max)
            halo_particles[i] = {
                'coords': all_coords[idx],
                'masses': all_masses[idx]
            }
        else:
            halo_particles[i] = {
                'coords': np.zeros((0, 3), dtype=np.float32),
                'masses': np.zeros(0, dtype=np.float32)
            }
    
    return halo_particles


def main():
    parser = argparse.ArgumentParser(description='Generate radial density profiles')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, required=True,
                        choices=[625, 1250, 2500], help='Simulation resolution')
    parser.add_argument('--mass-min', type=float, default=12.5,
                        help='log10(M_min) for halos')
    parser.add_argument('--mass-max', type=float, default=None,
                        help='log10(M_max) for halos')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--bcm-models', type=str, nargs='+',
                        default=['Arico20', 'Schneider19', 'Schneider25'],
                        help='BCM models to run')
    parser.add_argument('--skip-bcm', action='store_true',
                        help='Skip BCM profiles (only compute DMO and Hydro)')
    
    args = parser.parse_args()
    
    sim_res = args.sim_res
    snapNum = args.snap
    log_mass_min = args.mass_min
    log_mass_max = args.mass_max
    
    sim_config = SIM_PATHS[sim_res]
    dmo_basePath = sim_config['dmo']
    hydro_basePath = sim_config['hydro']
    dmo_mass = sim_config['dmo_mass']
    hydro_dm_mass = sim_config['hydro_dm_mass']
    
    output_dir = args.output_dir or CONFIG['output_dir']
    snap_dir = f"{output_dir}/L205n{sim_res}TNG/snap{snapNum:03d}"
    
    box_size = CONFIG['box_size']
    r_max_r200 = CONFIG['r_max_r200']
    n_bins = CONFIG['n_radial_bins']
    r_min_r200 = CONFIG['r_min_r200']
    
    # Radial bins (log-spaced in units of R_200)
    r_bins = np.logspace(np.log10(r_min_r200), np.log10(r_max_r200), n_bins + 1)
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    
    if rank == 0:
        os.makedirs(snap_dir, exist_ok=True)
        print("=" * 70)
        print(f"Generating Radial Profiles")
        print(f"Snapshot: {snapNum}")
        print(f"Simulation: L205n{sim_res}TNG")
        print(f"Mass range: {log_mass_min} - {log_mass_max if log_mass_max else 'inf'}")
        print(f"Radial range: {r_min_r200:.2f} - {r_max_r200:.1f} R_200")
        print(f"Number of bins: {n_bins}")
        print("=" * 70)
    
    comm.Barrier()
    
    # ========================================================================
    # Load halo catalogs and matches
    # ========================================================================
    if rank == 0:
        print("\n[1/5] Loading halo catalogs and matches...")
    
    halo_dmo = groupcat.loadHalos(
        dmo_basePath, snapNum,
        fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
    )
    halo_hydro = groupcat.loadHalos(
        hydro_basePath, snapNum,
        fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
    )
    
    # Load matches
    matches_file = f"{output_dir}/L205n{sim_res}TNG/matches/matches_snap{snapNum:03d}.npz"
    if not os.path.exists(matches_file):
        if rank == 0:
            print(f"ERROR: No matches file at {matches_file}")
        return
    
    matches = np.load(matches_file)
    matched_dmo_idx = matches['dmo_indices']
    matched_hydro_idx = matches['hydro_indices']
    
    # Apply mass filter to matched halos
    dmo_masses_all = halo_dmo['Group_M_Crit200'][matched_dmo_idx] * CONFIG['mass_unit']
    log_masses = np.log10(dmo_masses_all)
    
    if log_mass_max is not None:
        mass_mask = (log_masses >= log_mass_min) & (log_masses < log_mass_max)
    else:
        mass_mask = log_masses >= log_mass_min
    
    selected_indices = np.where(mass_mask)[0]
    n_halos = len(selected_indices)
    
    if rank == 0:
        print(f"  Total matched halos: {len(matched_dmo_idx)}")
        print(f"  Halos in mass range: {n_halos}")
    
    # Get properties of selected halos
    sel_dmo_idx = matched_dmo_idx[selected_indices]
    sel_hydro_idx = matched_hydro_idx[selected_indices]
    
    dmo_positions = halo_dmo['GroupPos'][sel_dmo_idx] / 1e3  # Mpc/h
    dmo_radii = halo_dmo['Group_R_Crit200'][sel_dmo_idx] / 1e3  # Mpc/h
    dmo_masses = halo_dmo['Group_M_Crit200'][sel_dmo_idx] * CONFIG['mass_unit']
    
    hydro_positions = halo_hydro['GroupPos'][sel_hydro_idx] / 1e3
    hydro_radii = halo_hydro['Group_R_Crit200'][sel_hydro_idx] / 1e3
    hydro_masses = halo_hydro['Group_M_Crit200'][sel_hydro_idx] * CONFIG['mass_unit']
    
    # ========================================================================
    # Distribute halos across ranks
    # ========================================================================
    my_halo_indices = [i for i in range(n_halos) if i % size == rank]
    n_my_halos = len(my_halo_indices)
    
    if rank == 0:
        print(f"\n  Each rank processes ~{n_halos // size} halos")
    
    # Initialize profile storage
    profiles_dmo = np.zeros((n_my_halos, n_bins), dtype=np.float32)
    profiles_hydro = np.zeros((n_my_halos, n_bins), dtype=np.float32)
    profiles_bcm = {name: np.zeros((n_my_halos, n_bins), dtype=np.float32) 
                    for name in args.bcm_models}
    
    # ========================================================================
    # Load DMO particles (for DMO and BCM profiles)
    # ========================================================================
    if rank == 0:
        print("\n[2/5] Loading DMO particles...")
    
    dmo_particles = load_particles_around_halos(
        dmo_basePath, snapNum, 
        dmo_positions[my_halo_indices], dmo_radii[my_halo_indices],
        r_max_r200, box_size, ['PartType1'], 
        dm_mass=dmo_mass, mass_unit=CONFIG['mass_unit']
    )
    
    # ========================================================================
    # Compute DMO profiles
    # ========================================================================
    if rank == 0:
        print("\n[3/5] Computing DMO profiles...")
    
    for local_i, global_i in enumerate(my_halo_indices):
        coords = dmo_particles[local_i]['coords']
        masses = dmo_particles[local_i]['masses']
        
        density, _, _ = compute_radial_profile(
            coords, masses, dmo_positions[global_i], dmo_radii[global_i],
            box_size, r_bins
        )
        profiles_dmo[local_i] = density
    
    if rank == 0:
        print(f"  Computed {n_my_halos} DMO profiles on rank 0")
    
    # ========================================================================
    # Compute BCM profiles
    # ========================================================================
    if not args.skip_bcm:
        import gc
        cdict = build_cosmodict(COSMO)
        
        for bcm_name in args.bcm_models:
            if rank == 0:
                print(f"\n[4/5] Computing BCM profiles: {bcm_name}...")
                t_start = time.time()
            
            # Setup BCM model
            bcm_model = setup_bcm_model(bcm_name)
            
            for local_i, global_i in enumerate(my_halo_indices):
                coords = dmo_particles[local_i]['coords']
                masses = dmo_particles[local_i]['masses']
                
                if len(coords) == 0:
                    continue
                
                # Create snapshot for this halo's particles
                Snap = bfg.ParticleSnapshot(
                    x=coords[:, 0].astype(np.float64),
                    y=coords[:, 1].astype(np.float64),
                    z=coords[:, 2].astype(np.float64),
                    L=box_size,
                    redshift=0,
                    cosmo=cdict,
                    M=masses[0] if len(masses) > 0 else dmo_mass * CONFIG['mass_unit']
                )
                
                # Single halo catalog (just this halo)
                HCat = bfg.HaloNDCatalog(
                    x=np.array([dmo_positions[global_i, 0]]),
                    y=np.array([dmo_positions[global_i, 1]]),
                    z=np.array([dmo_positions[global_i, 2]]),
                    M=np.array([dmo_masses[global_i]]),
                    redshift=0,
                    cosmo=cdict
                )
                
                # Baryonify
                try:
                    Runner = bfg.Runners.BaryonifySnapshot(
                        HCat, Snap, epsilon_max=r_max_r200, model=bcm_model,
                        KDTree_kwargs={'leafsize': 100, 'balanced_tree': False},
                        verbose=False
                    )
                    Baryonified = Runner.process()
                    
                    bcm_coords = np.array([
                        Baryonified['x'][:],
                        Baryonified['y'][:],
                        Baryonified['z'][:]
                    ]).T.astype(np.float32)
                    
                    # Compute profile
                    density, _, _ = compute_radial_profile(
                        bcm_coords, masses, dmo_positions[global_i], dmo_radii[global_i],
                        box_size, r_bins
                    )
                    profiles_bcm[bcm_name][local_i] = density
                    
                except Exception as e:
                    if rank == 0 and local_i == 0:
                        print(f"    Warning: BCM failed for halo {global_i}: {e}")
            
            # Clean up
            del bcm_model
            gc.collect()
            
            if rank == 0:
                print(f"    Done in {time.time() - t_start:.1f}s")
    
    # Free DMO particles
    del dmo_particles
    
    # ========================================================================
    # Load Hydro particles and compute profiles
    # ========================================================================
    if rank == 0:
        print("\n[5/5] Loading Hydro particles and computing profiles...")
    
    hydro_particles = load_particles_around_halos(
        hydro_basePath, snapNum,
        hydro_positions[my_halo_indices], hydro_radii[my_halo_indices],
        r_max_r200, box_size, ['PartType0', 'PartType1', 'PartType4'],
        dm_mass=hydro_dm_mass, mass_unit=CONFIG['mass_unit']
    )
    
    for local_i, global_i in enumerate(my_halo_indices):
        coords = hydro_particles[local_i]['coords']
        masses = hydro_particles[local_i]['masses']
        
        density, _, _ = compute_radial_profile(
            coords, masses, hydro_positions[global_i], hydro_radii[global_i],
            box_size, r_bins
        )
        profiles_hydro[local_i] = density
    
    del hydro_particles
    
    # ========================================================================
    # Gather all profiles to rank 0
    # ========================================================================
    if rank == 0:
        print("\nGathering results...")
    
    # Gather halo indices
    all_halo_indices = comm.gather(my_halo_indices, root=0)
    all_profiles_dmo = comm.gather(profiles_dmo, root=0)
    all_profiles_hydro = comm.gather(profiles_hydro, root=0)
    all_profiles_bcm = {name: comm.gather(profiles_bcm[name], root=0) 
                        for name in args.bcm_models}
    
    # ========================================================================
    # Save profiles
    # ========================================================================
    if rank == 0:
        print("\nSaving profiles...")
        
        # Reconstruct full arrays in correct order
        final_dmo = np.zeros((n_halos, n_bins), dtype=np.float32)
        final_hydro = np.zeros((n_halos, n_bins), dtype=np.float32)
        final_bcm = {name: np.zeros((n_halos, n_bins), dtype=np.float32)
                     for name in args.bcm_models}
        
        for r in range(size):
            for local_i, global_i in enumerate(all_halo_indices[r]):
                final_dmo[global_i] = all_profiles_dmo[r][local_i]
                final_hydro[global_i] = all_profiles_hydro[r][local_i]
                for name in args.bcm_models:
                    final_bcm[name][global_i] = all_profiles_bcm[name][r][local_i]
        
        # Build mass label
        if log_mass_max is not None:
            mass_label = f"M{log_mass_min:.1f}-{log_mass_max:.1f}"
        else:
            mass_label = f"Mgt{log_mass_min:.1f}"
        
        output_file = f"{snap_dir}/profiles_{mass_label}.h5"
        
        with h5py.File(output_file, 'w') as f:
            # Metadata
            f.attrs['snapshot'] = snapNum
            f.attrs['sim_resolution'] = sim_res
            f.attrs['n_halos'] = n_halos
            f.attrs['log_mass_min'] = log_mass_min
            f.attrs['log_mass_max'] = log_mass_max if log_mass_max else -1
            f.attrs['r_max_r200'] = r_max_r200
            f.attrs['n_radial_bins'] = n_bins
            
            # Radial bins
            f.create_dataset('r_bins', data=r_bins)
            f.create_dataset('r_centers', data=r_centers)
            
            # Halo properties
            f.create_dataset('dmo_halo_indices', data=sel_dmo_idx)
            f.create_dataset('hydro_halo_indices', data=sel_hydro_idx)
            f.create_dataset('dmo_masses', data=dmo_masses)
            f.create_dataset('hydro_masses', data=hydro_masses)
            f.create_dataset('dmo_positions', data=dmo_positions)
            f.create_dataset('hydro_positions', data=hydro_positions)
            f.create_dataset('dmo_radii', data=dmo_radii)
            f.create_dataset('hydro_radii', data=hydro_radii)
            
            # Profiles (density in Msun/h / (Mpc/h)^3)
            grp = f.create_group('profiles')
            grp.create_dataset('dmo', data=final_dmo)
            grp.create_dataset('hydro', data=final_hydro)
            for name in args.bcm_models:
                grp.create_dataset(f'bcm_{name.lower()}', data=final_bcm[name])
        
        print(f"Saved: {output_file}")
        print(f"  Shape: ({n_halos}, {n_bins})")
        print(f"  Profiles: dmo, hydro, " + ", ".join(f"bcm_{n.lower()}" for n in args.bcm_models))
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Done!")
        print("=" * 70)


if __name__ == "__main__":
    main()
