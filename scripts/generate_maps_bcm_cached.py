#!/usr/bin/env python
"""
Generate BCM 2D density maps using particle cache.

Applies BCM displacement to DMO particles around matched halos and
generates projected density maps.

The approach:
1. Start with full DMO particle field
2. For each halo above mass threshold, apply BCM displacement to particles within 5×R200
3. Project displaced particles to 2D

Usage:
    mpirun -np 4 python generate_maps_bcm_cached.py --sim-res 625 --snap 99 --bcm-model Arico20
"""

import numpy as np
import h5py
import argparse
import os
import sys
import time
import glob
from mpi4py import MPI
import MAS_library as MASL

import pyccl as ccl
import BaryonForge as bfg

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================================
# Configuration
# ============================================================================

h = 0.6774
COSMO = ccl.Cosmology(
    Omega_c=0.2589, Omega_b=0.0486, h=h,
    sigma8=0.8159, n_s=0.9667,
    transfer_function='boltzmann_camb',
    matter_power_spectrum='halofit'
)

SIM_PATHS = {
    2500: {'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output', 'dmo_mass': 0.0047271638660809},
    1250: {'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG_DM/output', 'dmo_mass': 0.0378173109},
    625: {'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG_DM/output', 'dmo_mass': 0.3025384873},
}

CACHE_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields'
BOX_SIZE = 205.0
MASS_UNIT = 1e10
GRID_SIZE = 4096

BCM_PARAMS = {
    'Arico20': dict(
        M_c=3.3e13, M1_0=8.63e11, eta=0.54, beta=0.12,
        mu=0.31, M_inn=3.3e13, theta_inn=0.1, theta_out=3,
        epsilon_h=0.015, alpha_g=2,
        epsilon_hydro=np.sqrt(5), theta_rg=0.3, sigma_rg=0.1,
        a=0.3, n=2, p=0.3, q=0.707,
        alpha_fsat=1, M1_fsat=1, delta_fsat=1, gamma_fsat=1, eps_fsat=1,
        M_r=1e16, beta_r=2, A_nt=0.495, alpha_nt=0.1,
    ),
    'Schneider19': dict(
        theta_ej=4, theta_co=0.1, M_c=1e14/h, mu_beta=0.4,
        gamma=2, delta=7, eta=0.3, eta_delta=0.3, tau=-1.5, tau_delta=0,
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
    ),
}


def build_cosmodict(cosmo):
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
    Displacement.setup_interpolator(z_min=0, z_max=3, z_linear_sampling=True, N_samples_R=10000, Rdelta_sampling=True)
    return Displacement


def load_dmo_particles_distributed(basePath, snapshot, sim_res):
    """Load DMO particles distributed across MPI ranks."""
    dm_mass = SIM_PATHS[sim_res]['dmo_mass']
    
    snap_dir = f"{basePath}/snapdir_{snapshot:03d}/"
    all_files = sorted(glob.glob(f"{snap_dir}/snap_{snapshot:03d}.*.hdf5"))
    my_files = [f for i, f in enumerate(all_files) if i % size == rank]
    
    coords_list = []
    ids_list = []
    
    for filepath in my_files:
        with h5py.File(filepath, 'r') as f:
            if 'PartType1' not in f:
                continue
            coords_list.append(f['PartType1']['Coordinates'][:].astype(np.float32) / 1e3)
            ids_list.append(f['PartType1']['ParticleIDs'][:])
    
    if coords_list:
        local_coords = np.concatenate(coords_list)
        local_ids = np.concatenate(ids_list)
    else:
        local_coords = np.zeros((0, 3), dtype=np.float32)
        local_ids = np.zeros(0, dtype=np.int64)
    
    local_masses = np.full(len(local_coords), dm_mass * MASS_UNIT, dtype=np.float32)
    
    return local_coords, local_masses, local_ids


def apply_bcm_to_region(coords, masses, halo_center, halo_mass, r200, radius_factor,
                        redshift, displacement_model, cosmo_dict):
    """Apply BCM displacement to particles within radius_factor × R200 of halo."""
    if len(coords) == 0:
        return coords.copy()
    
    # Find particles within radius
    dx = coords - halo_center
    dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
    dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
    r = np.linalg.norm(dx, axis=1)
    
    in_region = r < radius_factor * r200
    if not in_region.any():
        return coords.copy()
    
    # Apply BCM to particles in region
    region_coords = coords[in_region]
    region_masses = masses[in_region]
    
    try:
        Snap = bfg.ParticleSnapshot(
            x=region_coords[:, 0], y=region_coords[:, 1], z=region_coords[:, 2],
            L=BOX_SIZE / h, redshift=redshift, cosmo=cosmo_dict,
            M=region_masses[0] if len(region_masses) > 0 else 1e10
        )
        HCat = bfg.HaloNDCatalog(
            x=np.array([halo_center[0]]), y=np.array([halo_center[1]]), z=np.array([halo_center[2]]),
            M_200c=np.array([halo_mass]), redshift=redshift, cosmo=cosmo_dict, is_central=np.array([True])
        )
        
        disp_x, disp_y, disp_z = displacement_model.displace(Snap, HCat, verbose=False)
        
        displaced = coords.copy()
        displaced[in_region, 0] += disp_x
        displaced[in_region, 1] += disp_y
        displaced[in_region, 2] += disp_z
        displaced = displaced % BOX_SIZE
        
        return displaced
    except Exception as e:
        return coords.copy()


def project_to_2d(coords, masses, grid_size, box_size, axis=2):
    """Project particles to 2D using TSC."""
    pos_2d = np.delete(coords, axis, axis=1).astype(np.float32)
    
    field = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # Manual TSC in 2D
    cell_size = box_size / grid_size
    for i in range(len(pos_2d)):
        x, y = pos_2d[i] / cell_size
        ix, iy = int(x), int(y)
        
        fx, fy = x - ix, y - iy
        
        # TSC weights
        wx = np.array([0.5 * (0.5 - fx)**2, 0.75 - (fx - 0.5)**2, 0.5 * (fx - 0.5)**2])
        wy = np.array([0.5 * (0.5 - fy)**2, 0.75 - (fy - 0.5)**2, 0.5 * (fy - 0.5)**2])
        
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ii = (ix + di) % grid_size
                jj = (iy + dj) % grid_size
                field[jj, ii] += masses[i] * wx[di+1] * wy[dj+1]
    
    # Convert to surface density
    cell_area = (box_size / grid_size) ** 2
    field /= cell_area
    
    return field


def main():
    parser = argparse.ArgumentParser(description='Generate BCM 2D density maps')
    parser.add_argument('--snap', type=int, required=True)
    parser.add_argument('--sim-res', type=int, default=625, choices=[625, 1250, 2500])
    parser.add_argument('--bcm-model', type=str, default='Arico20', choices=['Arico20', 'Schneider19', 'Schneider25'])
    parser.add_argument('--mass-min', type=float, default=12.5)
    parser.add_argument('--mass-max', type=float, default=None)
    parser.add_argument('--radius-factor', type=float, default=5.0)
    parser.add_argument('--grid-size', type=int, default=GRID_SIZE)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    
    if rank == 0:
        print("=" * 70)
        print(f"BCM MAP GENERATION - {args.bcm_model}")
        print("=" * 70)
        print(f"Snapshot: {args.snap}, Resolution: L205n{args.sim_res}TNG")
        print(f"Grid: {args.grid_size}, BCM radius: {args.radius_factor}×R200")
        print("=" * 70)
    
    t_start = time.time()
    
    # Setup BCM
    if rank == 0:
        print("\n[1/5] Setting up BCM model...")
    displacement_model = setup_bcm_model(args.bcm_model)
    cosmo_dict = build_cosmodict(COSMO)
    
    # Load halo info from cache
    cache_file = os.path.join(CACHE_BASE, f'L205n{args.sim_res}TNG', 'particle_cache', f'cache_snap{args.snap:03d}.h5')
    
    if rank == 0:
        print(f"\n[2/5] Loading halo info from cache...")
        with h5py.File(cache_file, 'r') as f:
            halo_info = f['halo_info']
            positions = halo_info['positions_dmo'][:]
            radii = halo_info['radii_dmo'][:]
            masses = halo_info['masses'][:]
            log_masses = np.log10(masses)
            
            mass_mask = log_masses >= args.mass_min
            if args.mass_max:
                mass_mask &= log_masses <= args.mass_max
            
            halo_positions = positions[mass_mask]
            halo_radii = radii[mass_mask]
            halo_masses = masses[mass_mask]
        
        n_halos = len(halo_positions)
        print(f"  {n_halos} halos in mass range")
    else:
        n_halos = halo_positions = halo_radii = halo_masses = None
    
    n_halos = comm.bcast(n_halos, root=0)
    halo_positions = comm.bcast(halo_positions, root=0)
    halo_radii = comm.bcast(halo_radii, root=0)
    halo_masses = comm.bcast(halo_masses, root=0)
    
    # Load DMO particles
    if rank == 0:
        print(f"\n[3/5] Loading DMO particles...")
    
    basePath = SIM_PATHS[args.sim_res]['dmo']
    local_coords, local_masses, local_ids = load_dmo_particles_distributed(basePath, args.snap, args.sim_res)
    
    local_n = len(local_coords)
    total_n = comm.reduce(local_n, op=MPI.SUM, root=0)
    if rank == 0:
        print(f"  Total particles: {total_n:,}")
    
    # Apply BCM displacements
    if rank == 0:
        print(f"\n[4/5] Applying BCM displacements to {n_halos} halos...")
    
    redshift = 0.0
    displaced_coords = local_coords.copy()
    
    # Each rank applies BCM to its local particles for all halos
    for i in range(n_halos):
        if rank == 0 and i % 100 == 0:
            print(f"    Halo {i+1}/{n_halos}...")
        
        displaced_coords = apply_bcm_to_region(
            displaced_coords, local_masses,
            halo_positions[i], halo_masses[i], halo_radii[i],
            args.radius_factor, redshift, displacement_model, cosmo_dict
        )
    
    # Project to 2D
    if rank == 0:
        print(f"\n[5/5] Projecting to 2D grid ({args.grid_size}x{args.grid_size})...")
    
    local_field = project_to_2d(displaced_coords, local_masses, args.grid_size, BOX_SIZE)
    
    # Reduce across ranks
    if rank == 0:
        global_field = np.zeros((args.grid_size, args.grid_size), dtype=np.float32)
    else:
        global_field = None
    
    comm.Reduce(local_field, global_field, op=MPI.SUM, root=0)
    
    # Save
    if rank == 0:
        output_dir = args.output_dir or os.path.join(
            CACHE_BASE, f'L205n{args.sim_res}TNG', f'snap{args.snap:03d}', 'projected'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        model_name_lower = args.bcm_model.lower()
        mass_str = f"M{args.mass_min}".replace('.', 'p')
        output_file = os.path.join(output_dir, f'bcm_{model_name_lower}_{mass_str}.npz')
        
        # Handle NaN
        global_field = np.nan_to_num(global_field, nan=0.0)
        
        np.savez_compressed(output_file, field=global_field)
        
        print(f"\nSaved: {output_file}")
        print(f"  Shape: {global_field.shape}")
        print(f"  Range: {global_field.min():.2e} - {global_field.max():.2e}")
        print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
