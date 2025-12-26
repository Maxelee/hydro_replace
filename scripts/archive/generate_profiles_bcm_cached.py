#!/usr/bin/env python
"""
Generate BCM density profiles using particle cache.

Applies BCM displacement to DMO particles and computes stacked
density profiles by mass bin. Profiles are centered on DMO halo centers.

Usage:
    mpirun -np 4 python generate_profiles_bcm_cached.py --sim-res 625 --snap 99 --bcm-model Arico20
"""

import numpy as np
import h5py
import argparse
import os
import sys
import time
import glob
from mpi4py import MPI

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


class DistributedParticleData:
    def __init__(self, snapshot, sim_res, verbose=True):
        self.snapshot = snapshot
        self.sim_res = sim_res
        self.verbose = verbose and (rank == 0)
        self.sim_config = SIM_PATHS[sim_res]
        self._load_local_data()
    
    def _load_local_data(self):
        basePath = self.sim_config['dmo']
        dm_mass = self.sim_config['dmo_mass']
        
        snap_dir = f"{basePath}/snapdir_{self.snapshot:03d}/"
        all_files = sorted(glob.glob(f"{snap_dir}/snap_{self.snapshot:03d}.*.hdf5"))
        my_files = [f for i, f in enumerate(all_files) if i % size == rank]
        
        coords_list, masses_list, ids_list = [], [], []
        
        for filepath in my_files:
            with h5py.File(filepath, 'r') as f:
                if 'PartType1' not in f:
                    continue
                n_part = f['PartType1']['Coordinates'].shape[0]
                if n_part == 0:
                    continue
                coords_list.append(f['PartType1']['Coordinates'][:].astype(np.float32) / 1e3)
                ids_list.append(f['PartType1']['ParticleIDs'][:])
                masses_list.append(np.full(n_part, dm_mass * MASS_UNIT, dtype=np.float32))
        
        if coords_list:
            self.local_coords = np.concatenate(coords_list)
            self.local_masses = np.concatenate(masses_list)
            self.local_ids = np.concatenate(ids_list)
        else:
            self.local_coords = np.zeros((0, 3), dtype=np.float32)
            self.local_masses = np.zeros(0, dtype=np.float32)
            self.local_ids = np.zeros(0, dtype=np.int64)
        
        self.local_id_to_idx = {int(pid): i for i, pid in enumerate(self.local_ids)}
    
    def query_particles(self, particle_ids, halo_center, r200):
        n_query = comm.bcast(len(particle_ids) if rank == 0 else None, root=0)
        if rank != 0:
            particle_ids = np.empty(n_query, dtype=np.int64)
        comm.Bcast(particle_ids, root=0)
        halo_center = comm.bcast(halo_center, root=0)
        r200 = comm.bcast(r200, root=0)
        
        local_indices = [self.local_id_to_idx[int(pid)] for pid in particle_ids if int(pid) in self.local_id_to_idx]
        local_indices = np.array(local_indices, dtype=np.int64)
        
        if len(local_indices) > 0:
            local_coords = self.local_coords[local_indices]
            local_masses = self.local_masses[local_indices]
        else:
            local_coords = np.zeros((0, 3), dtype=np.float32)
            local_masses = np.zeros(0, dtype=np.float32)
        
        all_coords = comm.gather(local_coords, root=0)
        all_masses = comm.gather(local_masses, root=0)
        
        if rank == 0:
            return (np.concatenate(all_coords) if all_coords else np.zeros((0, 3), dtype=np.float32),
                    np.concatenate(all_masses) if all_masses else np.zeros(0, dtype=np.float32))
        return np.zeros((0, 3)), np.zeros(0)


def apply_bcm_displacement(coords, masses, halo_center, halo_mass, r200, redshift, displacement_model, cosmo_dict):
    if len(coords) == 0:
        return coords.copy()
    
    try:
        Snap = bfg.ParticleSnapshot(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            L=BOX_SIZE / h, redshift=redshift, cosmo=cosmo_dict,
            M=masses[0] if len(masses) > 0 else 1e10
        )
        HCat = bfg.HaloNDCatalog(
            x=np.array([halo_center[0]]), y=np.array([halo_center[1]]), z=np.array([halo_center[2]]),
            M_200c=np.array([halo_mass]), redshift=redshift, cosmo=cosmo_dict, is_central=np.array([True])
        )
        dx, dy, dz = displacement_model.displace(Snap, HCat, verbose=False)
        displaced = coords.copy()
        displaced[:, 0] += dx
        displaced[:, 1] += dy
        displaced[:, 2] += dz
        return displaced % BOX_SIZE
    except:
        return coords.copy()


def compute_radial_profile(coords, masses, halo_center, r200, r_bins):
    dx = coords - halo_center
    dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
    dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
    r = np.linalg.norm(dx, axis=1)
    r_norm = r / r200
    
    density = np.zeros(len(r_bins) - 1, dtype=np.float64)
    for j in range(len(r_bins) - 1):
        mask = (r_norm >= r_bins[j]) & (r_norm < r_bins[j+1])
        if mask.any():
            shell_volume = 4/3 * np.pi * r200**3 * (r_bins[j+1]**3 - r_bins[j]**3)
            density[j] = np.sum(masses[mask]) / shell_volume
    return density


def main():
    parser = argparse.ArgumentParser(description='Generate BCM density profiles')
    parser.add_argument('--snap', type=int, required=True)
    parser.add_argument('--sim-res', type=int, default=625, choices=[625, 1250, 2500])
    parser.add_argument('--bcm-model', type=str, default='Arico20', choices=['Arico20', 'Schneider19', 'Schneider25'])
    parser.add_argument('--mass-min', type=float, default=12.5)
    parser.add_argument('--mass-max', type=float, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    
    # Profile configuration
    r_bins = np.logspace(-2, np.log10(5), 31)
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    mass_bins = np.array([12.5, 13.0, 13.5, 14.0, 15.0])
    n_mass_bins = len(mass_bins) - 1
    
    if rank == 0:
        print("=" * 70)
        print(f"BCM PROFILE GENERATION - {args.bcm_model}")
        print("=" * 70)
        print(f"Snapshot: {args.snap}, Resolution: L205n{args.sim_res}TNG")
        print(f"Mass range: 10^{args.mass_min} - 10^{args.mass_max or '∞'}")
        print("=" * 70)
    
    t_start = time.time()
    
    # Setup BCM
    if rank == 0:
        print("\n[1/4] Setting up BCM model...")
    displacement_model = setup_bcm_model(args.bcm_model)
    cosmo_dict = build_cosmodict(COSMO)
    displacement_model = comm.bcast(displacement_model, root=0)
    cosmo_dict = comm.bcast(cosmo_dict, root=0)
    
    # Load cache
    cache_file = os.path.join(CACHE_BASE, f'L205n{args.sim_res}TNG', 'particle_cache', f'cache_snap{args.snap:03d}.h5')
    
    if rank == 0:
        print(f"\n[2/4] Loading cache: {cache_file}")
        with h5py.File(cache_file, 'r') as f:
            halo_info = f['halo_info']
            positions = halo_info['positions_dmo'][:]
            radii = halo_info['radii_dmo'][:]
            masses = halo_info['masses'][:]
            log_masses = np.log10(masses)
            
            mass_mask = log_masses >= args.mass_min
            if args.mass_max:
                mass_mask &= log_masses <= args.mass_max
            
            selected = np.where(mass_mask)[0]
            halo_positions = positions[mass_mask]
            halo_radii = radii[mass_mask]
            halo_masses = masses[mass_mask]
            halo_log_masses = log_masses[mass_mask]
            
            dmo_particle_ids = [f[f'dmo/halo_{i}'][:] for i in selected]
        
        n_halos = len(selected)
        print(f"  {n_halos} halos in mass range")
    else:
        n_halos = halo_positions = halo_radii = halo_masses = halo_log_masses = dmo_particle_ids = None
    
    n_halos = comm.bcast(n_halos, root=0)
    halo_positions = comm.bcast(halo_positions, root=0)
    halo_radii = comm.bcast(halo_radii, root=0)
    halo_masses = comm.bcast(halo_masses, root=0)
    halo_log_masses = comm.bcast(halo_log_masses, root=0)
    dmo_particle_ids = comm.bcast(dmo_particle_ids, root=0)
    
    # Load particles
    if rank == 0:
        print(f"\n[3/4] Loading DMO particles...")
    dmo_data = DistributedParticleData(args.snap, args.sim_res)
    comm.Barrier()
    
    # Initialize stacked profiles
    if rank == 0:
        stacked_dmo = np.zeros((n_mass_bins, len(r_centers)), dtype=np.float64)
        stacked_bcm = np.zeros((n_mass_bins, len(r_centers)), dtype=np.float64)
        stacked_counts = np.zeros(n_mass_bins, dtype=np.int32)
    
    redshift = 0.0
    
    if rank == 0:
        print(f"\n[4/4] Processing {n_halos} halos...")
    
    for i in range(n_halos):
        if rank == 0 and i % 50 == 0:
            print(f"    Halo {i+1}/{n_halos}...")
        
        dmo_ids = comm.bcast(dmo_particle_ids[i] if rank == 0 else None, root=0)
        pos = comm.bcast(halo_positions[i] if rank == 0 else None, root=0)
        r200 = comm.bcast(halo_radii[i] if rank == 0 else None, root=0)
        mass = comm.bcast(halo_masses[i] if rank == 0 else None, root=0)
        log_mass = comm.bcast(halo_log_masses[i] if rank == 0 else None, root=0)
        
        dmo_coords, dmo_masses_arr = dmo_data.query_particles(dmo_ids, pos, r200)
        
        if rank == 0 and len(dmo_coords) > 10:
            # DMO profile
            prof_dmo = compute_radial_profile(dmo_coords, dmo_masses_arr, pos, r200, r_bins)
            
            # BCM profile
            bcm_coords = apply_bcm_displacement(dmo_coords, dmo_masses_arr, pos, mass, r200, redshift, displacement_model, cosmo_dict)
            prof_bcm = compute_radial_profile(bcm_coords, dmo_masses_arr, pos, r200, r_bins)
            
            # Add to stack
            bin_idx = np.searchsorted(mass_bins, log_mass) - 1
            if 0 <= bin_idx < n_mass_bins:
                stacked_dmo[bin_idx] += prof_dmo
                stacked_bcm[bin_idx] += prof_bcm
                stacked_counts[bin_idx] += 1
    
    # Average stacks
    if rank == 0:
        for b in range(n_mass_bins):
            if stacked_counts[b] > 0:
                stacked_dmo[b] /= stacked_counts[b]
                stacked_bcm[b] /= stacked_counts[b]
        
        # Save
        output_dir = args.output_dir or os.path.join(CACHE_BASE, f'L205n{args.sim_res}TNG', 'profiles')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'profiles_bcm_{args.bcm_model}_snap{args.snap:03d}.h5')
        
        with h5py.File(output_file, 'w') as f:
            f.attrs['snapshot'] = args.snap
            f.attrs['bcm_model'] = args.bcm_model
            f.attrs['mass_min'] = args.mass_min
            f.create_dataset('r_bins', data=r_bins)
            f.create_dataset('r_centers', data=r_centers)
            f.create_dataset('mass_bins', data=mass_bins)
            f.create_dataset('stacked_dmo', data=stacked_dmo)
            f.create_dataset('stacked_bcm', data=stacked_bcm)
            f.create_dataset('stacked_counts', data=stacked_counts)
        
        print(f"\nSaved: {output_file}")
        print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
