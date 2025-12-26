#!/usr/bin/env python
"""
Compute BCM statistics using particle cache.

This applies BCM displacement to DMO particles and computes:
- Density profiles around DMO halo centers
- Mass enclosed at various radii

Uses the DMO particle cache for fast particle lookup.

Usage:
    mpirun -np 4 python compute_bcm_statistics.py --sim-res 625 --snap 99 --bcm-model Arico20
"""

import numpy as np
import h5py
import argparse
import os
import sys
import time
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from mpi4py import MPI

import pyccl as ccl
import BaryonForge as bfg

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
        'dmo_mass': 0.0047271638660809,
    },
    1250: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG_DM/output',
        'dmo_mass': 0.0378173109,
    },
    625: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG_DM/output',
        'dmo_mass': 0.3025384873,
    },
}

CACHE_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields'
BOX_SIZE = 205.0  # Mpc/h
MASS_UNIT = 1e10  # Convert to Msun/h

# BCM parameters
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
    ),
}


def build_cosmodict(cosmo):
    """Extract cosmological parameters from pyccl Cosmology object."""
    return {
        'Omega_m': cosmo.cosmo.params.Omega_m,
        'Omega_b': cosmo.cosmo.params.Omega_b,
        'sigma8': cosmo.cosmo.params.sigma8,
        'h': cosmo.cosmo.params.h,
        'n_s': cosmo.cosmo.params.n_s,
        'w0': cosmo.cosmo.params.w0,
        'wa': cosmo.cosmo.params.wa,
    }


def setup_bcm_model(model_name, redshift=0):
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
        z_min=0, z_max=3, z_linear_sampling=True,
        N_samples_R=10000, Rdelta_sampling=True
    )
    return Displacement


class DistributedParticleData:
    """Manages distributed particle data across MPI ranks."""
    
    def __init__(self, snapshot: int, sim_res: int, verbose: bool = True):
        self.snapshot = snapshot
        self.sim_res = sim_res
        self.verbose = verbose and (rank == 0)
        self.sim_config = SIM_PATHS[sim_res]
        
        self.local_ids = None
        self.local_coords = None
        self.local_masses = None
        self.local_id_to_idx = None
        
        self._load_local_data()
    
    def _load_local_data(self):
        """Load snapshot files assigned to this rank."""
        basePath = self.sim_config['dmo']
        dm_mass = self.sim_config['dmo_mass']
        
        snap_dir = f"{basePath}/snapdir_{self.snapshot:03d}/"
        all_files = sorted(glob.glob(f"{snap_dir}/snap_{self.snapshot:03d}.*.hdf5"))
        n_files = len(all_files)
        
        if self.verbose:
            print(f"\n[DMO] Loading distributed particle data...")
            print(f"  Total snapshot files: {n_files}")
        
        my_files = [f for i, f in enumerate(all_files) if i % size == rank]
        
        t0 = time.time()
        
        coords_list = []
        masses_list = []
        ids_list = []
        
        for filepath in my_files:
            with h5py.File(filepath, 'r') as f:
                pt_key = 'PartType1'
                if pt_key not in f:
                    continue
                
                n_part = f[pt_key]['Coordinates'].shape[0]
                if n_part == 0:
                    continue
                
                coords = f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3  # kpc/h -> Mpc/h
                coords_list.append(coords)
                
                pids = f[pt_key]['ParticleIDs'][:]
                ids_list.append(pids)
                
                m = np.full(n_part, dm_mass * MASS_UNIT, dtype=np.float32)
                masses_list.append(m)
        
        if coords_list:
            self.local_coords = np.concatenate(coords_list)
            self.local_masses = np.concatenate(masses_list)
            self.local_ids = np.concatenate(ids_list)
        else:
            self.local_coords = np.zeros((0, 3), dtype=np.float32)
            self.local_masses = np.zeros(0, dtype=np.float32)
            self.local_ids = np.zeros(0, dtype=np.int64)
        
        self.local_id_to_idx = {int(pid): i for i, pid in enumerate(self.local_ids)}
        
        local_n = len(self.local_ids)
        total_n = comm.reduce(local_n, op=MPI.SUM, root=0)
        
        if self.verbose:
            print(f"  Rank 0: {local_n:,} particles")
            print(f"  Total particles: {total_n:,}")
            print(f"  Load time: {time.time()-t0:.1f}s")
    
    def query_particles(self, particle_ids: np.ndarray, halo_center: np.ndarray, 
                        r200: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Query particle coordinates, masses, and radii for given IDs."""
        if rank == 0:
            n_query = len(particle_ids)
        else:
            n_query = None
        n_query = comm.bcast(n_query, root=0)
        
        if rank != 0:
            particle_ids = np.empty(n_query, dtype=np.int64)
        comm.Bcast(particle_ids, root=0)
        
        halo_center = comm.bcast(halo_center, root=0)
        r200 = comm.bcast(r200, root=0)
        
        local_indices = []
        for pid in particle_ids:
            if int(pid) in self.local_id_to_idx:
                local_indices.append(self.local_id_to_idx[int(pid)])
        
        local_indices = np.array(local_indices, dtype=np.int64)
        
        if len(local_indices) > 0:
            local_coords = self.local_coords[local_indices]
            local_masses = self.local_masses[local_indices]
            
            dx = local_coords - halo_center
            dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
            dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
            local_radii = np.linalg.norm(dx, axis=1)
            local_radii_r200 = local_radii / r200
        else:
            local_coords = np.zeros((0, 3), dtype=np.float32)
            local_masses = np.zeros(0, dtype=np.float32)
            local_radii_r200 = np.zeros(0, dtype=np.float32)
        
        all_coords = comm.gather(local_coords, root=0)
        all_masses = comm.gather(local_masses, root=0)
        all_radii_r200 = comm.gather(local_radii_r200, root=0)
        
        if rank == 0:
            coords = np.concatenate(all_coords) if all_coords else np.zeros((0, 3), dtype=np.float32)
            masses = np.concatenate(all_masses) if all_masses else np.zeros(0, dtype=np.float32)
            radii_r200 = np.concatenate(all_radii_r200) if all_radii_r200 else np.zeros(0, dtype=np.float32)
            return coords, masses, radii_r200
        else:
            return np.zeros((0, 3)), np.zeros(0), np.zeros(0)


def apply_bcm_displacement(coords, masses, halo_center, halo_mass, halo_r200, 
                           redshift, displacement_model, cosmo_dict):
    """Apply BCM displacement to particles around a halo."""
    if len(coords) == 0:
        return coords.copy()
    
    # Create particle snapshot for BaryonForge
    Snap = bfg.ParticleSnapshot(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        L=BOX_SIZE / h, redshift=redshift, cosmo=cosmo_dict,
        M=masses[0] if len(masses) > 0 else 1e10  # Particle mass
    )
    
    # Create halo catalog with single halo
    HCat = bfg.HaloNDCatalog(
        x=np.array([halo_center[0]]),
        y=np.array([halo_center[1]]),
        z=np.array([halo_center[2]]),
        M_200c=np.array([halo_mass]),
        redshift=redshift,
        cosmo=cosmo_dict,
        is_central=np.array([True])
    )
    
    # Apply displacement
    try:
        dx, dy, dz = displacement_model.displace(Snap, HCat, verbose=False)
        displaced = coords.copy()
        displaced[:, 0] += dx
        displaced[:, 1] += dy
        displaced[:, 2] += dz
        
        # Apply periodic boundary
        displaced = displaced % BOX_SIZE
        
        return displaced
    except Exception as e:
        if rank == 0:
            print(f"    Warning: BCM displacement failed: {e}")
        return coords.copy()


def compute_profile(masses, radii_r200, radii_mult):
    """Compute mass enclosed at multiple radii."""
    n_radii = len(radii_mult)
    m_enclosed = np.zeros(n_radii, dtype=np.float64)
    
    for i, r_mult in enumerate(radii_mult):
        mask = radii_r200 <= r_mult
        m_enclosed[i] = np.sum(masses[mask])
    
    return m_enclosed


def main():
    parser = argparse.ArgumentParser(description='Compute BCM statistics')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, default=625, choices=[625, 1250, 2500])
    parser.add_argument('--bcm-model', type=str, default='Arico20',
                       choices=['Arico20', 'Schneider19', 'Schneider25'])
    parser.add_argument('--mass-min', type=float, default=12.5,
                       help='Minimum log10(M200c/Msun/h)')
    parser.add_argument('--mass-max', type=float, default=None,
                       help='Maximum log10(M200c/Msun/h)')
    parser.add_argument('--output-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Radii at which to compute statistics
    radii_mult = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    n_radii = len(radii_mult)
    
    if rank == 0:
        print("=" * 70)
        print(f"BCM STATISTICS COMPUTATION - {args.bcm_model}")
        print("=" * 70)
        print(f"Snapshot: {args.snap}")
        print(f"Resolution: L205n{args.sim_res}TNG")
        print(f"BCM Model: {args.bcm_model}")
        print(f"Mass range: 10^{args.mass_min} - 10^{args.mass_max or '∞'} Msun/h")
        print("=" * 70)
        sys.stdout.flush()
    
    t_start = time.time()
    
    # Setup BCM model (rank 0 only initially, then broadcast)
    if rank == 0:
        print(f"\n[1/5] Setting up BCM model...")
        displacement_model = setup_bcm_model(args.bcm_model)
        cosmo_dict = build_cosmodict(COSMO)
        print(f"  BCM model ready")
    else:
        displacement_model = None
        cosmo_dict = None
    
    # Broadcast BCM model setup
    displacement_model = comm.bcast(displacement_model, root=0)
    cosmo_dict = comm.bcast(cosmo_dict, root=0)
    
    # Load particle cache
    cache_file = os.path.join(
        CACHE_BASE, f'L205n{args.sim_res}TNG',
        'particle_cache', f'cache_snap{args.snap:03d}.h5'
    )
    
    if rank == 0:
        print(f"\n[2/5] Loading halo cache: {cache_file}")
        
        with h5py.File(cache_file, 'r') as f:
            halo_info = f['halo_info']
            dmo_indices = halo_info['halo_indices'][:]
            positions = halo_info['positions_dmo'][:]
            radii = halo_info['radii_dmo'][:]
            masses = halo_info['masses'][:]
            
            # Filter by mass
            log_masses = np.log10(masses)
            mass_mask = log_masses >= args.mass_min
            if args.mass_max:
                mass_mask &= log_masses <= args.mass_max
            
            halo_cache_indices = np.where(mass_mask)[0]
            halo_positions = positions[mass_mask]
            halo_radii = radii[mass_mask]
            halo_masses = masses[mass_mask]
            halo_log_masses = log_masses[mass_mask]
            
            n_halos = len(halo_cache_indices)
            print(f"  Total halos in mass range: {n_halos}")
            
            # Load DMO particle IDs
            dmo_particle_ids = []
            for cache_idx in halo_cache_indices:
                dmo_ids = f[f'dmo/halo_{cache_idx}'][:]
                dmo_particle_ids.append(dmo_ids)
        
        print(f"  Loaded particle IDs for {n_halos} halos")
    else:
        n_halos = None
        halo_cache_indices = None
        halo_positions = None
        halo_radii = None
        halo_masses = None
        halo_log_masses = None
        dmo_particle_ids = None
    
    # Broadcast halo info
    n_halos = comm.bcast(n_halos, root=0)
    halo_positions = comm.bcast(halo_positions, root=0)
    halo_radii = comm.bcast(halo_radii, root=0)
    halo_masses = comm.bcast(halo_masses, root=0)
    halo_log_masses = comm.bcast(halo_log_masses, root=0)
    dmo_particle_ids = comm.bcast(dmo_particle_ids, root=0)
    
    # Load distributed particle data
    if rank == 0:
        print(f"\n[3/5] Loading distributed DMO particle data...")
    
    dmo_data = DistributedParticleData(args.snap, args.sim_res, verbose=True)
    
    comm.Barrier()
    
    # Get redshift from snapshot
    # For now assume z=0 for snap 99, could read from header
    redshift = 0.0 if args.snap >= 99 else 0.5  # Approximate
    
    # Process halos
    if rank == 0:
        print(f"\n[4/5] Processing {n_halos} halos with BCM...")
        
        results = {
            'log_masses': halo_log_masses.astype(np.float32),
            'positions': halo_positions.astype(np.float32),
            'radii': halo_radii.astype(np.float32),
            'm_dmo': np.zeros((n_halos, n_radii), dtype=np.float64),
            'm_bcm': np.zeros((n_halos, n_radii), dtype=np.float64),
            'ratio_bcm_dmo': np.zeros((n_halos, n_radii), dtype=np.float32),
        }
    
    t_proc = time.time()
    
    for i in range(n_halos):
        if rank == 0:
            if i % 20 == 0:
                print(f"    Halo {i+1}/{n_halos} (M={halo_log_masses[i]:.2f})...")
                sys.stdout.flush()
            
            dmo_ids = dmo_particle_ids[i]
            pos = halo_positions[i]
            r200 = halo_radii[i]
            mass = halo_masses[i]
        else:
            dmo_ids = None
            pos = None
            r200 = None
            mass = None
        
        # Query DMO particles
        dmo_ids = comm.bcast(dmo_ids, root=0)
        pos = comm.bcast(pos, root=0)
        r200 = comm.bcast(r200, root=0)
        mass = comm.bcast(mass, root=0)
        
        dmo_coords, dmo_masses, dmo_radii_r200 = dmo_data.query_particles(
            dmo_ids, pos, r200
        )
        
        # Compute DMO profile and BCM profile (rank 0 only)
        if rank == 0:
            # DMO profile
            m_dmo = compute_profile(dmo_masses, dmo_radii_r200, radii_mult)
            results['m_dmo'][i] = m_dmo
            
            # Apply BCM displacement
            if len(dmo_coords) > 0:
                bcm_coords = apply_bcm_displacement(
                    dmo_coords, dmo_masses, pos, mass, r200,
                    redshift, displacement_model, cosmo_dict
                )
                
                # Recompute radii after displacement
                dx = bcm_coords - pos
                dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
                dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
                bcm_radii = np.linalg.norm(dx, axis=1)
                bcm_radii_r200 = bcm_radii / r200
                
                m_bcm = compute_profile(dmo_masses, bcm_radii_r200, radii_mult)
                results['m_bcm'][i] = m_bcm
                
                # Ratio
                with np.errstate(divide='ignore', invalid='ignore'):
                    results['ratio_bcm_dmo'][i] = np.where(
                        m_dmo > 0, m_bcm / m_dmo, 0
                    )
    
    if rank == 0:
        print(f"  Processing time: {time.time()-t_proc:.1f}s")
    
    # Save results
    if rank == 0:
        print(f"\n[5/5] Saving results...")
        
        output_dir = args.output_dir or os.path.join(
            CACHE_BASE, f'L205n{args.sim_res}TNG', 'analysis'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(
            output_dir, f'bcm_statistics_{args.bcm_model}_snap{args.snap:03d}.h5'
        )
        
        with h5py.File(output_file, 'w') as f:
            f.attrs['snapshot'] = args.snap
            f.attrs['sim_res'] = args.sim_res
            f.attrs['bcm_model'] = args.bcm_model
            f.attrs['mass_min'] = args.mass_min
            f.attrs['mass_max'] = args.mass_max if args.mass_max else -1
            f.attrs['radii_r200'] = radii_mult
            f.attrs['n_halos'] = n_halos
            
            for key, arr in results.items():
                f.create_dataset(key, data=arr, compression='gzip')
        
        print(f"  Saved: {output_file}")
        print(f"\n{'='*70}")
        print(f"COMPLETE - Total time: {time.time()-t_start:.1f}s")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
