#!/usr/bin/env python
"""
Generate density profiles using pre-computed particle cache.

This is faster than computing profiles from scratch because we skip KDTree queries.
The cache stores particle IDs for each halo at multiple radii (up to 5×R200).

Key advantage: Once particles are loaded, we can quickly compute profiles with
different binning schemes, mass cuts, or radius ranges without re-querying.

Usage:
    mpirun -np 4 python generate_profiles_cached.py --snap 99 --sim-res 625

Output:
    - Stacked profiles for DMO, Hydro, Replace in mass bins
    - Individual halo profiles (optional)
"""

import numpy as np
import h5py
import argparse
import os
import sys
import time
import glob

from mpi4py import MPI

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

CACHE_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields'
BOX_SIZE = 205.0
MASS_UNIT = 1e10


# ============================================================================
# Distributed Particle Data (reuse from statistics script)
# ============================================================================

class DistributedParticleData:
    """Load particles distributed across MPI ranks."""
    
    def __init__(self, snapshot: int, sim_res: int, mode: str, verbose: bool = True):
        self.snapshot = snapshot
        self.sim_res = sim_res
        self.mode = mode
        self.verbose = verbose and (rank == 0)
        self.sim_config = SIM_PATHS[sim_res]
        
        self.local_ids = None
        self.local_coords = None
        self.local_masses = None
        self.local_types = None
        self.local_id_to_idx = None
        
        self._load()
    
    def _load(self):
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
        
        if self.verbose:
            print(f"\n[{self.mode.upper()}] Loading distributed particle data...")
            print(f"  Files: {len(all_files)} total, {len(my_files)} for rank 0")
        
        t0 = time.time()
        
        coords_list, masses_list, ids_list, types_list = [], [], [], []
        
        for filepath in my_files:
            with h5py.File(filepath, 'r') as f:
                for ptype in particle_types:
                    pt_key = f'PartType{ptype}'
                    if pt_key not in f or f[pt_key]['Coordinates'].shape[0] == 0:
                        continue
                    
                    n_part = f[pt_key]['Coordinates'].shape[0]
                    coords_list.append(f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3)
                    ids_list.append(f[pt_key]['ParticleIDs'][:])
                    
                    if 'Masses' in f[pt_key]:
                        masses_list.append(f[pt_key]['Masses'][:].astype(np.float32) * MASS_UNIT)
                    else:
                        masses_list.append(np.full(n_part, dm_mass * MASS_UNIT, dtype=np.float32))
                    
                    types_list.append(np.full(n_part, ptype, dtype=np.int8))
        
        if coords_list:
            self.local_coords = np.concatenate(coords_list)
            self.local_masses = np.concatenate(masses_list)
            self.local_ids = np.concatenate(ids_list)
            self.local_types = np.concatenate(types_list)
        else:
            self.local_coords = np.zeros((0, 3), dtype=np.float32)
            self.local_masses = np.zeros(0, dtype=np.float32)
            self.local_ids = np.zeros(0, dtype=np.int64)
            self.local_types = np.zeros(0, dtype=np.int8)
        
        self.local_id_to_idx = {int(pid): i for i, pid in enumerate(self.local_ids)}
        
        if self.verbose:
            print(f"  Rank 0: {len(self.local_ids):,} particles")
            print(f"  Load time: {time.time()-t0:.1f}s")
    
    def query_particles(self, particle_ids: np.ndarray, halo_center: np.ndarray, r200: float):
        """Query particle data and compute radii (collective operation)."""
        
        # Broadcast query from rank 0
        n_query = comm.bcast(len(particle_ids) if rank == 0 else None, root=0)
        if rank != 0:
            particle_ids = np.empty(n_query, dtype=np.int64)
        comm.Bcast(particle_ids, root=0)
        
        halo_center = comm.bcast(halo_center, root=0)
        r200 = comm.bcast(r200, root=0)
        
        # Find local matches
        local_idx = [self.local_id_to_idx[int(pid)] for pid in particle_ids 
                     if int(pid) in self.local_id_to_idx]
        
        if local_idx:
            local_idx = np.array(local_idx)
            local_coords = self.local_coords[local_idx]
            local_masses = self.local_masses[local_idx]
            local_types = self.local_types[local_idx]
            
            # Compute radii
            dx = local_coords - halo_center
            dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
            dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
            local_radii = np.linalg.norm(dx, axis=1)
            local_radii_r200 = local_radii / r200
        else:
            local_masses = np.zeros(0, dtype=np.float32)
            local_types = np.zeros(0, dtype=np.int8)
            local_radii_r200 = np.zeros(0, dtype=np.float32)
        
        # Gather to rank 0
        all_masses = comm.gather(local_masses, root=0)
        all_types = comm.gather(local_types, root=0)
        all_radii = comm.gather(local_radii_r200, root=0)
        
        if rank == 0:
            return (np.concatenate(all_masses), np.concatenate(all_types), 
                    np.concatenate(all_radii))
        return np.zeros(0), np.zeros(0, dtype=np.int8), np.zeros(0)


# ============================================================================
# Profile Computation
# ============================================================================

def compute_density_profile(masses, radii_r200, r_bins):
    """Compute spherically-averaged density profile."""
    # Bin masses by radius
    mass_profile, _ = np.histogram(radii_r200, bins=r_bins, weights=masses)
    count_profile, _ = np.histogram(radii_r200, bins=r_bins)
    
    # Shell volumes (in units of R200^3)
    volumes = 4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    
    # Density = mass / volume
    density = np.where(volumes > 0, mass_profile / volumes, 0)
    
    return density, mass_profile, count_profile


def compute_cumulative_mass(masses, radii_r200, r_bins):
    """Compute cumulative mass profile M(<r)."""
    cumulative = np.zeros(len(r_bins) - 1)
    for i, r_max in enumerate(r_bins[1:]):
        mask = radii_r200 <= r_max
        cumulative[i] = np.sum(masses[mask])
    return cumulative


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate profiles from cache')
    parser.add_argument('--snap', type=int, required=True)
    parser.add_argument('--sim-res', type=int, default=625, choices=[625, 1250, 2500])
    parser.add_argument('--mass-min', type=float, default=12.0)
    parser.add_argument('--n-bins', type=int, default=30, help='Number of radial bins')
    parser.add_argument('--r-max', type=float, default=5.0, help='Max radius in R200')
    parser.add_argument('--output-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Radial bins
    r_bins = np.logspace(-2, np.log10(args.r_max), args.n_bins + 1)
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    
    # Mass bins for stacking
    mass_bins = np.array([12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 16.0])
    n_mass_bins = len(mass_bins) - 1
    
    if rank == 0:
        print("=" * 70)
        print("CACHED PROFILE GENERATION")
        print("=" * 70)
        print(f"Snapshot: {args.snap}")
        print(f"Resolution: L205n{args.sim_res}TNG")
        print(f"Mass minimum: 10^{args.mass_min}")
        print(f"Radial bins: {args.n_bins} from 0.01 to {args.r_max} R200")
        print(f"MPI ranks: {size}")
        print("=" * 70)
        sys.stdout.flush()
    
    t_start = time.time()
    
    # Load cache
    cache_file = os.path.join(
        CACHE_BASE, f'L205n{args.sim_res}TNG',
        'particle_cache', f'cache_snap{args.snap:03d}.h5'
    )
    
    if rank == 0:
        print(f"\n[1/4] Loading cache...")
        sys.stdout.flush()
    
    with h5py.File(cache_file, 'r') as f:
        halo_info = f['halo_info']
        all_masses = halo_info['masses'][:]
        all_positions_dmo = halo_info['positions_dmo'][:]
        all_radii_dmo = halo_info['radii_dmo'][:]
        all_positions_hydro = halo_info['positions_hydro'][:]
        all_radii_hydro = halo_info['radii_hydro'][:]
        all_log_masses = np.log10(all_masses)
        
        mass_mask = all_log_masses >= args.mass_min
        selected = np.where(mass_mask)[0]
        n_halos = len(selected)
        
        if rank == 0:
            print(f"  Halos above 10^{args.mass_min}: {n_halos}")
        
        # Load particle IDs
        # DMO: particles around DMO halo center
        # Hydro: particles around HYDRO halo center (hydro_at_hydro)
        dmo_ids = [f[f'dmo/halo_{i}'][:] for i in selected]
        hydro_ids = [f[f'hydro_at_hydro/halo_{i}'][:] for i in selected]
        halo_log_masses = all_log_masses[selected]
        halo_positions_dmo = all_positions_dmo[selected]
        halo_positions_hydro = all_positions_hydro[selected]
        halo_radii_dmo = all_radii_dmo[selected]
        halo_radii_hydro = all_radii_hydro[selected]
    
    # Load distributed particle data
    if rank == 0:
        print(f"\n[2/4] Loading particles...")
        sys.stdout.flush()
    
    dmo_data = DistributedParticleData(args.snap, args.sim_res, 'dmo')
    hydro_data = DistributedParticleData(args.snap, args.sim_res, 'hydro')
    
    comm.Barrier()
    
    # Initialize stacked profiles
    if rank == 0:
        print(f"\n[3/4] Computing profiles for {n_halos} halos...")
        sys.stdout.flush()
        
        stacked_dmo = np.zeros((n_mass_bins, args.n_bins))
        stacked_hydro = np.zeros((n_mass_bins, args.n_bins))
        stacked_hydro_dm = np.zeros((n_mass_bins, args.n_bins))  # DM only in hydro
        stacked_hydro_gas = np.zeros((n_mass_bins, args.n_bins))
        stacked_hydro_stars = np.zeros((n_mass_bins, args.n_bins))
        stacked_counts = np.zeros(n_mass_bins, dtype=int)
    
    # Process halos
    for i in range(n_halos):
        if rank == 0 and i % 50 == 0:
            print(f"    Halo {i+1}/{n_halos}...")
            sys.stdout.flush()
        
        log_m = halo_log_masses[i]
        pos_dmo = halo_positions_dmo[i]
        pos_hydro = halo_positions_hydro[i]
        r200_dmo = halo_radii_dmo[i]
        r200_hydro = halo_radii_hydro[i]
        
        # Find mass bin
        mass_bin = np.digitize(log_m, mass_bins) - 1
        if mass_bin < 0 or mass_bin >= n_mass_bins:
            continue
        
        # Query DMO particles - centered on DMO halo
        dmo_masses_i, _, dmo_radii_i = dmo_data.query_particles(
            dmo_ids[i], pos_dmo, r200_dmo
        )
        
        # Query Hydro particles - centered on HYDRO halo
        hydro_masses_i, hydro_types_i, hydro_radii_i = hydro_data.query_particles(
            hydro_ids[i], pos_hydro, r200_hydro
        )
        
        if rank == 0:
            # Compute profiles
            dmo_dens, _, _ = compute_density_profile(dmo_masses_i, dmo_radii_i, r_bins)
            hydro_dens, _, _ = compute_density_profile(hydro_masses_i, hydro_radii_i, r_bins)
            
            # By particle type
            dm_mask = hydro_types_i == 1
            gas_mask = hydro_types_i == 0
            star_mask = hydro_types_i == 4
            
            dm_dens, _, _ = compute_density_profile(hydro_masses_i[dm_mask], 
                                                     hydro_radii_i[dm_mask], r_bins)
            gas_dens, _, _ = compute_density_profile(hydro_masses_i[gas_mask], 
                                                      hydro_radii_i[gas_mask], r_bins)
            star_dens, _, _ = compute_density_profile(hydro_masses_i[star_mask], 
                                                       hydro_radii_i[star_mask], r_bins)
            
            # Stack
            stacked_dmo[mass_bin] += dmo_dens
            stacked_hydro[mass_bin] += hydro_dens
            stacked_hydro_dm[mass_bin] += dm_dens
            stacked_hydro_gas[mass_bin] += gas_dens
            stacked_hydro_stars[mass_bin] += star_dens
            stacked_counts[mass_bin] += 1
    
    # Save results
    if rank == 0:
        print(f"\n[4/4] Saving results...")
        
        # Normalize stacked profiles
        for i in range(n_mass_bins):
            if stacked_counts[i] > 0:
                stacked_dmo[i] /= stacked_counts[i]
                stacked_hydro[i] /= stacked_counts[i]
                stacked_hydro_dm[i] /= stacked_counts[i]
                stacked_hydro_gas[i] /= stacked_counts[i]
                stacked_hydro_stars[i] /= stacked_counts[i]
        
        output_dir = args.output_dir or os.path.join(
            CACHE_BASE, f'L205n{args.sim_res}TNG', 'profiles'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f'profiles_snap{args.snap:03d}.h5')
        
        with h5py.File(output_file, 'w') as f:
            f.attrs['snapshot'] = args.snap
            f.attrs['sim_res'] = args.sim_res
            f.attrs['n_bins'] = args.n_bins
            f.attrs['r_max'] = args.r_max
            
            f.create_dataset('r_bins', data=r_bins)
            f.create_dataset('r_centers', data=r_centers)
            f.create_dataset('mass_bins', data=mass_bins)
            f.create_dataset('stacked_counts', data=stacked_counts)
            
            f.create_dataset('stacked_dmo', data=stacked_dmo)
            f.create_dataset('stacked_hydro', data=stacked_hydro)
            f.create_dataset('stacked_hydro_dm', data=stacked_hydro_dm)
            f.create_dataset('stacked_hydro_gas', data=stacked_hydro_gas)
            f.create_dataset('stacked_hydro_stars', data=stacked_hydro_stars)
        
        print(f"  Saved: {output_file}")
        print(f"\n{'='*70}")
        print(f"COMPLETE - Total time: {time.time()-t_start:.1f}s")
        print(f"{'='*70}")
        
        # Summary
        print("\nHalos per mass bin:")
        for i in range(n_mass_bins):
            print(f"  [{mass_bins[i]:.1f}, {mass_bins[i+1]:.1f}): {stacked_counts[i]}")


if __name__ == '__main__':
    main()
