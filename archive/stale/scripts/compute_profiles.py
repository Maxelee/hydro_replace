#!/usr/bin/env python
"""
Compute density profiles for matched halos in DMO, Hydro, and BCM.

Based on working code from:
  /mnt/home/mlee1/Hydro_replacement/BCM.py

Usage:
    mpirun -np 32 python compute_profiles.py --snap 99
    python compute_profiles.py --snap 99 --n-halos 100
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
import pyccl as ccl
import BaryonForge as bfg

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'dmo_basePath': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output',
    'hydro_basePath': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output',
    'matches_file': '/mnt/home/mlee1/ceph/halo_matches.npz',
    'box_size': 205.0,  # Mpc/h
    'dmo_mass': 0.0047271638660809,  # 10^10 Msun/h
    'hydro_dm_mass': 0.00398342749867548,
    'mass_unit': 1e10,
    'output_dir': '/mnt/home/mlee1/ceph/hydro_replace_fields',
    'min_halo_mass': 10**12.8,
}

# Cosmology
COSMO = ccl.Cosmology(
    Omega_c=0.3089 - 0.0486, Omega_b=0.0486, h=0.6774,
    sigma8=0.8158, n_s=0.9649, matter_power_spectrum='linear'
)
h = COSMO.cosmo.params.h


def apply_periodic_boundary(dx, box_size):
    """Apply minimum image convention."""
    return dx - np.round(dx / box_size) * box_size


def compute_density_profile(pos, mass, center, radius, box_size, n_bins=100):
    """
    Compute spherically-averaged density profile.
    
    Returns
    -------
    bin_centers : array
        Radial bin centers in units of R_200
    density : array
        Density in each bin [Msun/h / (Mpc/h)^3]
    """
    bins = np.logspace(-3, np.log10(5), n_bins + 1)  # 0.001 to 5 R_200
    
    dx = pos - center
    dx = apply_periodic_boundary(dx, box_size)
    r = np.linalg.norm(dx, axis=1) / radius  # Normalize by R_200
    
    prof = binned_statistic(r, mass, statistic=np.sum, bins=bins)
    volumes = 4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3) * radius**3
    density = prof.statistic / volumes
    
    # Bin centers (geometric mean)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    
    return bin_centers, density


def load_particles_for_halo(basePath, snapNum, center, search_radius, config):
    """
    Load particles within search_radius of center from distributed files.
    
    This loads particles from files assigned to this MPI rank.
    """
    snap_dir = f"{basePath}/snapdir_{snapNum:03d}/"
    files = sorted(glob.glob(f"{snap_dir}/snap_{snapNum:03d}.*.hdf5"))
    my_files = [f for i, f in enumerate(files) if i % size == rank]
    
    coords_list = []
    masses_list = []
    
    for filepath in my_files:
        with h5py.File(filepath, 'r') as f:
            # DM particles
            if 'PartType1' in f:
                c = f['PartType1']['Coordinates'][:].astype(np.float32) / 1e3 / h
                
                # Quick box filter (not exact due to periodic boundaries)
                mask = np.all(np.abs(c - center) < search_radius * 2, axis=1)
                c = c[mask]
                
                if len(c) > 0:
                    coords_list.append(c)
                    if 'DM' in basePath or 'TNG_DM' in basePath:
                        masses_list.append(np.ones(len(c)) * config['dmo_mass'] * config['mass_unit'])
                    else:
                        masses_list.append(np.ones(len(c)) * config['hydro_dm_mass'] * config['mass_unit'])
            
            # Gas particles (hydro only)
            if 'PartType0' in f:
                c = f['PartType0']['Coordinates'][:].astype(np.float32) / 1e3 / h
                m = f['PartType0']['Masses'][:].astype(np.float32) * config['mass_unit']
                
                mask = np.all(np.abs(c - center) < search_radius * 2, axis=1)
                if mask.sum() > 0:
                    coords_list.append(c[mask])
                    masses_list.append(m[mask])
            
            # Stars (hydro only)
            if 'PartType4' in f:
                c = f['PartType4']['Coordinates'][:].astype(np.float32) / 1e3 / h
                m = f['PartType4']['Masses'][:].astype(np.float32) * config['mass_unit']
                
                mask = np.all(np.abs(c - center) < search_radius * 2, axis=1)
                if mask.sum() > 0:
                    coords_list.append(c[mask])
                    masses_list.append(m[mask])
    
    if len(coords_list) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
    
    return np.concatenate(coords_list), np.concatenate(masses_list)


def main():
    parser = argparse.ArgumentParser(description='Compute density profiles')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--mass-min', type=float, default=12.8,
                        help='log10(M_min) for halos')
    parser.add_argument('--n-halos', type=int, default=None,
                        help='Number of halos to process (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    config = CONFIG.copy()
    config['min_halo_mass'] = 10**args.mass_min
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    snap_dir = f"{config['output_dir']}/snap{args.snap:03d}"
    if rank == 0:
        os.makedirs(snap_dir, exist_ok=True)
        print("=" * 70)
        print(f"Computing profiles for snapshot {args.snap}")
        print(f"Mass cut: log10(M) > {args.mass_min}")
        print("=" * 70)
    
    comm.Barrier()
    
    # Load halo matches
    if rank == 0:
        print("\nLoading halo matches...")
        matches = np.load(config['matches_file'])
        dmo_indices = matches['dmo_indices']
        hydro_indices = matches['hydro_indices']
        print(f"  {len(dmo_indices)} matched halos")
    else:
        dmo_indices = None
        hydro_indices = None
    
    dmo_indices = comm.bcast(dmo_indices, root=0)
    hydro_indices = comm.bcast(hydro_indices, root=0)
    
    # Load halo catalogs
    halo_dmo = groupcat.loadHalos(
        config['dmo_basePath'], args.snap,
        fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
    )
    halo_hydro = groupcat.loadHalos(
        config['hydro_basePath'], args.snap,
        fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
    )
    
    # Filter by mass
    masses = halo_dmo['Group_M_Crit200'][dmo_indices] * config['mass_unit']
    mass_mask = masses >= config['min_halo_mass']
    
    valid_indices = np.where(mass_mask)[0]
    if args.n_halos is not None:
        valid_indices = valid_indices[:args.n_halos]
    
    if rank == 0:
        print(f"\n{len(valid_indices)} halos above mass cut")
        print(f"Processing across {size} MPI ranks...\n")
    
    # Distribute halos across ranks
    my_indices = valid_indices[rank::size]
    
    # Storage for results
    results = []
    
    for i, idx in enumerate(my_indices):
        i_dmo = dmo_indices[idx]
        i_hydro = hydro_indices[idx]
        
        # DMO halo properties
        mass_dmo = halo_dmo['Group_M_Crit200'][i_dmo] * config['mass_unit']
        radius_dmo = halo_dmo['Group_R_Crit200'][i_dmo] / 1e3 / h
        center_dmo = halo_dmo['GroupPos'][i_dmo] / 1e3 / h
        
        # Hydro halo properties
        mass_hydro = halo_hydro['Group_M_Crit200'][i_hydro] * config['mass_unit']
        radius_hydro = halo_hydro['Group_R_Crit200'][i_hydro] / 1e3 / h
        center_hydro = halo_hydro['GroupPos'][i_hydro] / 1e3 / h
        
        if i % 10 == 0:
            print(f"Rank {rank}: Processing halo {i+1}/{len(my_indices)} (DMO idx {i_dmo})")
        
        # Search radius (5 R_200)
        search_radius = 5 * radius_dmo
        
        # Load DMO particles around halo
        dmo_coords, dmo_masses = load_particles_for_halo(
            config['dmo_basePath'], args.snap, center_dmo, search_radius, config
        )
        
        # Load Hydro particles around halo
        hydro_coords, hydro_masses = load_particles_for_halo(
            config['hydro_basePath'], args.snap, center_hydro, search_radius, config
        )
        
        # Compute profiles
        if len(dmo_coords) > 100:
            r_dmo, rho_dmo = compute_density_profile(
                dmo_coords, dmo_masses, center_dmo, radius_dmo, config['box_size'] / h
            )
        else:
            r_dmo = np.zeros(100)
            rho_dmo = np.zeros(100)
        
        if len(hydro_coords) > 100:
            r_hydro, rho_hydro = compute_density_profile(
                hydro_coords, hydro_masses, center_hydro, radius_hydro, config['box_size'] / h
            )
        else:
            r_hydro = np.zeros(100)
            rho_hydro = np.zeros(100)
        
        results.append({
            'dmo_index': i_dmo,
            'hydro_index': i_hydro,
            'mass_dmo': mass_dmo,
            'mass_hydro': mass_hydro,
            'radius_dmo': radius_dmo,
            'radius_hydro': radius_hydro,
            'center_dmo': center_dmo,
            'center_hydro': center_hydro,
            'r_bins': r_dmo,
            'rho_dmo': rho_dmo,
            'rho_hydro': rho_hydro,
        })
    
    # Gather results to rank 0
    all_results = comm.gather(results, root=0)
    
    if rank == 0:
        # Flatten results
        all_results_flat = [r for sublist in all_results for r in sublist]
        
        print(f"\nCollected {len(all_results_flat)} halo profiles")
        
        # Save to HDF5
        output_file = f"{snap_dir}/profiles.h5"
        
        with h5py.File(output_file, 'w') as f:
            f.attrs['snapshot'] = args.snap
            f.attrs['n_halos'] = len(all_results_flat)
            f.attrs['mass_min'] = config['min_halo_mass']
            f.attrs['box_size'] = config['box_size']
            
            # Save radial bins (same for all)
            if len(all_results_flat) > 0:
                f.create_dataset('r_bins', data=all_results_flat[0]['r_bins'])
            
            # Save per-halo data
            for i, r in enumerate(all_results_flat):
                grp = f.create_group(f'halo_{i:05d}')
                grp.attrs['dmo_index'] = r['dmo_index']
                grp.attrs['hydro_index'] = r['hydro_index']
                grp.attrs['mass_dmo'] = r['mass_dmo']
                grp.attrs['mass_hydro'] = r['mass_hydro']
                grp.attrs['radius_dmo'] = r['radius_dmo']
                grp.attrs['radius_hydro'] = r['radius_hydro']
                grp.create_dataset('center_dmo', data=r['center_dmo'])
                grp.create_dataset('center_hydro', data=r['center_hydro'])
                grp.create_dataset('rho_dmo', data=r['rho_dmo'])
                grp.create_dataset('rho_hydro', data=r['rho_hydro'])
        
        print(f"\nSaved profiles to: {output_file}")
        print("=" * 70)


if __name__ == "__main__":
    main()
