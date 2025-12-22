#!/usr/bin/env python
"""
Debug script: Compute profile for ONE halo correctly, loading particles from ALL files.
This verifies that the profile computation is correct when particles are loaded properly.

Usage:
    python debug_profile_single_halo.py --halo-idx 0 --sim-res 625

For quick testing, use 625 resolution (smallest simulation).
"""

import numpy as np
import h5py
import glob
import argparse
import time
import os

def apply_periodic_boundary(dx, box_size):
    """Apply minimum image convention."""
    return dx - np.round(dx / box_size) * box_size


def compute_radial_profile_correct(coords, masses, center, r200, box_size, r_bins):
    """
    Compute spherically-averaged density profile with proper periodic boundaries.
    """
    # Distance from center with periodic boundaries
    dx = coords - center
    dx = apply_periodic_boundary(dx, box_size)
    r = np.linalg.norm(dx, axis=1) / r200  # in units of R_200
    
    n_bins = len(r_bins) - 1
    density = np.zeros(n_bins)
    n_particles = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        n_particles[i] = mask.sum()
        
        if n_particles[i] > 0:
            mass_in_shell = masses[mask].sum()
            
            # Shell volume in physical units
            r_inner = r_bins[i] * r200
            r_outer = r_bins[i+1] * r200
            volume = 4/3 * np.pi * (r_outer**3 - r_inner**3)
            
            density[i] = mass_in_shell / volume
    
    return density, n_particles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--halo-idx', type=int, default=0, help='Halo index to test')
    parser.add_argument('--sim-res', type=int, default=625, choices=[625, 1250, 2500])
    parser.add_argument('--snap', type=int, default=99)
    args = parser.parse_args()
    
    # Configuration
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
    
    sim_res = args.sim_res
    snap = args.snap
    halo_idx = args.halo_idx
    
    sim_config = SIM_PATHS[sim_res]
    dmo_basePath = sim_config['dmo']
    hydro_basePath = sim_config['hydro']
    dmo_mass = sim_config['dmo_mass'] * 1e10  # Msun/h
    hydro_dm_mass = sim_config['hydro_dm_mass'] * 1e10
    
    box_size = 205.0
    r_max_r200 = 5.0
    n_bins = 25
    r_min_r200 = 0.01  # Use 0.01 instead of 0.001 (more realistic)
    
    r_bins = np.logspace(np.log10(r_min_r200), np.log10(r_max_r200), n_bins + 1)
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    
    # Load halo info from profile file
    profile_file = f'/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{sim_res}TNG/snap{snap:03d}/profiles_Mgt12.5.h5'
    if not os.path.exists(profile_file):
        # Fall back to older naming
        profile_file = f'/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{sim_res}TNG/snap{snap:03d}/profiles.h5'
    
    print("=" * 70)
    print(f"DEBUG: Single Halo Profile Test")
    print(f"Simulation: L205n{sim_res}TNG, Snap {snap}")
    print(f"Halo index: {halo_idx}")
    print("=" * 70)
    
    with h5py.File(profile_file, 'r') as f:
        dmo_positions = f['dmo_positions'][:]
        dmo_radii = f['dmo_radii'][:]
        dmo_masses_halos = f['dmo_masses'][:]
        hydro_positions = f['hydro_positions'][:]
        hydro_radii = f['hydro_radii'][:]
        
        # Compare with stored profile
        stored_dmo = f['profiles/dmo'][halo_idx]
        stored_hydro = f['profiles/hydro'][halo_idx]
    
    test_pos = dmo_positions[halo_idx]
    test_r200 = dmo_radii[halo_idx]
    test_mass = dmo_masses_halos[halo_idx]
    
    print(f"\nHalo properties:")
    print(f"  Position: {test_pos}")
    print(f"  R200: {test_r200:.4f} Mpc/h ({test_r200*1e3:.1f} kpc/h)")
    print(f"  Mass: {test_mass:.2e} Msun/h")
    
    # Load DMO particles from ALL files
    print(f"\n[1/2] Loading DMO particles...")
    snap_files = sorted(glob.glob(f"{dmo_basePath}/snapdir_{snap:03d}/snap_{snap:03d}.*.hdf5"))
    print(f"  Found {len(snap_files)} snapshot files")
    
    r_max = r_max_r200 * test_r200
    
    all_coords = []
    all_masses = []
    t0 = time.time()
    
    for i, snap_file in enumerate(snap_files):
        with h5py.File(snap_file, 'r') as f:
            coords = f['PartType1/Coordinates'][:] / 1e3  # kpc -> Mpc
        
        # Apply periodic boundary and distance filter
        dx = coords - test_pos
        dx = apply_periodic_boundary(dx, box_size)
        dist = np.linalg.norm(dx, axis=1)
        
        mask = dist < r_max
        n_in_file = mask.sum()
        
        if n_in_file > 0:
            all_coords.append(coords[mask])
            all_masses.append(np.full(n_in_file, dmo_mass, dtype=np.float32))
        
        if (i+1) % 10 == 0 or i == len(snap_files)-1:
            print(f"  Processed {i+1}/{len(snap_files)} files...")
    
    all_coords = np.concatenate(all_coords) if all_coords else np.zeros((0, 3))
    all_masses = np.concatenate(all_masses) if all_masses else np.zeros(0)
    
    print(f"  Loaded {len(all_coords):,} particles in {time.time()-t0:.1f}s")
    
    # Compute profile
    print(f"\nComputing DMO profile...")
    dmo_density, dmo_npart = compute_radial_profile_correct(
        all_coords, all_masses, test_pos, test_r200, box_size, r_bins
    )
    
    # Same for Hydro
    print(f"\n[2/2] Loading Hydro particles...")
    snap_files = sorted(glob.glob(f"{hydro_basePath}/snapdir_{snap:03d}/snap_{snap:03d}.*.hdf5"))
    
    hydro_pos = hydro_positions[halo_idx]
    hydro_r200 = hydro_radii[halo_idx]
    r_max_hydro = r_max_r200 * hydro_r200
    
    all_coords = []
    all_masses = []
    t0 = time.time()
    
    for i, snap_file in enumerate(snap_files):
        with h5py.File(snap_file, 'r') as f:
            for ptype in ['PartType0', 'PartType1', 'PartType4']:
                if ptype not in f:
                    continue
                coords = f[ptype]['Coordinates'][:] / 1e3
                
                if 'Masses' in f[ptype]:
                    masses = f[ptype]['Masses'][:] * 1e10
                else:
                    masses = np.full(len(coords), hydro_dm_mass, dtype=np.float32)
                
                # Distance filter
                dx = coords - hydro_pos
                dx = apply_periodic_boundary(dx, box_size)
                dist = np.linalg.norm(dx, axis=1)
                
                mask = dist < r_max_hydro
                if mask.sum() > 0:
                    all_coords.append(coords[mask])
                    all_masses.append(masses[mask])
        
        if (i+1) % 10 == 0 or i == len(snap_files)-1:
            print(f"  Processed {i+1}/{len(snap_files)} files...")
    
    all_coords = np.concatenate(all_coords) if all_coords else np.zeros((0, 3))
    all_masses = np.concatenate(all_masses) if all_masses else np.zeros(0)
    
    print(f"  Loaded {len(all_coords):,} particles in {time.time()-t0:.1f}s")
    
    # Compute profile
    print(f"\nComputing Hydro profile...")
    hydro_density, hydro_npart = compute_radial_profile_correct(
        all_coords, all_masses, hydro_pos, hydro_r200, box_size, r_bins
    )
    
    # Print results
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)
    print(f"{'r/R200':>10} {'N_DMO':>10} {'N_Hydro':>10} {'rho_DMO':>15} {'rho_Hydro':>15} {'Hydro/DMO':>12}")
    print("-" * 90)
    
    for i in range(n_bins):
        ratio = hydro_density[i] / dmo_density[i] if dmo_density[i] > 0 else np.nan
        print(f"{r_centers[i]:>10.4f} {dmo_npart[i]:>10,} {hydro_npart[i]:>10,} "
              f"{dmo_density[i]:>15.2e} {hydro_density[i]:>15.2e} {ratio:>12.3f}")
    
    print("\n" + "=" * 90)
    print("STORED vs COMPUTED DMO")
    print("=" * 90)
    print(f"{'r/R200':>10} {'Stored':>15} {'Computed':>15} {'Ratio':>10}")
    print("-" * 60)
    for i in range(n_bins):
        ratio = stored_dmo[i] / dmo_density[i] if dmo_density[i] > 0 else np.nan
        print(f"{r_centers[i]:>10.4f} {stored_dmo[i]:>15.2e} {dmo_density[i]:>15.2e} {ratio:>10.3f}")
    
    print("\n⚠️  If Stored/Computed ratio is far from 1.0, the bug is confirmed!")


if __name__ == "__main__":
    main()
