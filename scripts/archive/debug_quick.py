#!/usr/bin/env python
"""
Quick debug: Use illustris_python to load particles around ONE halo.
This is more memory-efficient and tests the correct approach.
"""

import numpy as np
import h5py
from illustris_python import groupcat, snapshot

def apply_periodic_boundary(dx, box_size):
    return dx - np.round(dx / box_size) * box_size

def main():
    # Use 625 resolution - smallest
    sim_res = 625
    snap = 99
    halo_idx = 0  # Use matched halo 0
    
    dmo_basePath = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L205n{sim_res}TNG_DM/output'
    hydro_basePath = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L205n{sim_res}TNG/output'
    
    dmo_mass = 0.3025384873 * 1e10  # Msun/h for 625
    box_size = 205.0
    
    # Load halo catalogs
    print("Loading halo catalogs...")
    halo_dmo = groupcat.loadHalos(dmo_basePath, snap, 
                                   fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos'])
    halo_hydro = groupcat.loadHalos(hydro_basePath, snap,
                                     fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos'])
    
    # Load matches
    matches_file = f'/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{sim_res}TNG/matches/matches_snap{snap:03d}.npz'
    matches = np.load(matches_file)
    
    dmo_idx = matches['dmo_indices'][halo_idx]
    hydro_idx = matches['hydro_indices'][halo_idx]
    
    dmo_pos = halo_dmo['GroupPos'][dmo_idx] / 1e3  # Mpc/h
    dmo_r200 = halo_dmo['Group_R_Crit200'][dmo_idx] / 1e3
    dmo_m200 = halo_dmo['Group_M_Crit200'][dmo_idx] * 1e10
    
    hydro_pos = halo_hydro['GroupPos'][hydro_idx] / 1e3
    hydro_r200 = halo_hydro['Group_R_Crit200'][hydro_idx] / 1e3
    
    print(f"\nDMO halo {dmo_idx}:")
    print(f"  Position: {dmo_pos}")
    print(f"  R200: {dmo_r200:.4f} Mpc/h ({dmo_r200*1e3:.1f} kpc/h)")
    print(f"  M200: {dmo_m200:.2e} Msun/h")
    
    # Load ALL DM particles (625 resolution is manageable)
    print("\nLoading ALL DMO particles...")
    dm_coords = snapshot.loadSubset(dmo_basePath, snap, 'dm', fields=['Coordinates']) / 1e3
    print(f"  Loaded {len(dm_coords):,} particles")
    
    # Find particles within 5*R200 of halo center
    r_max = 5.0 * dmo_r200
    dx = dm_coords - dmo_pos
    dx = apply_periodic_boundary(dx, box_size)
    dist = np.linalg.norm(dx, axis=1)
    
    within_5r200 = np.sum(dist < 5*dmo_r200)
    within_1r200 = np.sum(dist < dmo_r200)
    within_0p5r200 = np.sum(dist < 0.5*dmo_r200)
    within_0p1r200 = np.sum(dist < 0.1*dmo_r200)
    
    print(f"\nParticles near halo:")
    print(f"  Within 5×R200 ({5*dmo_r200:.3f} Mpc): {within_5r200:,}")
    print(f"  Within 1×R200 ({dmo_r200:.3f} Mpc): {within_1r200:,}")
    print(f"  Within 0.5×R200 ({0.5*dmo_r200:.3f} Mpc): {within_0p5r200:,}")
    print(f"  Within 0.1×R200 ({0.1*dmo_r200:.3f} Mpc): {within_0p1r200:,}")
    
    # Compute profile with correct approach
    r_min_r200 = 0.01
    r_max_r200 = 5.0
    n_bins = 25
    r_bins = np.logspace(np.log10(r_min_r200), np.log10(r_max_r200), n_bins + 1)
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    
    mask_all = dist < r_max
    coords_near = dm_coords[mask_all]
    masses_near = np.full(mask_all.sum(), dmo_mass)
    
    # Compute distances in R200 units
    dx_near = coords_near - dmo_pos
    dx_near = apply_periodic_boundary(dx_near, box_size)
    r_norm = np.linalg.norm(dx_near, axis=1) / dmo_r200
    
    print(f"\n{'r/R200':>10} {'N_part':>10} {'rho (Msun/h / (Mpc/h)^3)':>25}")
    print("-" * 50)
    
    for i in range(n_bins):
        mask = (r_norm >= r_bins[i]) & (r_norm < r_bins[i+1])
        n_part = mask.sum()
        
        if n_part > 0:
            mass_in_shell = masses_near[mask].sum()
            r_inner = r_bins[i] * dmo_r200
            r_outer = r_bins[i+1] * dmo_r200
            volume = 4/3 * np.pi * (r_outer**3 - r_inner**3)
            density = mass_in_shell / volume
        else:
            density = 0
        
        print(f"{r_centers[i]:>10.4f} {n_part:>10,} {density:>25.2e}")
    
    print("\n✓ This is what profiles SHOULD look like when loaded correctly!")
    print("  Non-zero particle counts at ALL radii (except maybe innermost bins)")


if __name__ == "__main__":
    main()
