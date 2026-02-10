#!/usr/bin/env python
"""
Generate individual halo density profiles for DMO and Hydro simulations.

This script computes density profiles for each halo individually, matching
the BCM pipeline output format. No stacking, no maps, no Replace fields.

Output format (profiles_individual_snap{snap:03d}.h5):
    Attributes:
        - snapshot, redshift, box_size, n_halos
        - radial_bins, mass_bin_edges
    
    Datasets:
        - halo_log_masses: (n_halos,) log10(M200c) in Msun/h
        - halo_radii: (n_halos,) R200c in Mpc/h
        - halo_positions: (n_halos, 3) DMO halo centers
        - halo_hydro_positions: (n_halos, 3) Hydro halo centers
        - halo_hydro_radii: (n_halos,) Hydro R200c
        - mass_bin_indices: (n_halos,) index into mass_bin_edges
        
        - individual_dmo_density: (n_halos, n_radial_bins) density profiles
        - individual_dmo_mass: (n_halos, n_radial_bins) mass per radial bin
        
        - individual_hydro_density: (n_halos, n_radial_bins) total density
        - individual_hydro_mass: (n_halos, n_radial_bins) total mass per bin
        - individual_hydro_density_dm: (n_halos, n_radial_bins) DM component
        - individual_hydro_density_gas: (n_halos, n_radial_bins) gas component
        - individual_hydro_density_stars: (n_halos, n_radial_bins) stellar component

Usage:
    mpirun -np 64 python generate_individual_profiles.py --snap 96 --sim-res 2500
"""

import numpy as np
import h5py
import argparse
import os
import sys
import time
import glob

from mpi4py import MPI
from scipy.spatial import cKDTree

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

OUTPUT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields'
BOX_SIZE = 205.0  # Mpc/h
MASS_UNIT = 1e10  # Convert to Msun/h

# Profile configuration
RADIAL_BINS = np.logspace(-2, np.log10(5), 31)  # 0.01 to 5 R200, 30 bins
MASS_BIN_EDGES = [12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 16.0]


# ============================================================================
# Data Loading
# ============================================================================

class DistributedParticles:
    """Load and distribute particles across MPI ranks."""
    
    def __init__(self, snapshot, sim_res, sim_type, radius_mult=5.0):
        self.snapshot = snapshot
        self.sim_res = sim_res
        self.sim_type = sim_type
        self.radius_mult = radius_mult
        self.coords = None
        self.masses = None
        self.types = None
        self.tree = None
        
    def load(self):
        """Load particles distributed across MPI ranks."""
        basePath = SIM_PATHS[self.sim_res][self.sim_type]
        snap_dir = f"{basePath}/snapdir_{self.snapshot:03d}"
        
        # Find all snapshot files
        snap_files = sorted(glob.glob(f"{snap_dir}/snap_{self.snapshot:03d}.*.hdf5"))
        n_files = len(snap_files)
        
        # Distribute files across ranks
        files_per_rank = n_files // size
        extra = n_files % size
        
        if rank < extra:
            start_file = rank * (files_per_rank + 1)
            end_file = start_file + files_per_rank + 1
        else:
            start_file = rank * files_per_rank + extra
            end_file = start_file + files_per_rank
        
        my_files = snap_files[start_file:end_file]
        
        # Load particles from assigned files
        all_coords = []
        all_masses = []
        all_types = []
        
        dm_mass = SIM_PATHS[self.sim_res][f'{self.sim_type}_dm_mass'] * MASS_UNIT
        
        for fpath in my_files:
            with h5py.File(fpath, 'r') as f:
                # DM particles (PartType1)
                if 'PartType1' in f:
                    coords_dm = f['PartType1/Coordinates'][:]
                    n_dm = len(coords_dm)
                    all_coords.append(coords_dm)
                    all_masses.append(np.full(n_dm, dm_mass, dtype=np.float32))
                    all_types.append(np.ones(n_dm, dtype=np.int8))
                
                # Gas particles (PartType0) - Hydro only
                if self.sim_type == 'hydro' and 'PartType0' in f:
                    coords_gas = f['PartType0/Coordinates'][:]
                    masses_gas = f['PartType0/Masses'][:] * MASS_UNIT
                    n_gas = len(coords_gas)
                    all_coords.append(coords_gas)
                    all_masses.append(masses_gas.astype(np.float32))
                    all_types.append(np.zeros(n_gas, dtype=np.int8))
                
                # Star particles (PartType4) - Hydro only
                if self.sim_type == 'hydro' and 'PartType4' in f:
                    coords_star = f['PartType4/Coordinates'][:]
                    masses_star = f['PartType4/Masses'][:] * MASS_UNIT
                    n_star = len(coords_star)
                    all_coords.append(coords_star)
                    all_masses.append(masses_star.astype(np.float32))
                    all_types.append(np.full(n_star, 4, dtype=np.int8))
        
        if all_coords:
            self.coords = np.vstack(all_coords).astype(np.float32)
            self.masses = np.concatenate(all_masses).astype(np.float32)
            self.types = np.concatenate(all_types)
            # Wrap coordinates to box (TNG coords can be slightly outside)
            self.coords = np.mod(self.coords, BOX_SIZE).astype(np.float32)
        else:
            self.coords = np.zeros((0, 3), dtype=np.float32)
            self.masses = np.zeros(0, dtype=np.float32)
            self.types = np.zeros(0, dtype=np.int8)
        
        return self
    
    def build_tree(self):
        """Build KDTree for particle queries."""
        if len(self.coords) > 0:
            self.tree = cKDTree(self.coords, boxsize=BOX_SIZE)
        return self
    
    def query_halo(self, center, r200):
        """Query particles within radius_mult * R200 of halo center."""
        if self.tree is None or len(self.coords) == 0:
            return np.array([], dtype=np.int64)
        
        radius = self.radius_mult * r200
        idx = self.tree.query_ball_point(center, radius)
        return np.array(idx, dtype=np.int64)
    
    def free(self):
        """Free memory."""
        self.coords = None
        self.masses = None
        self.types = None
        self.tree = None


# ============================================================================
# Profile Computation
# ============================================================================

def compute_profile(coords, masses, types, center, r200, radial_bins):
    """
    Compute density profile for a single halo.
    
    Returns:
        dict with density, mass profiles (total and by type)
    """
    n_bins = len(radial_bins) - 1
    
    if len(coords) == 0:
        return {
            'density': np.zeros(n_bins),
            'mass': np.zeros(n_bins),
            'density_dm': np.zeros(n_bins),
            'density_gas': np.zeros(n_bins),
            'density_stars': np.zeros(n_bins),
        }
    
    # Compute distances (with periodic BC)
    dx = coords - center
    dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
    dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
    r = np.linalg.norm(dx, axis=1) / r200  # In units of R200
    
    # Total profile
    mass_profile, _ = np.histogram(r, bins=radial_bins, weights=masses)
    
    # Shell volumes (in physical units: (R200 * Mpc/h)^3)
    volumes = 4/3 * np.pi * r200**3 * (radial_bins[1:]**3 - radial_bins[:-1]**3)
    density = np.where(volumes > 0, mass_profile / volumes, 0)
    
    # By particle type
    dm_mask = types == 1
    gas_mask = types == 0
    star_mask = types == 4
    
    mass_dm, _ = np.histogram(r[dm_mask], bins=radial_bins, weights=masses[dm_mask])
    mass_gas, _ = np.histogram(r[gas_mask], bins=radial_bins, weights=masses[gas_mask])
    mass_stars, _ = np.histogram(r[star_mask], bins=radial_bins, weights=masses[star_mask])
    
    density_dm = np.where(volumes > 0, mass_dm / volumes, 0)
    density_gas = np.where(volumes > 0, mass_gas / volumes, 0)
    density_stars = np.where(volumes > 0, mass_stars / volumes, 0)
    
    return {
        'density': density,
        'mass': mass_profile,
        'density_dm': density_dm,
        'density_gas': density_gas,
        'density_stars': density_stars,
    }


# ============================================================================
# Halo Loading
# ============================================================================

def load_halos(snapshot, sim_res, mass_min=12.0):
    """Load matched halo catalog."""
    hydro_path = SIM_PATHS[sim_res]['hydro']
    
    # Load Hydro group catalog (for matching)
    grp_file = f"{hydro_path}/groups_{snapshot:03d}/fof_subhalo_tab_{snapshot:03d}.0.hdf5"
    
    with h5py.File(grp_file, 'r') as f:
        # Group masses and positions
        group_mass = f['Group/GroupMass'][:] * MASS_UNIT
        group_pos = f['Group/GroupPos'][:]
        group_r200 = f['Group/Group_R_Crit200'][:]  # in kpc/h
    
    # Convert R200 to Mpc/h
    group_r200 = group_r200 / 1000.0
    
    # Wrap positions to box (TNG positions can be slightly outside)
    group_pos = np.mod(group_pos, BOX_SIZE)
    
    # Mass cut
    log_mass = np.log10(group_mass)
    mass_mask = log_mass >= mass_min
    
    hydro_masses = group_mass[mass_mask]
    hydro_positions = group_pos[mass_mask]
    hydro_radii = group_r200[mass_mask]
    hydro_log_masses = log_mass[mass_mask]
    
    # Load DMO group catalog
    dmo_path = SIM_PATHS[sim_res]['dmo']
    dmo_grp_file = f"{dmo_path}/groups_{snapshot:03d}/fof_subhalo_tab_{snapshot:03d}.0.hdf5"
    
    with h5py.File(dmo_grp_file, 'r') as f:
        dmo_group_mass = f['Group/GroupMass'][:] * MASS_UNIT
        dmo_group_pos = f['Group/GroupPos'][:]
        dmo_group_r200 = f['Group/Group_R_Crit200'][:] / 1000.0
    
    # Wrap DMO positions to box
    dmo_group_pos = np.mod(dmo_group_pos, BOX_SIZE)
    
    # Match halos by position (simple nearest neighbor)
    from scipy.spatial import cKDTree
    
    dmo_tree = cKDTree(dmo_group_pos, boxsize=BOX_SIZE)
    distances, dmo_indices = dmo_tree.query(hydro_positions)
    
    # Use DMO properties for matched halos
    dmo_masses = dmo_group_mass[dmo_indices]
    dmo_positions = dmo_group_pos[dmo_indices]
    dmo_radii = dmo_group_r200[dmo_indices]
    dmo_log_masses = np.log10(dmo_masses)
    
    # Assign mass bins
    mass_bin_indices = np.digitize(dmo_log_masses, MASS_BIN_EDGES) - 1
    
    return {
        'n_halos': len(hydro_masses),
        'dmo_masses': dmo_masses,
        'dmo_positions': dmo_positions,
        'dmo_radii': dmo_radii,
        'dmo_log_masses': dmo_log_masses,
        'hydro_masses': hydro_masses,
        'hydro_positions': hydro_positions,
        'hydro_radii': hydro_radii,
        'hydro_log_masses': hydro_log_masses,
        'mass_bin_indices': mass_bin_indices,
    }


def get_snapshot_redshift(snapshot, sim_res):
    """Get redshift for a given snapshot number."""
    basePath = SIM_PATHS[sim_res]['dmo']
    snap_file = f"{basePath}/snapdir_{snapshot:03d}/snap_{snapshot:03d}.0.hdf5"
    
    with h5py.File(snap_file, 'r') as f:
        z = f['Header'].attrs['Redshift']
    
    return z


# ============================================================================
# Main Pipeline
# ============================================================================

def run_profile_pipeline(args):
    """Run the individual profile computation pipeline."""
    
    t_start = time.time()
    
    if rank == 0:
        print("=" * 70)
        print("Individual Halo Profile Generation")
        print("=" * 70)
        print(f"Snapshot: {args.snap}")
        print(f"Simulation resolution: {args.sim_res}")
        print(f"Mass minimum: 10^{args.mass_min} Msun/h")
        print(f"Radius multiplier: {args.radius_mult}")
        print(f"MPI ranks: {size}")
        print("=" * 70)
        sys.stdout.flush()
    
    # Setup output directory
    output_dir = os.path.join(OUTPUT_BASE, f'L205n{args.sim_res}TNG')
    
    if rank == 0:
        os.makedirs(os.path.join(output_dir, 'profiles'), exist_ok=True)
    comm.Barrier()
    
    # Get redshift
    z_snap = get_snapshot_redshift(args.snap, args.sim_res)
    if rank == 0:
        print(f"\nRedshift: z = {z_snap:.4f}")
    
    # ========================================================================
    # Load halos
    # ========================================================================
    if rank == 0:
        print("\n[1/4] Loading halo catalog...")
        sys.stdout.flush()
    
    halos = load_halos(args.snap, args.sim_res, args.mass_min)
    n_halos = halos['n_halos']
    n_radial_bins = len(RADIAL_BINS) - 1
    
    if rank == 0:
        print(f"  Found {n_halos} halos above 10^{args.mass_min} Msun/h")
    
    # ========================================================================
    # Load DMO particles and compute profiles
    # ========================================================================
    if rank == 0:
        print("\n[2/4] Processing DMO simulation...")
        sys.stdout.flush()
    
    dmo = DistributedParticles(args.snap, args.sim_res, 'dmo', args.radius_mult)
    dmo.load().build_tree()
    
    # Initialize individual profile arrays
    local_dmo_density = np.zeros((n_halos, n_radial_bins), dtype=np.float64)
    local_dmo_mass = np.zeros((n_halos, n_radial_bins), dtype=np.float64)
    
    if rank == 0:
        print(f"  Computing profiles for {n_halos} halos...")
        t0 = time.time()
    
    for i in range(n_halos):
        center = halos['dmo_positions'][i]
        r200 = halos['dmo_radii'][i]
        
        # Query local particles
        local_idx = dmo.query_halo(center, r200)
        
        if len(local_idx) > 0:
            local_coords = dmo.coords[local_idx]
            local_masses = dmo.masses[local_idx]
            local_types = np.ones(len(local_idx), dtype=int)  # All DM for DMO
            
            profile = compute_profile(
                local_coords, local_masses, local_types, center, r200, RADIAL_BINS
            )
            
            local_dmo_density[i] = profile['density']
            local_dmo_mass[i] = profile['mass']
        
        if rank == 0 and (i + 1) % 500 == 0:
            print(f"    Halo {i+1}/{n_halos}...")
            sys.stdout.flush()
    
    if rank == 0:
        print(f"  DMO processing time: {time.time()-t0:.1f}s")
    
    # Reduce across ranks
    global_dmo_density = np.zeros_like(local_dmo_density)
    global_dmo_mass = np.zeros_like(local_dmo_mass)
    
    comm.Reduce(local_dmo_density, global_dmo_density, op=MPI.SUM, root=0)
    comm.Reduce(local_dmo_mass, global_dmo_mass, op=MPI.SUM, root=0)
    
    dmo.free()
    
    # ========================================================================
    # Load Hydro particles and compute profiles
    # ========================================================================
    if rank == 0:
        print("\n[3/4] Processing Hydro simulation...")
        sys.stdout.flush()
    
    hydro = DistributedParticles(args.snap, args.sim_res, 'hydro', args.radius_mult)
    hydro.load().build_tree()
    
    # Initialize individual profile arrays
    local_hydro_density = np.zeros((n_halos, n_radial_bins), dtype=np.float64)
    local_hydro_mass = np.zeros((n_halos, n_radial_bins), dtype=np.float64)
    local_hydro_density_dm = np.zeros((n_halos, n_radial_bins), dtype=np.float64)
    local_hydro_density_gas = np.zeros((n_halos, n_radial_bins), dtype=np.float64)
    local_hydro_density_stars = np.zeros((n_halos, n_radial_bins), dtype=np.float64)
    
    if rank == 0:
        print(f"  Computing profiles for {n_halos} halos...")
        t0 = time.time()
    
    for i in range(n_halos):
        # Use Hydro center and radius for Hydro profiles
        center = halos['hydro_positions'][i]
        r200 = halos['hydro_radii'][i]
        
        # Query local particles
        local_idx = hydro.query_halo(center, r200)
        
        if len(local_idx) > 0:
            local_coords = hydro.coords[local_idx]
            local_masses = hydro.masses[local_idx]
            local_types = hydro.types[local_idx]
            
            profile = compute_profile(
                local_coords, local_masses, local_types, center, r200, RADIAL_BINS
            )
            
            local_hydro_density[i] = profile['density']
            local_hydro_mass[i] = profile['mass']
            local_hydro_density_dm[i] = profile['density_dm']
            local_hydro_density_gas[i] = profile['density_gas']
            local_hydro_density_stars[i] = profile['density_stars']
        
        if rank == 0 and (i + 1) % 500 == 0:
            print(f"    Halo {i+1}/{n_halos}...")
            sys.stdout.flush()
    
    if rank == 0:
        print(f"  Hydro processing time: {time.time()-t0:.1f}s")
    
    # Reduce across ranks
    global_hydro_density = np.zeros_like(local_hydro_density)
    global_hydro_mass = np.zeros_like(local_hydro_mass)
    global_hydro_density_dm = np.zeros_like(local_hydro_density_dm)
    global_hydro_density_gas = np.zeros_like(local_hydro_density_gas)
    global_hydro_density_stars = np.zeros_like(local_hydro_density_stars)
    
    comm.Reduce(local_hydro_density, global_hydro_density, op=MPI.SUM, root=0)
    comm.Reduce(local_hydro_mass, global_hydro_mass, op=MPI.SUM, root=0)
    comm.Reduce(local_hydro_density_dm, global_hydro_density_dm, op=MPI.SUM, root=0)
    comm.Reduce(local_hydro_density_gas, global_hydro_density_gas, op=MPI.SUM, root=0)
    comm.Reduce(local_hydro_density_stars, global_hydro_density_stars, op=MPI.SUM, root=0)
    
    hydro.free()
    
    # ========================================================================
    # Save profiles
    # ========================================================================
    if rank == 0:
        print("\n[4/4] Saving individual profiles...")
        
        profile_file = os.path.join(output_dir, 'profiles', f'profiles_individual_snap{args.snap:03d}.h5')
        
        with h5py.File(profile_file, 'w') as f:
            # Attributes
            f.attrs['snapshot'] = args.snap
            f.attrs['redshift'] = z_snap
            f.attrs['box_size'] = BOX_SIZE
            f.attrs['n_halos'] = n_halos
            f.attrs['mass_min'] = args.mass_min
            f.attrs['radius_multiplier'] = args.radius_mult
            f.attrs['radial_bins'] = RADIAL_BINS
            f.attrs['mass_bin_edges'] = np.array(MASS_BIN_EDGES)
            
            # Halo properties
            f.create_dataset('halo_log_masses', data=halos['dmo_log_masses'].astype(np.float32))
            f.create_dataset('halo_radii', data=halos['dmo_radii'].astype(np.float32))
            f.create_dataset('halo_positions', data=halos['dmo_positions'].astype(np.float32))
            f.create_dataset('halo_hydro_log_masses', data=halos['hydro_log_masses'].astype(np.float32))
            f.create_dataset('halo_hydro_radii', data=halos['hydro_radii'].astype(np.float32))
            f.create_dataset('halo_hydro_positions', data=halos['hydro_positions'].astype(np.float32))
            f.create_dataset('mass_bin_indices', data=halos['mass_bin_indices'].astype(np.int32))
            
            # DMO individual profiles
            f.create_dataset('individual_dmo_density', data=global_dmo_density.astype(np.float32),
                           compression='gzip', compression_opts=4)
            f.create_dataset('individual_dmo_mass', data=global_dmo_mass.astype(np.float32),
                           compression='gzip', compression_opts=4)
            
            # Hydro individual profiles
            f.create_dataset('individual_hydro_density', data=global_hydro_density.astype(np.float32),
                           compression='gzip', compression_opts=4)
            f.create_dataset('individual_hydro_mass', data=global_hydro_mass.astype(np.float32),
                           compression='gzip', compression_opts=4)
            f.create_dataset('individual_hydro_density_dm', data=global_hydro_density_dm.astype(np.float32),
                           compression='gzip', compression_opts=4)
            f.create_dataset('individual_hydro_density_gas', data=global_hydro_density_gas.astype(np.float32),
                           compression='gzip', compression_opts=4)
            f.create_dataset('individual_hydro_density_stars', data=global_hydro_density_stars.astype(np.float32),
                           compression='gzip', compression_opts=4)
        
        print(f"    Saved: {profile_file}")
        print(f"    - {n_halos} halos × {n_radial_bins} radial bins")
        
        print("\n" + "=" * 70)
        print(f"Complete! Total time: {time.time()-t_start:.1f}s")
        print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate individual halo profiles for DMO and Hydro')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, choices=[625, 1250, 2500], default=2500,
                       help='Simulation resolution')
    parser.add_argument('--mass-min', type=float, default=12.0,
                       help='Minimum log10 halo mass (Msun/h)')
    parser.add_argument('--radius-mult', type=float, default=5.0,
                       help='Radius multiplier for particle queries')
    
    args = parser.parse_args()
    run_profile_pipeline(args)


if __name__ == '__main__':
    main()
