#!/usr/bin/env python
"""
Compute the exact mass filling fraction for Replace configurations.

For each (mass_bin, radius_shell) configuration, this computes:
- The total DMO mass excised (particles within radial shell around halos in mass bin)
- The mass fraction relative to total simulation mass

This is the "proper" computation of mass filling fraction, replacing the 
linear approximation M(<α×R200) = α × M200.

Usage:
    # Serial (for testing)
    python compute_mass_fractions.py --snap 96 --sim-res 2500
    
    # With MPI (recommended for speed)
    mpirun -np 16 python compute_mass_fractions.py --snap 96 --sim-res 2500

Output:
    /mnt/home/mlee1/ceph/hydro_replace_fields/L205n{sim_res}TNG/mass_fractions_snap{snap:03d}.npz
    
    Contains:
        - mass_fraction_matrix: (n_mass_bins, n_radius_shells) array in percent
        - mass_excised_matrix: raw masses in M_sun/h
        - particle_count_matrix: number of excised particles
        - total_sim_mass: total DMO simulation mass
        - config_labels: labels for each configuration
        - mass_edges, radius_edges: bin edges used
"""

import numpy as np
import h5py
import argparse
import os
import sys
import time
import glob
import gc

from scipy.spatial import cKDTree
from itertools import combinations

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    rank = 0
    size = 1
    comm = None


# ============================================================================
# Configuration
# ============================================================================

SIM_PATHS = {
    2500: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output',
        'dmo_dm_mass': 0.0047271638660809,
        'n_particles': 2500**3,
    },
    1250: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG_DM/output',
        'dmo_dm_mass': 0.0378173109,
        'n_particles': 1250**3,
    },
    625: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG_DM/output',
        'dmo_dm_mass': 0.3025384873,
        'n_particles': 625**3,
    },
}

OUTPUT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields'
BOX_SIZE = 205.0  # Mpc/h
MASS_UNIT = 1e10  # Convert to Msun/h

# Mass bin edges (log10 M_sun/h)
MASS_EDGES = [12.0, 12.5, 13.0, 13.5, 15.0]

# Radius bin edges (in units of R200)
RADIUS_EDGES = [0.0, 0.5, 1.0, 3.0, 5.0]


def generate_binned_configs():
    """Generate all (mass_bin, radius_shell) combinations.
    
    Returns:
        list of (M_lo, M_hi, R_inner, R_outer) tuples
        Total: C(5,2) × C(5,2) = 10 × 10 = 100 configurations
    """
    configs = []
    
    # All mass bin combinations (10 total)
    mass_bins = [(10**m_lo, 10**m_hi) for m_lo, m_hi in combinations(MASS_EDGES, 2)]
    
    # All radius shell combinations (10 total)
    radius_shells = list(combinations(RADIUS_EDGES, 2))
    
    for M_lo, M_hi in mass_bins:
        for R_inner, R_outer in radius_shells:
            configs.append((M_lo, M_hi, R_inner, R_outer))
    
    return configs


def get_config_label(M_lo, M_hi, R_inner, R_outer):
    """Generate config label."""
    return f"hydro_replace_Ml_{M_lo:.2e}_Mu_{M_hi:.2e}_Ri_{R_inner}_Ro_{R_outer}".replace('+', '')


# ============================================================================
# Particle Loading
# ============================================================================

def load_dmo_particles(snapshot, sim_res):
    """Load DMO particles distributed across MPI ranks."""
    t0 = time.time()
    
    sim_config = SIM_PATHS[sim_res]
    basePath = sim_config['dmo']
    dm_mass = sim_config['dmo_dm_mass'] * MASS_UNIT  # M_sun/h
    
    snap_dir = f"{basePath}/snapdir_{snapshot:03d}/"
    all_files = sorted(glob.glob(f"{snap_dir}/snap_{snapshot:03d}.*.hdf5"))
    
    if size > 1:
        my_files = [f for i, f in enumerate(all_files) if i % size == rank]
    else:
        my_files = all_files
    
    if rank == 0:
        print(f"  Loading DMO particles from snapshot {snapshot}...")
        print(f"    Files: {len(all_files)} total, {len(my_files)} per rank")
    
    coords_list = []
    
    for filepath in my_files:
        with h5py.File(filepath, 'r') as f:
            pt_key = 'PartType1'
            if pt_key in f and f[pt_key]['Coordinates'].shape[0] > 0:
                coords_list.append(f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3)  # kpc -> Mpc
    
    if coords_list:
        coords = np.concatenate(coords_list)
    else:
        coords = np.zeros((0, 3), dtype=np.float32)
    
    if rank == 0:
        print(f"    Rank 0: {len(coords):,} particles")
        print(f"    Load time: {time.time()-t0:.1f}s")
    
    return coords, dm_mass


def build_tree(coords):
    """Build KDTree for spatial queries."""
    t0 = time.time()
    if rank == 0:
        print(f"  Building KDTree...", end=" ", flush=True)
    
    tree = cKDTree(coords) if len(coords) > 0 else None
    
    if rank == 0:
        print(f"done ({time.time()-t0:.1f}s)")
    
    return tree


def query_halo_particles(tree, coords, center, radius, max_radius_mult=5.0):
    """Query particles within radius of halo center with periodic BC."""
    if tree is None or len(coords) == 0:
        return np.array([], dtype=int)
    
    search_radius = radius * max_radius_mult
    
    # Query tree
    indices = tree.query_ball_point(center, search_radius)
    
    if len(indices) == 0:
        return np.array([], dtype=int)
    
    indices = np.array(indices)
    
    # Check periodic images if near box edge
    if np.any(center < search_radius) or np.any(center > BOX_SIZE - search_radius):
        for dx in [-BOX_SIZE, 0, BOX_SIZE]:
            for dy in [-BOX_SIZE, 0, BOX_SIZE]:
                for dz in [-BOX_SIZE, 0, BOX_SIZE]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    shifted_center = center + np.array([dx, dy, dz])
                    if np.all(shifted_center >= -search_radius) and np.all(shifted_center <= BOX_SIZE + search_radius):
                        more_indices = tree.query_ball_point(shifted_center, search_radius)
                        if len(more_indices) > 0:
                            indices = np.unique(np.concatenate([indices, more_indices]))
    
    return indices


def precompute_halo_particle_data(coords, tree, halos, max_radius_mult=5.0):
    """Precompute particle indices and normalized distances for all halos.
    
    This queries KDTree ONCE per halo at max radius, then stores particle
    indices and their distances (in units of R200).
    """
    n_halos = len(halos['masses'])
    
    if rank == 0:
        print(f"    Querying {n_halos} halos at {max_radius_mult}×R200...")
        t0 = time.time()
    
    halo_data = []
    
    for i in range(n_halos):
        center = halos['positions'][i]
        r200 = halos['radii'][i]
        
        # Single KDTree query at max radius
        idx = query_halo_particles(tree, coords, center, r200, max_radius_mult)
        
        if len(idx) == 0:
            halo_data.append((np.array([], dtype=np.int64), np.array([], dtype=np.float32)))
            continue
        
        # Compute distances in units of R200
        halo_coords = coords[idx]
        dx = halo_coords - center
        dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
        dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
        dist_r200 = np.linalg.norm(dx, axis=1) / r200
        
        halo_data.append((np.array(idx, dtype=np.int64), dist_r200.astype(np.float32)))
        
        if rank == 0 and (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_halos - i - 1) / rate
            print(f"      {i+1}/{n_halos} halos ({rate:.1f}/s, ETA {eta:.0f}s)")
    
    if rank == 0:
        print(f"    Precomputation done: {time.time()-t0:.1f}s")
    
    return halo_data


def get_particles_in_shell(halo_data, halo_indices, r_inner, r_outer):
    """Get combined particle indices for halos in a mass bin within a radial shell."""
    all_particles = []
    for i in halo_indices:
        idx, dist = halo_data[i]
        if len(idx) == 0:
            continue
        # Fast numpy filtering by distance
        shell_mask = (dist >= r_inner) & (dist < r_outer)
        all_particles.append(idx[shell_mask])
    
    if all_particles:
        return np.unique(np.concatenate(all_particles))
    return np.array([], dtype=np.int64)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compute mass filling fractions')
    parser.add_argument('--snap', type=int, default=96, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, default=2500, choices=[625, 1250, 2500])
    parser.add_argument('--mass-min', type=float, default=12.0, help='Minimum halo mass (log10)')
    args = parser.parse_args()
    
    t_start = time.time()
    
    sim_config = SIM_PATHS[args.sim_res]
    output_dir = os.path.join(OUTPUT_BASE, f'L205n{args.sim_res}TNG')
    
    # Total simulation mass
    total_sim_mass = sim_config['n_particles'] * sim_config['dmo_dm_mass'] * MASS_UNIT
    dm_mass_per_particle = sim_config['dmo_dm_mass'] * MASS_UNIT
    
    if rank == 0:
        print("=" * 70)
        print("MASS FILLING FRACTION COMPUTATION")
        print("=" * 70)
        print(f"Snapshot: {args.snap}")
        print(f"Resolution: L205n{args.sim_res}TNG")
        print(f"Total simulation mass: {total_sim_mass:.4e} M☉/h")
        print(f"DMO particle mass: {dm_mass_per_particle:.4e} M☉/h")
        print(f"Total particles: {sim_config['n_particles']:,}")
        print("=" * 70)
        sys.stdout.flush()
    
    # Load halo catalog
    if rank == 0:
        print("\n[1/4] Loading halo catalog...")
        sys.stdout.flush()
    
    matches_file = os.path.join(output_dir, 'matches', f'matches_snap{args.snap:03d}.npz')
    
    with np.load(matches_file) as data:
        all_masses = data['dmo_masses'] * MASS_UNIT
        all_positions = data['dmo_positions'] / 1e3  # kpc -> Mpc
        all_radii = data['dmo_radii'] / 1e3  # kpc -> Mpc
    
    # Select halos above mass threshold
    log_masses = np.log10(all_masses)
    mass_mask = log_masses >= args.mass_min
    
    halos = {
        'masses': all_masses[mass_mask],
        'positions': all_positions[mass_mask],
        'radii': all_radii[mass_mask],
        'log_masses': log_masses[mass_mask],
    }
    
    if rank == 0:
        print(f"  Total halos: {len(all_masses)}")
        print(f"  Halos above 10^{args.mass_min}: {len(halos['masses'])}")
    
    # Load DMO particles and build KDTree
    if rank == 0:
        print("\n[2/4] Loading DMO particles...")
        sys.stdout.flush()
    
    coords, dm_mass = load_dmo_particles(args.snap, args.sim_res)
    tree = build_tree(coords)
    
    # Precompute halo particle data
    if rank == 0:
        print("\n[3/4] Precomputing halo particle data...")
        sys.stdout.flush()
    
    halo_data = precompute_halo_particle_data(coords, tree, halos, max_radius_mult=5.0)
    
    # Compute mass fractions for all configurations
    if rank == 0:
        print("\n[4/4] Computing mass fractions for all configurations...")
        sys.stdout.flush()
    
    # Generate bins
    mass_bins = list(combinations(MASS_EDGES, 2))
    radius_shells = list(combinations(RADIUS_EDGES, 2))
    
    n_mass_bins = len(mass_bins)
    n_radius_shells = len(radius_shells)
    
    # Results matrices
    local_mass_excised = np.zeros((n_mass_bins, n_radius_shells), dtype=np.float64)
    local_particle_count = np.zeros((n_mass_bins, n_radius_shells), dtype=np.int64)
    config_labels = []
    
    for i, (log_M_lo, log_M_hi) in enumerate(mass_bins):
        M_lo, M_hi = 10**log_M_lo, 10**log_M_hi
        
        # Select halos in this mass bin
        halo_mask = (halos['log_masses'] >= log_M_lo) & (halos['log_masses'] < log_M_hi)
        halo_indices = np.where(halo_mask)[0]
        n_halos_in_bin = len(halo_indices)
        
        if rank == 0:
            print(f"\n  Mass bin [{log_M_lo:.1f}, {log_M_hi:.1f}): {n_halos_in_bin} halos")
        
        for j, (R_inner, R_outer) in enumerate(radius_shells):
            # Get particles in shell
            particle_idx = get_particles_in_shell(halo_data, halo_indices, R_inner, R_outer)
            
            n_particles = len(particle_idx)
            local_mass_excised[i, j] = n_particles * dm_mass
            local_particle_count[i, j] = n_particles
            
            if rank == 0:
                label = get_config_label(M_lo, M_hi, R_inner, R_outer)
                config_labels.append(label)
                print(f"    R=[{R_inner:.1f}, {R_outer:.1f}): {n_particles:,} particles, "
                      f"{n_particles * dm_mass:.4e} M☉/h")
    
    # Reduce across MPI ranks
    if comm is not None and size > 1:
        global_mass_excised = np.zeros_like(local_mass_excised)
        global_particle_count = np.zeros_like(local_particle_count)
        comm.Reduce(local_mass_excised, global_mass_excised, op=MPI.SUM, root=0)
        comm.Reduce(local_particle_count, global_particle_count, op=MPI.SUM, root=0)
    else:
        global_mass_excised = local_mass_excised
        global_particle_count = local_particle_count
    
    # Save results
    if rank == 0:
        mass_fraction_matrix = global_mass_excised / total_sim_mass * 100  # percent
        
        print("\n" + "=" * 70)
        print("RESULTS: MASS FILLING FRACTIONS (%)")
        print("=" * 70)
        
        # Print matrix
        print("\nRadius shells →")
        print("Mass bins ↓", end="")
        for R_in, R_out in radius_shells:
            print(f" | [{R_in:.1f},{R_out:.1f})", end="")
        print()
        print("-" * 70)
        
        for i, (log_M_lo, log_M_hi) in enumerate(mass_bins):
            print(f"[{log_M_lo:.1f},{log_M_hi:.1f})", end="")
            for j in range(n_radius_shells):
                val = mass_fraction_matrix[i, j]
                if val >= 10:
                    print(f" | {val:7.2f}%", end="")
                elif val >= 1:
                    print(f" | {val:7.3f}%", end="")
                else:
                    print(f" | {val:7.4f}%", end="")
            print()
        
        print("\n" + "=" * 70)
        print(f"Total time: {time.time() - t_start:.1f}s")
        
        # Save to file
        output_file = os.path.join(output_dir, f'mass_fractions_snap{args.snap:03d}.npz')
        np.savez_compressed(
            output_file,
            mass_fraction_matrix=mass_fraction_matrix,
            mass_excised_matrix=global_mass_excised,
            particle_count_matrix=global_particle_count,
            total_sim_mass=total_sim_mass,
            dm_mass_per_particle=dm_mass_per_particle,
            mass_edges=np.array(MASS_EDGES),
            radius_edges=np.array(RADIUS_EDGES),
            mass_bins=np.array(mass_bins),
            radius_shells=np.array(radius_shells),
            snapshot=args.snap,
            sim_res=args.sim_res,
        )
        print(f"\nSaved to: {output_file}")


if __name__ == '__main__':
    main()
