#!/usr/bin/env python3
"""
Ramp Boundary Test for Replace Models
======================================

This script tests the effect of hard radial cutoffs vs smooth cosine ramps
in the Replace formalism. 

Experiment Design:
- Single snapshot (snap 96, z≈0.04)
- Single mass bin: M > 10^13 M⊙/h (cumulative)
- Four outer radii: α = 0.5, 1.0, 3.0, 5.0 R₂₀₀
- Three ramp widths: δ = 0 (hard), 0.1, 0.2, 0.5 R₂₀₀
- Inner radius always 0 (no inner boundary ramp needed)

The cosine ramp function at the outer boundary:
    w(r) = 1                                           if r < r_outer - δ
         = ½[1 + cos(π(r - r_outer + δ)/(2δ))]        if r_outer - δ ≤ r < r_outer + δ
         = 0                                           if r ≥ r_outer + δ

For the Replace field:
    ρ_Replace = (1-w) × ρ_DMO + w × ρ_Hydro
    
This is implemented at the particle level:
    - DMO particle: contributes mass × (1 - w(r))
    - Hydro particle: contributes mass × w(r)

Usage:
    # Single-node test (small scale)
    mpirun -np 8 python generate_ramp_test.py --snap 96 --test
    
    # Full run
    sbatch run_ramp_test.sh
"""

import os
import sys
import time
import argparse
import numpy as np
import h5py
from mpi4py import MPI

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import from generate_all_unified (only what actually exists)
from generate_all_unified import (
    BOX_SIZE, GRID_RES, LENSPLANE_CONFIG, MASS_UNIT, SIM_PATHS,
    DistributedParticles, TransformGenerator,
    precompute_halo_particle_data,
    project_lensplane,
    SNAPSHOT_ORDER, SNAPSHOT_TO_INDEX
)

# Matches file base path
MATCHES_BASE = '/mnt/home/mlee1/ceph/hydro_replace_fields/L205n2500TNG'

# ============================================================================
# Constants
# ============================================================================

# Test configuration
SNAP = 96  # z ≈ 0.04
M_LO = 1e13  # Lower mass bound (M⊙/h)
M_HI = 1e15  # Upper mass bound (cumulative)
R_INNER = 0.0  # Inner radius (always 0)

# Outer radii to test
ALPHA_VALUES = [0.5, 1.0, 3.0, 5.0]

# Ramp widths (δ in units of R₂₀₀)
DELTA_VALUES = [0.0, 0.1, 0.2, 0.5]

# Output paths
OUTPUT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_LP_ramp_test/L205n2500TNG'


# ============================================================================
# Cosine Ramp Weight Functions
# ============================================================================

def cosine_ramp_weight(r, r_outer, delta):
    """
    Compute weight for smooth transition at outer boundary.
    
    Args:
        r: distance in units of R200 (array or scalar)
        r_outer: outer radius cutoff in units of R200
        delta: ramp half-width in units of R200
    
    Returns:
        weight w in [0, 1] where:
            w = 1 for r < r_outer - delta (fully inside)
            w = smooth transition for r_outer - delta <= r < r_outer + delta
            w = 0 for r >= r_outer + delta (fully outside)
    """
    r = np.asarray(r)
    w = np.ones_like(r, dtype=np.float32)
    
    if delta <= 0:
        # Hard cutoff
        w[r >= r_outer] = 0.0
    else:
        # Transition region
        transition_start = r_outer - delta
        transition_end = r_outer + delta
        
        in_transition = (r >= transition_start) & (r < transition_end)
        beyond = r >= transition_end
        
        # Cosine ramp: 1 → 0 over [r_outer - δ, r_outer + δ]
        # w = 0.5 * [1 + cos(π * (r - r_outer + δ) / (2δ))]
        r_norm = (r[in_transition] - transition_start) / (2 * delta)
        w[in_transition] = 0.5 * (1 + np.cos(np.pi * r_norm))
        w[beyond] = 0.0
    
    return w


def get_particles_in_shell_weighted(halo_data, halo_indices, r_inner, r_outer, delta):
    """
    Get particle indices and weights for halos within a radial shell with smooth boundary.
    
    For each particle, computes the blending weight based on its distance from halo centers.
    If a particle is near multiple halos, it takes the maximum weight (most "inside").
    
    Args:
        halo_data: list of (indices, distances) from precompute_halo_particle_data
        halo_indices: array of halo indices in the mass bin
        r_inner: inner radius in units of R200
        r_outer: outer radius in units of R200
        delta: ramp half-width in units of R200
    
    Returns:
        particle_indices: unique particle indices that have non-zero weight
        particle_weights: corresponding weights in [0, 1]
    """
    # Dictionary to track max weight per particle
    particle_weight_dict = {}
    
    # Extended radius to capture particles in the transition region
    r_max_extended = r_outer + delta if delta > 0 else r_outer
    
    for i in halo_indices:
        idx, dist = halo_data[i]
        if len(idx) == 0:
            continue
        
        # Only consider particles within extended range
        in_range = (dist >= r_inner) & (dist < r_max_extended)
        if not np.any(in_range):
            continue
        
        idx_in_range = idx[in_range]
        dist_in_range = dist[in_range]
        
        # Compute weights
        weights = cosine_ramp_weight(dist_in_range, r_outer, delta)
        
        # Update max weight for each particle
        for pid, w in zip(idx_in_range, weights):
            if w > 0:
                if pid not in particle_weight_dict:
                    particle_weight_dict[pid] = w
                else:
                    particle_weight_dict[pid] = max(particle_weight_dict[pid], w)
    
    if not particle_weight_dict:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    
    particle_indices = np.array(list(particle_weight_dict.keys()), dtype=np.int64)
    particle_weights = np.array([particle_weight_dict[i] for i in particle_indices], dtype=np.float32)
    
    return particle_indices, particle_weights


# ============================================================================
# Lensplane Generation with Weighted Particles
# ============================================================================

def generate_ramp_lensplanes(
    snap, alpha, delta, 
    dmo_particles, hydro_particles, halos,
    dmo_halo_data, hydro_halo_data,
    transforms, comm, output_base
):
    """
    Generate lensplanes for a single (alpha, delta) configuration.
    
    Args:
        snap: snapshot number
        alpha: outer radius in units of R200
        delta: ramp half-width in units of R200
        dmo_particles: DistributedParticles for DMO
        hydro_particles: DistributedParticles for Hydro
        halos: halo catalog dict
        dmo_halo_data: precomputed halo-particle data for DMO
        hydro_halo_data: precomputed halo-particle data for Hydro
        transforms: TransformGenerator
        comm: MPI communicator
        output_base: output directory base
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    lp_grid = LENSPLANE_CONFIG['grid_res']
    pps = transforms.pps
    n_realizations = transforms.n_realizations
    
    config_label = f"ramp_alpha{alpha}_delta{delta}"
    
    if rank == 0:
        print(f"\n  Processing {config_label}...")
        t0 = time.time()
    
    # Get snapshot index
    if snap not in SNAPSHOT_TO_INDEX:
        if rank == 0:
            print(f"    Warning: snapshot {snap} not in SNAPSHOT_ORDER")
        return
    snapshot_idx = SNAPSHOT_TO_INDEX[snap]
    
    # Select halos in mass bin
    halo_mask = (halos['masses'] >= M_LO) & (halos['masses'] < M_HI)
    halo_indices = np.where(halo_mask)[0]
    n_selected = len(halo_indices)
    
    if rank == 0:
        print(f"    {n_selected} halos with M > 10^13 M⊙/h")
    
    if n_selected == 0:
        if rank == 0:
            print(f"    No halos, skipping")
        return
    
    # Get weighted particles
    dmo_idx, dmo_weights = get_particles_in_shell_weighted(
        dmo_halo_data, halo_indices, R_INNER, alpha, delta
    )
    hydro_idx, hydro_weights = get_particles_in_shell_weighted(
        hydro_halo_data, halo_indices, R_INNER, alpha, delta
    )
    
    n_dmo_affected = len(dmo_idx)
    n_hydro_affected = len(hydro_idx)
    
    if rank == 0:
        print(f"    DMO particles affected: {n_dmo_affected:,}")
        print(f"    Hydro particles affected: {n_hydro_affected:,}")
        if delta > 0:
            n_dmo_partial = np.sum((dmo_weights > 0) & (dmo_weights < 1))
            n_hydro_partial = np.sum((hydro_weights > 0) & (hydro_weights < 1))
            print(f"    DMO in transition: {n_dmo_partial:,}")
            print(f"    Hydro in transition: {n_hydro_partial:,}")
    
    # Build weighted particle arrays
    # For Replace: ρ_R = (1-w)*ρ_DMO + w*ρ_Hydro
    # DMO contribution: all particles with reduced mass in affected regions
    # Hydro contribution: affected particles with their weight
    
    # Create mask for affected DMO particles
    dmo_affected_mask = np.zeros(len(dmo_particles.coords), dtype=bool)
    dmo_affected_mask[dmo_idx] = True
    
    # DMO background: particles NOT affected (full mass)
    dmo_bg_coords = dmo_particles.coords[~dmo_affected_mask]
    dmo_bg_masses = dmo_particles.masses[~dmo_affected_mask]
    
    # DMO affected: particles with reduced mass (1 - w)
    dmo_aff_coords = dmo_particles.coords[dmo_idx]
    dmo_aff_masses = dmo_particles.masses[dmo_idx] * (1 - dmo_weights)
    
    # Hydro affected: particles with weight w
    hydro_aff_coords = hydro_particles.coords[hydro_idx]
    hydro_aff_masses = hydro_particles.masses[hydro_idx] * hydro_weights
    
    # Combine into Replace field
    local_pos_replace = np.concatenate([
        dmo_bg_coords,
        dmo_aff_coords,
        hydro_aff_coords
    ])
    local_mass_replace = np.concatenate([
        dmo_bg_masses,
        dmo_aff_masses,
        hydro_aff_masses
    ])
    
    # Remove particles with zero mass (fully replaced DMO)
    nonzero_mask = local_mass_replace > 0
    local_pos_replace = local_pos_replace[nonzero_mask]
    local_mass_replace = local_mass_replace[nonzero_mask]
    
    if rank == 0:
        print(f"    Total Replace particles: {len(local_pos_replace):,}")
    
    # Generate lensplanes
    for real_idx in range(n_realizations):
        t = transforms.get_transform(real_idx, snapshot_idx)
        
        for pps_slice in range(pps):
            file_idx = snapshot_idx * pps + pps_slice
            
            # Project local particles
            local_delta = project_lensplane(
                local_pos_replace, local_mass_replace, t,
                lp_grid, BOX_SIZE, pps_slice, pps
            )
            
            # MPI reduce
            if rank == 0:
                global_delta = np.zeros((lp_grid, lp_grid), dtype=np.float64)
            else:
                global_delta = None
            
            local_delta_contig = np.ascontiguousarray(local_delta.astype(np.float64))
            comm.Reduce(local_delta_contig, global_delta, op=MPI.SUM, root=0)
            
            if rank == 0:
                # Output path
                out_dir = os.path.join(
                    output_base, f'snap_{snap}', config_label, f'LP_{real_idx:02d}'
                )
                os.makedirs(out_dir, exist_ok=True)
                
                out_path = os.path.join(out_dir, f'lensplane_{file_idx:02d}.npz')
                np.savez_compressed(out_path, plane=global_delta.astype(np.float32))
    
    comm.Barrier()
    
    if rank == 0:
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ramp boundary test for Replace models')
    parser.add_argument('--snap', type=int, default=96, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, default=2500, help='TNG resolution')
    parser.add_argument('--test', action='store_true', help='Quick test mode (fewer realizations)')
    parser.add_argument('--alpha', type=float, default=None, help='Single alpha to test')
    parser.add_argument('--delta', type=float, default=None, help='Single delta to test')
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    snap = args.snap
    sim_res = args.sim_res
    
    # Determine which configurations to run
    if args.alpha is not None and args.delta is not None:
        configs = [(args.alpha, args.delta)]
    else:
        configs = [(a, d) for a in ALPHA_VALUES for d in DELTA_VALUES]
    
    if rank == 0:
        print("=" * 70)
        print("RAMP BOUNDARY TEST FOR REPLACE MODELS")
        print("=" * 70)
        print(f"Snapshot: {snap}")
        print(f"Mass bin: M > 10^13 M⊙/h (cumulative)")
        print(f"Alpha values: {ALPHA_VALUES}")
        print(f"Delta values: {DELTA_VALUES}")
        print(f"Total configurations: {len(configs)}")
        print(f"MPI ranks: {size}")
        print("=" * 70)
    
    # Load transforms
    transforms_file = f'/mnt/home/mlee1/ceph/hydro_replace_LP/L205n{sim_res}TNG/transforms.h5'
    if os.path.exists(transforms_file):
        transforms = TransformGenerator.load(transforms_file)
        if rank == 0:
            print(f"Loaded transforms from {transforms_file}")
    else:
        transforms = TransformGenerator(
            n_realizations=20 if not args.test else 2,
            n_snapshots=20,
            pps=2,
            seed=42,
            box_size=BOX_SIZE
        )
        if rank == 0:
            print("Created new transforms")
    
    # Reduce realizations for test mode
    if args.test:
        transforms.n_realizations = 2
    
    # Load halos from matches file
    if rank == 0:
        print(f"\nLoading halo catalog for snap {snap}...")
    
    matches_file = f'{MATCHES_BASE}/matches/matches_snap{snap:03d}.npz'
    with np.load(matches_file) as data:
        all_masses = data['dmo_masses'] * MASS_UNIT
        all_positions = data['dmo_positions'] / 1e3  # kpc -> Mpc
        all_radii = data['dmo_radii'] / 1e3  # kpc -> Mpc
    
    halos = {
        'masses': all_masses,
        'positions': all_positions,
        'radii': all_radii,
    }
    
    if rank == 0:
        print(f"  Total halos: {len(halos['masses']):,}")
        n_above_threshold = np.sum(halos['masses'] >= M_LO)
        print(f"  Halos with M > 10^13: {n_above_threshold:,}")
    
    # Load particles using DistributedParticles class
    if rank == 0:
        print(f"\nLoading DMO particles...")
    dmo_particles = DistributedParticles(snap, sim_res, 'dmo', radius_mult=1.0)
    dmo_particles.load()
    
    if rank == 0:
        print(f"\nLoading Hydro particles...")
    hydro_particles = DistributedParticles(snap, sim_res, 'hydro', radius_mult=1.0)
    hydro_particles.load()
    
    # Build KDTrees
    if rank == 0:
        print(f"\nBuilding KDTrees...")
    dmo_particles.build_tree()
    hydro_particles.build_tree()
    
    # Precompute halo-particle associations
    max_alpha = max(ALPHA_VALUES) + max(DELTA_VALUES)  # Extended for ramp
    
    if rank == 0:
        print(f"\nPrecomputing halo-particle data (max radius: {max_alpha}×R200)...")
    
    dmo_halo_data = precompute_halo_particle_data(
        dmo_particles, halos, max_alpha, comm
    )
    hydro_halo_data = precompute_halo_particle_data(
        hydro_particles, halos, max_alpha, comm
    )
    
    # Generate lensplanes for each configuration
    for alpha, delta in configs:
        generate_ramp_lensplanes(
            snap, alpha, delta,
            dmo_particles, hydro_particles, halos,
            dmo_halo_data, hydro_halo_data,
            transforms, comm, OUTPUT_BASE
        )
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("RAMP BOUNDARY TEST COMPLETE")
        print("=" * 70)


if __name__ == '__main__':
    main()
