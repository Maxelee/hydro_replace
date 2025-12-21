#!/usr/bin/env python
"""
Fast bijective halo matching using histogram binning.

Based on working code from:
  /mnt/home/mlee1/Hydro_replacement/bijective2.py

Generates matches for a specific simulation resolution and snapshot.
Matches are saved to: {output_dir}/matches/matches_snap{NNN}.npz

Usage:
    mpirun -np 32 python generate_matches.py --snap 99 --resolution 2500
    mpirun -np 32 python generate_matches.py --snap 99 --resolution 625
"""

import numpy as np
import argparse
import os
import time
from mpi4py import MPI
from illustris_python import snapshot, groupcat

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================================
# Simulation paths for different resolutions
# ============================================================================

SIM_PATHS = {
    # TNG300-1 (high resolution)
    2500: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output',
        'dmo_mass': 0.0047271638660809,  # 10^10 Msun/h
        'hydro_dm_mass': 0.00398342749867548,
    },
    # TNG300-2 (medium resolution)
    1250: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG/output',
        'dmo_mass': 0.0378173109,  # 8x higher mass
        'hydro_dm_mass': 0.0318674199,
    },
    # TNG300-3 (low resolution)
    625: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG/output',
        'dmo_mass': 0.3025384873,  # 64x higher mass
        'hydro_dm_mass': 0.2549393594,
    },
}

CONFIG = {
    'mass_unit': 1e10,  # Msun/h
    'min_halo_mass': 1e10,  # Msun/h - minimum mass to consider
    'min_overlap_fraction': 0.5,
    'min_particles': 100,
    'output_base': '/mnt/home/mlee1/ceph/hydro_replace_fields',
}


class SimulationData:
    """Container for pre-loaded simulation data."""
    
    def __init__(self, basePath, snapNum, min_mass, rank):
        self.basePath = basePath
        self.snapNum = snapNum
        
        if rank == 0:
            print(f"  Loading halo catalog from {basePath.split('/')[-2]}...")
        
        # Load halo catalog
        halos = groupcat.loadHalos(
            basePath, snapNum,
            fields=['Group_M_Crit200', 'GroupPos', 'GroupVel', 'GroupLenType']
        )
        
        masses = halos['Group_M_Crit200'] * CONFIG['mass_unit']
        self.mass_mask = masses >= min_mass
        
        self.indices = np.where(self.mass_mask)[0]
        self.positions = halos['GroupPos'][self.mass_mask]
        self.velocities = halos['GroupVel'][self.mass_mask]
        self.masses = masses[self.mass_mask]
        
        # Load all DM particle IDs
        if rank == 0:
            print(f"  Loading DM particle IDs from {basePath.split('/')[-2]}...")
        
        self.particle_ids = snapshot.loadSubset(basePath, snapNum, 'dm', ['ParticleIDs'])
        
        # Create offset array from GroupLenType (DM is type 1)
        self.offsets = self._create_offsets(halos['GroupLenType'][:, 1])
        
        if rank == 0:
            print(f"  Loaded {len(self.indices)} halos, {len(self.particle_ids):,} particles")
    
    def _create_offsets(self, lengths):
        """Create cumulative offset array from particle counts per halo."""
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        return offsets
    
    def get_halo_particles(self, halo_idx):
        """Get particle IDs for a specific halo using offsets."""
        start = self.offsets[halo_idx]
        end = self.offsets[halo_idx + 1]
        return self.particle_ids[start:end]


def find_best_match(source_ids, target_data):
    """
    Find which halo in target_data has the most overlap with source_ids.
    Uses histogram binning for O(N) complexity.
    """
    # Find indices in target particle array that match source IDs
    matched_indices = np.nonzero(np.isin(target_data.particle_ids, source_ids))[0]
    
    if len(matched_indices) == 0:
        return None, 0
    
    # Histogram bins matched particles into halos using offset array
    histogram, _ = np.histogram(matched_indices, bins=target_data.offsets)
    
    # Find halo with maximum overlap
    best_halo = np.argmax(histogram)
    max_overlap = histogram[best_halo]
    
    return best_halo, max_overlap


def match_single_halo_bijective(dmo_idx, dmo_data, hydro_data, config):
    """
    Perform bijective matching for a single DMO halo.
    
    Returns (hydro_idx, overlap_fraction) if bijective match found, else None.
    """
    # Get DMO halo particles
    dmo_particles = dmo_data.get_halo_particles(dmo_idx)
    
    if len(dmo_particles) < config['min_particles']:
        return None
    
    # Forward match: DMO -> Hydro
    hydro_idx, forward_overlap = find_best_match(dmo_particles, hydro_data)
    
    if hydro_idx is None:
        return None
    
    forward_fraction = forward_overlap / len(dmo_particles)
    if forward_fraction < config['min_overlap_fraction']:
        return None
    
    # Get matched hydro halo particles
    hydro_particles = hydro_data.get_halo_particles(hydro_idx)
    
    if len(hydro_particles) < config['min_particles']:
        return None
    
    # Reverse match: Hydro -> DMO
    dmo_reverse_idx, reverse_overlap = find_best_match(hydro_particles, dmo_data)
    
    if dmo_reverse_idx != dmo_idx:
        return None  # Not bijective
    
    reverse_fraction = reverse_overlap / len(hydro_particles)
    if reverse_fraction < config['min_overlap_fraction']:
        return None
    
    # Use minimum of both fractions as quality metric
    final_fraction = min(forward_fraction, reverse_fraction)
    
    return (hydro_idx, final_fraction)


def generate_matches(snapNum, resolution, output_dir):
    """Generate bijective halo matches for given snapshot and resolution."""
    
    if resolution not in SIM_PATHS:
        raise ValueError(f"Unknown resolution: {resolution}. Available: {list(SIM_PATHS.keys())}")
    
    sim_config = SIM_PATHS[resolution]
    
    if rank == 0:
        print("=" * 70)
        print(f"Bijective Halo Matching")
        print("=" * 70)
        print(f"Resolution: {resolution}^3")
        print(f"Snapshot: {snapNum}")
        print(f"MPI ranks: {size}")
        print(f"Min overlap fraction: {CONFIG['min_overlap_fraction']}")
        print(f"Min particles: {CONFIG['min_particles']}")
        print("=" * 70)
    
    t_start = time.time()
    
    # Load simulation data (each rank loads independently)
    if rank == 0:
        print("\n[1/3] Loading simulation data...")
    
    comm.Barrier()
    t_load_start = time.time()
    
    dmo_data = SimulationData(
        sim_config['dmo'], snapNum,
        CONFIG['min_halo_mass'], rank
    )
    hydro_data = SimulationData(
        sim_config['hydro'], snapNum,
        CONFIG['min_halo_mass'], rank
    )
    
    comm.Barrier()
    t_load = time.time() - t_load_start
    
    if rank == 0:
        print(f"  Loading took {t_load/60:.1f} minutes")
        print(f"\n[2/3] Matching halos across {size} ranks...")
    
    # Distribute halos across ranks
    my_dmo_indices = dmo_data.indices[rank::size]
    my_matches = {}
    
    for i, dmo_idx in enumerate(my_dmo_indices):
        if i % 100 == 0 and i > 0:
            print(f"Rank {rank}: {i}/{len(my_dmo_indices)} halos processed")
        
        result = match_single_halo_bijective(dmo_idx, dmo_data, hydro_data, CONFIG)
        
        if result is not None:
            hydro_idx, overlap = result
            my_matches[dmo_idx] = (hydro_idx, overlap)
    
    print(f"Rank {rank}: Found {len(my_matches)} bijective matches")
    
    # Gather results to rank 0
    all_matches = comm.gather(my_matches, root=0)
    comm.Barrier()
    
    # Save results
    if rank == 0:
        print("\n[3/3] Saving results...")
        
        bijective_matches = {}
        for matches in all_matches:
            bijective_matches.update(matches)
        
        print(f"  Total bijective matches: {len(bijective_matches)}")
        
        if len(bijective_matches) == 0:
            print("\nWARNING: No bijective matches found!")
            return None
        
        # Convert to arrays
        dmo_matched = np.array(list(bijective_matches.keys()))
        hydro_matched = np.array([h for h, _ in bijective_matches.values()])
        overlaps = np.array([o for _, o in bijective_matches.values()])
        
        # Get halo properties for matched halos
        dmo_map = {idx: i for i, idx in enumerate(dmo_data.indices)}
        hydro_map = {idx: i for i, idx in enumerate(hydro_data.indices)}
        
        dmo_positions = np.array([dmo_data.positions[dmo_map[i]] for i in dmo_matched])
        dmo_masses = np.array([dmo_data.masses[dmo_map[i]] for i in dmo_matched])
        
        hydro_positions = np.array([hydro_data.positions[hydro_map[i]] for i in hydro_matched])
        hydro_masses = np.array([hydro_data.masses[hydro_map[i]] for i in hydro_matched])
        
        # Create output directory
        matches_dir = f"{output_dir}/L205n{resolution}TNG/matches"
        os.makedirs(matches_dir, exist_ok=True)
        
        output_file = f"{matches_dir}/matches_snap{snapNum:03d}.npz"
        
        np.savez_compressed(
            output_file,
            dmo_indices=dmo_matched,
            hydro_indices=hydro_matched,
            overlap_fractions=overlaps,
            dmo_positions=dmo_positions,
            dmo_masses=dmo_masses,
            hydro_positions=hydro_positions,
            hydro_masses=hydro_masses,
            min_overlap_fraction=CONFIG['min_overlap_fraction'],
            min_halo_mass=CONFIG['min_halo_mass'],
            min_particles=CONFIG['min_particles'],
            snapNum=snapNum,
            resolution=resolution,
        )
        
        print(f"\nSaved to: {output_file}")
        print(f"\nStatistics:")
        print(f"  Total matches: {len(dmo_matched)}")
        print(f"  Mass range: {np.log10(dmo_masses.min()):.2f} - {np.log10(dmo_masses.max()):.2f} log10(Msun/h)")
        print(f"  Mean overlap: {np.mean(overlaps):.3f}")
        
        t_total = time.time() - t_start
        print(f"\nTotal time: {t_total/60:.1f} minutes")
        print("=" * 70)
        
        return output_file
    
    return None


def check_matches_exist(snapNum, resolution, output_dir):
    """Check if matches file already exists."""
    matches_file = f"{output_dir}/L205n{resolution}TNG/matches/matches_snap{snapNum:03d}.npz"
    return os.path.exists(matches_file), matches_file


def main():
    parser = argparse.ArgumentParser(description='Generate bijective halo matches')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--resolution', type=int, required=True,
                        choices=[625, 1250, 2500], help='Simulation resolution')
    parser.add_argument('--output-dir', type=str, default=CONFIG['output_base'],
                        help='Output directory')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if matches exist')
    
    args = parser.parse_args()
    
    # Check if matches already exist
    exists, matches_file = check_matches_exist(args.snap, args.resolution, args.output_dir)
    
    if exists and not args.force:
        if rank == 0:
            print(f"Matches already exist: {matches_file}")
            print("Use --force to regenerate")
        return
    
    generate_matches(args.snap, args.resolution, args.output_dir)


if __name__ == "__main__":
    main()
