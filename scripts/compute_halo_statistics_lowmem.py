#!/usr/bin/env python
"""
Low-Memory Distributed Halo Statistics Computation.

Optimized version that avoids broadcasting large particle ID lists.
Instead, all ranks read particle IDs from cache on-demand.

Key differences from compute_halo_statistics_distributed.py:
- All ranks read the cache file (read-only, parallel-safe)
- Particle IDs are read per-halo as needed, not loaded all at once
- Smaller memory footprint enables running with fewer tasks

Usage:
    mpirun -np 32 python compute_halo_statistics_lowmem.py --snap 99 --sim-res 1250
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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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
BOX_SIZE = 205.0  # Mpc/h
MASS_UNIT = 1e10  # Convert to Msun/h


# ============================================================================
# Distributed Particle Data Manager
# ============================================================================

class DistributedParticleData:
    """
    Manages distributed particle data across MPI ranks.
    
    Each rank loads a subset of snapshot files and builds local lookups.
    Particle queries are handled by broadcasting IDs and gathering results.
    """
    
    def __init__(self, snapshot: int, sim_res: int, mode: str, verbose: bool = True):
        """
        Initialize distributed particle data.
        
        Args:
            snapshot: Snapshot number
            sim_res: Simulation resolution (625, 1250, 2500)
            mode: 'dmo' or 'hydro'
            verbose: Print progress info
        """
        self.snapshot = snapshot
        self.sim_res = sim_res
        self.mode = mode
        self.verbose = verbose
        
        sim_config = SIM_PATHS[sim_res]
        self.dm_mass = sim_config[f'{mode}_dm_mass'] * MASS_UNIT
        
        if mode == 'dmo':
            self.basePath = sim_config['dmo']
            self.particle_types = [1]
        else:
            self.basePath = sim_config['hydro']
            self.particle_types = [0, 1, 4]
        
        # Load distributed particle data
        self._load_distributed_data()
    
    def _load_distributed_data(self):
        """Load particle data distributed across ranks."""
        snap_dir = f"{self.basePath}/snapdir_{self.snapshot:03d}/"
        all_files = sorted(glob.glob(f"{snap_dir}/snap_{self.snapshot:03d}.*.hdf5"))
        
        # Distribute files across ranks
        my_files = [f for i, f in enumerate(all_files) if i % size == rank]
        
        coords_list = []
        masses_list = []
        pids_list = []
        types_list = []
        
        for filepath in my_files:
            with h5py.File(filepath, 'r') as f:
                for ptype in self.particle_types:
                    pt_key = f'PartType{ptype}'
                    if pt_key not in f:
                        continue
                    
                    n_part = f[pt_key]['Coordinates'].shape[0]
                    if n_part == 0:
                        continue
                    
                    coords = f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3  # kpc to Mpc
                    pids = f[pt_key]['ParticleIDs'][:]
                    
                    if 'Masses' in f[pt_key]:
                        m = f[pt_key]['Masses'][:].astype(np.float32) * MASS_UNIT
                    else:
                        m = np.full(n_part, self.dm_mass, dtype=np.float32)
                    
                    coords_list.append(coords)
                    masses_list.append(m)
                    pids_list.append(pids)
                    types_list.append(np.full(n_part, ptype, dtype=np.int8))
        
        if coords_list:
            self.coords = np.concatenate(coords_list)
            self.masses = np.concatenate(masses_list)
            self.pids = np.concatenate(pids_list)
            self.types = np.concatenate(types_list)
        else:
            self.coords = np.zeros((0, 3), dtype=np.float32)
            self.masses = np.zeros(0, dtype=np.float32)
            self.pids = np.zeros(0, dtype=np.int64)
            self.types = np.zeros(0, dtype=np.int8)
        
        # Build ID lookup
        self.id_to_idx = {int(pid): i for i, pid in enumerate(self.pids)}
        
        # Gather statistics
        local_count = len(self.pids)
        total_count = comm.reduce(local_count, op=MPI.SUM, root=0)
        
        if rank == 0 and self.verbose:
            print(f"    Loaded {total_count:,} {self.mode.upper()} particles across {size} ranks")
            sys.stdout.flush()
    
    def query_particles(self, particle_ids, center, r200):
        """
        Query particles by ID and compute radii from center.
        
        This is a collective operation - all ranks must call it.
        
        Returns:
            coords, masses, radii_r200, types (gathered on rank 0, None on others)
        """
        # Find local matches
        local_indices = []
        for pid in particle_ids:
            idx = self.id_to_idx.get(int(pid))
            if idx is not None:
                local_indices.append(idx)
        
        if local_indices:
            local_coords = self.coords[local_indices]
            local_masses = self.masses[local_indices]
            local_types = self.types[local_indices]
            
            # Compute radii from center (with periodic wrapping)
            delta = local_coords - center
            delta = np.where(delta > BOX_SIZE/2, delta - BOX_SIZE, delta)
            delta = np.where(delta < -BOX_SIZE/2, delta + BOX_SIZE, delta)
            local_radii = np.sqrt(np.sum(delta**2, axis=1)) / r200
        else:
            local_coords = np.zeros((0, 3), dtype=np.float32)
            local_masses = np.zeros(0, dtype=np.float32)
            local_radii = np.zeros(0, dtype=np.float32)
            local_types = np.zeros(0, dtype=np.int8)
        
        # Gather to rank 0
        all_coords = comm.gather(local_coords, root=0)
        all_masses = comm.gather(local_masses, root=0)
        all_radii = comm.gather(local_radii, root=0)
        all_types = comm.gather(local_types, root=0)
        
        if rank == 0:
            coords = np.concatenate(all_coords) if any(len(c) > 0 for c in all_coords) else np.zeros((0, 3), dtype=np.float32)
            masses = np.concatenate(all_masses) if any(len(m) > 0 for m in all_masses) else np.zeros(0, dtype=np.float32)
            radii = np.concatenate(all_radii) if any(len(r) > 0 for r in all_radii) else np.zeros(0, dtype=np.float32)
            types = np.concatenate(all_types) if any(len(t) > 0 for t in all_types) else np.zeros(0, dtype=np.int8)
            return coords, masses, radii, types
        else:
            return None, None, None, None


# ============================================================================
# Statistics Functions
# ============================================================================

def compute_baryon_fraction(masses, radii_r200, types, radii_mult):
    """Compute baryon fractions at various radii."""
    n_radii = len(radii_mult)
    
    f_baryon = np.zeros(n_radii, dtype=np.float32)
    f_gas = np.zeros(n_radii, dtype=np.float32)
    f_stellar = np.zeros(n_radii, dtype=np.float32)
    m_total = np.zeros(n_radii, dtype=np.float64)
    m_gas = np.zeros(n_radii, dtype=np.float64)
    m_stellar = np.zeros(n_radii, dtype=np.float64)
    m_dm = np.zeros(n_radii, dtype=np.float64)
    
    for i, r_mult in enumerate(radii_mult):
        mask = radii_r200 <= r_mult
        
        if np.sum(mask) == 0:
            continue
        
        m_tot = np.sum(masses[mask])
        m_g = np.sum(masses[mask & (types == 0)])  # Gas
        m_s = np.sum(masses[mask & (types == 4)])  # Stars
        m_d = np.sum(masses[mask & (types == 1)])  # DM
        
        m_total[i] = m_tot
        m_gas[i] = m_g
        m_stellar[i] = m_s
        m_dm[i] = m_d
        
        if m_tot > 0:
            f_baryon[i] = (m_g + m_s) / m_tot
            f_gas[i] = m_g / m_tot
            f_stellar[i] = m_s / m_tot
    
    return {
        'f_baryon': f_baryon,
        'f_gas': f_gas,
        'f_stellar': f_stellar,
        'm_total': m_total,
        'm_gas': m_gas,
        'm_stellar': m_stellar,
        'm_dm': m_dm,
    }


def compute_mass_conservation(dmo_masses, dmo_radii, hydro_masses, hydro_radii, hydro_types, radii_mult):
    """Compute mass conservation ratios between DMO and Hydro."""
    n_radii = len(radii_mult)
    
    m_dmo = np.zeros(n_radii, dtype=np.float64)
    ratio_total = np.zeros(n_radii, dtype=np.float32)
    ratio_dm = np.zeros(n_radii, dtype=np.float32)
    
    for i, r_mult in enumerate(radii_mult):
        dmo_mask = dmo_radii <= r_mult
        hydro_mask = hydro_radii <= r_mult
        
        m_dmo_r = np.sum(dmo_masses[dmo_mask])
        m_hydro_total = np.sum(hydro_masses[hydro_mask])
        m_hydro_dm = np.sum(hydro_masses[hydro_mask & (hydro_types == 1)])
        
        m_dmo[i] = m_dmo_r
        
        if m_dmo_r > 0:
            ratio_total[i] = m_hydro_total / m_dmo_r
            ratio_dm[i] = m_hydro_dm / m_dmo_r
    
    return {
        'm_dmo': m_dmo,
        'ratio_total': ratio_total,
        'ratio_dm': ratio_dm,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compute halo statistics (low memory)')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, default=625, choices=[625, 1250, 2500])
    parser.add_argument('--mass-min', type=float, default=12.5, help='log10(M_min / Msun/h)')
    parser.add_argument('--mass-max', type=float, default=None, help='log10(M_max / Msun/h)')
    parser.add_argument('--output-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Radii for statistics (in units of R200)
    radii_mult = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    n_radii = len(radii_mult)
    
    if rank == 0:
        print("=" * 70)
        print("DISTRIBUTED HALO STATISTICS (LOW MEMORY)")
        print("=" * 70)
        print(f"Snapshot: {args.snap}")
        print(f"Resolution: L205n{args.sim_res}TNG")
        print(f"Mass minimum: 10^{args.mass_min} Msun/h")
        print(f"Radii: {radii_mult}")
        print(f"MPI ranks: {size}")
        print("=" * 70)
        sys.stdout.flush()
    
    t_start = time.time()
    
    # ========================================================================
    # Load cache metadata (all ranks read the same file)
    # ========================================================================
    cache_file = os.path.join(
        CACHE_BASE, f'L205n{args.sim_res}TNG', 
        'particle_cache', f'cache_snap{args.snap:03d}.h5'
    )
    
    if rank == 0:
        print(f"\n[1/5] Loading halo cache: {cache_file}")
        sys.stdout.flush()
    
    # ALL ranks read the cache file for halo metadata (this is small)
    with h5py.File(cache_file, 'r') as f:
        halo_info = f['halo_info']
        dmo_indices = halo_info['halo_indices'][:]
        positions_dmo = halo_info['positions_dmo'][:]
        radii_dmo = halo_info['radii_dmo'][:]
        positions_hydro = halo_info['positions_hydro'][:]
        radii_hydro = halo_info['radii_hydro'][:]
        masses = halo_info['masses'][:]
        
        # Check cache format
        has_hydro_at_hydro = 'hydro_at_hydro' in f
    
    # Filter by mass
    log_masses = np.log10(masses)
    mass_mask = log_masses >= args.mass_min
    if args.mass_max:
        mass_mask &= log_masses < args.mass_max
    
    halo_cache_indices = np.where(mass_mask)[0]
    halo_dmo_indices = dmo_indices[mass_mask]
    halo_positions_dmo = positions_dmo[mass_mask]
    halo_radii_dmo = radii_dmo[mass_mask]
    halo_positions_hydro = positions_hydro[mass_mask]
    halo_radii_hydro = radii_hydro[mass_mask]
    halo_log_masses = log_masses[mass_mask]
    
    n_halos = len(halo_cache_indices)
    
    if rank == 0:
        print(f"  Total halos above M > 10^{args.mass_min}: {n_halos}")
        sys.stdout.flush()
    
    # ========================================================================
    # Load distributed particle data
    # ========================================================================
    if rank == 0:
        print(f"\n[2/5] Loading distributed DMO particle data...")
        sys.stdout.flush()
    
    dmo_data = DistributedParticleData(args.snap, args.sim_res, 'dmo', verbose=True)
    
    if rank == 0:
        print(f"\n[3/5] Loading distributed Hydro particle data...")
        sys.stdout.flush()
    
    hydro_data = DistributedParticleData(args.snap, args.sim_res, 'hydro', verbose=True)
    
    comm.Barrier()
    
    # ========================================================================
    # Process halos - particle IDs read on-demand from cache
    # ========================================================================
    if rank == 0:
        print(f"\n[4/5] Processing {n_halos} halos...")
        sys.stdout.flush()
        
        # Pre-allocate results
        results = {
            'dmo_indices': halo_dmo_indices.astype(np.int32),
            'log_masses': halo_log_masses.astype(np.float32),
            'positions_dmo': halo_positions_dmo.astype(np.float32),
            'radii_dmo': halo_radii_dmo.astype(np.float32),
            'positions_hydro': halo_positions_hydro.astype(np.float32),
            'radii_hydro': halo_radii_hydro.astype(np.float32),
            
            # Baryon fractions
            'f_baryon': np.zeros((n_halos, n_radii), dtype=np.float32),
            'f_gas': np.zeros((n_halos, n_radii), dtype=np.float32),
            'f_stellar': np.zeros((n_halos, n_radii), dtype=np.float32),
            'm_total': np.zeros((n_halos, n_radii), dtype=np.float64),
            'm_gas': np.zeros((n_halos, n_radii), dtype=np.float64),
            'm_stellar': np.zeros((n_halos, n_radii), dtype=np.float64),
            'm_dm_hydro': np.zeros((n_halos, n_radii), dtype=np.float64),
            
            # Mass conservation
            'm_dmo': np.zeros((n_halos, n_radii), dtype=np.float64),
            'ratio_total': np.zeros((n_halos, n_radii), dtype=np.float32),
            'ratio_dm': np.zeros((n_halos, n_radii), dtype=np.float32),
        }
    
    t_proc = time.time()
    
    # Open cache file for reading particle IDs on-demand
    # All ranks open their own handle (read-only, safe for parallel access)
    cache_handle = h5py.File(cache_file, 'r')
    
    for i in range(n_halos):
        cache_idx = halo_cache_indices[i]
        
        # Read particle IDs from cache (each rank reads independently)
        dmo_ids = cache_handle[f'dmo/halo_{cache_idx}'][:]
        
        if has_hydro_at_hydro:
            hydro_ids = cache_handle[f'hydro_at_hydro/halo_{cache_idx}'][:]
        else:
            hydro_ids = cache_handle[f'hydro_at_dmo/halo_{cache_idx}'][:]
        
        pos_dmo = halo_positions_dmo[i]
        r200_dmo = halo_radii_dmo[i]
        pos_hydro = halo_positions_hydro[i]
        r200_hydro = halo_radii_hydro[i]
        
        if rank == 0 and i % 50 == 0:
            print(f"    Halo {i+1}/{n_halos} (M={halo_log_masses[i]:.2f})...")
            sys.stdout.flush()
        
        # Query DMO particles (collective operation) - centered on DMO halo
        dmo_coords, dmo_masses, dmo_radii_r200, dmo_types = dmo_data.query_particles(
            dmo_ids, pos_dmo, r200_dmo
        )
        
        # Query Hydro particles (collective operation) - centered on HYDRO halo
        hydro_coords, hydro_masses, hydro_radii_r200, hydro_types = hydro_data.query_particles(
            hydro_ids, pos_hydro, r200_hydro
        )
        
        # Compute statistics (rank 0 only)
        if rank == 0:
            # Baryon fractions
            bf = compute_baryon_fraction(hydro_masses, hydro_radii_r200, hydro_types, radii_mult)
            results['f_baryon'][i] = bf['f_baryon']
            results['f_gas'][i] = bf['f_gas']
            results['f_stellar'][i] = bf['f_stellar']
            results['m_total'][i] = bf['m_total']
            results['m_gas'][i] = bf['m_gas']
            results['m_stellar'][i] = bf['m_stellar']
            results['m_dm_hydro'][i] = bf['m_dm']
            
            # Mass conservation
            mc = compute_mass_conservation(
                dmo_masses, dmo_radii_r200,
                hydro_masses, hydro_radii_r200, hydro_types,
                radii_mult
            )
            results['m_dmo'][i] = mc['m_dmo']
            results['ratio_total'][i] = mc['ratio_total']
            results['ratio_dm'][i] = mc['ratio_dm']
    
    cache_handle.close()
    
    if rank == 0:
        print(f"  Processing time: {time.time()-t_proc:.1f}s")
    
    # ========================================================================
    # Save results
    # ========================================================================
    if rank == 0:
        print(f"\n[5/5] Saving results...")
        
        output_dir = args.output_dir or os.path.join(
            CACHE_BASE, f'L205n{args.sim_res}TNG', 'analysis'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f'halo_statistics_snap{args.snap:03d}.h5')
        
        with h5py.File(output_file, 'w') as f:
            # Metadata
            f.attrs['snapshot'] = args.snap
            f.attrs['sim_res'] = args.sim_res
            f.attrs['mass_min'] = args.mass_min
            f.attrs['radii_r200'] = radii_mult
            f.attrs['n_halos'] = n_halos
            
            # Per-halo data
            for key, arr in results.items():
                f.create_dataset(key, data=arr, compression='gzip')
        
        print(f"  Saved: {output_file}")
        print(f"\n{'='*70}")
        print(f"COMPLETE - Total time: {time.time()-t_start:.1f}s")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
