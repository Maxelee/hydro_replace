#!/usr/bin/env python
"""
Distributed Halo Statistics Computation.

Uses MPI to distribute particle data across ranks:
- Each rank loads a subset of snapshot files
- Each rank builds a LOCAL id_to_idx mapping for its particles only
- Halo queries are distributed: broadcast IDs → local lookup → gather results

This avoids loading the entire snapshot on any single node.

Usage:
    mpirun -np 32 python compute_halo_statistics_distributed.py --snap 99 --sim-res 625
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
        
        Parameters:
        -----------
        snapshot : int
            Snapshot number
        sim_res : int
            Simulation resolution (625, 1250, 2500)
        mode : str
            'dmo' or 'hydro'
        verbose : bool
            Print status messages (rank 0 only)
        """
        self.snapshot = snapshot
        self.sim_res = sim_res
        self.mode = mode
        self.verbose = verbose and (rank == 0)
        
        self.sim_config = SIM_PATHS[sim_res]
        
        # Local data (only particles from files assigned to this rank)
        self.local_ids = None
        self.local_coords = None
        self.local_masses = None
        self.local_types = None
        self.local_id_to_idx = None  # LOCAL mapping only
        
        self._load_local_data()
    
    def _load_local_data(self):
        """Load snapshot files assigned to this rank."""
        if self.mode == 'dmo':
            basePath = self.sim_config['dmo']
            dm_mass = self.sim_config['dmo_dm_mass']
            particle_types = [1]  # DM only
        else:
            basePath = self.sim_config['hydro']
            dm_mass = self.sim_config['hydro_dm_mass']
            particle_types = [0, 1, 4]  # Gas, DM, Stars
        
        snap_dir = f"{basePath}/snapdir_{self.snapshot:03d}/"
        all_files = sorted(glob.glob(f"{snap_dir}/snap_{self.snapshot:03d}.*.hdf5"))
        n_files = len(all_files)
        
        if self.verbose:
            print(f"\n[{self.mode.upper()}] Loading distributed particle data...")
            print(f"  Total snapshot files: {n_files}")
            print(f"  MPI ranks: {size}")
        
        # Distribute files across ranks
        my_files = [f for i, f in enumerate(all_files) if i % size == rank]
        
        if rank == 0 and self.verbose:
            print(f"  Files per rank: ~{len(my_files)}")
        
        t0 = time.time()
        
        coords_list = []
        masses_list = []
        ids_list = []
        types_list = []
        
        for filepath in my_files:
            with h5py.File(filepath, 'r') as f:
                for ptype in particle_types:
                    pt_key = f'PartType{ptype}'
                    if pt_key not in f:
                        continue
                    
                    n_part = f[pt_key]['Coordinates'].shape[0]
                    if n_part == 0:
                        continue
                    
                    # Coordinates (kpc/h -> Mpc/h)
                    coords = f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3
                    coords_list.append(coords)
                    
                    # Particle IDs
                    pids = f[pt_key]['ParticleIDs'][:]
                    ids_list.append(pids)
                    
                    # Masses
                    if 'Masses' in f[pt_key]:
                        m = f[pt_key]['Masses'][:].astype(np.float32) * MASS_UNIT
                    else:
                        m = np.full(n_part, dm_mass * MASS_UNIT, dtype=np.float32)
                    masses_list.append(m)
                    
                    # Particle types
                    types_list.append(np.full(n_part, ptype, dtype=np.int8))
        
        # Concatenate local data
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
        
        # Build LOCAL id -> index mapping
        self.local_id_to_idx = {int(pid): i for i, pid in enumerate(self.local_ids)}
        
        # Gather statistics
        local_n = len(self.local_ids)
        total_n = comm.reduce(local_n, op=MPI.SUM, root=0)
        
        if self.verbose:
            local_mem = (self.local_coords.nbytes + self.local_masses.nbytes + 
                        self.local_ids.nbytes + self.local_types.nbytes) / 1e9
            dict_mem = len(self.local_id_to_idx) * 56 / 1e9  # Estimate
            print(f"  Rank 0: {local_n:,} particles, {local_mem:.2f} GB arrays, {dict_mem:.2f} GB dict")
            print(f"  Total particles: {total_n:,}")
            print(f"  Load time: {time.time()-t0:.1f}s")
    
    def query_particles(self, particle_ids: np.ndarray, halo_center: np.ndarray, 
                        r200: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Query particle data for a set of particle IDs.
        
        This is a collective operation - all ranks must call this.
        
        Parameters:
        -----------
        particle_ids : np.ndarray
            Particle IDs to query (broadcast from rank 0)
        halo_center : np.ndarray
            Halo center position [3,] in Mpc/h
        r200 : float
            R200c radius in Mpc/h
        
        Returns:
        --------
        coords, masses, radii_r200, particle_types : arrays gathered on rank 0
            (Other ranks return empty arrays)
        """
        # Broadcast query IDs from rank 0
        if rank == 0:
            n_query = len(particle_ids)
        else:
            n_query = None
        n_query = comm.bcast(n_query, root=0)
        
        if rank != 0:
            particle_ids = np.empty(n_query, dtype=np.int64)
        comm.Bcast(particle_ids, root=0)
        
        # Broadcast halo center and r200
        halo_center = comm.bcast(halo_center, root=0)
        r200 = comm.bcast(r200, root=0)
        
        # Each rank finds matching particles in its local data
        local_indices = []
        for pid in particle_ids:
            pid_int = int(pid)
            if pid_int in self.local_id_to_idx:
                local_indices.append(self.local_id_to_idx[pid_int])
        
        local_indices = np.array(local_indices, dtype=np.int64)
        
        # Extract local matching data
        if len(local_indices) > 0:
            local_coords = self.local_coords[local_indices]
            local_masses = self.local_masses[local_indices]
            local_types = self.local_types[local_indices]
            
            # Compute radii from halo center
            dx = local_coords - halo_center
            # Periodic boundary
            dx = np.where(dx > BOX_SIZE/2, dx - BOX_SIZE, dx)
            dx = np.where(dx < -BOX_SIZE/2, dx + BOX_SIZE, dx)
            local_radii = np.linalg.norm(dx, axis=1)
            local_radii_r200 = local_radii / r200
        else:
            local_coords = np.zeros((0, 3), dtype=np.float32)
            local_masses = np.zeros(0, dtype=np.float32)
            local_types = np.zeros(0, dtype=np.int8)
            local_radii_r200 = np.zeros(0, dtype=np.float32)
        
        # Gather all results to rank 0
        all_coords = comm.gather(local_coords, root=0)
        all_masses = comm.gather(local_masses, root=0)
        all_types = comm.gather(local_types, root=0)
        all_radii_r200 = comm.gather(local_radii_r200, root=0)
        
        if rank == 0:
            coords = np.concatenate(all_coords) if all_coords else np.zeros((0, 3), dtype=np.float32)
            masses = np.concatenate(all_masses) if all_masses else np.zeros(0, dtype=np.float32)
            types = np.concatenate(all_types) if all_types else np.zeros(0, dtype=np.int8)
            radii_r200 = np.concatenate(all_radii_r200) if all_radii_r200 else np.zeros(0, dtype=np.float32)
            return coords, masses, radii_r200, types
        else:
            return np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int8)


# ============================================================================
# Statistics Computation Functions
# ============================================================================

def compute_baryon_fraction(masses: np.ndarray, radii_r200: np.ndarray, 
                           particle_types: np.ndarray, radii_mult: np.ndarray):
    """
    Compute baryon fractions at multiple radii.
    
    Returns dict with f_baryon, f_gas, f_stellar, m_total, m_gas, m_stellar, m_dm
    """
    n_radii = len(radii_mult)
    results = {
        'f_baryon': np.zeros(n_radii, dtype=np.float32),
        'f_gas': np.zeros(n_radii, dtype=np.float32),
        'f_stellar': np.zeros(n_radii, dtype=np.float32),
        'm_total': np.zeros(n_radii, dtype=np.float64),
        'm_gas': np.zeros(n_radii, dtype=np.float64),
        'm_stellar': np.zeros(n_radii, dtype=np.float64),
        'm_dm': np.zeros(n_radii, dtype=np.float64),
    }
    
    for i, r_mult in enumerate(radii_mult):
        mask = radii_r200 <= r_mult
        m_sel = masses[mask]
        t_sel = particle_types[mask]
        
        m_gas = np.sum(m_sel[t_sel == 0])
        m_dm = np.sum(m_sel[t_sel == 1])
        m_stars = np.sum(m_sel[t_sel == 4])
        m_total = m_gas + m_dm + m_stars
        
        results['m_total'][i] = m_total
        results['m_gas'][i] = m_gas
        results['m_stellar'][i] = m_stars
        results['m_dm'][i] = m_dm
        
        if m_total > 0:
            results['f_baryon'][i] = (m_gas + m_stars) / m_total
            results['f_gas'][i] = m_gas / m_total
            results['f_stellar'][i] = m_stars / m_total
    
    return results


def compute_mass_conservation(dmo_masses: np.ndarray, dmo_radii_r200: np.ndarray,
                              hydro_masses: np.ndarray, hydro_radii_r200: np.ndarray,
                              hydro_types: np.ndarray, radii_mult: np.ndarray):
    """
    Compute mass conservation ratio M_hydro/M_dmo at multiple radii.
    """
    n_radii = len(radii_mult)
    results = {
        'm_dmo': np.zeros(n_radii, dtype=np.float64),
        'ratio_total': np.zeros(n_radii, dtype=np.float32),
        'ratio_dm': np.zeros(n_radii, dtype=np.float32),
    }
    
    for i, r_mult in enumerate(radii_mult):
        # DMO mass within radius
        dmo_mask = dmo_radii_r200 <= r_mult
        m_dmo = np.sum(dmo_masses[dmo_mask])
        
        # Hydro masses within radius
        hydro_mask = hydro_radii_r200 <= r_mult
        m_hydro_total = np.sum(hydro_masses[hydro_mask])
        m_hydro_dm = np.sum(hydro_masses[hydro_mask & (hydro_types == 1)])
        
        results['m_dmo'][i] = m_dmo
        
        if m_dmo > 0:
            results['ratio_total'][i] = m_hydro_total / m_dmo
            results['ratio_dm'][i] = m_hydro_dm / m_dmo
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Distributed halo statistics')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, default=2500, choices=[625, 1250, 2500])
    parser.add_argument('--mass-min', type=float, default=12.0,
                        help='Minimum log10(M200c/Msun/h)')
    parser.add_argument('--output-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Radii at which to compute statistics
    radii_mult = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    n_radii = len(radii_mult)
    
    if rank == 0:
        print("=" * 70)
        print("DISTRIBUTED HALO STATISTICS COMPUTATION")
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
    # Load particle cache (rank 0 only, then broadcast halo info)
    # ========================================================================
    cache_file = os.path.join(
        CACHE_BASE, f'L205n{args.sim_res}TNG', 
        'particle_cache', f'cache_snap{args.snap:03d}.h5'
    )
    
    if rank == 0:
        print(f"\n[1/5] Loading halo cache: {cache_file}")
        sys.stdout.flush()
        
        with h5py.File(cache_file, 'r') as f:
            # Load halo info from halo_info group
            halo_info = f['halo_info']
            dmo_indices = halo_info['halo_indices'][:]
            positions = halo_info['positions_dmo'][:]  # Use DMO positions for matching
            radii = halo_info['radii_dmo'][:]  # Use DMO R200
            masses = halo_info['masses'][:]
            
            # Filter by mass
            log_masses = np.log10(masses)
            mass_mask = log_masses >= args.mass_min
            
            halo_cache_indices = np.where(mass_mask)[0]
            halo_dmo_indices = dmo_indices[mass_mask]
            halo_positions = positions[mass_mask]
            halo_radii = radii[mass_mask]
            halo_log_masses = log_masses[mass_mask]
            
            n_halos = len(halo_cache_indices)
            print(f"  Total halos above M > 10^{args.mass_min}: {n_halos}")
            
            # Load particle IDs for all halos
            # Check cache format (old vs new)
            has_new_format = 'hydro_at_dmo' in f
            
            dmo_particle_ids = []
            hydro_particle_ids = []
            
            for cache_idx in halo_cache_indices:
                dmo_ids = f[f'dmo/halo_{cache_idx}'][:]
                dmo_particle_ids.append(dmo_ids)
                
                if has_new_format:
                    hydro_ids = f[f'hydro_at_dmo/halo_{cache_idx}'][:]
                else:
                    hydro_ids = f[f'hydro/halo_{cache_idx}'][:]
                hydro_particle_ids.append(hydro_ids)
        
        print(f"  Loaded particle IDs for {n_halos} halos")
        sys.stdout.flush()
    else:
        n_halos = None
        halo_cache_indices = None
        halo_dmo_indices = None
        halo_positions = None
        halo_radii = None
        halo_log_masses = None
        dmo_particle_ids = None
        hydro_particle_ids = None
    
    # Broadcast halo info
    n_halos = comm.bcast(n_halos, root=0)
    halo_positions = comm.bcast(halo_positions, root=0)
    halo_radii = comm.bcast(halo_radii, root=0)
    halo_log_masses = comm.bcast(halo_log_masses, root=0)
    halo_dmo_indices = comm.bcast(halo_dmo_indices, root=0)
    dmo_particle_ids = comm.bcast(dmo_particle_ids, root=0)
    hydro_particle_ids = comm.bcast(hydro_particle_ids, root=0)
    
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
    # Process halos
    # ========================================================================
    if rank == 0:
        print(f"\n[4/5] Processing {n_halos} halos...")
        sys.stdout.flush()
        
        # Pre-allocate results
        results = {
            'dmo_indices': halo_dmo_indices.astype(np.int32),
            'log_masses': halo_log_masses.astype(np.float32),
            'positions': halo_positions.astype(np.float32),
            'radii': halo_radii.astype(np.float32),
            
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
    
    for i in range(n_halos):
        if rank == 0:
            if i % 20 == 0:
                print(f"    Halo {i+1}/{n_halos} (M={halo_log_masses[i]:.2f})...")
                sys.stdout.flush()
            
            dmo_ids = dmo_particle_ids[i]
            hydro_ids = hydro_particle_ids[i]
            pos = halo_positions[i]
            r200 = halo_radii[i]
        else:
            dmo_ids = None
            hydro_ids = None
            pos = None
            r200 = None
        
        # Query DMO particles (collective operation)
        dmo_ids = comm.bcast(dmo_ids, root=0)
        pos = comm.bcast(pos, root=0)
        r200 = comm.bcast(r200, root=0)
        
        dmo_coords, dmo_masses, dmo_radii_r200, dmo_types = dmo_data.query_particles(
            dmo_ids, pos, r200
        )
        
        # Query Hydro particles (collective operation)
        hydro_ids = comm.bcast(hydro_ids, root=0)
        
        hydro_coords, hydro_masses, hydro_radii_r200, hydro_types = hydro_data.query_particles(
            hydro_ids, pos, r200
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
