#!/usr/bin/env python
"""
Particle Access Library for Matched Halos.

Provides a clean interface to load and analyze particles from matched DMO/Hydro halos
using the pre-computed particle ID cache.

Key Features:
- Load matched halo information (positions, masses, radii)
- Access particle IDs for each halo from cache
- Load full particle data (coords, masses) from snapshots
- Compute baryon fractions, mass conservation, profiles

Usage:
    from particle_access import MatchedHaloSnapshot
    
    mh = MatchedHaloSnapshot(snapshot=99, sim_res=2500)
    
    # Get halo info
    halos = mh.get_halo_info()
    
    # Load particles for a specific halo
    dmo_data = mh.get_particles('dmo', halo_idx=100)
    hydro_data = mh.get_particles('hydro', halo_idx=100)
    
    # Compute profiles
    from particle_analysis import compute_radial_profile
    profile = compute_radial_profile(dmo_data, halos[100])
"""

import numpy as np
import h5py
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import time


# ============================================================================
# Configuration
# ============================================================================

SIM_PATHS = {
    2500: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output',
        'dmo_dm_mass': 0.0047271638660809,      # 10^10 Msun/h
        'hydro_dm_mass': 0.00398342749867548,   # 10^10 Msun/h
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

DEFAULT_CONFIG = {
    'cache_base': '/mnt/home/mlee1/ceph/hydro_replace_fields',
    'box_size': 205.0,  # Mpc/h
    'mass_unit': 1e10,  # Convert to Msun/h
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HaloInfo:
    """Container for matched halo information."""
    cache_idx: int          # Index in cache arrays (0, 1, 2, ...)
    dmo_idx: int            # Original DMO halo index from catalog
    position: np.ndarray    # Halo position in Mpc/h [3,]
    radius: float           # R200c in Mpc/h
    mass: float             # M200c in Msun/h
    log_mass: float         # log10(M200c / Msun/h)
    

@dataclass
class ParticleData:
    """Container for particle data around a halo."""
    particle_ids: np.ndarray        # Particle IDs
    coords: Optional[np.ndarray]    # Coordinates in Mpc/h [N, 3]
    masses: Optional[np.ndarray]    # Masses in Msun/h [N,]
    radii: Optional[np.ndarray]     # Distance from halo center in Mpc/h [N,]
    radii_r200: Optional[np.ndarray]  # Distance in units of R200 [N,]
    particle_types: Optional[np.ndarray]  # Particle type (0=gas, 1=DM, 4=stars)
    
    @property
    def n_particles(self) -> int:
        return len(self.particle_ids)
    
    @property
    def total_mass(self) -> float:
        if self.masses is not None:
            return float(np.sum(self.masses))
        return 0.0
    
    def select_radius(self, r_max_r200: float) -> 'ParticleData':
        """Return particles within r_max_r200 × R200."""
        if self.radii_r200 is None:
            raise ValueError("Radii not computed. Load with include_coords=True")
        
        mask = self.radii_r200 <= r_max_r200
        return ParticleData(
            particle_ids=self.particle_ids[mask],
            coords=self.coords[mask] if self.coords is not None else None,
            masses=self.masses[mask] if self.masses is not None else None,
            radii=self.radii[mask] if self.radii is not None else None,
            radii_r200=self.radii_r200[mask],
            particle_types=self.particle_types[mask] if self.particle_types is not None else None,
        )
    
    def select_type(self, ptype: int) -> 'ParticleData':
        """Return particles of a specific type (0=gas, 1=DM, 4=stars)."""
        if self.particle_types is None:
            raise ValueError("Particle types not available")
        
        mask = self.particle_types == ptype
        return ParticleData(
            particle_ids=self.particle_ids[mask],
            coords=self.coords[mask] if self.coords is not None else None,
            masses=self.masses[mask] if self.masses is not None else None,
            radii=self.radii[mask] if self.radii is not None else None,
            radii_r200=self.radii_r200[mask] if self.radii_r200 is not None else None,
            particle_types=self.particle_types[mask],
        )


# ============================================================================
# Main Class: MatchedHaloSnapshot
# ============================================================================

class MatchedHaloSnapshot:
    """
    Interface for accessing matched DMO/Hydro halo particles.
    
    This class manages:
    1. Loading particle ID cache
    2. Loading full snapshot data on demand
    3. Mapping cached IDs to particle properties
    
    Example:
        mh = MatchedHaloSnapshot(snapshot=99, sim_res=2500)
        
        # Get halo information
        halos = mh.get_halo_info(mass_range=[13.0, 14.0])
        
        # Load particles for a halo
        dmo_data = mh.get_particles('dmo', halo_idx=100, include_coords=True)
        hydro_data = mh.get_particles('hydro', halo_idx=100, include_coords=True)
    """
    
    def __init__(self, snapshot: int, sim_res: int = 2500, 
                 cache_base: str = None, verbose: bool = True):
        """
        Initialize matched halo snapshot interface.
        
        Parameters:
        -----------
        snapshot : int
            Snapshot number
        sim_res : int
            Simulation resolution (625, 1250, 2500)
        cache_base : str
            Base directory for cache files
        verbose : bool
            Print status messages
        """
        self.snapshot = snapshot
        self.sim_res = sim_res
        self.verbose = verbose
        
        # Paths
        self.cache_base = cache_base or DEFAULT_CONFIG['cache_base']
        self.sim_config = SIM_PATHS[sim_res]
        self.box_size = DEFAULT_CONFIG['box_size']
        self.mass_unit = DEFAULT_CONFIG['mass_unit']
        
        # Cache file path
        self.cache_file = os.path.join(
            self.cache_base,
            f'L205n{sim_res}TNG',
            'particle_cache',
            f'cache_snap{snapshot:03d}.h5'
        )
        
        # Lazy-loaded data
        self._halo_info = None
        self._cache_handle = None
        self._dmo_snapshot_data = None
        self._hydro_snapshot_data = None
        self._dmo_id_to_idx = None
        self._hydro_id_to_idx = None
        
        # Validate cache exists
        if not os.path.exists(self.cache_file):
            raise FileNotFoundError(f"Cache file not found: {self.cache_file}")
        
        if self.verbose:
            print(f"MatchedHaloSnapshot initialized")
            print(f"  Snapshot: {snapshot}")
            print(f"  Resolution: L205n{sim_res}TNG")
            print(f"  Cache: {self.cache_file}")
    
    def _load_halo_info(self):
        """Load halo information from cache."""
        if self._halo_info is not None:
            return
        
        if self.verbose:
            print("Loading halo info from cache...")
            t0 = time.time()
        
        with h5py.File(self.cache_file, 'r') as f:
            halo_indices = f['halo_info/halo_indices'][:]
            
            # Handle both old and new cache formats
            if 'halo_info/positions_dmo' in f:
                # New format: separate DMO and Hydro positions
                positions = f['halo_info/positions_dmo'][:]
                radii = f['halo_info/radii_dmo'][:]
                self._has_hydro_positions = 'halo_info/positions_hydro' in f
                if self._has_hydro_positions:
                    self._hydro_positions = f['halo_info/positions_hydro'][:]
                    self._hydro_radii = f['halo_info/radii_hydro'][:]
            else:
                # Old format: single positions array (DMO positions)
                positions = f['halo_info/positions'][:]
                radii = f['halo_info/radii'][:]
                self._has_hydro_positions = False
            
            masses = f['halo_info/masses'][:]
            self._cache_radius_mult = f.attrs['radius_multiplier']
            
            # Check which hydro groups exist
            self._has_hydro_at_dmo = 'hydro_at_dmo' in f
            self._has_hydro_at_hydro = 'hydro_at_hydro' in f
            self._has_legacy_hydro = 'hydro' in f
        
        self._halo_info = {}
        self._cache_idx_to_dmo_idx = {}
        
        for cache_idx, (dmo_idx, pos, r200, m200) in enumerate(
            zip(halo_indices, positions, radii, masses)
        ):
            self._halo_info[int(dmo_idx)] = HaloInfo(
                cache_idx=cache_idx,
                dmo_idx=int(dmo_idx),
                position=pos,
                radius=r200,
                mass=m200,
                log_mass=np.log10(m200) if m200 > 0 else 0,
            )
            self._cache_idx_to_dmo_idx[cache_idx] = int(dmo_idx)
        
        if self.verbose:
            print(f"  Loaded {len(self._halo_info)} halos in {time.time()-t0:.2f}s")
            print(f"  Cache radius: {self._cache_radius_mult}×R200")
            print(f"  Hydro positions: {self._has_hydro_positions}")
            print(f"  Groups: dmo={True}, hydro_at_dmo={self._has_hydro_at_dmo}, "
                  f"hydro_at_hydro={self._has_hydro_at_hydro}, legacy_hydro={self._has_legacy_hydro}")
    
    def get_halo_info(self, mass_range: Tuple[float, float] = None,
                      halo_indices: List[int] = None) -> Dict[int, HaloInfo]:
        """
        Get halo information, optionally filtered.
        
        Parameters:
        -----------
        mass_range : tuple (log_min, log_max)
            Filter by log10(M200c / Msun/h)
        halo_indices : list
            Specific DMO halo indices to return
        
        Returns:
        --------
        dict : dmo_idx -> HaloInfo
        """
        self._load_halo_info()
        
        result = {}
        for dmo_idx, halo in self._halo_info.items():
            # Filter by mass
            if mass_range is not None:
                if halo.log_mass < mass_range[0] or halo.log_mass > mass_range[1]:
                    continue
            
            # Filter by index
            if halo_indices is not None and dmo_idx not in halo_indices:
                continue
            
            result[dmo_idx] = halo
        
        return result
    
    def get_cached_particle_ids(self, mode: str, dmo_idx: int) -> np.ndarray:
        """
        Get particle IDs for a halo from cache.
        
        Parameters:
        -----------
        mode : str
            'dmo' - DMO particles at DMO halo center
            'hydro' or 'hydro_at_dmo' - Hydro particles at DMO halo center (for replacement)
            'hydro_at_hydro' - Hydro particles at Hydro halo center (for true profiles)
        dmo_idx : int
            DMO halo index
        
        Returns:
        --------
        np.ndarray : Particle IDs
        """
        self._load_halo_info()
        
        if dmo_idx not in self._halo_info:
            raise KeyError(f"Halo {dmo_idx} not in cache")
        
        cache_idx = self._halo_info[dmo_idx].cache_idx
        
        # Map mode to cache group name
        if mode == 'dmo':
            group = 'dmo'
        elif mode in ('hydro', 'hydro_at_dmo'):
            # Prefer new format, fall back to legacy
            if self._has_hydro_at_dmo:
                group = 'hydro_at_dmo'
            elif self._has_legacy_hydro:
                group = 'hydro'
            else:
                raise ValueError("No hydro particle cache available")
        elif mode == 'hydro_at_hydro':
            if not self._has_hydro_at_hydro:
                raise ValueError("hydro_at_hydro not available in this cache")
            group = 'hydro_at_hydro'
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'dmo', 'hydro', 'hydro_at_dmo', or 'hydro_at_hydro'")
        
        with h5py.File(self.cache_file, 'r') as f:
            key = f'{group}/halo_{cache_idx}'
            if key not in f:
                return np.array([], dtype=np.int64)
            return f[key][:]
    
    def _load_snapshot_data(self, mode: str):
        """
        Load full snapshot data (coords, masses, IDs) into memory.
        
        This is expensive but enables fast particle lookups.
        """
        if mode == 'dmo':
            if self._dmo_snapshot_data is not None:
                return
            basePath = self.sim_config['dmo']
            dm_mass = self.sim_config['dmo_dm_mass']
            particle_types = [1]  # DM only
        else:
            if self._hydro_snapshot_data is not None:
                return
            basePath = self.sim_config['hydro']
            dm_mass = self.sim_config['hydro_dm_mass']
            particle_types = [0, 1, 4]  # Gas, DM, Stars
        
        if self.verbose:
            print(f"Loading {mode.upper()} snapshot {self.snapshot}...")
            t0 = time.time()
        
        snap_dir = f"{basePath}/snapdir_{self.snapshot:03d}/"
        files = sorted(glob.glob(f"{snap_dir}/snap_{self.snapshot:03d}.*.hdf5"))
        
        coords_list = []
        masses_list = []
        ids_list = []
        types_list = []
        
        for filepath in files:
            with h5py.File(filepath, 'r') as f:
                for ptype in particle_types:
                    pt_key = f'PartType{ptype}'
                    if pt_key not in f:
                        continue
                    
                    n_part = f[pt_key]['Coordinates'].shape[0]
                    
                    # Coordinates (kpc/h -> Mpc/h)
                    coords = f[pt_key]['Coordinates'][:].astype(np.float32) / 1e3
                    coords_list.append(coords)
                    
                    # Particle IDs
                    pids = f[pt_key]['ParticleIDs'][:]
                    ids_list.append(pids)
                    
                    # Masses
                    if 'Masses' in f[pt_key]:
                        m = f[pt_key]['Masses'][:].astype(np.float32) * self.mass_unit
                    else:
                        m = np.full(n_part, dm_mass * self.mass_unit, dtype=np.float32)
                    masses_list.append(m)
                    
                    # Particle types
                    types_list.append(np.full(n_part, ptype, dtype=np.int8))
        
        # Concatenate all
        all_coords = np.concatenate(coords_list)
        all_masses = np.concatenate(masses_list)
        all_ids = np.concatenate(ids_list)
        all_types = np.concatenate(types_list)
        
        # Build ID -> index lookup
        id_to_idx = {int(pid): i for i, pid in enumerate(all_ids)}
        
        snapshot_data = {
            'coords': all_coords,
            'masses': all_masses,
            'particle_ids': all_ids,
            'particle_types': all_types,
        }
        
        if mode == 'dmo':
            self._dmo_snapshot_data = snapshot_data
            self._dmo_id_to_idx = id_to_idx
        else:
            self._hydro_snapshot_data = snapshot_data
            self._hydro_id_to_idx = id_to_idx
        
        if self.verbose:
            print(f"  Loaded {len(all_ids):,} particles in {time.time()-t0:.1f}s")
            print(f"  Memory: ~{all_coords.nbytes / 1e9:.2f} GB coords, "
                  f"~{all_masses.nbytes / 1e9:.2f} GB masses")
    
    def _apply_periodic_boundary(self, dx: np.ndarray) -> np.ndarray:
        """Apply minimum image convention."""
        return dx - np.round(dx / self.box_size) * self.box_size
    
    def get_particles(self, mode: str, dmo_idx: int, 
                      radius_mult: float = None,
                      include_coords: bool = True) -> ParticleData:
        """
        Get particle data for a halo.
        
        Parameters:
        -----------
        mode : str
            'dmo' - DMO particles at DMO halo center
            'hydro' or 'hydro_at_dmo' - Hydro particles at DMO center (for replacement)
            'hydro_at_hydro' - Hydro particles at Hydro center (for true profiles)
        dmo_idx : int
            DMO halo index
        radius_mult : float
            Maximum radius in units of R200 (default: use full cache)
        include_coords : bool
            If True, load full snapshot and compute coordinates/masses
            If False, return only particle IDs (faster)
        
        Returns:
        --------
        ParticleData : Container with particle information
        """
        self._load_halo_info()
        
        if dmo_idx not in self._halo_info:
            raise KeyError(f"Halo {dmo_idx} not in cache")
        
        halo = self._halo_info[dmo_idx]
        
        # Get cached particle IDs
        particle_ids = self.get_cached_particle_ids(mode, dmo_idx)
        
        if not include_coords:
            return ParticleData(
                particle_ids=particle_ids,
                coords=None,
                masses=None,
                radii=None,
                radii_r200=None,
                particle_types=None,
            )
        
        # Determine which snapshot to load (DMO or Hydro)
        snapshot_mode = 'dmo' if mode == 'dmo' else 'hydro'
        self._load_snapshot_data(snapshot_mode)
        
        if snapshot_mode == 'dmo':
            snapshot_data = self._dmo_snapshot_data
            id_to_idx = self._dmo_id_to_idx
        else:
            snapshot_data = self._hydro_snapshot_data
            id_to_idx = self._hydro_id_to_idx
        
        # Map particle IDs to indices
        valid_ids = []
        indices = []
        for pid in particle_ids:
            pid_int = int(pid)
            if pid_int in id_to_idx:
                valid_ids.append(pid)
                indices.append(id_to_idx[pid_int])
        
        if len(indices) == 0:
            return ParticleData(
                particle_ids=np.array([], dtype=np.int64),
                coords=np.zeros((0, 3), dtype=np.float32),
                masses=np.zeros(0, dtype=np.float32),
                radii=np.zeros(0, dtype=np.float32),
                radii_r200=np.zeros(0, dtype=np.float32),
                particle_types=np.zeros(0, dtype=np.int8),
            )
        
        indices = np.array(indices)
        particle_ids = np.array(valid_ids)
        
        # Extract properties
        coords = snapshot_data['coords'][indices]
        masses = snapshot_data['masses'][indices]
        particle_types = snapshot_data['particle_types'][indices]
        
        # Compute radii from halo center
        dx = coords - halo.position
        dx = self._apply_periodic_boundary(dx)
        radii = np.linalg.norm(dx, axis=1)
        radii_r200 = radii / halo.radius
        
        result = ParticleData(
            particle_ids=particle_ids,
            coords=coords,
            masses=masses,
            radii=radii,
            radii_r200=radii_r200,
            particle_types=particle_types,
        )
        
        # Filter by radius if requested
        if radius_mult is not None and radius_mult < self._cache_radius_mult:
            result = result.select_radius(radius_mult)
        
        return result
    
    def iter_halos(self, mass_range: Tuple[float, float] = None,
                   preload_snapshots: bool = True):
        """
        Iterate over matched halos.
        
        Parameters:
        -----------
        mass_range : tuple
            Filter by log10(M200c)
        preload_snapshots : bool
            If True, load snapshot data before iteration (faster)
        
        Yields:
        -------
        HaloInfo for each matched halo
        """
        halos = self.get_halo_info(mass_range=mass_range)
        
        if preload_snapshots:
            self._load_snapshot_data('dmo')
            self._load_snapshot_data('hydro')
        
        for dmo_idx in sorted(halos.keys()):
            yield halos[dmo_idx]
    
    def close(self):
        """Release memory from loaded snapshot data."""
        self._dmo_snapshot_data = None
        self._hydro_snapshot_data = None
        self._dmo_id_to_idx = None
        self._hydro_id_to_idx = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ============================================================================
# Convenience Functions
# ============================================================================

def list_available_snapshots(sim_res: int = 2500, 
                              cache_base: str = None) -> List[int]:
    """List snapshots with available particle caches."""
    cache_base = cache_base or DEFAULT_CONFIG['cache_base']
    cache_dir = os.path.join(cache_base, f'L205n{sim_res}TNG', 'particle_cache')
    
    if not os.path.exists(cache_dir):
        return []
    
    snapshots = []
    for f in glob.glob(os.path.join(cache_dir, 'cache_snap*.h5')):
        snap = int(os.path.basename(f).replace('cache_snap', '').replace('.h5', ''))
        snapshots.append(snap)
    
    return sorted(snapshots)


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == '__main__':
    # List available snapshots
    print("Available snapshots:")
    snaps = list_available_snapshots(sim_res=2500)
    print(f"  {snaps}")
    
    if len(snaps) > 0:
        snap = snaps[0]
        print(f"\nTesting with snapshot {snap}...")
        
        mh = MatchedHaloSnapshot(snapshot=snap, sim_res=2500)
        
        # Get halo info
        halos = mh.get_halo_info()
        print(f"\nTotal matched halos: {len(halos)}")
        
        # Get massive halos
        massive = mh.get_halo_info(mass_range=[13.5, 15.0])
        print(f"Halos with M > 10^13.5: {len(massive)}")
        
        if len(massive) > 0:
            # Pick first massive halo
            dmo_idx = list(massive.keys())[0]
            halo = massive[dmo_idx]
            print(f"\nTest halo {dmo_idx}:")
            print(f"  log10(M200c) = {halo.log_mass:.2f}")
            print(f"  R200c = {halo.radius:.3f} Mpc/h")
            print(f"  Position = {halo.position}")
            
            # Get particles (IDs only, fast)
            dmo_ids = mh.get_cached_particle_ids('dmo', dmo_idx)
            hydro_ids = mh.get_cached_particle_ids('hydro', dmo_idx)
            print(f"\nCached particle IDs:")
            print(f"  DMO: {len(dmo_ids):,} particles")
            print(f"  Hydro: {len(hydro_ids):,} particles")
            
            # Get full particle data (slower, loads snapshot)
            print("\nLoading full particle data...")
            dmo_data = mh.get_particles('dmo', dmo_idx, include_coords=True)
            hydro_data = mh.get_particles('hydro', dmo_idx, include_coords=True)
            
            print(f"\nDMO particles:")
            print(f"  N = {dmo_data.n_particles:,}")
            print(f"  Total mass = {dmo_data.total_mass:.3e} Msun/h")
            
            print(f"\nHydro particles:")
            print(f"  N = {hydro_data.n_particles:,}")
            print(f"  Total mass = {hydro_data.total_mass:.3e} Msun/h")
            
            # Filter by radius
            dmo_r200 = dmo_data.select_radius(1.0)
            hydro_r200 = hydro_data.select_radius(1.0)
            print(f"\nWithin R200:")
            print(f"  DMO: {dmo_r200.n_particles:,} particles, M = {dmo_r200.total_mass:.3e}")
            print(f"  Hydro: {hydro_r200.n_particles:,} particles, M = {hydro_r200.total_mass:.3e}")
            
            # Filter by type (hydro only)
            gas = hydro_data.select_type(0)
            dm = hydro_data.select_type(1)
            stars = hydro_data.select_type(4)
            print(f"\nHydro by particle type:")
            print(f"  Gas: {gas.n_particles:,}, M = {gas.total_mass:.3e}")
            print(f"  DM: {dm.n_particles:,}, M = {dm.total_mass:.3e}")
            print(f"  Stars: {stars.n_particles:,}, M = {stars.total_mass:.3e}")
        
        mh.close()
        print("\nTest complete!")
