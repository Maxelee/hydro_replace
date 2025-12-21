"""
Particle Extraction Module
==========================

Functions for extracting particles around halos with proper handling
of periodic boundary conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.spatial import cKDTree

# Try to import MPI
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

# Try to import illustris_python
try:
    from illustris_python import snapshot
    HAS_ILLUSTRIS = True
except ImportError:
    HAS_ILLUSTRIS = False

from .bijective_matching import MatchedCatalog

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractedHalo:
    """
    Container for extracted halo particle data.

    Attributes
    ----------
    halo_id : int
        Halo index in the catalog.
    center : ndarray
        Halo center position (3,) in Mpc/h.
    radius : float
        R_200c in Mpc/h.
    mass : float
        M_200c in Msun/h.
    extraction_radius : float
        Radius used for extraction (multiple of R_200c) in Mpc/h.
    coordinates : dict
        Particle coordinates by type {'dm': (N,3), 'gas': (M,3), ...}.
    masses : dict
        Particle masses by type {'dm': (N,), 'gas': (M,), ...}.
    particle_ids : dict
        Particle IDs by type (optional).
    n_particles : dict
        Number of particles by type.
    """
    
    halo_id: int
    center: np.ndarray
    radius: float
    mass: float
    extraction_radius: float
    coordinates: Dict[str, np.ndarray]
    masses: Dict[str, np.ndarray]
    particle_ids: Optional[Dict[str, np.ndarray]] = None
    
    @property
    def n_particles(self) -> Dict[str, int]:
        """Number of particles by type."""
        return {ptype: len(coords) for ptype, coords in self.coordinates.items()}
    
    @property
    def total_particles(self) -> int:
        """Total number of particles."""
        return sum(self.n_particles.values())
    
    @property
    def total_mass(self) -> float:
        """Total mass of extracted particles."""
        return sum(m.sum() for m in self.masses.values())
    
    def get_all_coordinates(self) -> np.ndarray:
        """Get all particle coordinates as single array."""
        coords_list = [self.coordinates[pt] for pt in sorted(self.coordinates.keys())]
        return np.concatenate(coords_list) if coords_list else np.empty((0, 3))
    
    def get_all_masses(self) -> np.ndarray:
        """Get all particle masses as single array."""
        mass_list = [self.masses[pt] for pt in sorted(self.masses.keys())]
        return np.concatenate(mass_list) if mass_list else np.empty(0)
    
    def save_hdf5(self, filepath: Union[str, Path]) -> None:
        """
        Save extracted halo to HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['halo_id'] = self.halo_id
            f.attrs['mass'] = self.mass
            f.attrs['radius'] = self.radius
            f.attrs['extraction_radius'] = self.extraction_radius
            f.create_dataset('center', data=self.center)
            
            # Particle data by type
            for ptype in self.coordinates:
                grp = f.create_group(ptype)
                grp.create_dataset('Coordinates', data=self.coordinates[ptype],
                                   compression='gzip', compression_opts=4)
                grp.create_dataset('Masses', data=self.masses[ptype],
                                   compression='gzip', compression_opts=4)
                if self.particle_ids and ptype in self.particle_ids:
                    grp.create_dataset('ParticleIDs', data=self.particle_ids[ptype],
                                       compression='gzip', compression_opts=4)
        
        logger.debug(f"Saved halo {self.halo_id} to {filepath}")
    
    @classmethod
    def load_hdf5(cls, filepath: Union[str, Path]) -> 'ExtractedHalo':
        """
        Load extracted halo from HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Input file path.

        Returns
        -------
        halo : ExtractedHalo
            Loaded halo data.
        """
        with h5py.File(filepath, 'r') as f:
            coordinates = {}
            masses = {}
            particle_ids = {}
            
            for ptype in f.keys():
                if ptype == 'center':
                    continue
                coordinates[ptype] = f[ptype]['Coordinates'][:]
                masses[ptype] = f[ptype]['Masses'][:]
                if 'ParticleIDs' in f[ptype]:
                    particle_ids[ptype] = f[ptype]['ParticleIDs'][:]
            
            return cls(
                halo_id=f.attrs['halo_id'],
                center=f['center'][:],
                radius=f.attrs['radius'],
                mass=f.attrs['mass'],
                extraction_radius=f.attrs['extraction_radius'],
                coordinates=coordinates,
                masses=masses,
                particle_ids=particle_ids if particle_ids else None,
            )


# =============================================================================
# Particle Extractor Class
# =============================================================================

class ParticleExtractor:
    """
    Extract particles around halos from simulation snapshots.

    This class handles loading particle data and extracting particles
    within specified radii of halo centers, with proper handling of
    periodic boundary conditions.

    Parameters
    ----------
    basePath : str
        Path to simulation output.
    snapNum : int
        Snapshot number.
    box_size : float
        Box size in Mpc/h.
    particle_types : list of str
        Particle types to extract ('dm', 'gas', 'stars', 'bh').
    mass_unit : float
        Mass unit conversion (default: 1e10).
    length_unit : float
        Length unit conversion (default: 1e-3).

    Examples
    --------
    >>> extractor = ParticleExtractor(
    ...     basePath='/path/to/TNG300',
    ...     snapNum=99,
    ...     box_size=205.0,
    ... )
    >>> halo = extractor.extract(center=[100, 100, 100], radius=0.5)
    """
    
    # Particle type mapping
    TYPE_MAP = {
        'gas': 0,
        'dm': 1,
        'tracers': 3,
        'stars': 4,
        'bh': 5,
    }
    
    # Default DM masses (TNG300)
    DEFAULT_DM_MASS = {
        'hydro': 0.00398342749867548,  # 10^10 Msun/h
        'dmo': 0.0047271638660809,
    }
    
    def __init__(
        self,
        basePath: str,
        snapNum: int,
        box_size: float,
        particle_types: List[str] = ['dm', 'gas', 'stars'],
        mass_unit: float = 1e10,
        length_unit: float = 1e-3,
        sim_type: str = 'hydro',
    ):
        self.basePath = basePath
        self.snapNum = snapNum
        self.box_size = box_size
        self.particle_types = particle_types
        self.mass_unit = mass_unit
        self.length_unit = length_unit
        self.sim_type = sim_type
        
        # Cached data
        self._particles: Dict[str, Dict] = {}
        self._trees: Dict[str, cKDTree] = {}
        self._loaded = False
        
        # MPI setup
        if HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
    
    def load_particles(self, load_ids: bool = False) -> None:
        """
        Load all particles into memory and build KD-trees.

        Parameters
        ----------
        load_ids : bool
            Whether to load particle IDs (memory intensive).
        """
        if self._loaded:
            return
        
        logger.info(f"Loading particles from {self.basePath}...")
        
        for ptype in self.particle_types:
            type_num = self.TYPE_MAP[ptype]
            
            logger.info(f"  Loading {ptype} particles (type {type_num})...")
            
            try:
                if HAS_ILLUSTRIS:
                    # Use illustris_python for loading
                    fields = ['Coordinates']
                    if ptype in ['gas', 'stars', 'bh']:
                        fields.append('Masses')
                    if load_ids:
                        fields.append('ParticleIDs')
                    
                    data = snapshot.loadSubset(
                        self.basePath, self.snapNum, type_num, fields
                    )
                    
                    # Handle single field case
                    if isinstance(data, np.ndarray):
                        if len(fields) == 1:
                            data = {fields[0]: data}
                        else:
                            data = {'Coordinates': data}
                    
                    coords = data.get('Coordinates', np.empty((0, 3)))
                    coords = coords * self.length_unit  # Convert to Mpc/h
                    
                    if ptype in ['gas', 'stars', 'bh']:
                        masses = data.get('Masses', np.ones(len(coords)))
                        masses = masses * self.mass_unit  # Convert to Msun/h
                    else:
                        # DM particles have fixed mass
                        dm_mass = self.DEFAULT_DM_MASS[self.sim_type] * self.mass_unit
                        masses = np.full(len(coords), dm_mass, dtype=np.float32)
                    
                    ids = data.get('ParticleIDs', None)
                    
                else:
                    # Fallback: load from HDF5 directly
                    coords, masses, ids = self._load_particles_hdf5(type_num, load_ids)
                
                self._particles[ptype] = {
                    'coords': coords.astype(np.float32),
                    'masses': masses.astype(np.float32),
                    'ids': ids,
                }
                
                logger.info(f"    Loaded {len(coords):,} {ptype} particles")
                
                # Build KD-tree
                if len(coords) > 0:
                    self._trees[ptype] = cKDTree(coords)
                
            except Exception as e:
                logger.warning(f"    Failed to load {ptype}: {e}")
                self._particles[ptype] = {
                    'coords': np.empty((0, 3), dtype=np.float32),
                    'masses': np.empty(0, dtype=np.float32),
                    'ids': None,
                }
        
        self._loaded = True
    
    def _load_particles_hdf5(
        self,
        part_type: int,
        load_ids: bool,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load particles directly from HDF5 files."""
        import glob
        
        snap_dir = f"{self.basePath}/snapdir_{self.snapNum:03d}/"
        files = sorted(glob.glob(f"{snap_dir}/snap_{self.snapNum:03d}.*.hdf5"))
        
        if len(files) == 0:
            # Try single file format
            single_file = f"{self.basePath}/snap_{self.snapNum:03d}.hdf5"
            if Path(single_file).exists():
                files = [single_file]
            else:
                raise FileNotFoundError(f"No snapshot files found in {snap_dir}")
        
        coords_list = []
        masses_list = []
        ids_list = []
        
        ptype_key = f'PartType{part_type}'
        
        for filepath in files:
            with h5py.File(filepath, 'r') as f:
                if ptype_key not in f:
                    continue
                
                coords_list.append(f[ptype_key]['Coordinates'][:] * self.length_unit)
                
                if 'Masses' in f[ptype_key]:
                    masses_list.append(f[ptype_key]['Masses'][:] * self.mass_unit)
                else:
                    n_part = len(coords_list[-1])
                    dm_mass = self.DEFAULT_DM_MASS[self.sim_type] * self.mass_unit
                    masses_list.append(np.full(n_part, dm_mass))
                
                if load_ids and 'ParticleIDs' in f[ptype_key]:
                    ids_list.append(f[ptype_key]['ParticleIDs'][:])
        
        coords = np.concatenate(coords_list) if coords_list else np.empty((0, 3))
        masses = np.concatenate(masses_list) if masses_list else np.empty(0)
        ids = np.concatenate(ids_list) if ids_list else None
        
        return coords, masses, ids
    
    def extract(
        self,
        center: np.ndarray,
        radius: float,
        halo_id: int = 0,
        mass: float = 0.0,
    ) -> ExtractedHalo:
        """
        Extract particles within a given radius of a center.

        Parameters
        ----------
        center : ndarray
            Center position (3,) in Mpc/h.
        radius : float
            Extraction radius in Mpc/h.
        halo_id : int
            Halo ID for metadata.
        mass : float
            Halo mass for metadata.

        Returns
        -------
        halo : ExtractedHalo
            Extracted particle data.
        """
        if not self._loaded:
            self.load_particles()
        
        center = np.asarray(center)
        coordinates = {}
        masses = {}
        particle_ids = {}
        
        for ptype in self.particle_types:
            if ptype not in self._trees or self._trees[ptype] is None:
                coordinates[ptype] = np.empty((0, 3), dtype=np.float32)
                masses[ptype] = np.empty(0, dtype=np.float32)
                continue
            
            # Handle periodic boundaries by querying with wrapped coordinates
            indices = self._query_periodic(center, radius, ptype)
            
            coords = self._particles[ptype]['coords'][indices]
            mass_arr = self._particles[ptype]['masses'][indices]
            
            # Apply periodic wrapping to coordinates
            coords = self._apply_periodic(coords, center)
            
            coordinates[ptype] = coords
            masses[ptype] = mass_arr
            
            if self._particles[ptype]['ids'] is not None:
                particle_ids[ptype] = self._particles[ptype]['ids'][indices]
        
        return ExtractedHalo(
            halo_id=halo_id,
            center=center,
            radius=radius,
            mass=mass,
            extraction_radius=radius,
            coordinates=coordinates,
            masses=masses,
            particle_ids=particle_ids if particle_ids else None,
        )
    
    def _query_periodic(
        self,
        center: np.ndarray,
        radius: float,
        ptype: str,
    ) -> np.ndarray:
        """
        Query KD-tree with periodic boundary handling.

        For queries near box boundaries, we need to check multiple
        periodic images.
        """
        tree = self._trees[ptype]
        
        # Simple query (most common case)
        indices = tree.query_ball_point(center, radius)
        
        # Check if we need to handle periodic boundaries
        for dim in range(3):
            if center[dim] - radius < 0:
                # Query from opposite side
                shifted_center = center.copy()
                shifted_center[dim] += self.box_size
                indices.extend(tree.query_ball_point(shifted_center, radius))
            
            if center[dim] + radius > self.box_size:
                shifted_center = center.copy()
                shifted_center[dim] -= self.box_size
                indices.extend(tree.query_ball_point(shifted_center, radius))
        
        return np.unique(indices)
    
    def _apply_periodic(
        self,
        coords: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        """Apply minimum image convention for periodic boundaries."""
        if len(coords) == 0:
            return coords
        
        dx = coords - center
        dx = dx - np.round(dx / self.box_size) * self.box_size
        return center + dx


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_halo_particles(
    basePath: str,
    snapNum: int,
    center: np.ndarray,
    radius: float,
    box_size: float,
    particle_types: List[str] = ['dm', 'gas', 'stars'],
    mass_unit: float = 1e10,
    length_unit: float = 1e-3,
    halo_id: int = 0,
    mass: float = 0.0,
) -> ExtractedHalo:
    """
    Extract particles around a halo center.

    This is a convenience function that creates a ParticleExtractor
    and performs a single extraction.

    Parameters
    ----------
    basePath : str
        Path to simulation output.
    snapNum : int
        Snapshot number.
    center : ndarray
        Halo center position (3,) in Mpc/h.
    radius : float
        Extraction radius in Mpc/h.
    box_size : float
        Simulation box size in Mpc/h.
    particle_types : list of str
        Particle types to extract.
    mass_unit : float
        Mass unit conversion.
    length_unit : float
        Length unit conversion.
    halo_id : int
        Halo ID for metadata.
    mass : float
        Halo mass for metadata.

    Returns
    -------
    halo : ExtractedHalo
        Extracted particle data.

    Examples
    --------
    >>> halo = extract_halo_particles(
    ...     basePath='/path/to/TNG300',
    ...     snapNum=99,
    ...     center=[100.0, 100.0, 100.0],
    ...     radius=1.0,
    ...     box_size=205.0,
    ... )
    >>> print(f"Extracted {halo.total_particles} particles")
    """
    extractor = ParticleExtractor(
        basePath=basePath,
        snapNum=snapNum,
        box_size=box_size,
        particle_types=particle_types,
        mass_unit=mass_unit,
        length_unit=length_unit,
    )
    
    return extractor.extract(
        center=center,
        radius=radius,
        halo_id=halo_id,
        mass=mass,
    )


def extract_matched_halos(
    matched_catalog: MatchedCatalog,
    hydro_basePath: str,
    dmo_basePath: str,
    snapNum: int,
    box_size: float,
    radius_mult: float = 5.0,
    output_dir: Optional[str] = None,
    particle_types: List[str] = ['dm', 'gas', 'stars'],
) -> Tuple[List[ExtractedHalo], List[ExtractedHalo]]:
    """
    Extract particles for all matched halos.

    Parameters
    ----------
    matched_catalog : MatchedCatalog
        Matched halo catalog from bijective matching.
    hydro_basePath : str
        Path to hydro simulation.
    dmo_basePath : str
        Path to DMO simulation.
    snapNum : int
        Snapshot number.
    box_size : float
        Box size in Mpc/h.
    radius_mult : float
        Radius multiplier (extraction_radius = radius_mult * R_200c).
    output_dir : str, optional
        If provided, save extracted halos to this directory.
    particle_types : list of str
        Particle types to extract.

    Returns
    -------
    hydro_halos : list of ExtractedHalo
        Extracted hydro halo data.
    dmo_halos : list of ExtractedHalo
        Extracted DMO halo data.
    """
    # Create extractors
    hydro_extractor = ParticleExtractor(
        basePath=hydro_basePath,
        snapNum=snapNum,
        box_size=box_size,
        particle_types=particle_types,
        sim_type='hydro',
    )
    
    dmo_extractor = ParticleExtractor(
        basePath=dmo_basePath,
        snapNum=snapNum,
        box_size=box_size,
        particle_types=['dm'],
        sim_type='dmo',
    )
    
    # Load particles
    hydro_extractor.load_particles()
    dmo_extractor.load_particles()
    
    hydro_halos = []
    dmo_halos = []
    
    for i in range(matched_catalog.n_matches):
        dmo_idx = matched_catalog.dmo_indices[i]
        hydro_idx = matched_catalog.hydro_indices[i]
        
        dmo_center = matched_catalog.dmo_positions[i]
        hydro_center = matched_catalog.hydro_positions[i]
        
        dmo_radius = matched_catalog.dmo_radii[i]
        hydro_radius = matched_catalog.hydro_radii[i]
        
        dmo_mass = matched_catalog.dmo_masses[i]
        hydro_mass = matched_catalog.hydro_masses[i]
        
        # Extract
        hydro_halo = hydro_extractor.extract(
            center=hydro_center,
            radius=hydro_radius * radius_mult,
            halo_id=hydro_idx,
            mass=hydro_mass,
        )
        
        dmo_halo = dmo_extractor.extract(
            center=dmo_center,
            radius=dmo_radius * radius_mult,
            halo_id=dmo_idx,
            mass=dmo_mass,
        )
        
        hydro_halos.append(hydro_halo)
        dmo_halos.append(dmo_halo)
        
        # Save if output_dir provided
        if output_dir is not None:
            output_path = Path(output_dir)
            hydro_halo.save_hdf5(output_path / f'hydro_halo_{hydro_idx}.h5')
            dmo_halo.save_hdf5(output_path / f'dmo_halo_{dmo_idx}.h5')
        
        if (i + 1) % 100 == 0:
            logger.info(f"Extracted {i+1}/{matched_catalog.n_matches} halo pairs")
    
    return hydro_halos, dmo_halos
