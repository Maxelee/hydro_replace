"""
Simulation Loading Module
=========================

Functions for loading TNG and CAMELS simulation data with proper unit conversions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import yaml

# Try to import illustris_python, fall back gracefully
try:
    from illustris_python import groupcat, snapshot
    HAS_ILLUSTRIS = True
except ImportError:
    HAS_ILLUSTRIS = False
    logging.warning("illustris_python not found. Some functionality will be limited.")

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Loading
# =============================================================================

def load_simulation_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load simulation configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    config : dict
        Configuration dictionary with simulation paths and parameters.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the YAML file is malformed.

    Examples
    --------
    >>> config = load_simulation_config('config/simulation_paths.yaml')
    >>> print(config['tng300']['hydro']['basePath'])
    '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output'
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    
    # Expand environment variables in paths
    config = _expand_paths(config)
    
    return config


def _expand_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand path variables in configuration."""
    if isinstance(config, dict):
        # Handle ${base_dir} style references
        base_dir = config.get('base_dir', '')
        result = {}
        for key, value in config.items():
            if isinstance(value, str) and '${base_dir}' in value:
                result[key] = value.replace('${base_dir}', base_dir)
            else:
                result[key] = _expand_paths(value)
        return result
    elif isinstance(config, list):
        return [_expand_paths(item) for item in config]
    else:
        return config


def load_analysis_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load analysis parameters from YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    config : dict
        Configuration dictionary with analysis parameters.
    """
    return load_simulation_config(config_path)


# =============================================================================
# Simulation Data Container
# =============================================================================

@dataclass
class SimulationData:
    """
    Container for simulation data with lazy loading support.

    This class provides a unified interface for accessing simulation data
    from TNG and CAMELS simulations.

    Parameters
    ----------
    basePath : str
        Base path to the simulation output directory.
    snapNum : int
        Snapshot number to load.
    name : str, optional
        Human-readable name for the simulation.
    box_size : float, optional
        Box size in Mpc/h. Will be read from header if not provided.
    dm_particle_mass : float, optional
        DM particle mass in 10^10 Msun/h.

    Attributes
    ----------
    header : dict
        Snapshot header information.
    redshift : float
        Redshift of the snapshot.
    n_particles : dict
        Number of particles by type.

    Examples
    --------
    >>> sim = SimulationData('/path/to/TNG300-1/output', 99)
    >>> print(sim.redshift)
    0.0
    >>> print(sim.box_size)
    205.0
    """
    
    basePath: str
    snapNum: int
    name: str = ""
    box_size: float = 0.0
    dm_particle_mass: float = 0.0
    
    # Lazy-loaded attributes
    _header: Optional[Dict] = field(default=None, repr=False)
    _halo_catalog: Optional[Dict] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize derived attributes."""
        if not Path(self.basePath).exists():
            raise FileNotFoundError(f"Simulation path does not exist: {self.basePath}")
        
        # Load header to get box size and redshift
        self._load_header()
        
        if self.box_size == 0.0:
            self.box_size = self._header.get('BoxSize', 0.0) / 1e3  # Convert to Mpc/h
        
        logger.info(f"Initialized SimulationData: {self.name or self.basePath}")
        logger.info(f"  Snapshot: {self.snapNum}, z={self.redshift:.3f}")
        logger.info(f"  Box size: {self.box_size:.1f} Mpc/h")
    
    def _load_header(self) -> None:
        """Load snapshot header."""
        if self._header is not None:
            return
        
        if HAS_ILLUSTRIS:
            self._header = snapshot.loadHeader(self.basePath, self.snapNum)
        else:
            self._header = load_snapshot_header(self.basePath, self.snapNum)
    
    @property
    def header(self) -> Dict:
        """Get snapshot header."""
        self._load_header()
        return self._header
    
    @property
    def redshift(self) -> float:
        """Get snapshot redshift."""
        return self.header.get('Redshift', 0.0)
    
    @property
    def scale_factor(self) -> float:
        """Get scale factor a = 1/(1+z)."""
        return 1.0 / (1.0 + self.redshift)
    
    @property
    def n_particles(self) -> Dict[str, int]:
        """Get number of particles by type."""
        n_part = self.header.get('NumPart_Total', [0]*6)
        return {
            'gas': n_part[0],
            'dm': n_part[1],
            'tracers': n_part[3],
            'stars': n_part[4],
            'bh': n_part[5],
        }
    
    @property
    def h(self) -> float:
        """Get Hubble parameter h."""
        return self.header.get('HubbleParam', 0.6774)
    
    @property
    def omega_m(self) -> float:
        """Get matter density parameter."""
        return self.header.get('Omega0', 0.3089)
    
    @property
    def omega_lambda(self) -> float:
        """Get dark energy density parameter."""
        return self.header.get('OmegaLambda', 0.6911)
    
    def load_particles(
        self,
        part_type: Union[int, str],
        fields: List[str],
        halo_id: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load particle data.

        Parameters
        ----------
        part_type : int or str
            Particle type (0='gas', 1='dm', 4='stars', 5='bh') or string name.
        fields : list of str
            Fields to load (e.g., ['Coordinates', 'Masses', 'ParticleIDs']).
        halo_id : int, optional
            If provided, load only particles belonging to this halo.

        Returns
        -------
        data : dict
            Dictionary with field names as keys and arrays as values.
        """
        # Convert string type to integer
        type_map = {'gas': 0, 'dm': 1, 'tracers': 3, 'stars': 4, 'bh': 5}
        if isinstance(part_type, str):
            part_type = type_map.get(part_type.lower(), part_type)
        
        if not HAS_ILLUSTRIS:
            raise RuntimeError("illustris_python required for particle loading")
        
        if halo_id is not None:
            # Load specific halo
            data = {}
            for field in fields:
                try:
                    data[field] = snapshot.loadHalo(
                        self.basePath, self.snapNum, halo_id, part_type, [field]
                    )
                except Exception as e:
                    logger.warning(f"Failed to load {field} for halo {halo_id}: {e}")
                    data[field] = np.array([])
        else:
            # Load all particles
            data = snapshot.loadSubset(
                self.basePath, self.snapNum, part_type, fields
            )
            # Handle single field case
            if len(fields) == 1:
                data = {fields[0]: data}
        
        return data
    
    def load_halo_catalog(
        self,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load halo catalog.

        Parameters
        ----------
        fields : list of str, optional
            Specific fields to load. If None, loads default fields.

        Returns
        -------
        catalog : dict
            Dictionary with halo catalog data.
        """
        if fields is None:
            fields = [
                'Group_M_Crit200',
                'Group_R_Crit200', 
                'GroupPos',
                'GroupVel',
                'GroupMassType',
                'GroupLenType',
                'GroupFirstSub',
            ]
        
        if not HAS_ILLUSTRIS:
            raise RuntimeError("illustris_python required for catalog loading")
        
        return groupcat.loadHalos(self.basePath, self.snapNum, fields=fields)


# =============================================================================
# Utility Functions
# =============================================================================

def load_snapshot_header(basePath: str, snapNum: int) -> Dict[str, Any]:
    """
    Load snapshot header directly from HDF5 file.

    This is a fallback for when illustris_python is not available.

    Parameters
    ----------
    basePath : str
        Base path to simulation output.
    snapNum : int
        Snapshot number.

    Returns
    -------
    header : dict
        Snapshot header dictionary.
    """
    # Try different file patterns
    patterns = [
        f"{basePath}/snapdir_{snapNum:03d}/snap_{snapNum:03d}.0.hdf5",
        f"{basePath}/snap_{snapNum:03d}.hdf5",
        f"{basePath}/snapshot_{snapNum:03d}.hdf5",
    ]
    
    for pattern in patterns:
        if Path(pattern).exists():
            with h5py.File(pattern, 'r') as f:
                header = dict(f['Header'].attrs)
            return header
    
    raise FileNotFoundError(f"Could not find snapshot {snapNum} in {basePath}")


def get_snapshot_redshift(basePath: str, snapNum: int) -> float:
    """
    Get redshift for a specific snapshot.

    Parameters
    ----------
    basePath : str
        Base path to simulation output.
    snapNum : int
        Snapshot number.

    Returns
    -------
    redshift : float
        Redshift of the snapshot.
    """
    header = load_snapshot_header(basePath, snapNum)
    return header.get('Redshift', 0.0)


def get_available_snapshots(basePath: str) -> List[int]:
    """
    Get list of available snapshot numbers.

    Parameters
    ----------
    basePath : str
        Base path to simulation output.

    Returns
    -------
    snapshots : list of int
        List of available snapshot numbers.
    """
    basePath = Path(basePath)
    
    # Check for snapdir_XXX directories
    snapdirs = list(basePath.glob("snapdir_*"))
    if snapdirs:
        return sorted([int(d.name.split('_')[1]) for d in snapdirs])
    
    # Check for individual snapshot files
    snapfiles = list(basePath.glob("snap_*.hdf5"))
    if snapfiles:
        return sorted([int(f.stem.split('_')[1]) for f in snapfiles])
    
    return []


def convert_units(
    data: np.ndarray,
    field: str,
    h: float = 0.6774,
    scale_factor: float = 1.0,
    to_physical: bool = True,
) -> np.ndarray:
    """
    Convert simulation units to physical units.

    Parameters
    ----------
    data : ndarray
        Data array to convert.
    field : str
        Field name (e.g., 'Coordinates', 'Masses', 'Velocities').
    h : float
        Hubble parameter.
    scale_factor : float
        Scale factor a = 1/(1+z).
    to_physical : bool
        If True, convert to physical units. If False, keep comoving.

    Returns
    -------
    converted : ndarray
        Data in converted units.
    """
    field_lower = field.lower()
    
    # Coordinate conversion: ckpc/h -> Mpc/h (comoving) or Mpc (physical)
    if 'coordinate' in field_lower or 'pos' in field_lower:
        data = data / 1e3  # kpc/h -> Mpc/h
        if to_physical:
            data = data * scale_factor / h  # Mpc/h -> Mpc
    
    # Mass conversion: 10^10 Msun/h -> Msun/h or Msun
    elif 'mass' in field_lower:
        data = data * 1e10  # 10^10 Msun/h -> Msun/h
        if to_physical:
            data = data / h  # Msun/h -> Msun
    
    # Velocity conversion: km/s (already in physical units)
    elif 'velocity' in field_lower or 'vel' in field_lower:
        # Velocities are typically already in km/s
        pass
    
    # Radius conversion: ckpc/h -> Mpc/h or Mpc
    elif 'radius' in field_lower or 'r_' in field_lower.lower():
        data = data / 1e3  # kpc/h -> Mpc/h
        if to_physical:
            data = data * scale_factor / h  # Mpc/h -> Mpc
    
    return data
