"""
Lens Plane Projector: Generate 2D projected density maps for lux ray-tracing.

This module creates pre-projected 2D density maps that can be read by a modified
version of lux, avoiding the need to write full 3D particle snapshots.

The workflow mirrors lux's internal processing:
1. Apply rotation (projection direction)
2. Apply translation and flip transformations
3. Project particles to 2D grid using CIC/TSC
4. Split into lens planes (typically 2 per snapshot, each 102.5 Mpc/h thick)

Output format: HDF5 files with projected density maps that lux can read directly.

Author: Max Lee
Date: December 2025
"""

import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class LensPlaneConfig:
    """Configuration for lens plane projection."""
    
    # Grid parameters
    grid_size: int = 4096  # LP_grid in lux
    planes_per_snapshot: int = 2  # pps in lux
    
    # Box parameters  
    box_size: float = 205.0  # Mpc/h (L in lux)
    
    # Random seed for reproducibility
    random_seed: int = 2020
    
    # Transformation flags
    apply_translation: bool = True
    apply_rotation: bool = True
    apply_flip: bool = True
    
    # Projection direction (0=x, 1=y, 2=z, -1=random)
    projection_direction: int = -1
    
    # Whether to stack (double transverse size)
    stack: bool = False
    
    @property
    def transverse_size(self) -> float:
        """Transverse box size (Lt in lux)."""
        return 2.0 * self.box_size if self.stack else self.box_size
    
    @property
    def longitudinal_size(self) -> float:
        """Longitudinal box size (Ll in lux)."""
        return self.box_size
    
    @property
    def plane_thickness(self) -> float:
        """Thickness of each lens plane in Mpc/h."""
        return self.longitudinal_size / self.planes_per_snapshot
    
    @property
    def cell_size(self) -> float:
        """Size of each grid cell in Mpc/h."""
        return self.transverse_size / self.grid_size


class LensPlaneProjector:
    """
    Project 3D particle distributions to 2D lens planes.
    
    This class replicates the projection logic from lux's lenspot.cpp,
    allowing us to pre-compute projected density maps in Python.
    """
    
    def __init__(self, config: LensPlaneConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        
        # Pre-compute transformation parameters
        self._setup_transformations()
    
    def _setup_transformations(self):
        """Setup random transformations (mimics lux's random setup)."""
        cfg = self.config
        
        # Projection direction
        if cfg.projection_direction in [0, 1, 2]:
            self.proj_dir = cfg.projection_direction
        else:
            self.proj_dir = self.rng.integers(0, 3)
        
        # Translation displacement (in Mpc/h)
        if cfg.apply_translation:
            self.disp = self.rng.uniform(0, cfg.box_size, size=3)
        else:
            self.disp = np.zeros(3)
        
        # Flip flags
        if cfg.apply_flip:
            self.flip = self.rng.integers(0, 2, size=3).astype(bool)
        else:
            self.flip = np.zeros(3, dtype=bool)
        
        logger.info(f"  Projection direction: {['x', 'y', 'z'][self.proj_dir]}")
        logger.info(f"  Displacement: {self.disp}")
        logger.info(f"  Flip: {self.flip}")
    
    def _apply_transformation(self, x: np.ndarray, y: np.ndarray, z: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply rotation, translation, and flip to particle coordinates.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Original particle coordinates in Mpc/h
            
        Returns
        -------
        x_t, y_t, z_t : np.ndarray
            Transformed coordinates (transverse x, transverse y, longitudinal z)
        """
        cfg = self.config
        
        # Map coordinates based on projection direction
        # ix, iy, iz determine which original axis maps to transverse (x,y) and longitudinal (z)
        if self.proj_dir == 0:  # Project along x
            ix, iy, iz = 1, 2, 0
            coords = [x, y, z]
        elif self.proj_dir == 1:  # Project along y
            ix, iy, iz = 2, 0, 1
            coords = [x, y, z]
        else:  # Project along z
            ix, iy, iz = 0, 1, 2
            coords = [x, y, z]
        
        # Select coordinates based on projection
        x_t = coords[ix].copy()
        y_t = coords[iy].copy()
        z_t = coords[iz].copy()
        
        # Apply displacement
        x_t += self.disp[ix]
        y_t += self.disp[iy]
        z_t += self.disp[iz]
        
        # Apply flip
        if self.flip[ix]:
            x_t = -x_t
        if self.flip[iy]:
            y_t = -y_t
        if self.flip[iz]:
            z_t = -z_t
        
        # Apply periodic boundary conditions
        Lt = cfg.transverse_size
        Ll = cfg.longitudinal_size
        
        if not cfg.stack:
            x_t = np.mod(x_t, Lt)
            y_t = np.mod(y_t, Lt)
            z_t = np.mod(z_t, Ll)
        else:
            # For stacked snapshots, wrap to central half
            x_t = np.mod(x_t - Lt/4, Lt/2) + Lt/4
            y_t = np.mod(y_t - Lt/4, Lt/2) + Lt/4
            z_t = np.mod(z_t, Ll)
        
        return x_t, y_t, z_t
    
    def _tsc_weights(self, x: float, cell_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute TSC (Triangular Shaped Cloud) weights for a position.
        
        Returns cell indices and weights for the 3 cells affected.
        """
        # Cell index of particle
        nx = int(np.floor(x / cell_size))
        
        # Distance to cell center
        dx = x / cell_size - nx - 0.5
        
        # TSC weights
        w_m1 = 0.5 * (0.5 - dx)**2  # left cell
        w_0 = 0.75 - dx**2           # center cell
        w_p1 = 0.5 * (0.5 + dx)**2  # right cell
        
        return np.array([nx-1, nx, nx+1]), np.array([w_m1, w_0, w_p1])
    
    def project_particles(self, positions: np.ndarray, masses: np.ndarray
                         ) -> List[np.ndarray]:
        """
        Project particles to 2D lens plane grids.
        
        Parameters
        ----------
        positions : np.ndarray
            Particle positions (N, 3) in Mpc/h
        masses : np.ndarray
            Particle masses in Msun/h (or 10^10 Msun/h)
            
        Returns
        -------
        planes : List[np.ndarray]
            List of 2D density grids, one per plane
        """
        cfg = self.config
        n_planes = cfg.planes_per_snapshot
        grid = cfg.grid_size
        cell = cfg.cell_size
        plane_thick = cfg.plane_thickness
        
        logger.info(f"  Projecting {len(positions):,} particles to {n_planes} planes...")
        logger.info(f"  Grid: {grid}x{grid}, cell size: {cell:.4f} Mpc/h")
        logger.info(f"  Plane thickness: {plane_thick:.2f} Mpc/h")
        
        # Initialize density grids
        planes = [np.zeros((grid, grid), dtype=np.float64) for _ in range(n_planes)]
        
        # Transform coordinates
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        x_t, y_t, z_t = self._apply_transformation(x, y, z)
        
        # Handle scalar mass (all particles same mass)
        if np.isscalar(masses) or len(masses) == 1:
            m = np.full(len(positions), masses if np.isscalar(masses) else masses[0])
        else:
            m = masses
        
        # Assign particles to planes using TSC
        # This is vectorized for efficiency
        
        # Determine which plane each particle belongs to
        plane_idx = np.floor(z_t / plane_thick).astype(int)
        plane_idx = np.clip(plane_idx, 0, n_planes - 1)
        
        # Grid indices
        ix = np.floor(x_t / cell).astype(int) % grid
        iy = np.floor(y_t / cell).astype(int) % grid
        
        # For TSC, we need to distribute to 3x3 cells
        # For efficiency, we'll use CIC (2x2) here - can upgrade to TSC if needed
        
        # CIC weights
        dx = x_t / cell - np.floor(x_t / cell)
        dy = y_t / cell - np.floor(y_t / cell)
        
        wx0 = 1.0 - dx
        wx1 = dx
        wy0 = 1.0 - dy
        wy1 = dy
        
        # Assign to 4 neighboring cells (CIC)
        for p in range(n_planes):
            mask = (plane_idx == p)
            if not np.any(mask):
                continue
            
            m_p = m[mask]
            ix_p = ix[mask]
            iy_p = iy[mask]
            wx0_p = wx0[mask]
            wx1_p = wx1[mask]
            wy0_p = wy0[mask]
            wy1_p = wy1[mask]
            
            ix1 = (ix_p + 1) % grid
            iy1 = (iy_p + 1) % grid
            
            # Add contributions using np.add.at for thread safety
            np.add.at(planes[p], (ix_p, iy_p), m_p * wx0_p * wy0_p)
            np.add.at(planes[p], (ix1, iy_p), m_p * wx1_p * wy0_p)
            np.add.at(planes[p], (ix_p, iy1), m_p * wx0_p * wy1_p)
            np.add.at(planes[p], (ix1, iy1), m_p * wx1_p * wy1_p)
        
        logger.info(f"  Projection complete")
        
        return planes
    
    def compute_density_contrast(self, planes: List[np.ndarray], 
                                  Omega_m: float = 0.3089) -> List[np.ndarray]:
        """
        Convert mass maps to density contrast (delta) as done in lux.
        
        lux computes: delta = (rho / bar_rho - 1) * plane_thickness
        
        Parameters
        ----------
        planes : List[np.ndarray]
            Mass maps from project_particles
        Omega_m : float
            Matter density parameter
            
        Returns
        -------
        delta_planes : List[np.ndarray]
            Density contrast maps ready for lux FFT
        """
        cfg = self.config
        
        # Critical density in 10^10 Msun/h / (Mpc/h)^3
        rho_c0 = 27.7536627
        
        # Mean mass per cell
        # bar = rho_c0 * Omega_m * Lt^2 * Ll / pps / grid^2
        bar = (rho_c0 * Omega_m * cfg.transverse_size**2 * 
               cfg.longitudinal_size / cfg.planes_per_snapshot / cfg.grid_size**2)
        
        delta_planes = []
        for plane in planes:
            # Convert to overdensity and multiply by plane thickness
            delta = (plane / bar - 1.0) * cfg.plane_thickness
            delta_planes.append(delta)
        
        return delta_planes
    
    def save_lens_planes(self, planes: List[np.ndarray], output_path: Path,
                        snap_num: int, mode: str, redshift: float = 0.0):
        """
        Save lens plane maps in format readable by modified lux.
        
        Parameters
        ----------
        planes : List[np.ndarray]
            Density contrast maps
        output_path : Path
            Output directory
        snap_num : int
            Snapshot number
        mode : str
            Pipeline mode (dmo, hydro, replace, bcm)
        redshift : float
            Snapshot redshift
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cfg = self.config
        
        for p, plane in enumerate(planes):
            plane_num = snap_num * cfg.planes_per_snapshot + p + 1
            
            # Save in HDF5 format for easy reading
            fname = output_path / f'lensplane_{plane_num:02d}.h5'
            
            with h5py.File(fname, 'w') as f:
                # Store the density contrast map
                f.create_dataset('delta', data=plane, compression='gzip')
                
                # Store metadata
                f.attrs['grid_size'] = cfg.grid_size
                f.attrs['transverse_size'] = cfg.transverse_size
                f.attrs['longitudinal_size'] = cfg.longitudinal_size
                f.attrs['plane_thickness'] = cfg.plane_thickness
                f.attrs['snapshot'] = snap_num
                f.attrs['plane_index'] = p
                f.attrs['plane_number'] = plane_num
                f.attrs['redshift'] = redshift
                f.attrs['mode'] = mode
                f.attrs['projection_direction'] = self.proj_dir
                f.attrs['displacement'] = self.disp
                f.attrs['flip'] = self.flip
            
            logger.info(f"  Saved lens plane {plane_num}: {fname}")
        
        # Also save in binary format matching lux's expected format
        # This allows direct reading by lux without modification
        for p, plane in enumerate(planes):
            plane_num = snap_num * cfg.planes_per_snapshot + p + 1
            fname = output_path / f'density{plane_num:02d}.dat'
            
            with open(fname, 'wb') as f:
                # lux binary format: int(grid), double[grid*grid], int(grid)
                grid = np.array([cfg.grid_size], dtype=np.int32)
                grid.tofile(f)
                plane.astype(np.float64).tofile(f)
                grid.tofile(f)
            
            logger.info(f"  Saved lux-format density: {fname}")
    
    def save_config(self, output_path: Path, n_snapshots: int, 
                    scale_factors: List[float], chi_values: List[float]):
        """
        Save configuration file matching lux's config.dat format.
        
        This allows ray-tracing to proceed without re-reading snapshots.
        """
        output_path = Path(output_path)
        cfg = self.config
        
        Np = n_snapshots * cfg.planes_per_snapshot
        Ns = n_snapshots
        
        fname = output_path / 'config.dat'
        
        with open(fname, 'wb') as f:
            np.array([Np], dtype=np.int32).tofile(f)
            np.array([Ns], dtype=np.int32).tofile(f)
            np.array(scale_factors, dtype=np.float64).tofile(f)
            np.array(chi_values, dtype=np.float64).tofile(f)
            # chi_out values
            np.array(chi_values[1:], dtype=np.float64).tofile(f)
            # Ll values (all same for TNG)
            np.full(Ns, cfg.longitudinal_size, dtype=np.float64).tofile(f)
            # Lt values
            np.full(Ns, cfg.transverse_size, dtype=np.float64).tofile(f)
            # proj_dirs
            np.full(Ns, self.proj_dir, dtype=np.int32).tofile(f)
            # disp (3*Ns values)
            np.tile(self.disp, Ns).astype(np.float64).tofile(f)
            # flip (3*Ns values)
            np.tile(self.flip, Ns).astype(np.bool_).tofile(f)
        
        logger.info(f"  Saved config: {fname}")


def create_lens_planes_from_particles(positions: np.ndarray, 
                                       masses: np.ndarray,
                                       output_dir: Path,
                                       snap_num: int,
                                       mode: str = 'dmo',
                                       redshift: float = 0.0,
                                       config: Optional[LensPlaneConfig] = None,
                                       stack: bool = False) -> LensPlaneProjector:
    """
    Convenience function to create lens planes from particle data.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions (N, 3) in Mpc/h
    masses : np.ndarray
        Particle masses in 10^10 Msun/h (matching TNG convention)
    output_dir : Path
        Directory to save lens planes
    snap_num : int
        Snapshot number (for naming output files)
    mode : str
        Pipeline mode for metadata
    redshift : float
        Snapshot redshift
    config : LensPlaneConfig, optional
        Configuration parameters
    stack : bool
        Whether to use stacking (doubles transverse size)
        
    Returns
    -------
    projector : LensPlaneProjector
        The projector instance (useful for accessing transformation params)
    """
    if config is None:
        config = LensPlaneConfig(stack=stack)
    else:
        config.stack = stack
    
    projector = LensPlaneProjector(config)
    
    # Project particles
    planes = projector.project_particles(positions, masses)
    
    # Convert to density contrast
    delta_planes = projector.compute_density_contrast(planes)
    
    # Save
    projector.save_lens_planes(delta_planes, output_dir, snap_num, mode, redshift)
    
    return projector
