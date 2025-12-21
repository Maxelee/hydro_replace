"""
Power Spectrum Module
=====================

Functions for computing matter power spectra and suppression ratios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

# Try to import Pylians
try:
    import Pk_library as PKL
    import MAS_library as MASL
    HAS_PYLIANS = True
except ImportError:
    HAS_PYLIANS = False

logger = logging.getLogger(__name__)


@dataclass  
class PowerSpectrum:
    """
    Container for power spectrum data.

    Attributes
    ----------
    k : ndarray
        Wavenumber array (h/Mpc).
    pk : ndarray
        Power spectrum P(k) ((Mpc/h)^3).
    n_modes : ndarray
        Number of modes in each k bin.
    box_size : float
        Box size in Mpc/h.
    grid_size : int
        Grid resolution used.
    mas : str
        Mass assignment scheme used.
    label : str
        Label for this spectrum.
    """
    
    k: np.ndarray
    pk: np.ndarray
    n_modes: np.ndarray
    box_size: float
    grid_size: int = 1024
    mas: str = "CIC"
    label: str = ""
    
    @property
    def k_min(self) -> float:
        """Minimum k."""
        return float(self.k[self.pk > 0].min())
    
    @property
    def k_max(self) -> float:
        """Maximum k."""
        return float(self.k[self.pk > 0].max())
    
    @property
    def log_k(self) -> np.ndarray:
        """Log10 of k."""
        return np.log10(self.k)
    
    @property
    def log_pk(self) -> np.ndarray:
        """Log10 of P(k)."""
        with np.errstate(divide='ignore'):
            return np.log10(self.pk)
    
    def get_pk_at_k(self, k_target: float) -> float:
        """
        Interpolate P(k) at a given k.

        Parameters
        ----------
        k_target : float
            Target wavenumber.

        Returns
        -------
        pk : float
            Interpolated power.
        """
        valid = self.pk > 0
        return np.interp(
            np.log10(k_target),
            np.log10(self.k[valid]),
            np.log10(self.pk[valid])
        )
    
    def save_hdf5(self, filepath: Union[str, Path], group_name: str = "power") -> None:
        """Save power spectrum to HDF5."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'a') as f:
            if group_name in f:
                del f[group_name]
            
            grp = f.create_group(group_name)
            grp.attrs['box_size'] = self.box_size
            grp.attrs['grid_size'] = self.grid_size
            grp.attrs['mas'] = self.mas
            grp.attrs['label'] = self.label
            
            grp.create_dataset('k', data=self.k)
            grp.create_dataset('pk', data=self.pk)
            grp.create_dataset('n_modes', data=self.n_modes)
    
    def save_txt(self, filepath: Union[str, Path]) -> None:
        """Save power spectrum to text file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        np.savetxt(
            filepath,
            np.column_stack([self.k, self.pk, self.n_modes]),
            header='k[h/Mpc] Pk[(Mpc/h)^3] Nmodes',
            fmt='%.6e'
        )
    
    @classmethod
    def load_hdf5(cls, filepath: Union[str, Path], group_name: str = "power") -> 'PowerSpectrum':
        """Load power spectrum from HDF5."""
        with h5py.File(filepath, 'r') as f:
            grp = f[group_name]
            return cls(
                k=grp['k'][:],
                pk=grp['pk'][:],
                n_modes=grp['n_modes'][:],
                box_size=grp.attrs['box_size'],
                grid_size=grp.attrs.get('grid_size', 1024),
                mas=grp.attrs.get('mas', 'CIC'),
                label=grp.attrs.get('label', ''),
            )


def compute_power_spectrum(
    coords: np.ndarray,
    masses: np.ndarray,
    box_size: float,
    grid_size: int = 1024,
    mas: str = 'CIC',
    axis: int = 0,
    threads: int = 1,
    label: str = "",
) -> PowerSpectrum:
    """
    Compute matter power spectrum from particle data.

    Parameters
    ----------
    coords : ndarray
        Particle coordinates (N, 3) in Mpc/h.
    masses : ndarray
        Particle masses (N,) in Msun/h.
    box_size : float
        Box size in Mpc/h.
    grid_size : int
        Grid resolution for FFT.
    mas : str
        Mass assignment scheme ('NGP', 'CIC', 'TSC', 'PCS').
    axis : int
        Axis for line-of-sight (0, 1, or 2).
    threads : int
        Number of OpenMP threads.
    label : str
        Label for this spectrum.

    Returns
    -------
    power : PowerSpectrum
        Computed power spectrum.

    Raises
    ------
    ImportError
        If Pylians is not installed.

    Examples
    --------
    >>> pk = compute_power_spectrum(
    ...     coords=particle_coords,
    ...     masses=particle_masses,
    ...     box_size=205.0,
    ...     grid_size=1024,
    ... )
    >>> print(f"P(k=1) = {pk.get_pk_at_k(1.0):.2e}")
    """
    if not HAS_PYLIANS:
        raise ImportError("Pylians required for power spectrum computation")
    
    logger.info(f"Computing power spectrum (grid={grid_size}, mas={mas})...")
    
    # Create density field
    delta = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    MASL.MA(
        coords.astype(np.float32),
        delta,
        box_size,
        MAS=mas,
        W=masses.astype(np.float32),
        verbose=False,
    )
    
    # Convert to overdensity
    mean_density = delta.mean()
    if mean_density > 0:
        delta = delta / mean_density - 1.0
    else:
        logger.warning("Zero mean density - empty field?")
        delta = delta * 0.0
    
    # Compute power spectrum
    pk = PKL.Pk(delta, box_size, axis, mas, threads, verbose=False)
    
    return PowerSpectrum(
        k=pk.k1D,
        pk=pk.Pk1D,
        n_modes=pk.Nmodes1D,
        box_size=box_size,
        grid_size=grid_size,
        mas=mas,
        label=label,
    )


def compute_power_from_grid(
    delta: np.ndarray,
    box_size: float,
    mas: str = 'CIC',
    axis: int = 0,
    threads: int = 1,
    label: str = "",
) -> PowerSpectrum:
    """
    Compute power spectrum from a pre-computed density grid.

    Parameters
    ----------
    delta : ndarray
        3D density or overdensity field.
    box_size : float
        Box size in Mpc/h.
    mas : str
        Mass assignment scheme used to create the grid.
    axis : int
        Axis for line-of-sight.
    threads : int
        Number of OpenMP threads.
    label : str
        Label for this spectrum.

    Returns
    -------
    power : PowerSpectrum
        Computed power spectrum.
    """
    if not HAS_PYLIANS:
        raise ImportError("Pylians required for power spectrum computation")
    
    grid_size = delta.shape[0]
    
    # Convert to overdensity if needed
    mean = delta.mean()
    if mean > 0 and not np.isclose(mean, 0.0, atol=1e-3):
        # Likely density field, convert to overdensity
        delta_use = delta / mean - 1.0
    else:
        # Already overdensity
        delta_use = delta
    
    pk = PKL.Pk(delta_use.astype(np.float32), box_size, axis, mas, threads, verbose=False)
    
    return PowerSpectrum(
        k=pk.k1D,
        pk=pk.Pk1D,
        n_modes=pk.Nmodes1D,
        box_size=box_size,
        grid_size=grid_size,
        mas=mas,
        label=label,
    )


def compute_suppression(
    pk_hydro: PowerSpectrum,
    pk_dmo: PowerSpectrum,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectrum suppression S(k) = P_hydro / P_dmo.

    Parameters
    ----------
    pk_hydro : PowerSpectrum
        Hydro (or baryon-affected) power spectrum.
    pk_dmo : PowerSpectrum
        DMO power spectrum.

    Returns
    -------
    k : ndarray
        Wavenumber array.
    suppression : ndarray
        Suppression ratio S(k).

    Examples
    --------
    >>> k, S = compute_suppression(pk_hydro, pk_dmo)
    >>> print(f"Suppression at k=1: {S[np.argmin(np.abs(k-1))]:.3f}")
    """
    # Check that k arrays match
    if not np.allclose(pk_hydro.k, pk_dmo.k):
        logger.warning("k arrays don't match - interpolating")
        # Use DMO k as reference
        k = pk_dmo.k
        pk_hydro_interp = 10**np.interp(
            np.log10(k),
            np.log10(pk_hydro.k),
            np.log10(pk_hydro.pk)
        )
        suppression = pk_hydro_interp / pk_dmo.pk
    else:
        k = pk_dmo.k
        with np.errstate(divide='ignore', invalid='ignore'):
            suppression = pk_hydro.pk / pk_dmo.pk
    
    return k, suppression


def compute_suppression_ratio(
    pk_test: PowerSpectrum,
    pk_reference: PowerSpectrum,
    pk_target: PowerSpectrum,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute how well a test spectrum matches a target relative to reference.

    ratio = (P_test / P_ref) / (P_target / P_ref) = P_test / P_target

    Parameters
    ----------
    pk_test : PowerSpectrum
        Test power spectrum (e.g., replacement).
    pk_reference : PowerSpectrum
        Reference power spectrum (e.g., DMO).
    pk_target : PowerSpectrum
        Target power spectrum (e.g., hydro).

    Returns
    -------
    k : ndarray
        Wavenumber array.
    ratio : ndarray
        Ratio of suppressions.
    """
    k1, s_test = compute_suppression(pk_test, pk_reference)
    k2, s_target = compute_suppression(pk_target, pk_reference)
    
    # Ensure same k
    if not np.allclose(k1, k2):
        raise ValueError("Incompatible k arrays")
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = s_test / s_target
    
    return k1, ratio


class PowerSpectrumAnalyzer:
    """
    Analyze and compare multiple power spectra.

    Parameters
    ----------
    spectra : dict
        Dictionary mapping labels to PowerSpectrum objects.
    """
    
    def __init__(self, spectra: Dict[str, PowerSpectrum]):
        self.spectra = spectra
    
    @property
    def k(self) -> np.ndarray:
        """Common k array."""
        return list(self.spectra.values())[0].k
    
    def compute_ratio(self, label1: str, label2: str) -> np.ndarray:
        """Compute P_1(k) / P_2(k)."""
        pk1 = self.spectra[label1].pk
        pk2 = self.spectra[label2].pk
        
        with np.errstate(divide='ignore', invalid='ignore'):
            return pk1 / pk2
    
    def compute_residuals(self, label1: str, label2: str) -> np.ndarray:
        """Compute (P_1 - P_2) / P_2."""
        pk1 = self.spectra[label1].pk
        pk2 = self.spectra[label2].pk
        
        with np.errstate(divide='ignore', invalid='ignore'):
            return (pk1 - pk2) / pk2
    
    def save_all(self, output_dir: Union[str, Path], prefix: str = "") -> None:
        """Save all spectra to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for label, pk in self.spectra.items():
            filename = f"{prefix}{label}_pk.h5" if prefix else f"{label}_pk.h5"
            pk.save_hdf5(output_dir / filename)
