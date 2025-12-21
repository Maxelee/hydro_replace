"""
Convergence Map Module
======================

Tools for working with weak lensing convergence maps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMap:
    """
    Container for weak lensing convergence map.

    Attributes
    ----------
    kappa : ndarray
        2D convergence field.
    fov : float
        Field of view in degrees.
    redshift_source : float
        Source redshift.
    label : str
        Label for this map.
    """
    
    kappa: np.ndarray
    fov: float
    redshift_source: float = 1.0
    label: str = ""
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Map shape (ny, nx)."""
        return self.kappa.shape
    
    @property
    def npix(self) -> int:
        """Number of pixels per side."""
        return self.kappa.shape[0]
    
    @property
    def pixel_scale(self) -> float:
        """Pixel scale in degrees."""
        return self.fov / self.npix
    
    @property
    def pixel_scale_arcmin(self) -> float:
        """Pixel scale in arcminutes."""
        return self.pixel_scale * 60.0
    
    @property
    def mean(self) -> float:
        """Mean convergence."""
        return float(np.nanmean(self.kappa))
    
    @property
    def std(self) -> float:
        """Standard deviation of convergence."""
        return float(np.nanstd(self.kappa))
    
    @property
    def kappa_normalized(self) -> np.ndarray:
        """Convergence normalized by standard deviation (SNR map)."""
        return (self.kappa - self.mean) / self.std
    
    def smooth(
        self,
        smoothing_scale: float,
        scale_unit: str = 'arcmin',
    ) -> 'ConvergenceMap':
        """
        Apply Gaussian smoothing.

        Parameters
        ----------
        smoothing_scale : float
            Smoothing scale.
        scale_unit : str
            'arcmin' or 'pixel'.

        Returns
        -------
        smoothed : ConvergenceMap
            Smoothed convergence map.
        """
        if scale_unit == 'arcmin':
            sigma_pix = smoothing_scale / self.pixel_scale_arcmin
        else:
            sigma_pix = smoothing_scale
        
        kappa_smoothed = smooth_map(self.kappa, sigma_pix)
        
        return ConvergenceMap(
            kappa=kappa_smoothed,
            fov=self.fov,
            redshift_source=self.redshift_source,
            label=f"{self.label}_smooth{smoothing_scale:.1f}{scale_unit}",
        )
    
    def save_fits(self, filepath: Union[str, Path]) -> None:
        """Save to FITS file."""
        from astropy.io import fits
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        hdu = fits.PrimaryHDU(self.kappa.astype(np.float32))
        hdu.header['FOV'] = self.fov
        hdu.header['ZSOURCE'] = self.redshift_source
        hdu.header['LABEL'] = self.label
        hdu.header['PIXSCALE'] = self.pixel_scale_arcmin
        
        hdu.writeto(filepath, overwrite=True)
        logger.info(f"Saved convergence map to {filepath}")
    
    def save_hdf5(self, filepath: Union[str, Path], group_name: str = "kappa") -> None:
        """Save to HDF5 file."""
        import h5py
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'a') as f:
            if group_name in f:
                del f[group_name]
            
            grp = f.create_group(group_name)
            grp.attrs['fov'] = self.fov
            grp.attrs['redshift_source'] = self.redshift_source
            grp.attrs['label'] = self.label
            
            grp.create_dataset('kappa', data=self.kappa.astype(np.float32))
    
    @classmethod
    def load_fits(cls, filepath: Union[str, Path]) -> 'ConvergenceMap':
        """Load from FITS file."""
        from astropy.io import fits
        
        with fits.open(filepath) as hdul:
            kappa = hdul[0].data
            header = hdul[0].header
            
            return cls(
                kappa=kappa,
                fov=header.get('FOV', 5.0),
                redshift_source=header.get('ZSOURCE', 1.0),
                label=header.get('LABEL', ''),
            )
    
    @classmethod
    def load_hdf5(cls, filepath: Union[str, Path], group_name: str = "kappa") -> 'ConvergenceMap':
        """Load from HDF5 file."""
        import h5py
        
        with h5py.File(filepath, 'r') as f:
            grp = f[group_name]
            return cls(
                kappa=grp['kappa'][:],
                fov=grp.attrs['fov'],
                redshift_source=grp.attrs.get('redshift_source', 1.0),
                label=grp.attrs.get('label', ''),
            )


def smooth_map(
    data: np.ndarray,
    sigma: float,
    boundary: str = 'wrap',
) -> np.ndarray:
    """
    Apply Gaussian smoothing to a 2D map.

    Parameters
    ----------
    data : ndarray
        2D data array.
    sigma : float
        Smoothing scale in pixels.
    boundary : str
        Boundary mode: 'wrap', 'reflect', 'constant'.

    Returns
    -------
    smoothed : ndarray
        Smoothed data.
    """
    return ndimage.gaussian_filter(data, sigma=sigma, mode=boundary)


def compute_power_spectrum_2d(
    kappa: np.ndarray,
    pixel_scale: float,
    n_bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D power spectrum of convergence map.

    Parameters
    ----------
    kappa : ndarray
        2D convergence field.
    pixel_scale : float
        Pixel scale in arcminutes.
    n_bins : int
        Number of radial bins.

    Returns
    -------
    ell : ndarray
        Multipole moments.
    cl : ndarray
        Power spectrum C(ell).

    Examples
    --------
    >>> ell, cl = compute_power_spectrum_2d(kappa_map.kappa, kappa_map.pixel_scale_arcmin)
    >>> plt.loglog(ell, ell**2 * cl / (2*np.pi))
    """
    ny, nx = kappa.shape
    
    # 2D FFT
    kappa_fft = np.fft.fft2(kappa)
    power_2d = np.abs(kappa_fft)**2 / (nx * ny)
    
    # Compute ell for each pixel
    # Wavenumber in 1/arcmin
    kx = np.fft.fftfreq(nx, d=pixel_scale)
    ky = np.fft.fftfreq(ny, d=pixel_scale)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Convert to multipole: ell = 2 * pi * k, with k in 1/arcmin
    # and convert arcmin to radians: 1 arcmin = pi / (180 * 60) rad
    k_to_ell = 2 * np.pi * 180 * 60 / np.pi  # ~10313
    ell_grid = k_grid * k_to_ell
    
    # Bin in ell
    ell_max = ell_grid.max()
    ell_edges = np.linspace(0, ell_max, n_bins + 1)
    ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
    
    cl = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (ell_grid >= ell_edges[i]) & (ell_grid < ell_edges[i+1])
        if mask.any():
            cl[i] = np.mean(power_2d[mask])
            counts[i] = mask.sum()
    
    # Normalize: multiply by pixel area in steradians
    # pixel_scale is in arcmin, convert to radians
    pixel_sr = (pixel_scale * np.pi / (180 * 60))**2
    cl = cl * pixel_sr
    
    return ell_centers[counts > 0], cl[counts > 0]


def kaiser_squires(
    kappa: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Kaiser-Squires inversion to get shear from convergence.

    Parameters
    ----------
    kappa : ndarray
        2D convergence field.

    Returns
    -------
    gamma1 : ndarray
        First shear component.
    gamma2 : ndarray
        Second shear component.

    Notes
    -----
    This implements the inverse Kaiser-Squires relation:
    gamma_hat = D* kappa_hat
    where D = (l1^2 - l2^2 + 2i l1 l2) / (l1^2 + l2^2)
    """
    ny, nx = kappa.shape
    
    # FFT of convergence
    kappa_fft = np.fft.fft2(kappa)
    
    # Compute wavenumbers
    lx = np.fft.fftfreq(nx)
    ly = np.fft.fftfreq(ny)
    lx_grid, ly_grid = np.meshgrid(lx, ly)
    
    # Kaiser-Squires kernel
    l_sq = lx_grid**2 + ly_grid**2
    l_sq[0, 0] = 1  # Avoid division by zero
    
    D1 = (lx_grid**2 - ly_grid**2) / l_sq
    D2 = 2 * lx_grid * ly_grid / l_sq
    D1[0, 0] = 0
    D2[0, 0] = 0
    
    # Apply kernel
    gamma1_fft = D1 * kappa_fft
    gamma2_fft = D2 * kappa_fft
    
    # Inverse FFT
    gamma1 = np.real(np.fft.ifft2(gamma1_fft))
    gamma2 = np.real(np.fft.ifft2(gamma2_fft))
    
    return gamma1, gamma2


def convergence_from_shear(
    gamma1: np.ndarray,
    gamma2: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct convergence from shear using Kaiser-Squires.

    Parameters
    ----------
    gamma1 : ndarray
        First shear component.
    gamma2 : ndarray
        Second shear component.

    Returns
    -------
    kappa : ndarray
        Reconstructed convergence (B-mode filtered).
    """
    ny, nx = gamma1.shape
    
    # FFT of shear components
    gamma1_fft = np.fft.fft2(gamma1)
    gamma2_fft = np.fft.fft2(gamma2)
    
    # Compute wavenumbers
    lx = np.fft.fftfreq(nx)
    ly = np.fft.fftfreq(ny)
    lx_grid, ly_grid = np.meshgrid(lx, ly)
    
    # Inverse Kaiser-Squires kernel (for E-mode)
    l_sq = lx_grid**2 + ly_grid**2
    l_sq[0, 0] = 1  # Avoid division by zero
    
    D1 = (lx_grid**2 - ly_grid**2) / l_sq
    D2 = 2 * lx_grid * ly_grid / l_sq
    D1[0, 0] = 0
    D2[0, 0] = 0
    
    # Reconstruct convergence (E-mode only)
    kappa_fft = D1 * gamma1_fft + D2 * gamma2_fft
    kappa = np.real(np.fft.ifft2(kappa_fft))
    
    return kappa


def aperture_mass(
    kappa: np.ndarray,
    theta: float,
    pixel_scale: float,
) -> np.ndarray:
    """
    Compute aperture mass Map.

    Parameters
    ----------
    kappa : ndarray
        Convergence field.
    theta : float
        Aperture radius in arcminutes.
    pixel_scale : float
        Pixel scale in arcminutes.

    Returns
    -------
    m_ap : ndarray
        Aperture mass map.
    """
    # Aperture radius in pixels
    theta_pix = theta / pixel_scale
    
    # Create filter (polynomial compensation filter)
    ny, nx = kappa.shape
    y, x = np.mgrid[:ny, :nx]
    y = y - ny // 2
    x = x - nx // 2
    r = np.sqrt(x**2 + y**2)
    
    # Compensated filter (integral = 0)
    u = r / theta_pix
    filter_func = np.where(
        u < 1,
        9 / np.pi * (1 - u**2) * (1/3 - u**2),
        0
    )
    
    # Normalize
    filter_func = filter_func / (theta_pix**2)
    
    # Convolve
    m_ap = ndimage.convolve(kappa, filter_func, mode='wrap')
    
    return m_ap


def compute_pdf(
    kappa: np.ndarray,
    n_bins: int = 50,
    kappa_range: Optional[Tuple[float, float]] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute probability distribution function of convergence.

    Parameters
    ----------
    kappa : ndarray
        Convergence field.
    n_bins : int
        Number of bins.
    kappa_range : tuple, optional
        Range of kappa values.
    normalize : bool
        If True, normalize by standard deviation.

    Returns
    -------
    bin_centers : ndarray
        Bin centers.
    pdf : ndarray
        PDF values.
    """
    if normalize:
        kappa_use = (kappa - np.nanmean(kappa)) / np.nanstd(kappa)
    else:
        kappa_use = kappa
    
    if kappa_range is None:
        kappa_range = (np.nanmin(kappa_use), np.nanmax(kappa_use))
    
    counts, bin_edges = np.histogram(
        kappa_use.ravel(),
        bins=n_bins,
        range=kappa_range,
        density=True,
    )
    
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    return bin_centers, counts
