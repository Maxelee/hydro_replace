"""
Utility functions for computing statistics from density and convergence fields.
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter


def load_kappa(path, ng=1024):
    """
    Load convergence map from lux binary format.
    
    Parameters
    ----------
    path : str
        Path to kappa*.dat file
    ng : int, optional
        Grid resolution (default: 1024)
    
    Returns
    -------
    np.ndarray
        2D convergence map of shape (ng, ng)
    """
    with open(path, 'rb') as f:
        dummy = np.fromfile(f, dtype="int32", count=1)
        kappa = np.fromfile(f, dtype="float", count=ng*ng)
        dummy = np.fromfile(f, dtype="int32", count=1)
    return kappa.reshape(ng, ng)


def read_lensplane(path):
    """
    Load lensplane from binary dat format.
    
    Lensplanes are stored as binary files (lenspotXX.dat) with format:
    - Fortran record markers (int32)
    - Float32 array of size ng*ng
    
    Parameters
    ----------
    path : str
        Path to lenspot*.dat file
    
    Returns
    -------
    np.ndarray
        2D mass plane
    """
    # Try to infer grid size from file size
    file_size = os.path.getsize(path)
    
    # With Fortran record markers: 4 bytes (marker) + ng*ng*4 bytes (data) + 4 bytes (marker)
    # Common sizes: 4096^2 * 4 = 67108864 bytes for data
    # Total with markers: 67108872 bytes
    
    if file_size == 67108872:  # 4096^2 grid
        ng = 4096
    elif file_size == 16777224:  # 2048^2 grid
        ng = 2048
    elif file_size == 4194312:  # 1024^2 grid
        ng = 1024
    else:
        # Try default
        ng = 4096
    
    try:
        with open(path, 'rb') as f:
            # Read Fortran record marker
            dummy = np.fromfile(f, dtype='int32', count=1)
            # Read data
            data = np.fromfile(f, dtype='float32', count=ng*ng)
            # Read closing marker
            dummy = np.fromfile(f, dtype='int32', count=1)
        
        return data.reshape(ng, ng)
    except Exception as e:
        # If that fails, try without markers (raw float32)
        data = np.fromfile(path, dtype='float32')
        ng = int(np.sqrt(len(data)))
        if ng * ng != len(data):
            raise ValueError(f"Cannot determine grid size from file: {path}")
        return data.reshape(ng, ng)


def compute_Pk_2d(field, box_size=205.0):
    """
    Compute 2D power spectrum using Pylians.
    
    Parameters
    ----------
    field : np.ndarray
        2D density field
    box_size : float, optional
        Box size in Mpc/h (default: 205.0)
    
    Returns
    -------
    k : np.ndarray
        Wavenumber bins (h/Mpc)
    Pk : np.ndarray
        Power spectrum values (Mpc/h)²
    """
    import Pk_library as PKL
    
    # Convert to overdensity
    mean_val = np.mean(field)
    if mean_val > 0:
        delta = (field - mean_val) / mean_val
    else:
        delta = field.copy()
    delta = delta.astype(np.float32)
    
    # Compute P(k)
    Pk2D = PKL.Pk_plane(delta, box_size, 'None', 1)
    k = Pk2D.k
    Pk = Pk2D.Pk
    
    return k, Pk


def compute_Cl(kappa, fov_deg=5.0):
    """
    Compute angular power spectrum using Pylians.
    
    Parameters
    ----------
    kappa : np.ndarray
        2D convergence map
    fov_deg : float, optional
        Field of view in degrees (default: 5.0)
    
    Returns
    -------
    ell : np.ndarray
        Multipole bins
    Cl : np.ndarray
        Angular power spectrum
    """
    import Pk_library as PKL
    
    # Convert to float32
    kappa = kappa.astype(np.float32)
    
    # Compute angular power spectrum
    # Box size in degrees
    Pk2D = PKL.Pk_plane(kappa, fov_deg, 'None', 1)
    
    # Convert k to ell (ell ≈ k for small angles)
    # More precisely: ell = 2π k / (θ in radians)
    k = Pk2D.k  # cycles/degree
    ell = k * 360.0  # Convert to multipoles
    Cl = Pk2D.Pk
    
    return ell, Cl


def smooth_kappa(kappa, smoothing_arcmin, pixel_scale_arcmin):
    """
    Apply Gaussian smoothing to convergence map.
    
    Parameters
    ----------
    kappa : np.ndarray
        2D convergence map
    smoothing_arcmin : float
        Smoothing scale in arcminutes
    pixel_scale_arcmin : float
        Pixel scale in arcminutes per pixel
    
    Returns
    -------
    np.ndarray
        Smoothed convergence map
    """
    sigma_pix = smoothing_arcmin / pixel_scale_arcmin
    return gaussian_filter(kappa, sigma=sigma_pix)


def compute_peaks(kappa_smooth, rms, sn_bins):
    """
    Compute peak counts in S/N bins.
    
    Parameters
    ----------
    kappa_smooth : np.ndarray
        Smoothed convergence map
    rms : float
        RMS normalization (from DMO)
    sn_bins : np.ndarray
        S/N bin edges
    
    Returns
    -------
    np.ndarray
        Peak counts in each bin
    """
    # Detect peaks (local maxima)
    is_peak = (kappa_smooth == maximum_filter(kappa_smooth, size=3))
    peak_values = kappa_smooth[is_peak]
    
    # Convert to S/N
    peak_sn = peak_values / rms
    
    # Histogram
    counts, _ = np.histogram(peak_sn, bins=sn_bins)
    
    return counts


def compute_minima(kappa_smooth, rms, sn_bins):
    """
    Compute minima counts in S/N bins.
    
    Parameters
    ----------
    kappa_smooth : np.ndarray
        Smoothed convergence map
    rms : float
        RMS normalization (from DMO)
    sn_bins : np.ndarray
        S/N bin edges
    
    Returns
    -------
    np.ndarray
        Minima counts in each bin
    """
    # Detect minima (local minima)
    is_minimum = (kappa_smooth == minimum_filter(kappa_smooth, size=3))
    minimum_values = kappa_smooth[is_minimum]
    
    # Convert to S/N
    minimum_sn = minimum_values / rms
    
    # Histogram
    counts, _ = np.histogram(minimum_sn, bins=sn_bins)
    
    return counts


def compute_pdf(kappa_smooth, rms, sn_bins):
    """
    Compute PDF in S/N bins.
    
    Parameters
    ----------
    kappa_smooth : np.ndarray
        Smoothed convergence map
    rms : float
        RMS normalization (from DMO)
    sn_bins : np.ndarray
        S/N bin edges
    
    Returns
    -------
    np.ndarray
        PDF counts in each bin
    """
    # Convert all pixels to S/N
    sn_values = kappa_smooth.flatten() / rms
    
    # Histogram
    counts, _ = np.histogram(sn_values, bins=sn_bins)
    
    return counts
