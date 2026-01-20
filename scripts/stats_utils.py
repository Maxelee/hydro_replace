"""
Utility functions for computing statistics from density and convergence fields.
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from scipy.fft import fft2, ifft2, fftfreq


# =============================================================================
# Wavelet Scattering Transform Functions
# =============================================================================

def lanusse_scaling_function_2d(r, kc=np.pi):
    """
    Lanusse isotropic scaling function in 2D physical space (adapted from Lanusse+2012).
    This function is zero in 2D Fourier space beyond kc.
    
    Parameters
    ----------
    r : np.ndarray
        Grid of radial coordinates
    kc : float
        Cutoff frequency (default: π)
    
    Returns
    -------
    np.ndarray
        Scaling function values on the grid
    """
    res = np.zeros_like(r)
    c = 1 / (4 * np.pi)  # 2D normalization constant
    mask = r != 0.0
    r_masked = r[mask]
    
    # 2D version: Bessel function-based scaling
    # For 2D isotropic wavelets: phi(r) ~ J_1(kc*r) / r
    from scipy.special import j1
    res[mask] = c * kc * j1(kc * r_masked) / r_masked
    res[~mask] = c * kc**2 / 2  # Limit as r -> 0
    
    return res


def lanusse_mother_wavelet_2d(r, kc=np.pi):
    """
    Lanusse isotropic wavelet in 2D (adapted from Lanusse+2012).
    Defined as difference of scaling functions at 2*kc and kc.
    
    Parameters
    ----------
    r : np.ndarray
        Grid of radial coordinates
    kc : float
        Cutoff frequency (default: π)
    
    Returns
    -------
    np.ndarray
        Mother wavelet values on the grid
    """
    return lanusse_scaling_function_2d(r, kc=2*kc) - lanusse_scaling_function_2d(r, kc=kc)


def build_wavelet_bank_2d(grid_size, J, Q=1, kc=np.pi):
    """
    Build a bank of 2D isotropic wavelets in Fourier space.
    
    The wavelets are bump-steerable wavelets following Lanusse+2012 design,
    adapted for 2D fields. Each wavelet probes a different scale j with
    cutoff frequency kc * 2^(-j/Q).
    
    Parameters
    ----------
    grid_size : int or tuple
        Grid size (assumed square if int)
    J : int
        Number of octaves (j = 0, 1, ..., J*Q - 1)
    Q : int
        Number of scales per octave (default: 1)
    kc : float
        Cutoff frequency of mother wavelet (default: π)
    
    Returns
    -------
    np.ndarray
        Wavelet bank in Fourier space, shape (J*Q, grid_size, grid_size)
    """
    if isinstance(grid_size, int):
        ny, nx = grid_size, grid_size
    else:
        ny, nx = grid_size
    
    # Build frequency grid
    kx = fftfreq(nx) * 2 * np.pi  # Angular frequencies
    ky = fftfreq(ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # Number of wavelets
    n_wavelets = J * Q
    
    # Scale factors: wavelets at different scales
    # kc_j = kc * 2^(-j/Q) for j = 0, 1, ..., J*Q-1
    scalings = 2.0 ** (-np.arange(n_wavelets) / Q)
    kcs = kc * scalings
    
    # Build wavelets in Fourier space
    fwavelets = np.zeros((n_wavelets, ny, nx), dtype=np.float32)
    
    for j, kc_j in enumerate(kcs):
        # Bump wavelet in Fourier space
        # The Fourier transform of the Lanusse wavelet is a smooth bump
        # supported on [kc_j/2, 2*kc_j]
        
        # Use a smooth bump function: cos^2 taper
        k_low = kc_j / 2
        k_high = 2 * kc_j
        
        # Smooth bump with cos taper
        w = np.zeros_like(K)
        
        # Rising edge: k_low to kc_j
        mask_rise = (K >= k_low) & (K < kc_j)
        if np.any(mask_rise):
            t = (K[mask_rise] - k_low) / (kc_j - k_low)
            w[mask_rise] = np.sin(np.pi/2 * t)**2
        
        # Falling edge: kc_j to k_high
        mask_fall = (K >= kc_j) & (K <= k_high)
        if np.any(mask_fall):
            t = (K[mask_fall] - kc_j) / (k_high - kc_j)
            w[mask_fall] = np.cos(np.pi/2 * t)**2
        
        fwavelets[j] = w.astype(np.float32)
    
    return fwavelets


def compute_wavelet_scattering_2d(field, J=5, Q=1, kc=np.pi, moments=(0.5, 1.0, 2.0)):
    """
    Compute 2D Wavelet Scattering Transform coefficients.
    
    The WST computes multi-scale statistics of a field:
    - S0(p): <|field|^p>  (zeroth order - moments of absolute field)
    - S1(j, p): <|field * ψ_j|^p>  (first order - moments of wavelet-filtered field)
    
    where ψ_j are wavelets at different scales j, and p are moment orders.
    
    This implementation follows Allys+2020 and Regaldo-Saint Blancard+2022
    for cosmological applications.
    
    Parameters
    ----------
    field : np.ndarray
        2D input field (e.g., convergence map)
    J : int
        Number of octaves for wavelet scales (default: 5)
    Q : int
        Number of scales per octave (default: 1)
    kc : float
        Cutoff frequency of mother wavelet (default: π)
    moments : tuple of float
        Moment orders p to compute (default: (0.5, 1.0, 2.0))
    
    Returns
    -------
    S0 : np.ndarray
        Zeroth order coefficients, shape (n_moments,)
    S1 : np.ndarray
        First order coefficients, shape (J*Q, n_moments)
    scales : np.ndarray
        Scale factors 2^(j/Q) for each wavelet (useful for plotting)
    """
    ny, nx = field.shape
    n_moments = len(moments)
    n_wavelets = J * Q
    
    # Build wavelet bank
    fwavelets = build_wavelet_bank_2d((ny, nx), J, Q, kc)
    
    # FFT of input field
    f_field = fft2(field)
    
    # S0: moments of absolute field
    S0 = np.zeros(n_moments, dtype=np.float64)
    abs_field = np.abs(field)
    for i, p in enumerate(moments):
        S0[i] = np.mean(abs_field**p)
    
    # S1: moments of wavelet-filtered field
    S1 = np.zeros((n_wavelets, n_moments), dtype=np.float64)
    
    for j in range(n_wavelets):
        # Wavelet transform via FFT convolution
        f_filtered = f_field * fwavelets[j]
        filtered = ifft2(f_filtered)
        
        # Modulus of wavelet transform
        abs_filtered = np.abs(filtered)
        
        # Compute moments
        for i, p in enumerate(moments):
            S1[j, i] = np.mean(abs_filtered**p)
    
    # Scale factors for reference
    scales = 2.0 ** (np.arange(n_wavelets) / Q)
    
    return S0, S1, scales


# =============================================================================
# Bispectrum Functions
# =============================================================================

def compute_bispectrum_2d(field, fov_deg=5.0, ell_bins=None, config='equilateral'):
    """
    Compute 2D bispectrum for weak lensing convergence maps.
    
    The bispectrum B(ℓ₁, ℓ₂, ℓ₃) measures non-Gaussian correlations
    in the field. We compute it for specific triangle configurations.
    
    Parameters
    ----------
    field : np.ndarray
        2D convergence map
    fov_deg : float
        Field of view in degrees (default: 5.0)
    ell_bins : np.ndarray, optional
        Multipole bin edges. If None, uses logarithmic bins from 100 to 10000.
    config : str
        Triangle configuration:
        - 'equilateral': ℓ₁ ≈ ℓ₂ ≈ ℓ₃
        - 'squeezed': ℓ₁ ≪ ℓ₂ ≈ ℓ₃ (ℓ₁ fixed at lowest bin)
        - 'folded': ℓ₁ ≈ 2ℓ₂ ≈ 2ℓ₃
        - 'isosceles': ℓ₁ = ℓ₂, varying ℓ₃
    
    Returns
    -------
    ell_centers : np.ndarray
        Multipole bin centers
    bispectrum : np.ndarray
        Bispectrum values for the configuration
    """
    ny, nx = field.shape
    
    # Default ell bins (logarithmic spacing)
    if ell_bins is None:
        ell_bins = np.logspace(np.log10(100), np.log10(10000), 15)
    
    n_bins = len(ell_bins) - 1
    ell_centers = np.sqrt(ell_bins[:-1] * ell_bins[1:])  # Geometric mean
    
    # FFT of the field
    field_fft = fft2(field)
    
    # Build ell grid
    # Convert pixel frequencies to multipoles
    # ell = 2π * k / θ where θ is in radians
    fov_rad = np.deg2rad(fov_deg)
    kx = fftfreq(nx, d=fov_rad/nx) * 2 * np.pi
    ky = fftfreq(ny, d=fov_rad/ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    ELL = np.sqrt(KX**2 + KY**2)
    
    # Bin assignment for each pixel
    bin_idx = np.digitize(ELL, ell_bins) - 1
    
    # Compute bispectrum based on configuration
    if config == 'equilateral':
        bispectrum = _bispectrum_equilateral(field_fft, bin_idx, n_bins, ELL, ell_bins)
    elif config == 'squeezed':
        bispectrum = _bispectrum_squeezed(field_fft, bin_idx, n_bins, ELL, ell_bins, KX, KY)
    elif config == 'folded':
        bispectrum = _bispectrum_folded(field_fft, bin_idx, n_bins, ELL, ell_bins)
    elif config == 'isosceles':
        bispectrum = _bispectrum_isosceles(field_fft, bin_idx, n_bins, ELL, ell_bins, KX, KY)
    else:
        raise ValueError(f"Unknown bispectrum configuration: {config}")
    
    return ell_centers, bispectrum


def _bispectrum_equilateral(field_fft, bin_idx, n_bins, ELL, ell_bins):
    """
    Compute equilateral bispectrum: ℓ₁ ≈ ℓ₂ ≈ ℓ₃.
    
    For equilateral triangles, we use the estimator:
    B_eq(ℓ) = ⟨|κ̃(ℓ)|² κ̃(ℓ)⟩ approximation via filtering
    """
    bispectrum = np.zeros(n_bins, dtype=np.float64)
    
    for i in range(n_bins):
        # Select modes in this ell bin
        mask = (bin_idx == i)
        if np.sum(mask) < 10:
            bispectrum[i] = np.nan
            continue
        
        # Filter field to this ell band
        filtered_fft = np.where(mask, field_fft, 0)
        filtered = ifft2(filtered_fft)
        
        # Equilateral bispectrum ≈ ⟨filtered³⟩ (real part)
        # This captures the skewness at scale ℓ
        bispectrum[i] = np.mean(np.real(filtered)**3)
    
    return bispectrum


def _bispectrum_squeezed(field_fft, bin_idx, n_bins, ELL, ell_bins, KX, KY):
    """
    Compute squeezed bispectrum: ℓ₁ ≪ ℓ₂ ≈ ℓ₃.
    
    Fix ℓ₁ at low-ℓ (first few bins combined) and vary ℓ₂ ≈ ℓ₃.
    B_sq(ℓ) = ⟨κ_low * |κ̃(ℓ)|²⟩
    
    Note: We use multiple low-ℓ bins (0-2) combined to ensure enough modes
    for robust estimation, since individual low-ℓ bins may have too few modes
    on small fields.
    """
    bispectrum = np.zeros(n_bins, dtype=np.float64)
    
    # Low-ℓ filtered field: combine first 3 bins to get enough modes
    # This gives us ℓ < ~270 as the "squeezed" large-scale mode
    mask_low = (bin_idx >= 0) & (bin_idx <= 2)
    n_low_modes = np.sum(mask_low)
    
    if n_low_modes < 10:
        # Fall back to even more bins if needed
        mask_low = (bin_idx >= 0) & (bin_idx <= 4)
        n_low_modes = np.sum(mask_low)
        if n_low_modes < 10:
            return np.full(n_bins, np.nan)
    
    low_fft = np.where(mask_low, field_fft, 0)
    low_field = np.real(ifft2(low_fft))
    
    # Start from bin 3 (or later) for high-ℓ to maintain squeezed condition
    # ℓ_high should be >> ℓ_low for squeezed limit
    for i in range(n_bins):
        mask = (bin_idx == i)
        if np.sum(mask) < 10:
            bispectrum[i] = np.nan
            continue
        
        # Only compute if this bin is significantly higher than our low-ℓ band
        # i.e., ℓ_high > 2 * ℓ_low_max for squeezed condition
        if i <= 3:  # Skip first few bins where squeezed approximation breaks down
            bispectrum[i] = np.nan
            continue
        
        # Filter to high-ℓ band
        high_fft = np.where(mask, field_fft, 0)
        high_field = ifft2(high_fft)
        
        # Squeezed: ⟨κ_low * |κ_high|²⟩
        bispectrum[i] = np.mean(low_field * np.abs(high_field)**2)
    
    return bispectrum


def _bispectrum_folded(field_fft, bin_idx, n_bins, ELL, ell_bins):
    """
    Compute folded bispectrum: ℓ₁ ≈ 2ℓ₂ ≈ 2ℓ₃.
    
    For folded triangles (ℓ₁ = ℓ₂ + ℓ₃ collinear).
    """
    bispectrum = np.zeros(n_bins, dtype=np.float64)
    
    for i in range(n_bins):
        ell_center = np.sqrt(ell_bins[i] * ell_bins[i+1])
        
        # Find bin corresponding to ℓ/2
        half_ell = ell_center / 2
        i_half = np.searchsorted(ell_bins, half_ell) - 1
        if i_half < 0 or i_half >= n_bins:
            bispectrum[i] = np.nan
            continue
        
        # Mask for ℓ and ℓ/2 bins
        mask_full = (bin_idx == i)
        mask_half = (bin_idx == i_half)
        
        if np.sum(mask_full) < 10 or np.sum(mask_half) < 10:
            bispectrum[i] = np.nan
            continue
        
        # Filter fields
        full_fft = np.where(mask_full, field_fft, 0)
        half_fft = np.where(mask_half, field_fft, 0)
        
        full_field = ifft2(full_fft)
        half_field = ifft2(half_fft)
        
        # Folded: ⟨κ(ℓ) * κ(ℓ/2)²⟩
        bispectrum[i] = np.mean(np.real(full_field) * np.real(half_field)**2)
    
    return bispectrum


def _bispectrum_isosceles(field_fft, bin_idx, n_bins, ELL, ell_bins, KX, KY):
    """
    Compute isosceles bispectrum: ℓ₁ = ℓ₂ ≠ ℓ₃.
    
    Returns a 2D array where [i, j] corresponds to ℓ₁=ℓ₂ at bin i, ℓ₃ at bin j.
    For simplicity, we return the diagonal (ℓ₁=ℓ₂=ℓ₃) and off-diagonal average.
    """
    # For isosceles, return 1D array: fix ℓ₁=ℓ₂ and average over valid ℓ₃
    bispectrum = np.zeros(n_bins, dtype=np.float64)
    
    for i in range(n_bins):
        mask_i = (bin_idx == i)
        if np.sum(mask_i) < 10:
            bispectrum[i] = np.nan
            continue
        
        # Filter for ℓ₁ = ℓ₂
        fft_i = np.where(mask_i, field_fft, 0)
        field_i = ifft2(fft_i)
        
        # For isosceles with ℓ₁=ℓ₂, average over ℓ₃ in neighboring bins
        # that satisfy triangle inequality
        bispec_sum = 0.0
        count = 0
        
        for j in range(max(0, i-2), min(n_bins, i+3)):
            # Triangle inequality: |ℓ₁ - ℓ₂| ≤ ℓ₃ ≤ ℓ₁ + ℓ₂
            # For isosceles ℓ₁=ℓ₂: 0 ≤ ℓ₃ ≤ 2ℓ₁
            mask_j = (bin_idx == j)
            if np.sum(mask_j) < 10:
                continue
            
            fft_j = np.where(mask_j, field_fft, 0)
            field_j = ifft2(fft_j)
            
            # B(ℓ, ℓ, ℓ') ≈ ⟨|κ(ℓ)|² κ(ℓ')⟩
            bispec_sum += np.mean(np.abs(field_i)**2 * np.real(field_j))
            count += 1
        
        if count > 0:
            bispectrum[i] = bispec_sum / count
        else:
            bispectrum[i] = np.nan
    
    return bispectrum


def compute_all_bispectra(field, fov_deg=5.0, ell_bins=None):
    """
    Compute bispectrum for all four triangle configurations.
    
    Parameters
    ----------
    field : np.ndarray
        2D convergence map
    fov_deg : float
        Field of view in degrees
    ell_bins : np.ndarray, optional
        Multipole bin edges
    
    Returns
    -------
    ell_centers : np.ndarray
        Multipole bin centers
    bispectra : dict
        Dictionary with keys 'equilateral', 'squeezed', 'folded', 'isosceles'
    """
    configs = ['equilateral', 'squeezed', 'folded', 'isosceles']
    bispectra = {}
    
    for config in configs:
        ell_centers, bispec = compute_bispectrum_2d(field, fov_deg, ell_bins, config)
        bispectra[config] = bispec
    
    return ell_centers, bispectra


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
    - Grid size (int32)
    - Float64 array of size ng*ng
    
    Parameters
    ----------
    path : str
        Path to lenspot*.dat file
    
    Returns
    -------
    np.ndarray
        2D mass plane
    """
    with open(path, 'rb') as f:
        # Read grid size from first int32
        grid_size = np.fromfile(f, dtype=np.int32, count=1)[0]
        # Read data as float64
        data = np.fromfile(f, dtype=np.float64, count=grid_size*grid_size)
    
    return data.reshape(grid_size, grid_size)


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
    Pk2D = PKL.Pk_plane(delta, box_size, 'None', 1, verbose=False)
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
    Pk2D = PKL.Pk_plane(kappa, fov_deg, 'None', 1, verbose=False)
    
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


def compute_minkowski_functionals(kappa_smooth, rms, thresholds):
    """
    Compute Minkowski functionals (V0, V1, V2) at multiple thresholds.
    
    In 2D, the three Minkowski functionals are:
    - V0: Area fraction (fraction of pixels above threshold)
    - V1: Perimeter (boundary length, normalized)
    - V2: Euler characteristic (# connected regions - # holes)
    
    Parameters
    ----------
    kappa_smooth : np.ndarray
        Smoothed convergence map
    rms : float
        RMS normalization (from DMO)
    thresholds : np.ndarray
        S/N threshold values at which to compute MFs
    
    Returns
    -------
    V0 : np.ndarray
        Area fraction at each threshold
    V1 : np.ndarray  
        Perimeter at each threshold
    V2 : np.ndarray
        Euler characteristic at each threshold
    """
    from scipy import ndimage
    
    # Convert to S/N field
    sn_field = kappa_smooth / rms
    n_pix = sn_field.size
    ny, nx = sn_field.shape
    
    n_thresh = len(thresholds)
    V0 = np.zeros(n_thresh)
    V1 = np.zeros(n_thresh)
    V2 = np.zeros(n_thresh)
    
    for i, nu in enumerate(thresholds):
        # Binary excursion set: pixels above threshold
        excursion = (sn_field >= nu).astype(np.int32)
        
        # V0: Area fraction
        V0[i] = np.sum(excursion) / n_pix
        
        # V1: Perimeter - count boundary pixels
        # Use gradient to find edges
        grad_x = np.abs(np.diff(excursion, axis=1))
        grad_y = np.abs(np.diff(excursion, axis=0))
        # Perimeter normalized by total perimeter possible
        perimeter = np.sum(grad_x) + np.sum(grad_y)
        V1[i] = perimeter / (2 * (nx + ny))
        
        # V2: Euler characteristic using labeled regions
        # Euler = #connected_regions - #holes
        # For 2D binary image: χ = C - H where C = # white regions, H = # black regions enclosed
        # Using the relation: χ = V - E + F for pixel grid
        
        # Label connected white regions (4-connectivity)
        labeled, n_white = ndimage.label(excursion, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        
        # Label connected black regions (holes) - invert and label
        labeled_inv, n_black = ndimage.label(1 - excursion, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        
        # Euler characteristic (normalized by area)
        # Subtract 1 from n_black to exclude the infinite background region
        V2[i] = (n_white - (n_black - 1)) / n_pix
    
    return V0, V1, V2
