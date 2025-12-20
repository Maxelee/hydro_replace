"""
Peak Finding Module
===================

Tools for finding and analyzing peaks in convergence maps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class Peak:
    """
    Single peak in a convergence map.

    Attributes
    ----------
    x : int
        X pixel coordinate.
    y : int
        Y pixel coordinate.
    kappa : float
        Convergence value at peak.
    snr : float
        Signal-to-noise ratio.
    """
    x: int
    y: int
    kappa: float
    snr: float


@dataclass
class PeakCatalog:
    """
    Catalog of peaks from convergence maps.

    Attributes
    ----------
    x : ndarray
        X pixel coordinates.
    y : ndarray
        Y pixel coordinates.
    kappa : ndarray
        Convergence values.
    snr : ndarray
        Signal-to-noise ratios.
    smoothing_scale : float
        Smoothing scale used (arcmin).
    map_label : str
        Label of source map.
    """
    
    x: np.ndarray
    y: np.ndarray
    kappa: np.ndarray
    snr: np.ndarray
    smoothing_scale: float = 0.0
    map_label: str = ""
    
    @property
    def n_peaks(self) -> int:
        """Number of peaks."""
        return len(self.x)
    
    @property
    def positions(self) -> np.ndarray:
        """Peak positions as (N, 2) array."""
        return np.column_stack([self.x, self.y])
    
    def filter_by_snr(self, snr_min: float, snr_max: Optional[float] = None) -> 'PeakCatalog':
        """Filter peaks by SNR range."""
        mask = self.snr >= snr_min
        if snr_max is not None:
            mask &= self.snr <= snr_max
        
        return PeakCatalog(
            x=self.x[mask],
            y=self.y[mask],
            kappa=self.kappa[mask],
            snr=self.snr[mask],
            smoothing_scale=self.smoothing_scale,
            map_label=self.map_label,
        )
    
    def get_snr_histogram(
        self,
        bins: Union[int, np.ndarray] = 20,
        snr_range: Tuple[float, float] = (-4, 6),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram of peak SNR.

        Returns
        -------
        bin_centers : ndarray
        counts : ndarray
        """
        counts, bin_edges = np.histogram(self.snr, bins=bins, range=snr_range)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centers, counts
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'kappa': self.kappa,
            'snr': self.snr,
            'smoothing_scale': self.smoothing_scale,
            'map_label': self.map_label,
        }
    
    def save_hdf5(self, filepath: Union[str, Path], group_name: str = "peaks") -> None:
        """Save to HDF5 file."""
        import h5py
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'a') as f:
            if group_name in f:
                del f[group_name]
            
            grp = f.create_group(group_name)
            grp.attrs['smoothing_scale'] = self.smoothing_scale
            grp.attrs['map_label'] = self.map_label
            grp.attrs['n_peaks'] = self.n_peaks
            
            grp.create_dataset('x', data=self.x)
            grp.create_dataset('y', data=self.y)
            grp.create_dataset('kappa', data=self.kappa)
            grp.create_dataset('snr', data=self.snr)
    
    @classmethod
    def load_hdf5(cls, filepath: Union[str, Path], group_name: str = "peaks") -> 'PeakCatalog':
        """Load from HDF5 file."""
        import h5py
        
        with h5py.File(filepath, 'r') as f:
            grp = f[group_name]
            return cls(
                x=grp['x'][:],
                y=grp['y'][:],
                kappa=grp['kappa'][:],
                snr=grp['snr'][:],
                smoothing_scale=grp.attrs.get('smoothing_scale', 0.0),
                map_label=grp.attrs.get('map_label', ''),
            )


def find_peaks(
    kappa: np.ndarray,
    snr_threshold: float = 0.0,
    smoothing_scale: float = 0.0,
    exclude_boundary: int = 10,
    return_snr_map: bool = False,
) -> Union[PeakCatalog, Tuple[PeakCatalog, np.ndarray]]:
    """
    Find peaks in a convergence map.

    Parameters
    ----------
    kappa : ndarray
        2D convergence field.
    snr_threshold : float
        Minimum SNR for detected peaks.
    smoothing_scale : float
        Smoothing scale in pixels (applied before detection).
    exclude_boundary : int
        Exclude peaks within this many pixels of boundary.
    return_snr_map : bool
        If True, also return the SNR map.

    Returns
    -------
    catalog : PeakCatalog
        Detected peaks.
    snr_map : ndarray, optional
        SNR map if return_snr_map=True.

    Examples
    --------
    >>> catalog = find_peaks(kappa_map.kappa, snr_threshold=3.0)
    >>> print(f"Found {catalog.n_peaks} peaks above 3 sigma")
    """
    # Apply smoothing if requested
    if smoothing_scale > 0:
        kappa_smoothed = ndimage.gaussian_filter(kappa, sigma=smoothing_scale)
    else:
        kappa_smoothed = kappa
    
    # Compute SNR map
    mean = np.nanmean(kappa_smoothed)
    std = np.nanstd(kappa_smoothed)
    snr_map = (kappa_smoothed - mean) / std
    
    # Find local maxima using maximum filter
    max_filtered = ndimage.maximum_filter(snr_map, size=3)
    peaks_mask = (snr_map == max_filtered) & (snr_map >= snr_threshold)
    
    # Exclude boundary
    if exclude_boundary > 0:
        ny, nx = kappa.shape
        peaks_mask[:exclude_boundary, :] = False
        peaks_mask[-exclude_boundary:, :] = False
        peaks_mask[:, :exclude_boundary] = False
        peaks_mask[:, -exclude_boundary:] = False
    
    # Extract peak positions
    y_peaks, x_peaks = np.where(peaks_mask)
    
    catalog = PeakCatalog(
        x=x_peaks,
        y=y_peaks,
        kappa=kappa_smoothed[y_peaks, x_peaks],
        snr=snr_map[y_peaks, x_peaks],
        smoothing_scale=smoothing_scale,
    )
    
    logger.info(f"Found {catalog.n_peaks} peaks (SNR >= {snr_threshold})")
    
    if return_snr_map:
        return catalog, snr_map
    return catalog


def find_peaks_2d(
    kappa: np.ndarray,
    sigma: float = 0.0,
    threshold: float = 0.0,
    exclude_boundary: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find peaks in 2D map (compatibility function).

    Parameters
    ----------
    kappa : ndarray
        2D convergence field.
    sigma : float
        Smoothing scale in pixels.
    threshold : float
        SNR threshold.
    exclude_boundary : int
        Boundary exclusion in pixels.

    Returns
    -------
    x_peaks : ndarray
    y_peaks : ndarray
    snr_peaks : ndarray
    """
    catalog = find_peaks(
        kappa,
        snr_threshold=threshold,
        smoothing_scale=sigma,
        exclude_boundary=exclude_boundary,
    )
    return catalog.x, catalog.y, catalog.snr


def compute_peak_counts(
    catalog: PeakCatalog,
    snr_bins: Union[int, np.ndarray] = 20,
    snr_range: Tuple[float, float] = (-4, 6),
    area: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute peak count histogram.

    Parameters
    ----------
    catalog : PeakCatalog
        Peak catalog.
    snr_bins : int or ndarray
        Number of bins or bin edges.
    snr_range : tuple
        Range of SNR values.
    area : float, optional
        Map area in square degrees for density normalization.

    Returns
    -------
    bin_centers : ndarray
        SNR bin centers.
    counts : ndarray
        Peak counts in each bin.
    errors : ndarray
        Poisson errors.

    Examples
    --------
    >>> snr, counts, errors = compute_peak_counts(catalog, area=25.0)
    >>> plt.errorbar(snr, counts, yerr=errors)
    """
    counts, bin_edges = np.histogram(catalog.snr, bins=snr_bins, range=snr_range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Poisson errors
    errors = np.sqrt(counts)
    
    # Normalize by area if provided
    if area is not None:
        bin_width = bin_edges[1] - bin_edges[0]
        counts = counts / area / bin_width
        errors = errors / area / bin_width
    
    return bin_centers, counts.astype(float), errors


def compute_peak_counts_multiple(
    catalogs: Dict[str, PeakCatalog],
    snr_bins: Union[int, np.ndarray] = 20,
    snr_range: Tuple[float, float] = (-4, 6),
    areas: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute peak counts for multiple catalogs.

    Parameters
    ----------
    catalogs : dict
        Dictionary mapping labels to PeakCatalog objects.
    snr_bins : int or ndarray
        Number of bins or bin edges.
    snr_range : tuple
        Range of SNR values.
    areas : dict, optional
        Map areas for each catalog.

    Returns
    -------
    results : dict
        Dictionary with peak count results for each catalog.
    """
    results = {}
    
    for label, catalog in catalogs.items():
        area = areas.get(label) if areas else None
        snr, counts, errors = compute_peak_counts(
            catalog,
            snr_bins=snr_bins,
            snr_range=snr_range,
            area=area,
        )
        results[label] = {
            'snr': snr,
            'counts': counts,
            'errors': errors,
        }
    
    return results


def compute_peak_ratio(
    catalog1: PeakCatalog,
    catalog2: PeakCatalog,
    snr_bins: Union[int, np.ndarray] = 20,
    snr_range: Tuple[float, float] = (-4, 6),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ratio of peak counts between two catalogs.

    Parameters
    ----------
    catalog1 : PeakCatalog
        First catalog (numerator).
    catalog2 : PeakCatalog
        Second catalog (denominator).
    snr_bins : int or ndarray
        Number of bins or bin edges.
    snr_range : tuple
        Range of SNR values.

    Returns
    -------
    bin_centers : ndarray
        SNR bin centers.
    ratio : ndarray
        Count ratio N1/N2.
    errors : ndarray
        Propagated errors.
    """
    snr, counts1, err1 = compute_peak_counts(catalog1, snr_bins, snr_range)
    _, counts2, err2 = compute_peak_counts(catalog2, snr_bins, snr_range)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = counts1 / counts2
        # Error propagation
        errors = ratio * np.sqrt((err1/counts1)**2 + (err2/counts2)**2)
    
    return snr, ratio, errors


class PeakStatistics:
    """
    Class for computing peak statistics across multiple maps.

    Parameters
    ----------
    catalogs : list
        List of PeakCatalog objects.
    """
    
    def __init__(self, catalogs: List[PeakCatalog]):
        self.catalogs = catalogs
    
    @property
    def n_maps(self) -> int:
        """Number of maps."""
        return len(self.catalogs)
    
    def compute_mean_counts(
        self,
        snr_bins: Union[int, np.ndarray] = 20,
        snr_range: Tuple[float, float] = (-4, 6),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute mean peak counts across maps.

        Returns
        -------
        bin_centers : ndarray
        mean_counts : ndarray
        std_counts : ndarray
        """
        all_counts = []
        
        for catalog in self.catalogs:
            snr, counts, _ = compute_peak_counts(catalog, snr_bins, snr_range)
            all_counts.append(counts)
        
        all_counts = np.array(all_counts)
        
        mean_counts = np.mean(all_counts, axis=0)
        std_counts = np.std(all_counts, axis=0) / np.sqrt(self.n_maps)
        
        return snr, mean_counts, std_counts
    
    def get_total_peaks(self, snr_min: float = 0.0) -> np.ndarray:
        """Get total number of peaks above threshold for each map."""
        return np.array([
            (cat.snr >= snr_min).sum() for cat in self.catalogs
        ])
    
    def compute_covariance(
        self,
        snr_bins: Union[int, np.ndarray] = 20,
        snr_range: Tuple[float, float] = (-4, 6),
    ) -> np.ndarray:
        """
        Compute covariance matrix of peak counts.

        Returns
        -------
        cov : ndarray
            Covariance matrix.
        """
        all_counts = []
        
        for catalog in self.catalogs:
            _, counts, _ = compute_peak_counts(catalog, snr_bins, snr_range)
            all_counts.append(counts)
        
        return np.cov(np.array(all_counts).T)
