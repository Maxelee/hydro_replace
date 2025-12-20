"""
Mass Bin Configuration Module
=============================

Utilities for defining and managing mass bins for halo replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class MassBinConfig:
    """
    Configuration for a mass bin.

    Attributes
    ----------
    min_mass : float
        Minimum mass in Msun/h.
    max_mass : float
        Maximum mass in Msun/h.
    label : str
        Human-readable label for the bin.
    is_cumulative : bool
        Whether this is a cumulative (>= threshold) bin.
    """
    
    min_mass: float
    max_mass: float
    label: str
    is_cumulative: bool = False
    
    @property
    def log_min(self) -> float:
        """Log10 of minimum mass."""
        return np.log10(self.min_mass)
    
    @property
    def log_max(self) -> float:
        """Log10 of maximum mass."""
        return np.log10(self.max_mass) if np.isfinite(self.max_mass) else np.inf
    
    def contains(self, mass: float) -> bool:
        """Check if a mass falls within this bin."""
        return self.min_mass <= mass < self.max_mass
    
    def __repr__(self) -> str:
        if self.is_cumulative:
            return f"MassBin(M >= {self.min_mass:.2e}, label='{self.label}')"
        return f"MassBin([{self.min_mass:.2e}, {self.max_mass:.2e}), label='{self.label}')"


def get_mass_bins(
    log_mass_edges: List[float] = [12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0],
) -> List[MassBinConfig]:
    """
    Create regular mass bins from log10 mass edges.

    Parameters
    ----------
    log_mass_edges : list of float
        Edges of mass bins in log10(M/Msun/h).

    Returns
    -------
    bins : list of MassBinConfig
        List of mass bin configurations.

    Examples
    --------
    >>> bins = get_mass_bins([12.0, 13.0, 14.0, 15.0])
    >>> for b in bins:
    ...     print(b.label)
    M12.0-13.0
    M13.0-14.0
    M14.0-15.0
    """
    bins = []
    
    for i in range(len(log_mass_edges) - 1):
        log_min = log_mass_edges[i]
        log_max = log_mass_edges[i + 1]
        
        bins.append(MassBinConfig(
            min_mass=10**log_min,
            max_mass=10**log_max,
            label=f"M{log_min:.1f}-{log_max:.1f}",
            is_cumulative=False,
        ))
    
    return bins


def get_cumulative_bins(
    log_mass_thresholds: List[float] = [12.0, 12.5, 13.0, 13.5, 14.0],
) -> List[MassBinConfig]:
    """
    Create cumulative mass bins (>= threshold).

    Parameters
    ----------
    log_mass_thresholds : list of float
        Mass thresholds in log10(M/Msun/h).

    Returns
    -------
    bins : list of MassBinConfig
        List of cumulative mass bin configurations.

    Examples
    --------
    >>> bins = get_cumulative_bins([12.0, 13.0, 14.0])
    >>> for b in bins:
    ...     print(b.label)
    M_gt12.0
    M_gt13.0
    M_gt14.0
    """
    bins = []
    
    for log_thresh in log_mass_thresholds:
        bins.append(MassBinConfig(
            min_mass=10**log_thresh,
            max_mass=np.inf,
            label=f"M_gt{log_thresh:.1f}",
            is_cumulative=True,
        ))
    
    return bins


def get_standard_bins() -> Tuple[List[MassBinConfig], List[MassBinConfig]]:
    """
    Get standard mass bin configurations for papers.

    Returns
    -------
    regular_bins : list of MassBinConfig
        Regular (differential) mass bins.
    cumulative_bins : list of MassBinConfig
        Cumulative mass bins.
    """
    regular = get_mass_bins([12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0])
    cumulative = get_cumulative_bins([12.0, 12.5, 13.0, 13.5, 14.0, 14.5])
    
    return regular, cumulative


def get_all_configurations(
    radius_mults: List[float] = [1.0, 3.0, 5.0],
) -> List[Dict]:
    """
    Generate all replacement configurations (mass bins × radius multipliers).

    Parameters
    ----------
    radius_mults : list of float
        Radius multipliers to use.

    Returns
    -------
    configs : list of dict
        List of configuration dictionaries with keys:
        'mass_bin', 'radius_mult', 'label'

    Examples
    --------
    >>> configs = get_all_configurations([1.0, 3.0])
    >>> print(f"Generated {len(configs)} configurations")
    Generated 24 configurations
    """
    regular_bins, cumulative_bins = get_standard_bins()
    all_bins = regular_bins + cumulative_bins
    
    configs = []
    
    for mass_bin in all_bins:
        for radius_mult in radius_mults:
            configs.append({
                'mass_bin': mass_bin,
                'radius_mult': radius_mult,
                'label': f"{mass_bin.label}_R{radius_mult:.0f}",
            })
    
    return configs


def bin_halos_by_mass(
    masses: np.ndarray,
    bins: List[MassBinConfig],
) -> Dict[str, np.ndarray]:
    """
    Bin halo indices by mass.

    Parameters
    ----------
    masses : ndarray
        Array of halo masses.
    bins : list of MassBinConfig
        Mass bin configurations.

    Returns
    -------
    binned : dict
        Dictionary mapping bin labels to arrays of halo indices.
    """
    binned = {}
    indices = np.arange(len(masses))
    
    for mass_bin in bins:
        mask = (masses >= mass_bin.min_mass) & (masses < mass_bin.max_mass)
        binned[mass_bin.label] = indices[mask]
    
    return binned


def summarize_bins(
    masses: np.ndarray,
    bins: List[MassBinConfig],
) -> None:
    """
    Print summary of halo counts in each bin.

    Parameters
    ----------
    masses : ndarray
        Array of halo masses.
    bins : list of MassBinConfig
        Mass bin configurations.
    """
    print(f"{'Bin Label':<20} {'N Halos':>10} {'Mass Range':>25}")
    print("-" * 60)
    
    for mass_bin in bins:
        mask = (masses >= mass_bin.min_mass) & (masses < mass_bin.max_mass)
        n_halos = mask.sum()
        
        if np.isinf(mass_bin.max_mass):
            mass_range = f">= {mass_bin.min_mass:.2e}"
        else:
            mass_range = f"[{mass_bin.min_mass:.2e}, {mass_bin.max_mass:.2e})"
        
        print(f"{mass_bin.label:<20} {n_halos:>10,} {mass_range:>25}")
