#!/usr/bin/env python
"""
08_peak_finding.py - Peak Detection and Statistics on Convergence Maps

This script finds weak lensing peaks in convergence maps and computes:
    - Peak counts n(ν) as function of S/N
    - Fractional differences between models
    - χ² statistics comparing models to true hydro

Usage:
    python 08_peak_finding.py
    python 08_peak_finding.py --smoothing 1.0 --snr_bins 0,1,2,3,4,5,6,7
    
Inputs:
    - convergence_maps_[DMO|Hydro|BCM|Replace]_r00-09.fits

Outputs:
    - peak_counts.csv: Peak counts per S/N bin per configuration
    - peak_statistics.h5: Full peak catalogs and statistics
    - figures/peak_counts_comparison.pdf

Dependencies:
    - astropy.io.fits
    - scipy.ndimage (for smoothing and peak detection)
    - hydro_replace package
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import List, Tuple, Dict
from glob import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hydro_replace.utils.logging_setup import setup_logging
from hydro_replace.utils.io_helpers import ensure_dir
from hydro_replace.raytrace.convergence_maps import ConvergenceMap
from hydro_replace.raytrace.peak_finding import PeakCatalog, find_peaks

logger = setup_logging("08_peak_finding")


def load_convergence_maps(map_dir: Path, config_name: str, 
                          n_realizations: int = 10) -> List[ConvergenceMap]:
    """
    Load all convergence maps for a given configuration.
    
    Parameters
    ----------
    map_dir : Path
        Directory containing FITS files
    config_name : str
        Configuration name ('DMO', 'Hydro', 'BCM', 'Replace')
    n_realizations : int
        Number of realizations to load
    
    Returns
    -------
    maps : list of ConvergenceMap
        List of loaded maps
    """
    maps = []
    
    for i in range(n_realizations):
        fits_file = map_dir / f"convergence_maps_{config_name}_r{i:02d}.fits"
        
        if fits_file.exists():
            kappa_map = ConvergenceMap.load_fits(fits_file)
            maps.append(kappa_map)
            logger.debug(f"  Loaded: {fits_file.name}")
        else:
            logger.warning(f"  Missing: {fits_file.name}")
    
    logger.info(f"  Loaded {len(maps)}/{n_realizations} maps for {config_name}")
    return maps


def compute_peak_counts(maps: List[ConvergenceMap], 
                        snr_bins: np.ndarray,
                        smoothing_arcmin: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average peak counts across realizations.
    
    Parameters
    ----------
    maps : list of ConvergenceMap
        Input convergence maps
    snr_bins : np.ndarray
        S/N bin edges
    smoothing_arcmin : float
        Gaussian smoothing scale
    
    Returns
    -------
    n_peaks_mean : np.ndarray
        Mean peak counts per bin (peaks per deg²)
    n_peaks_std : np.ndarray
        Standard deviation across realizations
    """
    n_bins = len(snr_bins) - 1
    all_counts = []
    
    for kappa_map in maps:
        # Apply smoothing and find peaks
        smoothed = kappa_map.smooth(sigma_arcmin=smoothing_arcmin)
        peaks = find_peaks(smoothed)
        
        # Compute S/N for each peak
        sigma_kappa = smoothed.compute_noise_std()
        snr = peaks.kappa_values / sigma_kappa
        
        # Histogram into S/N bins
        counts, _ = np.histogram(snr, bins=snr_bins)
        
        # Normalize to peaks per deg²
        area_deg2 = kappa_map.fov_deg ** 2
        counts_per_deg2 = counts / area_deg2
        
        all_counts.append(counts_per_deg2)
    
    all_counts = np.array(all_counts)
    
    # Compute mean and std across realizations
    n_peaks_mean = np.mean(all_counts, axis=0)
    n_peaks_std = np.std(all_counts, axis=0)
    
    return n_peaks_mean, n_peaks_std


def compute_chi_squared(n_model: np.ndarray, n_truth: np.ndarray,
                        sigma: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute χ² comparing model to truth.
    
    Parameters
    ----------
    n_model : np.ndarray
        Model peak counts
    n_truth : np.ndarray
        Truth (hydro) peak counts
    sigma : np.ndarray
        Uncertainty on truth counts
    
    Returns
    -------
    chi2_total : float
        Total χ² summed over bins
    chi2_per_bin : np.ndarray
        χ² contribution per bin
    """
    # Avoid division by zero
    valid = sigma > 0
    
    chi2_per_bin = np.zeros_like(n_model)
    chi2_per_bin[valid] = ((n_model[valid] - n_truth[valid]) / sigma[valid]) ** 2
    
    chi2_total = np.sum(chi2_per_bin)
    
    return chi2_total, chi2_per_bin


def create_summary_table(results: Dict, snr_bins: np.ndarray) -> pd.DataFrame:
    """
    Create summary table of peak counts for all configurations.
    
    Parameters
    ----------
    results : dict
        Dictionary with configuration names as keys,
        (mean, std) tuples as values
    snr_bins : np.ndarray
        S/N bin edges
    
    Returns
    -------
    df : pd.DataFrame
        Summary table
    """
    bin_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])
    bin_labels = [f"ν={snr_bins[i]:.0f}-{snr_bins[i+1]:.0f}" 
                  for i in range(len(snr_bins)-1)]
    
    data = {'snr_bin': bin_labels, 'snr_center': bin_centers}
    
    for config, (mean, std) in results.items():
        data[f'{config}_mean'] = mean
        data[f'{config}_std'] = std
    
    return pd.DataFrame(data)


def plot_peak_counts(results: Dict, snr_bins: np.ndarray, 
                     output_path: Path):
    """
    Create peak counts comparison plot.
    
    Parameters
    ----------
    results : dict
        Peak count results per configuration
    snr_bins : np.ndarray
        S/N bin edges
    output_path : Path
        Where to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 12
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    bin_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), 
                              gridspec_kw={'height_ratios': [2, 1]})
    
    # Color scheme
    colors = {
        'Hydro': 'black',
        'DMO': 'gray',
        'BCM': '#d62728',  # red
        'Replace': '#1f77b4',  # blue
    }
    
    # Top panel: Peak counts
    ax1 = axes[0]
    
    for config, (mean, std) in results.items():
        color = colors.get(config, 'purple')
        ax1.errorbar(bin_centers, mean, yerr=std, 
                    label=config, color=color, 
                    marker='o', capsize=3, markersize=6)
    
    ax1.set_xlabel('S/N (ν)')
    ax1.set_ylabel('n(ν) [peaks/deg²]')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_title('Weak Lensing Peak Counts')
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Fractional difference from Hydro
    ax2 = axes[1]
    
    if 'Hydro' in results:
        hydro_mean, hydro_std = results['Hydro']
        
        for config, (mean, std) in results.items():
            if config == 'Hydro':
                continue
            
            # Fractional difference
            frac_diff = (mean - hydro_mean) / hydro_mean
            frac_err = np.sqrt((std/hydro_mean)**2 + 
                              (mean * hydro_std / hydro_mean**2)**2)
            
            color = colors.get(config, 'purple')
            ax2.errorbar(bin_centers, frac_diff, yerr=frac_err,
                        label=config, color=color,
                        marker='o', capsize=3, markersize=6)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('S/N (ν)')
    ax2.set_ylabel('Δn/n (relative to Hydro)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 0.5)
    
    # Shade high-S/N region where BCM typically fails
    for ax in axes:
        ax.axvspan(4, snr_bins[-1], alpha=0.1, color='red',
                   label='BCM failure region' if ax == axes[0] else None)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved figure: {output_path}")


def main():
    """Main peak finding pipeline."""
    
    parser = argparse.ArgumentParser(description="Peak finding on convergence maps")
    parser.add_argument("--smoothing", type=float, default=1.0,
                        help="Gaussian smoothing scale in arcmin")
    parser.add_argument("--snr_bins", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated S/N bin edges")
    parser.add_argument("--n_realizations", type=int, default=10,
                        help="Number of realizations per configuration")
    parser.add_argument("--configurations", nargs='+',
                        default=['Hydro', 'DMO', 'BCM', 'Replace'],
                        help="Configurations to analyze")
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Peak Finding Pipeline")
    logger.info("="*60)
    
    # Parse S/N bins
    snr_bins = np.array([float(x) for x in args.snr_bins.split(',')])
    logger.info(f"S/N bins: {snr_bins}")
    logger.info(f"Smoothing: {args.smoothing} arcmin")
    
    # Load configuration for paths
    import yaml
    config_dir = project_root / "config"
    
    with open(config_dir / "raytrace_config.yaml") as f:
        raytrace_config = yaml.safe_load(f)
    
    with open(config_dir / "analysis_params.yaml") as f:
        analysis_config = yaml.safe_load(f)
    
    # Directories
    map_dir = Path(raytrace_config['output']['map_dir'])
    output_dir = Path(analysis_config['output']['base_dir'])
    fig_dir = output_dir / "figures"
    ensure_dir(fig_dir)
    
    # Results storage
    results = {}
    
    # Process each configuration
    for config_name in args.configurations:
        logger.info(f"\nProcessing: {config_name}")
        
        # Load maps
        maps = load_convergence_maps(map_dir, config_name, args.n_realizations)
        
        if len(maps) == 0:
            logger.warning(f"  No maps found for {config_name}, skipping")
            continue
        
        # Compute peak counts
        mean_counts, std_counts = compute_peak_counts(
            maps, snr_bins, smoothing_arcmin=args.smoothing
        )
        
        results[config_name] = (mean_counts, std_counts)
        
        logger.info(f"  Peak counts: {mean_counts.sum():.1f} total peaks/deg²")
    
    # Compute χ² if Hydro is available
    if 'Hydro' in results:
        logger.info("\nχ² Analysis (relative to Hydro):")
        hydro_mean, hydro_std = results['Hydro']
        
        for config_name, (mean, std) in results.items():
            if config_name == 'Hydro':
                continue
            
            # Combined uncertainty (model + truth)
            combined_sigma = np.sqrt(std**2 + hydro_std**2)
            chi2_total, chi2_per_bin = compute_chi_squared(
                mean, hydro_mean, combined_sigma
            )
            
            n_dof = len(snr_bins) - 1
            chi2_reduced = chi2_total / n_dof
            
            logger.info(f"  {config_name}: χ² = {chi2_total:.1f} "
                       f"(χ²/dof = {chi2_reduced:.2f})")
            
            # Find worst bins
            worst_bin = np.argmax(chi2_per_bin)
            logger.info(f"    Worst bin: ν = {snr_bins[worst_bin]:.0f}-"
                       f"{snr_bins[worst_bin+1]:.0f} (χ² = {chi2_per_bin[worst_bin]:.1f})")
    
    # Create summary table
    df = create_summary_table(results, snr_bins)
    
    csv_path = output_dir / "peak_counts.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"\nSaved: {csv_path}")
    
    # Save detailed HDF5
    h5_path = output_dir / "peak_statistics.h5"
    with h5py.File(h5_path, 'w') as f:
        f.attrs['smoothing_arcmin'] = args.smoothing
        f.create_dataset('snr_bins', data=snr_bins)
        
        for config, (mean, std) in results.items():
            grp = f.create_group(config)
            grp.create_dataset('mean', data=mean)
            grp.create_dataset('std', data=std)
    
    logger.info(f"Saved: {h5_path}")
    
    # Create figure
    fig_path = fig_dir / "peak_counts_comparison.pdf"
    plot_peak_counts(results, snr_bins, fig_path)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Peak Finding Complete")
    logger.info("="*60)
    
    if 'Hydro' in results and 'BCM' in results and 'Replace' in results:
        hydro_mean, _ = results['Hydro']
        bcm_mean, _ = results['BCM']
        replace_mean, _ = results['Replace']
        
        # High S/N bins (ν > 4)
        high_snr_mask = snr_bins[:-1] >= 4
        
        if np.any(high_snr_mask):
            hydro_high = hydro_mean[high_snr_mask].sum()
            bcm_high = bcm_mean[high_snr_mask].sum()
            replace_high = replace_mean[high_snr_mask].sum()
            
            bcm_deficit = (hydro_high - bcm_high) / hydro_high * 100
            replace_deficit = (hydro_high - replace_high) / hydro_high * 100
            
            logger.info(f"\nHigh-S/N (ν > 4) peak deficit:")
            logger.info(f"  BCM: {bcm_deficit:.1f}% fewer peaks than Hydro")
            logger.info(f"  Replace: {replace_deficit:.1f}% fewer peaks than Hydro")
            
            improvement = (abs(bcm_deficit) - abs(replace_deficit)) / abs(bcm_deficit) * 100
            logger.info(f"  Improvement from Hydro Replace: {improvement:.1f}%")


if __name__ == "__main__":
    main()
