#!/usr/bin/env python
"""
Particle Analysis Functions for Matched Halos.

Provides analysis routines built on top of particle_access.py:
- Baryon fractions (total, gas, stellar)
- Mass conservation between DMO and Hydro
- Radial density profiles
- Cumulative mass profiles

Usage:
    from particle_access import MatchedHaloSnapshot
    from particle_analysis import HaloAnalyzer
    
    mh = MatchedHaloSnapshot(snapshot=99, sim_res=2500)
    analyzer = HaloAnalyzer(mh)
    
    results = analyzer.analyze_halo(dmo_idx=100)
    print(results['baryon_fraction'])
    print(results['mass_conservation'])
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

# Import from particle_access (same directory)
from particle_access import MatchedHaloSnapshot, HaloInfo, ParticleData


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class BaryonFractionResult:
    """Baryon fraction measurements at various radii."""
    radius_r200: np.ndarray     # Radii in units of R200
    f_baryon: np.ndarray        # Total baryon fraction (gas + stars) / total
    f_gas: np.ndarray           # Gas fraction
    f_stellar: np.ndarray       # Stellar fraction
    m_total: np.ndarray         # Total mass enclosed (Msun/h)
    m_gas: np.ndarray           # Gas mass enclosed
    m_stellar: np.ndarray       # Stellar mass enclosed
    m_dm: np.ndarray            # DM mass enclosed
    
    # Universal baryon fraction for comparison
    f_baryon_cosmic: float = 0.0486 / 0.3089  # Omega_b / Omega_m for TNG


@dataclass
class MassConservationResult:
    """Mass conservation between DMO and Hydro."""
    radius_r200: np.ndarray     # Radii in units of R200
    m_dmo: np.ndarray           # DMO mass enclosed
    m_hydro_total: np.ndarray   # Hydro total mass enclosed
    m_hydro_dm: np.ndarray      # Hydro DM-only mass enclosed
    ratio_total: np.ndarray     # M_hydro_total / M_dmo
    ratio_dm: np.ndarray        # M_hydro_dm / M_dmo (should be ~0.84 for TNG)
    
    # Expected DM mass ratio from particle masses
    expected_dm_ratio: float = 0.00398342749867548 / 0.0047271638660809


@dataclass  
class RadialProfileResult:
    """Radial density and cumulative mass profiles."""
    r_bins: np.ndarray          # Bin edges in units of R200
    r_mid: np.ndarray           # Bin centers in units of R200
    
    # Density profiles (Msun/h / (Mpc/h)^3)
    rho_dmo: np.ndarray         # DMO density profile
    rho_hydro_total: np.ndarray # Hydro total density
    rho_hydro_dm: np.ndarray    # Hydro DM-only density
    rho_hydro_gas: np.ndarray   # Hydro gas density
    rho_hydro_stars: np.ndarray # Hydro stellar density
    
    # Cumulative mass profiles (Msun/h)
    m_enc_dmo: np.ndarray       # DMO cumulative mass
    m_enc_hydro_total: np.ndarray
    m_enc_hydro_dm: np.ndarray
    m_enc_hydro_gas: np.ndarray
    m_enc_hydro_stars: np.ndarray
    
    # Particle counts per bin
    n_dmo: np.ndarray
    n_hydro_total: np.ndarray


@dataclass
class HaloAnalysisResult:
    """Complete analysis results for a single halo."""
    dmo_idx: int
    halo_info: HaloInfo
    baryon_fraction: BaryonFractionResult
    mass_conservation: MassConservationResult
    radial_profile: RadialProfileResult


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_baryon_fraction(
    hydro_data: ParticleData,
    radii_r200: np.ndarray = None,
) -> BaryonFractionResult:
    """
    Compute baryon fractions as a function of radius.
    
    Parameters:
    -----------
    hydro_data : ParticleData
        Hydro particles with masses, radii, and types
    radii_r200 : array
        Radii at which to compute (in R200 units)
        Default: [0.5, 1.0, 2.0, 3.0, 5.0]
    
    Returns:
    --------
    BaryonFractionResult
    """
    if radii_r200 is None:
        radii_r200 = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    
    n_radii = len(radii_r200)
    
    f_baryon = np.zeros(n_radii)
    f_gas = np.zeros(n_radii)
    f_stellar = np.zeros(n_radii)
    m_total = np.zeros(n_radii)
    m_gas = np.zeros(n_radii)
    m_stellar = np.zeros(n_radii)
    m_dm = np.zeros(n_radii)
    
    for i, r_max in enumerate(radii_r200):
        # Select particles within radius
        subset = hydro_data.select_radius(r_max)
        
        if subset.n_particles == 0:
            continue
        
        # Separate by type
        gas_mask = subset.particle_types == 0
        dm_mask = subset.particle_types == 1
        star_mask = subset.particle_types == 4
        
        m_gas[i] = subset.masses[gas_mask].sum() if gas_mask.any() else 0
        m_dm[i] = subset.masses[dm_mask].sum() if dm_mask.any() else 0
        m_stellar[i] = subset.masses[star_mask].sum() if star_mask.any() else 0
        m_total[i] = m_gas[i] + m_dm[i] + m_stellar[i]
        
        if m_total[i] > 0:
            f_baryon[i] = (m_gas[i] + m_stellar[i]) / m_total[i]
            f_gas[i] = m_gas[i] / m_total[i]
            f_stellar[i] = m_stellar[i] / m_total[i]
    
    return BaryonFractionResult(
        radius_r200=radii_r200,
        f_baryon=f_baryon,
        f_gas=f_gas,
        f_stellar=f_stellar,
        m_total=m_total,
        m_gas=m_gas,
        m_stellar=m_stellar,
        m_dm=m_dm,
    )


def compute_mass_conservation(
    dmo_data: ParticleData,
    hydro_data: ParticleData,
    radii_r200: np.ndarray = None,
) -> MassConservationResult:
    """
    Compute mass conservation between DMO and Hydro.
    
    The ratio M_hydro_total / M_dmo should be ~1 if mass is conserved.
    The ratio M_hydro_dm / M_dmo should be ~0.84 (ratio of DM particle masses).
    
    Parameters:
    -----------
    dmo_data : ParticleData
        DMO particles
    hydro_data : ParticleData
        Hydro particles
    radii_r200 : array
        Radii at which to compute
    
    Returns:
    --------
    MassConservationResult
    """
    if radii_r200 is None:
        radii_r200 = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    
    n_radii = len(radii_r200)
    
    m_dmo = np.zeros(n_radii)
    m_hydro_total = np.zeros(n_radii)
    m_hydro_dm = np.zeros(n_radii)
    
    for i, r_max in enumerate(radii_r200):
        # DMO mass
        dmo_subset = dmo_data.select_radius(r_max)
        m_dmo[i] = dmo_subset.total_mass
        
        # Hydro total mass
        hydro_subset = hydro_data.select_radius(r_max)
        m_hydro_total[i] = hydro_subset.total_mass
        
        # Hydro DM-only mass
        if hydro_subset.particle_types is not None:
            dm_mask = hydro_subset.particle_types == 1
            m_hydro_dm[i] = hydro_subset.masses[dm_mask].sum() if dm_mask.any() else 0
    
    # Compute ratios (avoid division by zero)
    ratio_total = np.where(m_dmo > 0, m_hydro_total / m_dmo, 0)
    ratio_dm = np.where(m_dmo > 0, m_hydro_dm / m_dmo, 0)
    
    return MassConservationResult(
        radius_r200=radii_r200,
        m_dmo=m_dmo,
        m_hydro_total=m_hydro_total,
        m_hydro_dm=m_hydro_dm,
        ratio_total=ratio_total,
        ratio_dm=ratio_dm,
    )


def compute_radial_profile(
    dmo_data: ParticleData,
    hydro_data: ParticleData,
    halo: HaloInfo,
    n_bins: int = 25,
    r_min_r200: float = 0.01,
    r_max_r200: float = 5.0,
) -> RadialProfileResult:
    """
    Compute radial density and cumulative mass profiles.
    
    Parameters:
    -----------
    dmo_data : ParticleData
        DMO particles
    hydro_data : ParticleData
        Hydro particles
    halo : HaloInfo
        Halo information (for R200)
    n_bins : int
        Number of radial bins
    r_min_r200 : float
        Minimum radius in R200 units
    r_max_r200 : float
        Maximum radius in R200 units
    
    Returns:
    --------
    RadialProfileResult
    """
    # Log-spaced bins
    r_bins = np.logspace(np.log10(r_min_r200), np.log10(r_max_r200), n_bins + 1)
    r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])  # Geometric mean
    
    # Initialize arrays
    rho_dmo = np.zeros(n_bins)
    rho_hydro_total = np.zeros(n_bins)
    rho_hydro_dm = np.zeros(n_bins)
    rho_hydro_gas = np.zeros(n_bins)
    rho_hydro_stars = np.zeros(n_bins)
    
    n_dmo = np.zeros(n_bins, dtype=int)
    n_hydro_total = np.zeros(n_bins, dtype=int)
    
    r200_mpc = halo.radius  # R200 in Mpc/h
    
    for i in range(n_bins):
        r_inner = r_bins[i]
        r_outer = r_bins[i + 1]
        
        # Shell volume in physical units (Mpc/h)^3
        r_inner_mpc = r_inner * r200_mpc
        r_outer_mpc = r_outer * r200_mpc
        volume = 4/3 * np.pi * (r_outer_mpc**3 - r_inner_mpc**3)
        
        # DMO
        if dmo_data.radii_r200 is not None:
            mask = (dmo_data.radii_r200 >= r_inner) & (dmo_data.radii_r200 < r_outer)
            n_dmo[i] = mask.sum()
            if n_dmo[i] > 0:
                rho_dmo[i] = dmo_data.masses[mask].sum() / volume
        
        # Hydro
        if hydro_data.radii_r200 is not None:
            mask = (hydro_data.radii_r200 >= r_inner) & (hydro_data.radii_r200 < r_outer)
            n_hydro_total[i] = mask.sum()
            
            if n_hydro_total[i] > 0:
                rho_hydro_total[i] = hydro_data.masses[mask].sum() / volume
                
                # By particle type
                if hydro_data.particle_types is not None:
                    gas_mask = mask & (hydro_data.particle_types == 0)
                    dm_mask = mask & (hydro_data.particle_types == 1)
                    star_mask = mask & (hydro_data.particle_types == 4)
                    
                    if gas_mask.any():
                        rho_hydro_gas[i] = hydro_data.masses[gas_mask].sum() / volume
                    if dm_mask.any():
                        rho_hydro_dm[i] = hydro_data.masses[dm_mask].sum() / volume
                    if star_mask.any():
                        rho_hydro_stars[i] = hydro_data.masses[star_mask].sum() / volume
    
    # Cumulative mass profiles
    m_enc_dmo = np.cumsum(rho_dmo * np.diff(4/3 * np.pi * (r_bins * r200_mpc)**3))
    m_enc_hydro_total = np.cumsum(rho_hydro_total * np.diff(4/3 * np.pi * (r_bins * r200_mpc)**3))
    m_enc_hydro_dm = np.cumsum(rho_hydro_dm * np.diff(4/3 * np.pi * (r_bins * r200_mpc)**3))
    m_enc_hydro_gas = np.cumsum(rho_hydro_gas * np.diff(4/3 * np.pi * (r_bins * r200_mpc)**3))
    m_enc_hydro_stars = np.cumsum(rho_hydro_stars * np.diff(4/3 * np.pi * (r_bins * r200_mpc)**3))
    
    return RadialProfileResult(
        r_bins=r_bins,
        r_mid=r_mid,
        rho_dmo=rho_dmo,
        rho_hydro_total=rho_hydro_total,
        rho_hydro_dm=rho_hydro_dm,
        rho_hydro_gas=rho_hydro_gas,
        rho_hydro_stars=rho_hydro_stars,
        m_enc_dmo=m_enc_dmo,
        m_enc_hydro_total=m_enc_hydro_total,
        m_enc_hydro_dm=m_enc_hydro_dm,
        m_enc_hydro_gas=m_enc_hydro_gas,
        m_enc_hydro_stars=m_enc_hydro_stars,
        n_dmo=n_dmo,
        n_hydro_total=n_hydro_total,
    )


# ============================================================================
# HaloAnalyzer Class
# ============================================================================

class HaloAnalyzer:
    """
    High-level interface for analyzing matched halos.
    
    Example:
        mh = MatchedHaloSnapshot(snapshot=99, sim_res=2500)
        analyzer = HaloAnalyzer(mh)
        
        # Analyze single halo
        result = analyzer.analyze_halo(dmo_idx=100)
        
        # Analyze all halos in mass range
        results = analyzer.analyze_all(mass_range=[13.0, 14.0])
    """
    
    def __init__(self, matched_snapshot: MatchedHaloSnapshot,
                 radii_r200: np.ndarray = None,
                 n_profile_bins: int = 25):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        matched_snapshot : MatchedHaloSnapshot
            Loaded snapshot interface
        radii_r200 : array
            Radii for baryon fraction / mass conservation
        n_profile_bins : int
            Number of bins for radial profiles
        """
        self.mh = matched_snapshot
        self.radii_r200 = radii_r200 if radii_r200 is not None else np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        self.n_profile_bins = n_profile_bins
    
    def analyze_halo(self, dmo_idx: int, 
                     compute_profiles: bool = True) -> HaloAnalysisResult:
        """
        Perform full analysis on a single halo.
        
        Parameters:
        -----------
        dmo_idx : int
            DMO halo index
        compute_profiles : bool
            If True, compute radial profiles (slower)
        
        Returns:
        --------
        HaloAnalysisResult
        """
        # Get halo info
        halo = self.mh.get_halo_info(halo_indices=[dmo_idx])[dmo_idx]
        
        # Load particles
        dmo_data = self.mh.get_particles('dmo', dmo_idx, include_coords=True)
        hydro_data = self.mh.get_particles('hydro', dmo_idx, include_coords=True)
        
        # Compute baryon fractions
        baryon_result = compute_baryon_fraction(hydro_data, self.radii_r200)
        
        # Compute mass conservation
        conservation_result = compute_mass_conservation(dmo_data, hydro_data, self.radii_r200)
        
        # Compute profiles
        if compute_profiles:
            profile_result = compute_radial_profile(
                dmo_data, hydro_data, halo,
                n_bins=self.n_profile_bins,
            )
        else:
            profile_result = None
        
        return HaloAnalysisResult(
            dmo_idx=dmo_idx,
            halo_info=halo,
            baryon_fraction=baryon_result,
            mass_conservation=conservation_result,
            radial_profile=profile_result,
        )
    
    def analyze_all(self, mass_range: Tuple[float, float] = None,
                    max_halos: int = None,
                    compute_profiles: bool = True,
                    verbose: bool = True) -> Dict[int, HaloAnalysisResult]:
        """
        Analyze all halos in a mass range.
        
        Parameters:
        -----------
        mass_range : tuple
            log10(M) range
        max_halos : int
            Maximum number of halos to analyze
        compute_profiles : bool
            If True, compute radial profiles
        verbose : bool
            Print progress
        
        Returns:
        --------
        dict : dmo_idx -> HaloAnalysisResult
        """
        halos = self.mh.get_halo_info(mass_range=mass_range)
        
        if max_halos is not None and len(halos) > max_halos:
            # Sort by mass and take most massive
            sorted_indices = sorted(halos.keys(), 
                                   key=lambda x: halos[x].log_mass, 
                                   reverse=True)[:max_halos]
            halos = {idx: halos[idx] for idx in sorted_indices}
        
        results = {}
        n_halos = len(halos)
        
        for i, dmo_idx in enumerate(sorted(halos.keys())):
            if verbose:
                halo = halos[dmo_idx]
                print(f"  [{i+1}/{n_halos}] Halo {dmo_idx} "
                      f"(log M = {halo.log_mass:.2f})")
            
            results[dmo_idx] = self.analyze_halo(
                dmo_idx, compute_profiles=compute_profiles
            )
        
        return results
    
    def stack_results(self, results: Dict[int, HaloAnalysisResult],
                      mass_bins: np.ndarray = None) -> Dict:
        """
        Stack analysis results by halo mass.
        
        Parameters:
        -----------
        results : dict
            Results from analyze_all()
        mass_bins : array
            log10(M) bin edges
        
        Returns:
        --------
        dict with stacked profiles and statistics per mass bin
        """
        if mass_bins is None:
            mass_bins = np.array([12.0, 12.5, 13.0, 13.5, 14.0, 15.0])
        
        n_bins = len(mass_bins) - 1
        n_radii = len(self.radii_r200)
        
        # Initialize stacking arrays
        stacked = {
            'mass_bins': mass_bins,
            'mass_bin_centers': (mass_bins[:-1] + mass_bins[1:]) / 2,
            'n_halos': np.zeros(n_bins, dtype=int),
            
            # Baryon fractions (mean and std per mass bin)
            'f_baryon_mean': np.zeros((n_bins, n_radii)),
            'f_baryon_std': np.zeros((n_bins, n_radii)),
            'f_gas_mean': np.zeros((n_bins, n_radii)),
            'f_stellar_mean': np.zeros((n_bins, n_radii)),
            
            # Mass conservation
            'ratio_total_mean': np.zeros((n_bins, n_radii)),
            'ratio_total_std': np.zeros((n_bins, n_radii)),
            'ratio_dm_mean': np.zeros((n_bins, n_radii)),
        }
        
        # Collect results by mass bin
        for mass_bin_idx in range(n_bins):
            m_min, m_max = mass_bins[mass_bin_idx], mass_bins[mass_bin_idx + 1]
            
            f_baryon_list = []
            f_gas_list = []
            f_stellar_list = []
            ratio_total_list = []
            ratio_dm_list = []
            
            for dmo_idx, result in results.items():
                log_mass = result.halo_info.log_mass
                if m_min <= log_mass < m_max:
                    f_baryon_list.append(result.baryon_fraction.f_baryon)
                    f_gas_list.append(result.baryon_fraction.f_gas)
                    f_stellar_list.append(result.baryon_fraction.f_stellar)
                    ratio_total_list.append(result.mass_conservation.ratio_total)
                    ratio_dm_list.append(result.mass_conservation.ratio_dm)
            
            n_in_bin = len(f_baryon_list)
            stacked['n_halos'][mass_bin_idx] = n_in_bin
            
            if n_in_bin > 0:
                stacked['f_baryon_mean'][mass_bin_idx] = np.mean(f_baryon_list, axis=0)
                stacked['f_baryon_std'][mass_bin_idx] = np.std(f_baryon_list, axis=0)
                stacked['f_gas_mean'][mass_bin_idx] = np.mean(f_gas_list, axis=0)
                stacked['f_stellar_mean'][mass_bin_idx] = np.mean(f_stellar_list, axis=0)
                stacked['ratio_total_mean'][mass_bin_idx] = np.mean(ratio_total_list, axis=0)
                stacked['ratio_total_std'][mass_bin_idx] = np.std(ratio_total_list, axis=0)
                stacked['ratio_dm_mean'][mass_bin_idx] = np.mean(ratio_dm_list, axis=0)
        
        stacked['radii_r200'] = self.radii_r200
        return stacked


# ============================================================================
# Convenience Functions for Quick Analysis
# ============================================================================

def quick_baryon_fraction(snapshot: int, sim_res: int = 2500,
                          mass_range: Tuple[float, float] = None,
                          max_halos: int = 10) -> Dict:
    """
    Quick baryon fraction analysis for a few halos.
    
    Returns dict with summary statistics.
    """
    mh = MatchedHaloSnapshot(snapshot=snapshot, sim_res=sim_res)
    analyzer = HaloAnalyzer(mh)
    
    results = analyzer.analyze_all(
        mass_range=mass_range,
        max_halos=max_halos,
        compute_profiles=False,
    )
    
    stacked = analyzer.stack_results(results)
    mh.close()
    
    return stacked


def quick_mass_conservation(snapshot: int, sim_res: int = 2500,
                            mass_range: Tuple[float, float] = None,
                            max_halos: int = 10) -> Dict:
    """
    Quick mass conservation check for a few halos.
    """
    mh = MatchedHaloSnapshot(snapshot=snapshot, sim_res=sim_res)
    analyzer = HaloAnalyzer(mh)
    
    results = analyzer.analyze_all(
        mass_range=mass_range,
        max_halos=max_halos,
        compute_profiles=False,
    )
    
    # Extract mass conservation
    summary = {
        'radii_r200': analyzer.radii_r200,
        'halos': [],
    }
    
    for dmo_idx, result in results.items():
        summary['halos'].append({
            'dmo_idx': dmo_idx,
            'log_mass': result.halo_info.log_mass,
            'ratio_total': result.mass_conservation.ratio_total,
            'ratio_dm': result.mass_conservation.ratio_dm,
        })
    
    mh.close()
    return summary


# ============================================================================
# Main (Testing)
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Check for available snapshots
    from particle_access import list_available_snapshots
    
    snaps = list_available_snapshots(sim_res=2500)
    print(f"Available snapshots: {snaps}")
    
    if len(snaps) == 0:
        print("No cache files found. Run generate_particle_cache.py first.")
        sys.exit(1)
    
    snap = snaps[0]
    print(f"\n{'='*60}")
    print(f"Testing particle_analysis on snapshot {snap}")
    print(f"{'='*60}")
    
    # Initialize
    mh = MatchedHaloSnapshot(snapshot=snap, sim_res=2500)
    analyzer = HaloAnalyzer(mh)
    
    # Get a few massive halos
    halos = mh.get_halo_info(mass_range=[13.0, 15.0])
    print(f"\nFound {len(halos)} halos with M > 10^13 Msun/h")
    
    if len(halos) == 0:
        print("No massive halos found. Try a different mass range.")
        mh.close()
        sys.exit(1)
    
    # Analyze first 3 halos
    test_indices = sorted(halos.keys(), 
                          key=lambda x: halos[x].log_mass, 
                          reverse=True)[:3]
    
    print(f"\nAnalyzing {len(test_indices)} most massive halos...")
    
    for dmo_idx in test_indices:
        halo = halos[dmo_idx]
        print(f"\n--- Halo {dmo_idx} (log M = {halo.log_mass:.2f}) ---")
        
        result = analyzer.analyze_halo(dmo_idx, compute_profiles=True)
        
        # Print baryon fractions
        print("\nBaryon fractions:")
        print(f"  {'r/R200':<8} {'f_b':<8} {'f_gas':<8} {'f_star':<8}")
        for i, r in enumerate(result.baryon_fraction.radius_r200):
            fb = result.baryon_fraction.f_baryon[i]
            fg = result.baryon_fraction.f_gas[i]
            fs = result.baryon_fraction.f_stellar[i]
            print(f"  {r:<8.1f} {fb:<8.3f} {fg:<8.3f} {fs:<8.3f}")
        
        print(f"  (cosmic f_b = {result.baryon_fraction.f_baryon_cosmic:.3f})")
        
        # Print mass conservation
        print("\nMass conservation (M_hydro / M_dmo):")
        print(f"  {'r/R200':<8} {'total':<8} {'DM only':<8}")
        for i, r in enumerate(result.mass_conservation.radius_r200):
            rt = result.mass_conservation.ratio_total[i]
            rd = result.mass_conservation.ratio_dm[i]
            print(f"  {r:<8.1f} {rt:<8.3f} {rd:<8.3f}")
        
        print(f"  (expected DM ratio = {result.mass_conservation.expected_dm_ratio:.3f})")
        
        # Print profile summary
        if result.radial_profile is not None:
            print("\nDensity profile (first 5 bins):")
            print(f"  {'r/R200':<8} {'rho_DMO':<12} {'rho_Hydro':<12}")
            for i in range(min(5, len(result.radial_profile.r_mid))):
                r = result.radial_profile.r_mid[i]
                rho_d = result.radial_profile.rho_dmo[i]
                rho_h = result.radial_profile.rho_hydro_total[i]
                print(f"  {r:<8.3f} {rho_d:<12.2e} {rho_h:<12.2e}")
    
    # Stack results
    print("\n" + "="*60)
    print("Stacking all analyzed halos...")
    
    results = {idx: analyzer.analyze_halo(idx, compute_profiles=False) 
               for idx in test_indices}
    stacked = analyzer.stack_results(results)
    
    print(f"\nStacked baryon fractions at R200:")
    print(f"  Mass bin        N     f_b (mean±std)")
    for i in range(len(stacked['mass_bins']) - 1):
        m1, m2 = stacked['mass_bins'][i], stacked['mass_bins'][i+1]
        n = stacked['n_halos'][i]
        if n > 0:
            # Index for R200 (r=1.0)
            r200_idx = np.argmin(np.abs(stacked['radii_r200'] - 1.0))
            fb_mean = stacked['f_baryon_mean'][i, r200_idx]
            fb_std = stacked['f_baryon_std'][i, r200_idx]
            print(f"  [{m1:.1f}, {m2:.1f})   {n:<5} {fb_mean:.3f} ± {fb_std:.3f}")
    
    mh.close()
    print("\nTest complete!")
