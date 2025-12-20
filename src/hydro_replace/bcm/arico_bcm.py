"""
Arico BCM Implementation
========================

Interface to BaryonForge implementing Arico+2020 baryon correction model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Try to import BaryonForge
try:
    from BaryonForge.Profiles import Profiles as BFProfiles
    from BaryonForge.DensityProfiles import (
        BCM_DMO,
        BCM_7param,
        DarkMatterOnly,
    )
    HAS_BARYONFORGE = True
except ImportError:
    HAS_BARYONFORGE = False

logger = logging.getLogger(__name__)


@dataclass
class BCMParameters:
    """
    Baryon Correction Model parameters following Arico+2020.

    Parameters
    ----------
    M_c : float
        Characteristic mass for gas ejection (Msun/h).
    eta : float
        Power-law slope for ejection efficiency.
    beta : float
        Outer slope of gas profile.
    M_1_z0_cen : float
        Characteristic mass for stellar fraction (Msun/h).
    epsilon_0 : float
        Normalization for stellar fraction.
    epsilon_1 : float
        Additional stellar parameter.
    gamma : float
        Inner slope parameter.

    Notes
    -----
    Default values are from Arico+2020 Table 1 (TNG calibration).
    """
    
    # Gas ejection parameters
    M_c: float = 1e14  # Characteristic mass for ejection
    eta: float = 0.2   # Power-law slope
    beta: float = 0.6  # Outer slope of gas profile
    
    # Stellar parameters  
    M_1_z0_cen: float = 1e12  # Characteristic mass
    epsilon_0: float = 0.023  # Normalization
    epsilon_1: float = 0.0    # Additional parameter
    gamma: float = 2.5        # Inner slope
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'M_c': self.M_c,
            'eta': self.eta,
            'beta': self.beta,
            'M_1_z0_cen': self.M_1_z0_cen,
            'epsilon_0': self.epsilon_0,
            'epsilon_1': self.epsilon_1,
            'gamma': self.gamma,
        }
    
    @classmethod
    def tng_calibration(cls) -> 'BCMParameters':
        """Return TNG-calibrated parameters from Arico+2020."""
        return cls(
            M_c=1.2e14,
            eta=0.2,
            beta=0.6,
            M_1_z0_cen=1.1e12,
            epsilon_0=0.023,
            epsilon_1=0.0,
            gamma=2.5,
        )
    
    @classmethod
    def eagle_calibration(cls) -> 'BCMParameters':
        """Return EAGLE-calibrated parameters."""
        return cls(
            M_c=8e13,
            eta=0.25,
            beta=0.55,
            M_1_z0_cen=9e11,
            epsilon_0=0.025,
            epsilon_1=0.0,
            gamma=2.6,
        )


class AricoBCM:
    """
    Arico+2020 Baryon Correction Model interface.

    Parameters
    ----------
    params : BCMParameters
        BCM parameters.
    cosmology : dict
        Cosmological parameters.
    redshift : float
        Redshift for calculations.

    Examples
    --------
    >>> bcm = AricoBCM(BCMParameters.tng_calibration())
    >>> rho_bcm = bcm.compute_density_profile(
    ...     r=np.logspace(-2, 1, 50),
    ...     M200c=1e14,
    ...     c200c=5.0,
    ... )
    """
    
    def __init__(
        self,
        params: Optional[BCMParameters] = None,
        cosmology: Optional[Dict[str, float]] = None,
        redshift: float = 0.0,
    ):
        if not HAS_BARYONFORGE:
            raise ImportError(
                "BaryonForge not installed. Install with: "
                "pip install BaryonForge"
            )
        
        self.params = params or BCMParameters.tng_calibration()
        self.cosmology = cosmology or {
            'omega_m': 0.3089,
            'omega_b': 0.0486,
            'h': 0.6774,
            'sigma8': 0.8159,
            'ns': 0.9667,
        }
        self.redshift = redshift
        
        # Initialize BaryonForge profile calculator
        self._setup_baryonforge()
    
    def _setup_baryonforge(self) -> None:
        """Initialize BaryonForge calculator."""
        # Create parameter dictionary for BaryonForge
        self.bf_params = {
            'M_c': self.params.M_c,
            'eta': self.params.eta,
            'beta': self.params.beta,
            'M_1_z0_cen': self.params.M_1_z0_cen,
            'epsilon': self.params.epsilon_0,
            'gamma': self.params.gamma,
        }
        
        logger.debug(f"Initialized BCM with params: {self.bf_params}")
    
    def compute_density_profile(
        self,
        r: np.ndarray,
        M200c: float,
        c200c: float,
        component: str = 'total',
    ) -> np.ndarray:
        """
        Compute density profile using BCM.

        Parameters
        ----------
        r : ndarray
            Radii in Mpc/h.
        M200c : float
            Halo mass (Msun/h).
        c200c : float
            Concentration.
        component : str
            Which component: 'total', 'dm', 'gas', 'stars'.

        Returns
        -------
        rho : ndarray
            Density profile in Msun/h / (Mpc/h)^3.
        """
        # Use BaryonForge
        profile = BCM_7param(
            M200c=M200c,
            c200c=c200c,
            z=self.redshift,
            params=self.bf_params,
        )
        
        if component == 'total':
            return profile.rho_total(r)
        elif component == 'dm':
            return profile.rho_dm(r)
        elif component == 'gas':
            return profile.rho_gas(r)
        elif component == 'stars':
            return profile.rho_stars(r)
        else:
            raise ValueError(f"Unknown component: {component}")
    
    def compute_dmo_profile(
        self,
        r: np.ndarray,
        M200c: float,
        c200c: float,
    ) -> np.ndarray:
        """
        Compute DMO NFW profile.

        Parameters
        ----------
        r : ndarray
            Radii in Mpc/h.
        M200c : float
            Halo mass (Msun/h).
        c200c : float
            Concentration.

        Returns
        -------
        rho : ndarray
            DMO density profile.
        """
        profile = DarkMatterOnly(
            M200c=M200c,
            c200c=c200c,
            z=self.redshift,
        )
        return profile.rho(r)
    
    def compute_suppression_profile(
        self,
        r: np.ndarray,
        M200c: float,
        c200c: float,
    ) -> np.ndarray:
        """
        Compute density suppression rho_BCM / rho_DMO.

        Parameters
        ----------
        r : ndarray
            Radii in Mpc/h.
        M200c : float
            Halo mass (Msun/h).
        c200c : float
            Concentration.

        Returns
        -------
        suppression : ndarray
            Density ratio at each radius.
        """
        rho_bcm = self.compute_density_profile(r, M200c, c200c, component='total')
        rho_dmo = self.compute_dmo_profile(r, M200c, c200c)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            return rho_bcm / rho_dmo
    
    def compute_enclosed_mass(
        self,
        r: np.ndarray,
        M200c: float,
        c200c: float,
        component: str = 'total',
    ) -> np.ndarray:
        """
        Compute enclosed mass profile M(<r).

        Parameters
        ----------
        r : ndarray
            Radii in Mpc/h.
        M200c : float
            Halo mass (Msun/h).
        c200c : float
            Concentration.
        component : str
            Component to compute.

        Returns
        -------
        mass : ndarray
            Enclosed mass at each radius.
        """
        from scipy import integrate
        
        def integrand(r_prime):
            return 4 * np.pi * r_prime**2 * self.compute_density_profile(
                np.array([r_prime]), M200c, c200c, component
            )[0]
        
        mass = np.zeros_like(r)
        for i, r_i in enumerate(r):
            mass[i], _ = integrate.quad(integrand, 0, r_i)
        
        return mass
    
    def compute_baryon_fraction(
        self,
        r: np.ndarray,
        M200c: float,
        c200c: float,
    ) -> np.ndarray:
        """
        Compute baryon fraction profile f_b(<r) = M_baryon / M_total.

        Parameters
        ----------
        r : ndarray
            Radii in Mpc/h.
        M200c : float
            Halo mass (Msun/h).
        c200c : float
            Concentration.

        Returns
        -------
        f_b : ndarray
            Baryon fraction at each radius.
        """
        m_total = self.compute_enclosed_mass(r, M200c, c200c, 'total')
        m_gas = self.compute_enclosed_mass(r, M200c, c200c, 'gas')
        m_stars = self.compute_enclosed_mass(r, M200c, c200c, 'stars')
        
        with np.errstate(divide='ignore', invalid='ignore'):
            return (m_gas + m_stars) / m_total


def compute_bcm_profiles(
    halos: List[Dict],
    radii: np.ndarray,
    params: Optional[BCMParameters] = None,
    redshift: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Compute BCM profiles for multiple halos.

    Parameters
    ----------
    halos : list of dict
        List of halo dictionaries with 'M200c' and 'c200c' keys.
    radii : ndarray
        Radii at which to compute profiles.
    params : BCMParameters, optional
        BCM parameters.
    redshift : float
        Redshift.

    Returns
    -------
    profiles : dict
        Dictionary with stacked profiles.

    Examples
    --------
    >>> halos = [{'M200c': 1e14, 'c200c': 5}, {'M200c': 5e13, 'c200c': 6}]
    >>> profiles = compute_bcm_profiles(halos, np.logspace(-2, 1, 50))
    >>> rho_mean = profiles['rho_total_mean']
    """
    if not HAS_BARYONFORGE:
        raise ImportError("BaryonForge required")
    
    bcm = AricoBCM(params, redshift=redshift)
    
    rho_total = []
    rho_dm = []
    rho_gas = []
    rho_stars = []
    suppression = []
    
    for halo in halos:
        M200c = halo['M200c']
        c200c = halo.get('c200c', 5.0)  # Default concentration
        
        try:
            rho_total.append(bcm.compute_density_profile(radii, M200c, c200c, 'total'))
            rho_dm.append(bcm.compute_density_profile(radii, M200c, c200c, 'dm'))
            rho_gas.append(bcm.compute_density_profile(radii, M200c, c200c, 'gas'))
            rho_stars.append(bcm.compute_density_profile(radii, M200c, c200c, 'stars'))
            suppression.append(bcm.compute_suppression_profile(radii, M200c, c200c))
        except Exception as e:
            logger.warning(f"Failed to compute BCM for halo M200c={M200c}: {e}")
            continue
    
    return {
        'radii': radii,
        'rho_total': np.array(rho_total),
        'rho_total_mean': np.nanmean(rho_total, axis=0),
        'rho_total_std': np.nanstd(rho_total, axis=0),
        'rho_dm': np.array(rho_dm),
        'rho_dm_mean': np.nanmean(rho_dm, axis=0),
        'rho_gas': np.array(rho_gas),
        'rho_gas_mean': np.nanmean(rho_gas, axis=0),
        'rho_stars': np.array(rho_stars),
        'rho_stars_mean': np.nanmean(rho_stars, axis=0),
        'suppression': np.array(suppression),
        'suppression_mean': np.nanmean(suppression, axis=0),
        'suppression_std': np.nanstd(suppression, axis=0),
    }


def compute_bcm_power(
    k: np.ndarray,
    pk_dmo: np.ndarray,
    params: Optional[BCMParameters] = None,
    redshift: float = 0.0,
) -> np.ndarray:
    """
    Compute BCM-corrected power spectrum.

    Parameters
    ----------
    k : ndarray
        Wavenumbers in h/Mpc.
    pk_dmo : ndarray
        DMO power spectrum.
    params : BCMParameters, optional
        BCM parameters.
    redshift : float
        Redshift.

    Returns
    -------
    pk_bcm : ndarray
        BCM-corrected power spectrum.

    Notes
    -----
    This is a simplified implementation. Full implementation would
    use halo model framework.
    """
    # This is a placeholder - full implementation would integrate
    # over halo mass function and compute 1-halo + 2-halo terms
    
    logger.warning(
        "compute_bcm_power is a simplified placeholder. "
        "Use BaryonForge directly for accurate power spectrum calculations."
    )
    
    # Simple fitting formula from Schneider & Teyssier 2015
    params = params or BCMParameters.tng_calibration()
    
    # Characteristic scale
    k_star = 10.0  # h/Mpc
    
    # Suppression function
    A = 1.0 - params.eta * 0.5  # Simplified
    B = params.beta
    
    suppression = 1.0 - A * (k / k_star)**B / (1 + (k / k_star)**B)
    
    return pk_dmo * suppression
