"""
Response Visualization Module
=============================

Generalized framework for visualizing baryonic response fractions F_S(M_min, α, z)
for any scale-dependent statistic S (power spectrum, angular power spectrum, etc.).

Key Equations:
    F_S = (S_R - S_D) / (S_H - S_D)

where:
    S_D = statistic from DMO simulation
    S_H = statistic from Hydro simulation  
    S_R = statistic from Replace simulation

Usage:
    from response_visualization import (
        PK_CONFIG, CELL_CONFIG, 
        DensityStatsLoader, KappaStatsLoader,
        ResponseDataCube, ResponsePlotter
    )
    
    loader = DensityStatsLoader(stats_dir)
    cube = ResponseDataCube(PK_CONFIG, loader)
    cube.build()
    
    fig = ResponsePlotter.plot_redshift_evolution(cube)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import h5py
import pickle
from scipy.ndimage import uniform_filter1d


# =============================================================================
# Configuration
# =============================================================================

# Standard mass thresholds and radii for Replace models
MASS_THRESHOLDS = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']
MASS_LABELS = [r'$10^{12.0}$', r'$10^{12.5}$', r'$10^{13.0}$', r'$10^{13.5}$']
LOG_MASS = [12.0, 12.5, 13.0, 13.5]

RADII = ['0.5', '1.0', '3.0', '5.0']
ALPHA_VALUES = [0.5, 1.0, 3.0, 5.0]

# Snapshot mappings
SNAPSHOT_LIST = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
SNAPSHOT_TO_Z = {
    96: 0.02, 90: 0.10, 85: 0.18, 80: 0.27, 76: 0.35, 
    71: 0.46, 67: 0.55, 63: 0.65, 59: 0.76, 56: 0.85,
    52: 0.97, 49: 1.08, 46: 1.21, 43: 1.36, 41: 1.47,
    38: 1.63, 35: 1.82, 33: 1.97, 31: 2.14, 29: 2.32
}


# =============================================================================
# StatisticConfig: Describes a scale-dependent statistic
# =============================================================================

@dataclass
class StatisticConfig:
    """Configuration for a scale-dependent statistic.
    
    Attributes:
        name: Short identifier (e.g., "Pk", "C_ell", "peaks")
        label: LaTeX label for response fraction (e.g., r"$\\langle F_P \\rangle$")
        x_key: Key in stats dict for x-axis (e.g., "k", "ell", "kappa_bins")
        y_key: Key in stats dict for y values (e.g., "Pk", "C_ell", "peaks")
        x_label: LaTeX label for x-axis (e.g., r"$k$ [h/Mpc]")
        y_label: LaTeX label for the statistic itself (e.g., r"$P(k)$")
        x_min: Minimum x value for averaging
        x_max: Maximum x value for averaging (e.g., Nyquist frequency)
        threshold: Minimum |ΔS/S_D| to consider significant (default 0.01 = 1%)
        smooth_window: Window size for smoothing (default 9)
        log_x: Whether to use log scale for x-axis (default True)
    """
    name: str
    label: str
    x_key: str
    y_key: str
    x_label: str
    y_label: str
    x_min: float
    x_max: float
    threshold: float = 0.01
    smooth_window: int = 9
    log_x: bool = True


# Pre-built configurations for common statistics
PK_CONFIG = StatisticConfig(
    name="Pk",
    label=r"$\langle F_P \rangle_k$",
    x_key="k",
    y_key="Pk",
    x_label=r"$k$ [h/Mpc]",
    y_label=r"$P(k)$",
    x_min=1.0,
    x_max=np.pi * 4096 / 205,  # Nyquist for L205n4096
    threshold=0.01,
    smooth_window=9,
    log_x=True,
)

# Kappa map parameters: 1024^2 grid, 5 deg x 5 deg opening angle
# ell_min = 2*pi / theta = 360/5 = 72
# ell_Nyquist = pi * N / theta = pi * 1024 / (5 * pi/180) = 36864
KAPPA_ELL_MIN = 72
KAPPA_ELL_NYQUIST = 36864

CELL_CONFIG = StatisticConfig(
    name="C_ell",
    label=r"$\langle F_{C_\ell} \rangle$",
    x_key="ell",
    y_key="C_ell",
    x_label=r"$\ell$",
    y_label=r"$C_\ell$",
    x_min=100,
    x_max=KAPPA_ELL_NYQUIST,  # Nyquist for 1024^2 grid, 5 deg FOV
    threshold=0.01,
    smooth_window=5,
    log_x=True,
)

PEAKS_CONFIG = StatisticConfig(
    name="peaks",
    label=r"$\langle F_{\rm peaks} \rangle$",
    x_key="peak_kappa_bins",
    y_key="peaks",
    x_label=r"$\kappa$",
    y_label=r"Peak counts",
    x_min=-0.02,
    x_max=0.15,
    threshold=0.01,
    smooth_window=3,
    log_x=False,
)

PDF_CONFIG = StatisticConfig(
    name="pdf",
    label=r"$\langle F_{\rm PDF} \rangle$",
    x_key="pdf_kappa_bins",
    y_key="pdf",
    x_label=r"$\kappa$",
    y_label=r"PDF",
    x_min=-0.05,
    x_max=0.15,
    threshold=0.01,
    smooth_window=3,
    log_x=False,
)

V0_CONFIG = StatisticConfig(
    name="V0",
    label=r"$\langle F_{V_0} \rangle$",
    x_key="mf_thresholds",
    y_key="V0",
    x_label=r"$\kappa$ threshold",
    y_label=r"$V_0$",
    x_min=-0.05,
    x_max=0.12,
    threshold=0.01,
    smooth_window=3,
    log_x=False,
)


# =============================================================================
# StatsLoader: Abstract base class for loading statistics
# =============================================================================

class StatsLoader(ABC):
    """Abstract base class for loading statistics from cache files."""
    
    @abstractmethod
    def load(self, model_name: str, snapshot_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """Load statistics for a model at a given snapshot index (0-19).
        
        Returns None if data not available, and prints a warning.
        """
        pass
    
    @abstractmethod
    def get_redshifts(self) -> np.ndarray:
        """Return array of redshifts corresponding to snapshot indices 0-19."""
        pass


class DensityStatsLoader(StatsLoader):
    """Load density field statistics from {model}_density_stats.h5 files.
    
    File structure:
        - k: wavenumber bins
        - Pk: power spectra (n_samples, n_k)
        - snapshot_ids: which snapshot index each sample comes from (0-19)
    """
    
    def __init__(self, stats_dir: Path):
        self.stats_dir = Path(stats_dir)
        self._missing_warned = set()  # Track which files we've warned about
        
    def load(self, model_name: str, snapshot_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """Load statistics for a model at a given snapshot index."""
        fpath = self.stats_dir / f"{model_name}_density_stats.h5"
        
        if not fpath.exists():
            if model_name not in self._missing_warned:
                print(f"  ⚠ Missing: {fpath.name}")
                self._missing_warned.add(model_name)
            return None
        
        try:
            with h5py.File(fpath, 'r') as f:
                k = f['k'][:]
                
                # Filter by snapshot index
                if 'snapshot_ids' in f:
                    snap_ids = f['snapshot_ids'][:]
                    mask = snap_ids == snapshot_idx
                    n_samples = np.sum(mask)
                    
                    if n_samples == 0:
                        return None
                    
                    Pk_all = f['Pk'][mask]
                else:
                    Pk_all = f['Pk'][:]
                
                stats = {
                    'k': k,
                    'Pk': np.mean(Pk_all, axis=0),
                    'Pk_std': np.std(Pk_all, axis=0) if Pk_all.shape[0] > 1 else np.zeros_like(k),
                }
                
                # Load PDF if available
                if 'pdf_bins' in f:
                    stats['pdf_bins'] = f['pdf_bins'][:]
                    if 'snapshot_ids' in f:
                        stats['pdf_values'] = np.mean(f['pdf'][mask], axis=0)
                    else:
                        stats['pdf_values'] = np.mean(f['pdf'][:], axis=0)
                
            return stats
            
        except Exception as e:
            print(f"  ⚠ Error loading {fpath.name}: {e}")
            return None
    
    def get_redshifts(self) -> np.ndarray:
        """Return redshifts for snapshot indices 0-19."""
        return np.array([SNAPSHOT_TO_Z[SNAPSHOT_LIST[i]] for i in range(20)])


class KappaStatsLoader(StatsLoader):
    """Load kappa (convergence) statistics from {model}_z{ZZ}_stats.h5 files.
    
    File naming: z00 = snapshot_idx 0, z01 = snapshot_idx 1, etc.
    
    File structure:
        - ell: multipole bins
        - C_ell: angular power spectrum (n_realizations, n_ell)
        - peaks, minima, pdf, V0, V1, V2: other statistics
    """
    
    def __init__(self, stats_dir: Path):
        self.stats_dir = Path(stats_dir)
        self._missing_warned = set()
        
    def load(self, model_name: str, snapshot_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """Load statistics for a model at a given snapshot index."""
        z_label = f"z{snapshot_idx:02d}"
        fpath = self.stats_dir / f"{model_name}_{z_label}_stats.h5"
        
        if not fpath.exists():
            warn_key = (model_name, snapshot_idx)
            if warn_key not in self._missing_warned:
                print(f"  ⚠ Missing: {fpath.name}")
                self._missing_warned.add(warn_key)
            return None
        
        try:
            with h5py.File(fpath, 'r') as f:
                stats = {}
                
                # Load ell and C_ell (always present)
                if 'ell' in f:
                    stats['ell'] = f['ell'][:]
                    stats['C_ell'] = np.mean(f['C_ell'][:], axis=0)
                
                # Load optional statistics
                for key in ['peaks', 'minima', 'pdf', 'V0', 'V1', 'V2']:
                    if key in f:
                        stats[key] = np.mean(f[key][:], axis=0)
                
                # Load bin edges
                for key in ['peak_kappa_bins', 'minima_kappa_bins', 'pdf_kappa_bins', 'mf_thresholds']:
                    if key in f:
                        stats[key] = f[key][:]
                
            return stats
            
        except Exception as e:
            print(f"  ⚠ Error loading {fpath.name}: {e}")
            return None
    
    def get_redshifts(self) -> np.ndarray:
        """Return redshifts for snapshot indices 0-19."""
        return np.array([SNAPSHOT_TO_Z[SNAPSHOT_LIST[i]] for i in range(20)])


class KappaResponseLoader(StatsLoader):
    """Load pre-computed kappa response data from response_analysis_output_kappabins.
    
    File naming: response_{statistic}_z{ZZ}.npz and response_{statistic}_z{ZZ}_dicts.pkl
    where ZZ is 01-40 (source redshift bins).
    
    NPZ structure:
        - ell (or kappa bins): x-axis values
        - S_D, S_H, Delta_S: baseline statistics
        
    PKL structure:
        - cumulative_responses: dict with keys like 'M1.0e+12_a0.5'
            - F_S: pre-computed response fraction array
            - S_R: Replace statistic
    """
    
    # Map statistic name to file prefix
    STAT_FILE_PREFIX = {
        'C_ell': 'C_ell',
        'peaks': 'peaks',
        'minima': 'minima',
        'pdf': 'pdf',
        'V0': 'V0',
        'V1': 'V1',
        'V2': 'V2',
    }
    
    # Map cumulative_responses key format to our standard format
    MASS_KEY_MAP = {
        'M1.0e+12': '1.00e12',
        'M3.2e+12': '3.16e12', 
        'M1.0e+13': '1.00e13',
        'M3.2e+13': '3.16e13',
    }
    
    def __init__(self, stats_dir: Path, statistic: str = 'C_ell'):
        """
        Args:
            stats_dir: Path to response_analysis_output_kappabins directory
            statistic: Which statistic to load ('C_ell', 'peaks', 'pdf', 'V0', etc.)
        """
        self.stats_dir = Path(stats_dir)
        self.statistic = statistic
        self._missing_warned = set()
        self._data_cache = {}  # Cache loaded data by z_idx
        
    def _load_z_data(self, z_idx: int) -> Optional[Dict]:
        """Load and cache all data for a redshift index."""
        if z_idx in self._data_cache:
            return self._data_cache[z_idx]
        
        file_prefix = self.STAT_FILE_PREFIX.get(self.statistic, self.statistic)
        z_label = f"z{z_idx+1:02d}"  # File uses 1-indexed: z01, z02, ..., z40
        
        npz_path = self.stats_dir / f"response_{file_prefix}_{z_label}.npz"
        pkl_path = self.stats_dir / f"response_{file_prefix}_{z_label}_dicts.pkl"
        
        if not npz_path.exists() or not pkl_path.exists():
            if z_idx not in self._missing_warned:
                print(f"  ⚠ Missing: {npz_path.name} or {pkl_path.name}")
                self._missing_warned.add(z_idx)
            return None
        
        try:
            npz = np.load(npz_path, allow_pickle=True)
            with open(pkl_path, 'rb') as f:
                pkl = pickle.load(f)
            
            self._data_cache[z_idx] = {
                'npz': dict(npz),  # Convert to regular dict
                'pkl': pkl,
            }
            return self._data_cache[z_idx]
            
        except Exception as e:
            print(f"  ⚠ Error loading z{z_idx+1:02d}: {e}")
            return None
    
    def _get_cumulative_key(self, mass_str: str, radius_str: str) -> str:
        """Convert our format to the cumulative_responses key format."""
        # Our format: '1.00e12', '0.5' -> 'M1.0e+12_a0.5'
        mass_val = float(mass_str)
        if mass_val >= 1e13:
            mass_key = f"M{mass_val/1e13:.1f}e+13"
        else:
            mass_key = f"M{mass_val/1e12:.1f}e+12"
        return f"{mass_key}_a{radius_str}"
    
    def load(self, model_name: str, snapshot_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """Load statistics for a model at a given snapshot index.
        
        For baselines (dmo, hydro), returns x, S_D, S_H from NPZ.
        For Replace models, parses model_name to get S_R and recomputes F_S.
        
        Note: F_S is recomputed from S_R, S_D, S_H because the stored F_S in the
        pickle files was computed with an overly strict threshold (1e-15) that
        sets high-ell values to NaN incorrectly.
        """
        data = self._load_z_data(snapshot_idx)
        if data is None:
            return None
        
        npz = data['npz']
        pkl = data['pkl']
        
        # Determine x-axis key based on statistic
        if self.statistic == 'C_ell':
            x_key = 'ell'
            x = npz['ell']
        elif self.statistic in ['peaks', 'minima', 'pdf']:
            # These have kappa bin arrays instead of ell
            x_key = 'kappa_mid'
            x = npz['kappa_mid']
        else:
            x_key = 'ell'
            x = npz.get('ell')
        
        S_D = npz['S_D']
        S_H = npz['S_H']
        Delta_S = S_H - S_D
        
        if model_name == 'dmo':
            return {
                x_key: x,
                self.statistic: S_D,
                f'{self.statistic}_err': npz.get('S_D_err', np.zeros_like(S_D)),
            }
        elif model_name == 'hydro':
            return {
                x_key: x,
                self.statistic: S_H,
                f'{self.statistic}_err': npz.get('S_H_err', np.zeros_like(S_H)),
            }
        else:
            # Parse Replace model name: hydro_replace_Ml_{mass}_Mu_inf_R_{radius}
            if 'hydro_replace_Ml_' not in model_name:
                return None
            
            parts = model_name.replace('hydro_replace_Ml_', '').split('_Mu_inf_R_')
            if len(parts) != 2:
                return None
            
            mass_str, radius_str = parts
            cumul_key = self._get_cumulative_key(mass_str, radius_str)
            
            cumul_resp = pkl.get('cumulative_responses', {})
            if cumul_key not in cumul_resp:
                return None
            
            resp = cumul_resp[cumul_key]
            S_R = resp['S_R']
            
            # Recompute F_S with proper threshold based on relative difference
            # Use 1% relative threshold instead of absolute 1e-15
            threshold = 0.01
            significant = np.abs(Delta_S) > threshold * np.abs(S_D)
            F_S = np.full_like(S_R, np.nan)
            F_S[significant] = (S_R[significant] - S_D[significant]) / Delta_S[significant]
            
            return {
                x_key: x,
                self.statistic: S_R,
                f'{self.statistic}_err': resp.get('S_R_err', np.zeros_like(S_R)),
                'F_S': F_S,  # Recomputed response fraction
                'F_S_err': resp.get('F_S_err', np.zeros_like(S_R)),
            }
    
    def get_redshifts(self) -> np.ndarray:
        """Return redshifts for snapshot indices 0-39 (source z bins)."""
        # The kappa files use source redshift bins z01-z40
        # These correspond to different source redshifts, not the same as TNG snapshots
        # Return a placeholder array - actual z values depend on RT configuration
        return np.linspace(0.1, 2.5, 40)


# =============================================================================
# Response computation utilities
# =============================================================================

def smooth_array(x: np.ndarray, y: np.ndarray, window: int = 9) -> np.ndarray:
    """Smooth y(x) with a uniform moving average, handling NaN values."""
    valid = ~np.isnan(y)
    if valid.sum() < window:
        return y.copy()
    y_smooth = np.full_like(y, np.nan)
    y_smooth[valid] = uniform_filter1d(y[valid], size=window, mode='nearest')
    return y_smooth


def compute_response_fraction(
    S_r: np.ndarray, 
    S_d: np.ndarray, 
    S_h: np.ndarray,
    x: np.ndarray,
    x_min: float,
    x_max: float,
    threshold: float = 0.01,
) -> np.ndarray:
    """Compute scale-dependent response fraction F_S(x).
    
    F_S(x) = (S_R(x) - S_D(x)) / (S_H(x) - S_D(x))
    
    Returns array with NaN where |ΔS| < threshold * |S_D| or outside [x_min, x_max].
    """
    Delta_S = S_h - S_d
    
    # Mask: within range and significant baryonic effect
    in_range = (x >= x_min) & (x <= x_max)
    significant = np.abs(Delta_S) > threshold * np.abs(S_d)
    valid = in_range & significant
    
    F = np.full_like(S_d, np.nan)
    F[valid] = (S_r[valid] - S_d[valid]) / Delta_S[valid]
    
    return F


def compute_response_mean(
    S_r: np.ndarray,
    S_d: np.ndarray,
    S_h: np.ndarray,
    x: np.ndarray,
    x_min: float,
    x_max: float,
    threshold: float = 0.01,
) -> float:
    """Compute weighted mean response fraction over [x_min, x_max].
    
    Uses |ΔS(x)| as weights for the average.
    """
    F = compute_response_fraction(S_r, S_d, S_h, x, x_min, x_max, threshold)
    valid = ~np.isnan(F)
    
    if np.sum(valid) == 0:
        return np.nan
    
    Delta_S = S_h - S_d
    weights = np.abs(Delta_S[valid])
    
    return np.average(F[valid], weights=weights)


# =============================================================================
# ResponseDataCube: Build and store F_S(M_min, α, z)
# =============================================================================

class ResponseDataCube:
    """Data cube storing F_S(M_min, α, z) for a given statistic.
    
    Attributes:
        config: StatisticConfig describing the statistic
        loader: StatsLoader for loading data
        F_cube: numpy array of shape (n_mass, n_alpha, n_snap)
        z_array: numpy array of redshifts
    """
    
    def __init__(
        self,
        config: StatisticConfig,
        loader: StatsLoader,
        mass_thresholds: List[str] = None,
        radii: List[str] = None,
        n_snapshots: int = 20,
    ):
        self.config = config
        self.loader = loader
        self.mass_thresholds = mass_thresholds or MASS_THRESHOLDS
        self.radii = radii or RADII
        self.n_snapshots = n_snapshots
        
        # Will be populated by build()
        self.F_cube = None
        self.z_array = None
        self._dmo_cache = {}
        self._hydro_cache = {}
        
    def build(self, verbose: bool = True) -> 'ResponseDataCube':
        """Build the data cube by loading all models and computing F_S.
        
        Returns self for chaining.
        """
        n_mass = len(self.mass_thresholds)
        n_alpha = len(self.radii)
        n_snap = self.n_snapshots
        
        self.F_cube = np.full((n_mass, n_alpha, n_snap), np.nan)
        self.z_array = self.loader.get_redshifts()[:n_snap]
        
        if verbose:
            print(f"Building {self.config.name} response cube: "
                  f"{n_mass} masses × {n_alpha} radii × {n_snap} snapshots")
        
        for s_idx in range(n_snap):
            z = self.z_array[s_idx]
            
            # Load baselines
            dmo = self.loader.load('dmo', s_idx)
            hydro = self.loader.load('hydro', s_idx)
            
            if dmo is None or hydro is None:
                if verbose:
                    print(f"  Snapshot {s_idx} (z={z:.2f}): missing baselines, skipping")
                continue
            
            # Cache baselines
            self._dmo_cache[s_idx] = dmo
            self._hydro_cache[s_idx] = hydro
            
            # Get x and y arrays
            x = dmo[self.config.x_key]
            S_d = dmo[self.config.y_key]
            S_h = hydro[self.config.y_key]
            
            for i, ml in enumerate(self.mass_thresholds):
                for j, r in enumerate(self.radii):
                    model_name = f'hydro_replace_Ml_{ml}_Mu_inf_R_{r}'
                    replace_stats = self.loader.load(model_name, s_idx)
                    
                    if replace_stats is None:
                        continue
                    
                    S_r = replace_stats[self.config.y_key]
                    
                    F = compute_response_mean(
                        S_r, S_d, S_h, x,
                        self.config.x_min,
                        self.config.x_max,
                        self.config.threshold,
                    )
                    self.F_cube[i, j, s_idx] = F
            
            if verbose:
                n_valid = np.sum(~np.isnan(self.F_cube[:, :, s_idx]))
                print(f"  Snapshot {s_idx:2d} (z={z:.2f}): {n_valid}/{n_mass*n_alpha} valid")
        
        total_valid = np.sum(~np.isnan(self.F_cube))
        if verbose:
            print(f"\n✓ Cube built: {total_valid}/{self.F_cube.size} valid entries")
        
        return self
    
    def get_cumulative_models_at_snapshot(
        self, 
        snapshot_idx: int = 0
    ) -> Dict[tuple, Dict[str, np.ndarray]]:
        """Load all cumulative Replace models for a single snapshot.
        
        Returns dict mapping (mass_threshold, radius) -> stats dict.
        """
        models = {}
        for ml in self.mass_thresholds:
            for r in self.radii:
                model_name = f'hydro_replace_Ml_{ml}_Mu_inf_R_{r}'
                stats = self.loader.load(model_name, snapshot_idx)
                if stats is not None:
                    models[(ml, r)] = stats
        return models
    
    def get_baselines_at_snapshot(
        self,
        snapshot_idx: int = 0
    ) -> tuple:
        """Get (dmo_stats, hydro_stats) for a snapshot."""
        if snapshot_idx in self._dmo_cache:
            return self._dmo_cache[snapshot_idx], self._hydro_cache[snapshot_idx]
        
        dmo = self.loader.load('dmo', snapshot_idx)
        hydro = self.loader.load('hydro', snapshot_idx)
        return dmo, hydro
    
    def save_hdf5(self, path: Path):
        """Save cube to HDF5 file."""
        with h5py.File(path, 'w') as f:
            f.create_dataset('F_cube', data=self.F_cube)
            f.create_dataset('z_array', data=self.z_array)
            f.attrs['statistic_name'] = self.config.name
            f.attrs['x_min'] = self.config.x_min
            f.attrs['x_max'] = self.config.x_max
            # Store as encoded strings for HDF5 compatibility
            f.attrs['mass_thresholds'] = [m.encode() for m in self.mass_thresholds]
            f.attrs['radii'] = [r.encode() for r in self.radii]
    
    @classmethod
    def load_hdf5(cls, path: Path, config: StatisticConfig, loader: StatsLoader) -> 'ResponseDataCube':
        """Load cube from HDF5 file."""
        cube = cls(config, loader)
        with h5py.File(path, 'r') as f:
            cube.F_cube = f['F_cube'][:]
            cube.z_array = f['z_array'][:]
            # Decode bytes back to strings
            cube.mass_thresholds = [m.decode() if isinstance(m, bytes) else m 
                                    for m in f.attrs['mass_thresholds']]
            cube.radii = [r.decode() if isinstance(r, bytes) else r 
                         for r in f.attrs['radii']]
        return cube


# =============================================================================
# ResponsePlotter: Generate standard visualization figures
# =============================================================================

class ResponsePlotter:
    """Static methods for generating response visualization figures."""
    
    @staticmethod
    def plot_scale_dependent_response(
        config: StatisticConfig,
        loader: StatsLoader,
        snapshot_idx: int = 0,
        figsize: tuple = (20, 10),
        save_path: Path = None,
    ):
        """Generate Figure 1: S/S_DMO ratios + F_S(x) panels.
        
        Row 0: 4 ratio panels (one per M_min)
        Row 1: 4 F_S(x) panels (share x-axis with row 0)
        """
        import matplotlib.pyplot as plt
        
        # Load data
        dmo = loader.load('dmo', snapshot_idx)
        hydro = loader.load('hydro', snapshot_idx)
        
        if dmo is None or hydro is None:
            print("⚠ Cannot load baselines")
            return None
        
        cumulative_models = {}
        for ml in MASS_THRESHOLDS:
            for r in RADII:
                model_name = f'hydro_replace_Ml_{ml}_Mu_inf_R_{r}'
                stats = loader.load(model_name, snapshot_idx)
                if stats is not None:
                    cumulative_models[(ml, r)] = stats
        
        if not cumulative_models:
            print("⚠ No Replace models available")
            return None
        
        # Extract arrays
        x = dmo[config.x_key]
        S_d = dmo[config.y_key]
        S_h = hydro[config.y_key]
        Delta_S = S_h - S_d
        significant = np.abs(Delta_S) > config.threshold * np.abs(S_d)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.08, wspace=0.08)
        
        colors_alpha = plt.cm.jet(np.linspace(0.2, 0.9, len(RADII)))
        
        # Row 0: Ratio panels
        ax_ratio = [fig.add_subplot(gs[0, j]) for j in range(4)]
        for j in range(1, 4):
            ax_ratio[j].sharey(ax_ratio[0])
            ax_ratio[j].sharex(ax_ratio[0])
        
        ratio_hydro = S_h / S_d
        ratio_hydro_smooth = smooth_array(x, ratio_hydro, config.smooth_window)
        
        for ax_idx, ml in enumerate(MASS_THRESHOLDS):
            ax = ax_ratio[ax_idx]
            
            plot_fn = ax.semilogx if config.log_x else ax.plot
            
            if ax_idx == 1:
                plot_fn(x, ratio_hydro_smooth, lw=2.5, color='black', label='Hydro/DMO')
                ax.legend()
            else:
                plot_fn(x, ratio_hydro_smooth, lw=2.5, color='black')
            
            for r_idx, r in enumerate(RADII):
                key = (ml, r)
                if key in cumulative_models:
                    S_r = cumulative_models[key][config.y_key]
                    ratio_r = S_r / S_d
                    ratio_r_smooth = smooth_array(x, ratio_r, config.smooth_window)
                    plot_fn(x, ratio_r_smooth, lw=2, color=colors_alpha[r_idx], label=fr'$\alpha={r}$')
            
            ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.6)
            mass_log = np.log10(float(ml))
            ax.set_title(fr'$M_{{\rm min}}=10^{{{mass_log:.1f}}}\ M_\odot/h$')
            ax.set_xlim(config.x_min * 0.5, config.x_max)
            ax.grid(True, alpha=0.3)
            
            if ax_idx == 0:
                ax.legend(loc='lower left', ncol=2)
                ax.set_ylabel(f'{config.y_label} / {config.y_label}$_{{\\rm DMO}}$')
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
        
        # Row 1: F_S(x) panels
        ax_fk = [fig.add_subplot(gs[1, j], sharex=ax_ratio[j]) for j in range(4)]
        for j in range(1, 4):
            ax_fk[j].sharey(ax_fk[0])
        
        for ax in ax_ratio:
            plt.setp(ax.get_xticklabels(), visible=False)
        
        for ax_idx, ml in enumerate(MASS_THRESHOLDS):
            ax = ax_fk[ax_idx]
            plot_fn = ax.semilogx if config.log_x else ax.plot
            
            for r_idx, r in enumerate(RADII):
                key = (ml, r)
                if key in cumulative_models:
                    S_r = cumulative_models[key][config.y_key]
                    F = compute_response_fraction(
                        S_r, S_d, S_h, x,
                        config.x_min, config.x_max, config.threshold
                    )
                    F_smooth = smooth_array(x, F, config.smooth_window)
                    plot_fn(x, F_smooth, lw=2, color=colors_alpha[r_idx], label=fr'$\alpha={r}$')
            
            ax.axhline(1.0, color='red', ls='--', lw=1.2, alpha=0.6)
            ax.axhline(0.0, color='blue', ls=':', lw=1.2, alpha=0.6)
            ax.axhspan(0, 1, alpha=0.03, color='green')
            ax.set_xlim(config.x_min * 0.5, config.x_max)
            ax.set_ylim(-0.4, 1.5)
            ax.set_xlabel(config.x_label)
            ax.grid(True, alpha=0.3)
            
            if ax_idx == 0:
                ax.set_ylabel(f'$F_{{{config.name}}}$')
                ax.legend(loc='upper left', ncol=2)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_cumulative_summary(
        config: StatisticConfig,
        loader: StatsLoader,
        snapshot_idx: int = 0,
        figsize: tuple = (15, 10),
        save_path: Path = None,
    ):
        """Generate Figure 2: Cumulative response line plots + heatmap.
        
        Row 0: 2 panels (fixed α varying M_min, fixed M_min varying α)
        Row 1: Heatmap of F_S(M_min, α)
        """
        import matplotlib.pyplot as plt
        
        # Load data and compute F matrix
        dmo = loader.load('dmo', snapshot_idx)
        hydro = loader.load('hydro', snapshot_idx)
        
        if dmo is None or hydro is None:
            print("⚠ Cannot load baselines")
            return None
        
        n_mass = len(MASS_THRESHOLDS)
        n_alpha = len(RADII)
        F_matrix = np.full((n_mass, n_alpha), np.nan)
        
        x = dmo[config.x_key]
        S_d = dmo[config.y_key]
        S_h = hydro[config.y_key]
        
        cumulative_models = {}
        for i, ml in enumerate(MASS_THRESHOLDS):
            for j, r in enumerate(RADII):
                model_name = f'hydro_replace_Ml_{ml}_Mu_inf_R_{r}'
                stats = loader.load(model_name, snapshot_idx)
                if stats is not None:
                    cumulative_models[(ml, r)] = stats
                    S_r = stats[config.y_key]
                    F_matrix[i, j] = compute_response_mean(
                        S_r, S_d, S_h, x,
                        config.x_min, config.x_max, config.threshold
                    )
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.3, wspace=0.08)
        
        colors_alpha = plt.cm.jet(np.linspace(0.2, 0.9, len(RADII)))
        colors_mass = plt.cm.plasma(np.linspace(0.2, 0.9, len(MASS_THRESHOLDS)))
        markers = ['o', 's', '^', 'D']
        
        # Row 0: Line plots
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
        
        # Left: fixed α, varying M_min
        for r_idx, r in enumerate(RADII):
            means = [F_matrix[i, r_idx] for i in range(n_mass)]
            ax0.plot(LOG_MASS, means, marker=markers[r_idx], color=colors_alpha[r_idx],
                    lw=2, ms=8, label=fr'$\alpha={r}$')
        ax0.set_xlabel(r'$\log_{10}(M_{\rm min}/M_\odot h^{-1})$')
        ax0.set_ylabel(config.label)
        
        # Right: fixed M_min, varying α
        for m_idx, ml in enumerate(MASS_THRESHOLDS):
            means = [F_matrix[m_idx, j] for j in range(n_alpha)]
            ax1.plot(ALPHA_VALUES, means, marker=markers[m_idx], color=colors_mass[m_idx],
                    lw=2, ms=8, label=r'$M_{\rm min}=$' + MASS_LABELS[m_idx])
        ax1.set_xlabel(r'$\alpha$')
        plt.setp(ax1.get_yticklabels(), visible=False)
        
        for ax in (ax0, ax1):
            ax.axhline(1.0, color='red', ls='--', lw=1.2, alpha=0.7)
            ax.axhline(0.0, color='blue', ls=':', lw=1.2, alpha=0.7)
            ax.set_ylim(-0.4, 1.5)
            ax.grid(True, alpha=0.3)
            ax.axhspan(0, 1, alpha=0.03, color='green')
        ax0.legend(loc='best', ncol=2)
        ax1.legend(loc='best', ncol=2)
        
        # Row 1: Heatmap
        ax2 = fig.add_subplot(gs[1, :])
        im = ax2.imshow(F_matrix, cmap='Reds', vmin=0, vmax=1.2, aspect='auto', origin='lower')
        ax2.set_xticks(range(n_alpha))
        ax2.set_xticklabels(RADII)
        ax2.set_yticks(range(n_mass))
        ax2.set_yticklabels([f'{np.log10(float(m)):.1f}' for m in MASS_THRESHOLDS])
        ax2.set_xlabel(r'$\alpha$')
        ax2.set_ylabel(r'$\log_{10}M_{\rm min}$')
        ax2.grid(False)
        
        for i in range(n_mass):
            for j in range(n_alpha):
                val = F_matrix[i, j]
                if not np.isnan(val):
                    ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=10, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label(config.label)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_redshift_evolution(
        cube: ResponseDataCube,
        figsize: tuple = (20, 5),
        save_path: Path = None,
    ):
        """Generate redshift evolution plot: F_S vs z for each M_min.
        
        4 panels (one per M_min), lines colored by α.
        """
        import matplotlib.pyplot as plt
        
        config = cube.config
        n_mass = len(cube.mass_thresholds)
        n_alpha = len(cube.radii)
        
        fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True,
                                  gridspec_kw={'wspace': 0.03})
        
        colors_alpha = plt.cm.jet(np.linspace(0.1, 0.9, n_alpha))
        markers = ['o', 's', '^', 'D']
        alpha_labels = [fr'$\alpha = {r}$' for r in cube.radii]
        
        for ax_idx, (i, ml) in enumerate(zip(range(n_mass), cube.mass_thresholds)):
            ax = axes.flat[ax_idx]
            
            for j, (r, alpha_label) in enumerate(zip(cube.radii, alpha_labels)):
                F_vs_z = cube.F_cube[i, j, :]
                valid = ~np.isnan(F_vs_z)
                
                ax.plot(cube.z_array[valid], F_vs_z[valid], marker=markers[j], ms=5, lw=1.5,
                       color=colors_alpha[j], label=alpha_label)
            
            ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.6)
            ax.axhline(0.0, color='blue', ls=':', lw=1, alpha=0.6)
            ax.set_title(r'$M_{\rm min} = $' + MASS_LABELS[i] + r'$h^{-1}\,M_\odot$')
            ax.grid(True, alpha=0.3)
            ax.axhspan(0, 1, alpha=0.03, color='green')
            ax.set_xlabel('$z$')
            
            if ax_idx == 0:
                ax.set_ylabel(config.label)
        
        axes[0].legend(loc='lower left', ncol=2)
        axes[0].set_xlim(-0.1, 2.5)
        axes[0].set_ylim(-0.2, 1.4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
        
        return fig
