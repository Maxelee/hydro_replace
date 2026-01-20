"""
driver_baryonic_response_mpi_robust.py

MPI-parallelized analysis with robust error handling.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from scipy import ndimage
import h5py
import sys
import time
import warnings

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def print_rank0(msg):
    if rank == 0:
        print(msg)
        sys.stdout.flush()

try:
    import Pk_library as PKL
    PYLIANS_AVAILABLE = True
except ImportError:
    PYLIANS_AVAILABLE = False
    warnings.warn("Pylians not available")

# =============================================================================
# Configuration
# =============================================================================

RT_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG')
STATS_CACHE_DIR = Path('./statistics_cache')
if rank == 0:
    STATS_CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path('./response_analysis_output')
if rank == 0:
    OUTPUT_DIR.mkdir(exist_ok=True)

comm.Barrier()

RT_GRID = 1024
FIELD_SIZE_DEG = 5.0
N_REALIZATIONS_PER_LP = 50
N_LENS_PLANES = 10
SNAP_Z = 23

# Smoothing configuration
PIXEL_SIZE_ARCMIN = (FIELD_SIZE_DEG * 60.0) / RT_GRID  # ~0.293 arcmin/pixel
TARGET_SMOOTHING_ARCMIN = 2.0
SMOOTHING_PIX = TARGET_SMOOTHING_ARCMIN / PIXEL_SIZE_ARCMIN  # ~6.8 pixels

# Reference kappa RMS from DMO ensemble (500 maps, 2 arcmin smoothing)
KAPPA_RMS_DMO = 0.0107

# SNR bins for peaks and minima (optimized for baryonic signal detection)
PEAK_SNR_BINS = np.linspace(-2, 6, 17)     # 16 bins from -2 to +6
MINIMA_SNR_BINS = np.linspace(-6, 2, 17)  # 16 bins from -6 to +2
PEAK_SNR_MID = 0.5 * (PEAK_SNR_BINS[:-1] + PEAK_SNR_BINS[1:])
MINIMA_SNR_MID = 0.5 * (MINIMA_SNR_BINS[:-1] + MINIMA_SNR_BINS[1:])

MASS_BIN_EDGES = np.array([1.00e12, 3.16e12, 1.00e13, 3.16e13, 1.00e15])
ALPHA_EDGES = np.array([0.0, 0.5, 1.0, 3.0, 5.0])

CUMULATIVE_MODELS = [
    'hydro_replace_Ml_1.00e12_Mu_inf_R_0.5',
    'hydro_replace_Ml_1.00e12_Mu_inf_R_1.0',
    'hydro_replace_Ml_1.00e12_Mu_inf_R_3.0',
    'hydro_replace_Ml_1.00e12_Mu_inf_R_5.0',
    'hydro_replace_Ml_3.16e12_Mu_inf_R_0.5',
    'hydro_replace_Ml_3.16e12_Mu_inf_R_1.0',
    'hydro_replace_Ml_3.16e12_Mu_inf_R_3.0',
    'hydro_replace_Ml_3.16e12_Mu_inf_R_5.0',
    'hydro_replace_Ml_1.00e13_Mu_inf_R_0.5',
    'hydro_replace_Ml_1.00e13_Mu_inf_R_1.0',
    'hydro_replace_Ml_1.00e13_Mu_inf_R_3.0',
    'hydro_replace_Ml_1.00e13_Mu_inf_R_5.0',
    'hydro_replace_Ml_3.16e13_Mu_inf_R_0.5',
    'hydro_replace_Ml_3.16e13_Mu_inf_R_1.0',
    'hydro_replace_Ml_3.16e13_Mu_inf_R_3.0',
    'hydro_replace_Ml_3.16e13_Mu_inf_R_5.0',
]

DIFFERENTIAL_MODELS = [
    'hydro_replace_Ml_1.00e12_Mu_3.16e12_R_0.5',
    'hydro_replace_Ml_1.00e12_Mu_3.16e12_R_1.0',
    'hydro_replace_Ml_1.00e12_Mu_3.16e12_R_3.0',
    'hydro_replace_Ml_1.00e12_Mu_3.16e12_R_5.0',
    'hydro_replace_Ml_3.16e12_Mu_1.00e13_R_0.5',
    'hydro_replace_Ml_3.16e12_Mu_1.00e13_R_1.0',
    'hydro_replace_Ml_3.16e12_Mu_1.00e13_R_3.0',
    'hydro_replace_Ml_3.16e12_Mu_1.00e13_R_5.0',
    'hydro_replace_Ml_1.00e13_Mu_3.16e13_R_0.5',
    'hydro_replace_Ml_1.00e13_Mu_3.16e13_R_1.0',
    'hydro_replace_Ml_1.00e13_Mu_3.16e13_R_3.0',
    'hydro_replace_Ml_1.00e13_Mu_3.16e13_R_5.0',
    'hydro_replace_Ml_3.16e13_Mu_1.00e15_R_0.5',
    'hydro_replace_Ml_3.16e13_Mu_1.00e15_R_1.0',
    'hydro_replace_Ml_3.16e13_Mu_1.00e15_R_3.0',
    'hydro_replace_Ml_3.16e13_Mu_1.00e15_R_5.0',
]

ALL_MODELS = ['dmo', 'hydro'] + CUMULATIVE_MODELS + DIFFERENTIAL_MODELS

# =============================================================================
# Helper functions with robust error handling
# =============================================================================

def load_kappa(fname, ng=1024):
    """Load single kappa map with error handling."""
    try:
        with open(fname, 'rb') as f:
            dummy = np.fromfile(f, dtype="int32", count=1)
            kappa = np.fromfile(f, dtype="float", count=ng*ng)
            dummy = np.fromfile(f, dtype="int32", count=1)
        
        kappa = kappa.reshape(ng, ng)
        
        # Sanity check
        if np.any(~np.isfinite(kappa)):
            raise ValueError(f"Non-finite values in {fname}")
        if np.std(kappa) == 0:
            raise ValueError(f"Zero variance in {fname}")
        
        return kappa
    except Exception as e:
        # Return None on any error - caller will handle
        return None

def compute_Cl_single_map(kappa, field_size_deg=FIELD_SIZE_DEG):
    """Compute C_ell with error handling."""
    try:
        grid = kappa.shape[0]
        field_size_rad = np.radians(field_size_deg)
        box_size = field_size_rad
        
        delta_kappa = (kappa - np.mean(kappa)).astype(np.float32)
        
        if PYLIANS_AVAILABLE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Pk2D = PKL.Pk_plane(delta_kappa, box_size, 'None', threads=1, verbose=False)
            return Pk2D.k, Pk2D.Pk
        else:
            # Fallback FFT
            kappa_k = np.fft.fft2(delta_kappa)
            Pk_2d = np.abs(kappa_k)**2 * (field_size_rad / grid)**2
            
            lx = np.fft.fftfreq(grid, d=field_size_rad/grid) * 2 * np.pi
            ly = np.fft.fftfreq(grid, d=field_size_rad/grid) * 2 * np.pi
            lx, ly = np.meshgrid(lx, ly)
            ell_2d = np.sqrt(lx**2 + ly**2)
            
            ell_bins = np.logspace(np.log10(100), np.log10(20000), 30)
            ell = 0.5 * (ell_bins[:-1] + ell_bins[1:])
            
            C_ell = np.zeros(len(ell))
            for i in range(len(ell)):
                mask = (ell_2d >= ell_bins[i]) & (ell_2d < ell_bins[i+1])
                if np.sum(mask) > 0:
                    C_ell[i] = np.mean(Pk_2d[mask])
            
            return ell, C_ell
    except Exception as e:
        return None, None

def compute_peaks_single_map(kappa, smoothing_pix=SMOOTHING_PIX, snr_bins=PEAK_SNR_BINS, kappa_rms=KAPPA_RMS_DMO):
    """Compute peak counts with error handling.
    
    Uses a constant reference kappa_rms (from DMO ensemble) for SNR normalization
    to ensure consistent bin definitions across all models.
    """
    try:
        kappa_smooth = ndimage.gaussian_filter(kappa, smoothing_pix)
        
        data_max = ndimage.maximum_filter(kappa_smooth, size=3)
        peaks_mask = (kappa_smooth == data_max) & (kappa_smooth > 0)
        
        peaks_mask[:2, :] = False
        peaks_mask[-2:, :] = False
        peaks_mask[:, :2] = False
        peaks_mask[:, -2:] = False
        
        if np.sum(peaks_mask) == 0:
            return np.zeros(len(snr_bins) - 1)
        
        # Use constant reference sigma for consistent SNR bins across models
        sigma = kappa_rms
        
        peak_values = kappa_smooth[peaks_mask] / sigma
        peak_counts, _ = np.histogram(peak_values, bins=snr_bins)
        
        return peak_counts
    except Exception as e:
        return np.zeros(len(snr_bins) - 1)

def compute_minima_single_map(kappa, smoothing_pix=SMOOTHING_PIX, snr_bins=MINIMA_SNR_BINS, kappa_rms=KAPPA_RMS_DMO):
    """Compute minima counts with error handling.
    
    Uses a constant reference kappa_rms (from DMO ensemble) for SNR normalization.
    Minima have negative SNR values (nu = kappa / sigma < 0 for underdense regions).
    """
    try:
        kappa_smooth = ndimage.gaussian_filter(kappa, smoothing_pix)
        
        data_min = ndimage.minimum_filter(kappa_smooth, size=3)
        minima_mask = (kappa_smooth == data_min) & (kappa_smooth < 0)
        
        minima_mask[:2, :] = False
        minima_mask[-2:, :] = False
        minima_mask[:, :2] = False
        minima_mask[:, -2:] = False
        
        if np.sum(minima_mask) == 0:
            return np.zeros(len(snr_bins) - 1)
        
        # Use constant reference sigma for consistent SNR bins across models
        sigma = kappa_rms
        
        # SNR with natural sign (minima have negative nu)
        minima_values = kappa_smooth[minima_mask] / sigma
        minima_counts, _ = np.histogram(minima_values, bins=snr_bins)
        
        return minima_counts
    except Exception as e:
        return np.zeros(len(snr_bins) - 1)

# =============================================================================
# Stage 1: Compute statistics with robust error handling
# =============================================================================

def compute_all_statistics_for_model(model_name, z=SNAP_Z, force_recompute=False):
    """Compute statistics with robust file handling."""
    cache_file = STATS_CACHE_DIR / f'{model_name}_z{z:02d}_stats.h5'
    
    if cache_file.exists() and not force_recompute:
        print(f"[Rank {rank}] Statistics already cached for {model_name}, skipping")
        sys.stdout.flush()
        return cache_file
    
    print(f"[Rank {rank}] Computing statistics for {model_name}...")
    sys.stdout.flush()
    
    t_start = time.time()
    
    all_C_ell = []
    all_peaks = []
    all_minima = []
    all_LP_ids = []
    all_run_ids = []
    
    ell = None
    n_success = 0
    n_fail = 0
    
    # Loop over all lens planes and runs
    for LP_id in range(N_LENS_PLANES):
        for run_id in range(1, N_REALIZATIONS_PER_LP + 1):
            path = f'{RT_BASE}/{model_name}/LP_{LP_id:02d}/run{run_id:03d}/kappa{z:02d}.dat'
            
            # Try to load kappa
            kappa = load_kappa(path, ng=RT_GRID)
            if kappa is None:
                n_fail += 1
                continue
            
            # Compute C_ell
            ell_map, C_ell_map = compute_Cl_single_map(kappa)
            if ell_map is None or C_ell_map is None:
                n_fail += 1
                continue
            
            if ell is None:
                ell = ell_map
            
            # Compute peaks and minima
            peaks = compute_peaks_single_map(kappa)
            minima = compute_minima_single_map(kappa)
            
            # Store successful computation
            all_C_ell.append(C_ell_map)
            all_peaks.append(peaks)
            all_minima.append(minima)
            all_LP_ids.append(LP_id)
            all_run_ids.append(run_id)
            n_success += 1
    
    if n_success == 0:
        print(f"[Rank {rank}] ERROR: No valid maps found for {model_name} (0/{n_success+n_fail} succeeded)")
        sys.stdout.flush()
        return None
    
    print(f"[Rank {rank}]   {model_name}: {n_success} succeeded, {n_fail} failed")
    sys.stdout.flush()
    
    # Convert to arrays
    all_C_ell = np.array(all_C_ell)
    all_peaks = np.array(all_peaks)
    all_minima = np.array(all_minima)
    all_LP_ids = np.array(all_LP_ids)
    all_run_ids = np.array(all_run_ids)
    
    # Save to HDF5
    try:
        with h5py.File(cache_file, 'w') as f:
            f.create_dataset('ell', data=ell)
            f.create_dataset('peak_snr_bins', data=PEAK_SNR_BINS)
            f.create_dataset('minima_snr_bins', data=MINIMA_SNR_BINS)
            f.create_dataset('kappa_rms_ref', data=KAPPA_RMS_DMO)
            f.create_dataset('C_ell', data=all_C_ell)
            f.create_dataset('peaks', data=all_peaks)
            f.create_dataset('minima', data=all_minima)
            f.create_dataset('LP_ids', data=all_LP_ids)
            f.create_dataset('run_ids', data=all_run_ids)
            
            f.attrs['model_name'] = model_name
            f.attrs['z'] = z
            f.attrs['n_realizations'] = n_success
            f.attrs['n_failed'] = n_fail
        
        t_elapsed = time.time() - t_start
        print(f"[Rank {rank}]   Saved to {cache_file.name} ({t_elapsed:.1f}s)")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"[Rank {rank}] ERROR saving {cache_file}: {e}")
        sys.stdout.flush()
        return None
    
    return cache_file

def compute_all_statistics_all_models_mpi(models=ALL_MODELS, z=SNAP_Z, force_recompute=False):
    """Stage 1 with MPI parallelization."""
    if rank == 0:
        print("="*70)
        print(f"STAGE 1: Computing statistics with {size} MPI ranks")
        print(f"Total models: {len(models)}")
        print(f"Smoothing: {TARGET_SMOOTHING_ARCMIN:.1f} arcmin ({SMOOTHING_PIX:.1f} pixels)")
        print("="*70)
        sys.stdout.flush()
    
    # Divide models
    models_per_rank = len(models) // size
    remainder = len(models) % size
    
    if rank < remainder:
        start_idx = rank * (models_per_rank + 1)
        end_idx = start_idx + models_per_rank + 1
    else:
        start_idx = remainder * (models_per_rank + 1) + (rank - remainder) * models_per_rank
        end_idx = start_idx + models_per_rank
    
    my_models = models[start_idx:end_idx]
    
    print(f"[Rank {rank}] Processing {len(my_models)} models")
    sys.stdout.flush()
    
    # Process assigned models
    for i, model in enumerate(my_models):
        print(f"[Rank {rank}] [{i+1}/{len(my_models)}] {model}")
        sys.stdout.flush()
        
        compute_all_statistics_for_model(model, z=z, force_recompute=force_recompute)
    
    comm.Barrier()
    
    if rank == 0:
        print("\n" + "="*70)
        print("Stage 1 complete!")
        print("="*70)
        sys.stdout.flush()

# =============================================================================
# Stage 2: Analysis (same as before, rank 0 only)
# =============================================================================

# [Keep the same Stage 2 code from before - load_cached_statistics, etc.]
# I'll skip repeating it here for brevity



# =============================================================================
# STAGE 2: Load cached statistics and run response analysis (rank 0 only)
# =============================================================================

def load_cached_statistics(model_name, z=SNAP_Z):
    """Load pre-computed statistics for a model."""
    cache_file = STATS_CACHE_DIR / f'{model_name}_z{z:02d}_stats.h5'
    
    if not cache_file.exists():
        raise FileNotFoundError(f"Statistics not cached for {model_name}. Run Stage 1 first.")
    
    with h5py.File(cache_file, 'r') as f:
        result = {
            'ell': f['ell'][:],
            'C_ell': f['C_ell'][:],
            'peaks': f['peaks'][:],
            'minima': f['minima'][:],
            'LP_ids': f['LP_ids'][:],
            'run_ids': f['run_ids'][:],
        }
        # Handle both old and new formats
        if 'peak_snr_bins' in f:
            result['peak_snr_bins'] = f['peak_snr_bins'][:]
            result['minima_snr_bins'] = f['minima_snr_bins'][:]
            result['kappa_rms_ref'] = f['kappa_rms_ref'][()]
        else:
            # Old format compatibility
            result['snr_bins'] = f['snr_bins'][:]
        return result

def parse_model_name(model_name):
    """Parse model name to extract mass and radius info."""
    if not model_name.startswith('hydro_replace_'):
        return None
    
    parts = model_name.split('_')
    ml_idx = parts.index('Ml') + 1
    mu_idx = parts.index('Mu') + 1
    r_idx = parts.index('R') + 1
    
    M_lo = float(parts[ml_idx])
    M_hi_str = parts[mu_idx]
    M_hi = np.inf if M_hi_str == 'inf' else float(M_hi_str)
    r_factor = float(parts[r_idx])
    
    return M_lo, M_hi, r_factor

def model_to_tile_key(model_name, mass_edges, alpha_edges):
    """Convert differential model to tile key."""
    parsed = parse_model_name(model_name)
    if parsed is None or parsed[1] == np.inf:
        return None
    
    M_lo, M_hi, r_factor = parsed
    
    mass_bin_idx = None
    for i in range(len(mass_edges) - 1):
        if np.isclose(mass_edges[i], M_lo) and np.isclose(mass_edges[i+1], M_hi):
            mass_bin_idx = i
            break
    
    radius_shell_idx = None
    for i in range(len(alpha_edges) - 1):
        if np.isclose(alpha_edges[i+1], r_factor):
            radius_shell_idx = i
            break
    
    if mass_bin_idx is None or radius_shell_idx is None:
        return None
    
    return (mass_bin_idx, radius_shell_idx)

def model_to_cumulative_config(model_name):
    """Convert cumulative model to (M_min, alpha_max)."""
    parsed = parse_model_name(model_name)
    if parsed is None or parsed[1] != np.inf:
        return None
    
    return parsed[0], parsed[2]

def run_response_analysis_from_cache(stat_key='C_ell', z=SNAP_Z):
    """
    Stage 2 (rank 0 only): Load cached statistics and run response analysis.
    """
    if rank != 0:
        return None  # Only rank 0 runs this
    
    print("\n" + "="*70)
    print(f"STAGE 2: Response analysis for {stat_key}, z={z}")
    print("="*70)
    sys.stdout.flush()
    
    # Import response analysis tools
    from baryonic_response import lambda_pattern_from_Mmin_alpha, additive_response_from_tiles
    
    # Load baseline
    print("\nLoading DMO and Hydro statistics...")
    sys.stdout.flush()
    dmo_stats = load_cached_statistics('dmo', z=z)
    hydro_stats = load_cached_statistics('hydro', z=z)
    
    # Extract the relevant statistic
    S_D_all = dmo_stats[stat_key]
    S_H_all = hydro_stats[stat_key]
    
    # Compute means and errors
    S_D = np.mean(S_D_all, axis=0)
    S_H = np.mean(S_H_all, axis=0)
    Delta_S = S_H - S_D
    
    S_D_err = np.std(S_D_all, axis=0) / np.sqrt(len(S_D_all))
    S_H_err = np.std(S_H_all, axis=0) / np.sqrt(len(S_H_all))
    
    print(f"  DMO: {len(S_D_all)} realizations, stat shape {S_D.shape}")
    print(f"  Hydro: {len(S_H_all)} realizations")
    print(f"  Mean |Delta S / S_D|: {np.nanmean(np.abs(Delta_S / S_D)):.3f}")
    sys.stdout.flush()
    
    # Storage
    tile_responses = {}
    tile_responses_err = {}
    cumulative_responses = {}
    cumulative_responses_err = {}
    additivity_results = {}
    
    # Process differential (tile) models
    print("\nComputing tile responses...")
    sys.stdout.flush()
    
    for model in DIFFERENTIAL_MODELS:
        tile_key = model_to_tile_key(model, MASS_BIN_EDGES, ALPHA_EDGES)
        if tile_key is None:
            continue
        
        try:
            stats = load_cached_statistics(model, z=z)
            S_tile_all = stats[stat_key]
            
            S_tile = np.mean(S_tile_all, axis=0)
            S_tile_err = np.std(S_tile_all, axis=0) / np.sqrt(len(S_tile_all))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                Delta_F = (S_tile - S_D) / Delta_S
                Delta_F_err = np.sqrt(S_tile_err**2 + S_D_err**2) / np.abs(Delta_S)
            
            tile_responses[tile_key] = Delta_F
            tile_responses_err[tile_key] = Delta_F_err
            
        except Exception as e:
            print(f"  Skipping {model}: {e}")
            sys.stdout.flush()
    
    print(f"  Loaded {len(tile_responses)} tiles")
    sys.stdout.flush()
    
    # Process cumulative models
    print("\nComputing cumulative responses...")
    sys.stdout.flush()
    
    for model in CUMULATIVE_MODELS:
        config = model_to_cumulative_config(model)
        if config is None:
            continue
        
        M_min, alpha_max = config
        label = f"M{M_min:.1e}_a{alpha_max:.1f}"
        
        try:
            stats = load_cached_statistics(model, z=z)
            S_R_all = stats[stat_key]
            
            S_R = np.mean(S_R_all, axis=0)
            S_R_err = np.std(S_R_all, axis=0) / np.sqrt(len(S_R_all))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                F_S = (S_R - S_D) / Delta_S
                F_S_err = np.sqrt(S_R_err**2 + S_D_err**2) / np.abs(Delta_S)
            
            cumulative_responses[label] = {
                'M_min': M_min,
                'alpha_max': alpha_max,
                'F_S': F_S,
                'F_S_err': F_S_err,
            }
            
        except Exception as e:
            print(f"  Skipping {model}: {e}")
            sys.stdout.flush()
    
    print(f"  Loaded {len(cumulative_responses)} cumulative configs")
    sys.stdout.flush()
    
    # Test additivity
    print("\nTesting additivity...")
    sys.stdout.flush()
    
    for label, cum_data in cumulative_responses.items():
        M_min = cum_data['M_min']
        alpha_max = cum_data['alpha_max']
        F_true = cum_data['F_S']
        
        lam_pattern = lambda_pattern_from_Mmin_alpha(
            M_min, alpha_max, MASS_BIN_EDGES, ALPHA_EDGES
        )
        
        F_lin = additive_response_from_tiles(lam_pattern, tile_responses)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            eps = (F_true - F_lin) / F_true
        
        additivity_results[label] = {
            'F_true': F_true,
            'F_lin': F_lin,
            'epsilon': eps,
            'mean_eps': np.nanmean(np.abs(eps)),
        }
        
        print(f"  {label}: mean |epsilon| = {np.nanmean(np.abs(eps)):.3f}")
        sys.stdout.flush()
    
    # Package and save results
    # Determine SNR bins based on stat_key
    if stat_key == 'peaks':
        snr_bins = PEAK_SNR_BINS
        snr_mid = PEAK_SNR_MID
    elif stat_key == 'minima':
        snr_bins = MINIMA_SNR_BINS
        snr_mid = MINIMA_SNR_MID
    else:
        snr_bins = None
        snr_mid = None
    
    results = {
        'stat_key': stat_key,
        'z': z,
        'S_D': S_D,
        'S_H': S_H,
        'Delta_S': Delta_S,
        'S_D_err': S_D_err,
        'S_H_err': S_H_err,
        'tile_responses': tile_responses,
        'tile_responses_err': tile_responses_err,
        'cumulative_responses': cumulative_responses,
        'additivity_results': additivity_results,
        'mass_bin_edges': MASS_BIN_EDGES,
        'alpha_edges': ALPHA_EDGES,
        'ell': dmo_stats['ell'] if stat_key == 'C_ell' else None,
        'snr_bins': snr_bins,
        'snr_mid': snr_mid,
        'kappa_rms_ref': KAPPA_RMS_DMO,
    }
    
    # Save
    save_path = OUTPUT_DIR / f'response_{stat_key}_z{z:02d}.npz'
    np.savez(save_path, **{k: v for k, v in results.items() if not isinstance(v, dict)})
    
    import pickle
    pickle_path = OUTPUT_DIR / f'response_{stat_key}_z{z:02d}_dicts.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'tile_responses': tile_responses,
            'tile_responses_err': tile_responses_err,
            'cumulative_responses': cumulative_responses,
            'additivity_results': additivity_results,
        }, f)
    
    print(f"\nSaved results to {save_path}")
    print(f"Saved dicts to {pickle_path}")
    sys.stdout.flush()
    
    return results

# =============================================================================
# Main execution
# =============================================================================

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MPI-parallelized baryonic response analysis')
    parser.add_argument('--stage', type=str, default='both', choices=['1', '2', 'both'],
                        help='Stage to run: 1=compute stats, 2=response analysis, both=all')
    parser.add_argument('--force-recompute', action='store_true',
                        help='Force recomputation of statistics (overwrite cache)')
    parser.add_argument('--z', type=int, default=23, help='Redshift snapshot index')
    args = parser.parse_args()
    
    stage = args.stage
    Z_SNAP = args.z
    force_recompute = args.force_recompute
    
    if rank == 0:
        print("="*70)
        print("Baryonic Response Analysis")
        print(f"Stage: {stage}")
        print(f"z snapshot: {Z_SNAP}")
        print(f"Force recompute: {force_recompute}")
        print(f"KAPPA_RMS_DMO: {KAPPA_RMS_DMO}")
        print(f"PEAK_SNR_BINS: {PEAK_SNR_BINS[0]:.1f} to {PEAK_SNR_BINS[-1]:.1f} ({len(PEAK_SNR_BINS)-1} bins)")
        print(f"MINIMA_SNR_BINS: {MINIMA_SNR_BINS[0]:.1f} to {MINIMA_SNR_BINS[-1]:.1f} ({len(MINIMA_SNR_BINS)-1} bins)")
        print("="*70)
        sys.stdout.flush()
    
    if stage in ['1', 'both']:
        compute_all_statistics_all_models_mpi(
            models=ALL_MODELS,
            z=Z_SNAP,
            force_recompute=force_recompute
        )
    
    comm.Barrier()
    # Synchronize before Stage 2
    
    # Stage 2: Only rank 0
    if stage in ['2', 'both']:
        if rank == 0:
            for stat_key in ['C_ell', 'peaks', 'minima']:
                try:
                    results = run_response_analysis_from_cache(stat_key=stat_key, z=Z_SNAP)
                    print(f"\n{stat_key} analysis complete!")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"\nERROR in {stat_key}: {e}")
                    sys.stdout.flush()
                    import traceback
                    traceback.print_exc()
    
    # Final synchronization
    comm.Barrier()
    
    if rank == 0:
        print("\n" + "="*70)
        print("All analysis complete!")
        print("="*70)
        sys.stdout.flush()
