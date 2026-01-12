"""
driver_density_response_mpi.py

MPI-parallelized analysis of baryonic responses for 2D density planes (lenspots).
Computes power spectrum, PDF, and other statistics for the projected density fields.

These are the raw density planes before ray-tracing, allowing us to measure
baryonic effects directly on the matter distribution without lensing systematics.

Usage:
    mpirun -np 34 python driver_density_response_mpi.py [stage]
    
    stage = '1'    : Compute statistics only
    stage = '2'    : Run response analysis only  
    stage = 'both' : Run both stages (default)
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

LP_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG')
STATS_CACHE_DIR = Path('./density_statistics_cache')
if rank == 0:
    STATS_CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path('./density_response_output')
if rank == 0:
    OUTPUT_DIR.mkdir(exist_ok=True)

comm.Barrier()

# Grid and physical parameters
LP_GRID = 4096
BOX_SIZE = 205.0  # Mpc/h - full simulation box

# Lens plane structure
N_LENS_PLANES = 10      # LP_00 to LP_09
N_LENSPOTS = 40         # lenspot00.dat to lenspot39.dat (20 snapshots × 2 planes)
N_SNAPSHOTS = 20
PLANES_PER_SNAPSHOT = 2

# Snapshot mapping (index -> snapshot number)
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]

# PDF configuration
PDF_NBINS = 100
LOG_DELTA_MIN = -2.0  # log10(1 + delta) range
LOG_DELTA_MAX = 4.0

# Power spectrum ell bins (for 2D field)
ELL_MIN = 100
ELL_MAX = 50000
N_ELL_BINS = 50

# Mass bin edges for response analysis
MASS_BIN_EDGES = np.array([1.00e12, 3.16e12, 1.00e13, 3.16e13, 1.00e15])
ALPHA_EDGES = np.array([0.0, 0.5, 1.0, 3.0, 5.0])

# Model lists
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
# Helper functions
# =============================================================================

def load_lenspot(fname, ng=LP_GRID):
    """Load a lenspot .dat file (lux format).
    
    Format: int32(ng) + float64[ng*ng](field) + int32(ng)
    
    The field contains delta * dz (surface density × depth).
    """
    try:
        with open(fname, 'rb') as f:
            ng_read = np.fromfile(f, dtype='int32', count=1)[0]
            if ng_read != ng:
                raise ValueError(f"Grid size mismatch: expected {ng}, got {ng_read}")
            data = np.fromfile(f, dtype='float64', count=ng*ng)
            _ = np.fromfile(f, dtype='int32', count=1)
        
        field = data.reshape(ng, ng)
        
        # Sanity check
        if np.any(~np.isfinite(field)):
            raise ValueError(f"Non-finite values in {fname}")
        
        return field
    except Exception as e:
        return None


def compute_power_spectrum_2d(field, box_size=BOX_SIZE):
    """Compute 2D power spectrum P(k) for a density field.
    
    Parameters
    ----------
    field : np.ndarray, shape (N, N)
        2D density field (or overdensity).
    box_size : float
        Physical size of the field in Mpc/h.
        
    Returns
    -------
    k : np.ndarray
        Wavenumber bins in h/Mpc.
    Pk : np.ndarray
        Power spectrum P(k) in (Mpc/h)^2.
    """
    ng = field.shape[0]
    
    # Convert to overdensity if needed
    mean_val = np.mean(field)
    if mean_val > 0:
        delta = field / mean_val - 1.0
    else:
        delta = field - np.mean(field)
    
    delta = delta.astype(np.float32)
    
    if PYLIANS_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Pk2D = PKL.Pk_plane(delta, box_size, 'None', threads=1, verbose=False)
            return Pk2D.k, Pk2D.Pk
        except Exception as e:
            pass
    
    # Fallback: manual FFT-based power spectrum
    field_k = np.fft.fft2(delta)
    Pk_2d = np.abs(field_k)**2 * (box_size / ng)**2
    
    # Radial averaging
    kx = np.fft.fftfreq(ng, d=box_size/ng) * 2 * np.pi
    ky = np.fft.fftfreq(ng, d=box_size/ng) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    k_2d = np.sqrt(kx**2 + ky**2)
    
    k_bins = np.logspace(np.log10(2*np.pi/box_size), np.log10(np.pi*ng/box_size), N_ELL_BINS + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    Pk = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        mask = (k_2d >= k_bins[i]) & (k_2d < k_bins[i+1])
        if np.sum(mask) > 0:
            Pk[i] = np.mean(Pk_2d[mask])
    
    return k_centers, Pk


def compute_pdf(field, nbins=PDF_NBINS, log_min=LOG_DELTA_MIN, log_max=LOG_DELTA_MAX):
    """Compute PDF of the density field.
    
    Parameters
    ----------
    field : np.ndarray
        2D density field.
    nbins : int
        Number of bins for PDF.
    log_min, log_max : float
        Range for log10(1 + delta).
        
    Returns
    -------
    bin_centers : np.ndarray
        Bin centers in log10(1 + delta).
    pdf : np.ndarray
        Normalized PDF values.
    """
    # Convert to overdensity
    mean_val = np.mean(field)
    if mean_val > 0:
        delta = field / mean_val - 1.0
    else:
        delta = field - np.mean(field)
    
    # Compute log(1 + delta), handling negative values
    one_plus_delta = 1.0 + delta
    one_plus_delta = np.clip(one_plus_delta, 1e-10, None)  # Avoid log of zero/negative
    log_delta = np.log10(one_plus_delta)
    
    # Compute histogram
    bins = np.linspace(log_min, log_max, nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    hist, _ = np.histogram(log_delta.flatten(), bins=bins, density=True)
    
    return bin_centers, hist


def compute_variance(field):
    """Compute variance of overdensity field."""
    mean_val = np.mean(field)
    if mean_val > 0:
        delta = field / mean_val - 1.0
    else:
        delta = field - np.mean(field)
    return np.var(delta)


def compute_skewness(field):
    """Compute skewness of overdensity field."""
    mean_val = np.mean(field)
    if mean_val > 0:
        delta = field / mean_val - 1.0
    else:
        delta = field - np.mean(field)
    
    std = np.std(delta)
    if std == 0:
        return 0.0
    return np.mean(((delta - np.mean(delta)) / std)**3)


def compute_kurtosis(field):
    """Compute excess kurtosis of overdensity field."""
    mean_val = np.mean(field)
    if mean_val > 0:
        delta = field / mean_val - 1.0
    else:
        delta = field - np.mean(field)
    
    std = np.std(delta)
    if std == 0:
        return 0.0
    return np.mean(((delta - np.mean(delta)) / std)**4) - 3.0


# =============================================================================
# Stage 1: Compute statistics for all models
# =============================================================================

def compute_statistics_for_model(model_name, force_recompute=False):
    """Compute all statistics for a single model across all lens planes and snapshots."""
    
    cache_file = STATS_CACHE_DIR / f'{model_name}_density_stats.h5'
    
    if cache_file.exists() and not force_recompute:
        print(f"[Rank {rank}] Statistics already cached for {model_name}, skipping")
        sys.stdout.flush()
        return cache_file
    
    print(f"[Rank {rank}] Computing statistics for {model_name}...")
    sys.stdout.flush()
    
    t_start = time.time()
    
    # Storage for all statistics
    all_Pk = []
    all_pdf = []
    all_variance = []
    all_skewness = []
    all_kurtosis = []
    all_LP_ids = []
    all_lenspot_ids = []
    all_snapshot_ids = []
    
    k_bins = None
    pdf_bins = None
    
    n_success = 0
    n_fail = 0
    
    # Loop over all lens planes and lenspot files
    for LP_id in range(N_LENS_PLANES):
        LP_dir = LP_BASE / model_name / f'LP_{LP_id:02d}'
        
        if not LP_dir.exists():
            print(f"[Rank {rank}]   LP_{LP_id:02d} not found for {model_name}")
            sys.stdout.flush()
            continue
        
        for lenspot_id in range(N_LENSPOTS):
            fname = LP_dir / f'lenspot{lenspot_id:02d}.dat'
            
            if not fname.exists():
                n_fail += 1
                continue
            
            # Load field
            field = load_lenspot(fname, ng=LP_GRID)
            if field is None:
                n_fail += 1
                continue
            
            # Compute power spectrum
            k, Pk = compute_power_spectrum_2d(field, box_size=BOX_SIZE)
            if k is None or Pk is None:
                n_fail += 1
                continue
            
            if k_bins is None:
                k_bins = k
            
            # Compute PDF
            pdf_bin_centers, pdf = compute_pdf(field)
            if pdf_bins is None:
                pdf_bins = pdf_bin_centers
            
            # Compute moments
            var = compute_variance(field)
            skew = compute_skewness(field)
            kurt = compute_kurtosis(field)
            
            # Determine which snapshot this corresponds to
            snapshot_idx = lenspot_id // PLANES_PER_SNAPSHOT
            
            # Store
            all_Pk.append(Pk)
            all_pdf.append(pdf)
            all_variance.append(var)
            all_skewness.append(skew)
            all_kurtosis.append(kurt)
            all_LP_ids.append(LP_id)
            all_lenspot_ids.append(lenspot_id)
            all_snapshot_ids.append(snapshot_idx)
            
            n_success += 1
    
    if n_success == 0:
        print(f"[Rank {rank}] ERROR: No valid fields found for {model_name}")
        sys.stdout.flush()
        return None
    
    print(f"[Rank {rank}]   {model_name}: {n_success} succeeded, {n_fail} failed")
    sys.stdout.flush()
    
    # Convert to arrays
    all_Pk = np.array(all_Pk)
    all_pdf = np.array(all_pdf)
    all_variance = np.array(all_variance)
    all_skewness = np.array(all_skewness)
    all_kurtosis = np.array(all_kurtosis)
    all_LP_ids = np.array(all_LP_ids)
    all_lenspot_ids = np.array(all_lenspot_ids)
    all_snapshot_ids = np.array(all_snapshot_ids)
    
    # Save to HDF5
    try:
        with h5py.File(cache_file, 'w') as f:
            f.create_dataset('k', data=k_bins)
            f.create_dataset('pdf_bins', data=pdf_bins)
            f.create_dataset('Pk', data=all_Pk)
            f.create_dataset('pdf', data=all_pdf)
            f.create_dataset('variance', data=all_variance)
            f.create_dataset('skewness', data=all_skewness)
            f.create_dataset('kurtosis', data=all_kurtosis)
            f.create_dataset('LP_ids', data=all_LP_ids)
            f.create_dataset('lenspot_ids', data=all_lenspot_ids)
            f.create_dataset('snapshot_ids', data=all_snapshot_ids)
            
            f.attrs['model_name'] = model_name
            f.attrs['n_realizations'] = n_success
            f.attrs['n_failed'] = n_fail
            f.attrs['box_size'] = BOX_SIZE
            f.attrs['grid_size'] = LP_GRID
        
        t_elapsed = time.time() - t_start
        print(f"[Rank {rank}]   Saved to {cache_file.name} ({t_elapsed:.1f}s)")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"[Rank {rank}] ERROR saving {cache_file}: {e}")
        sys.stdout.flush()
        return None
    
    return cache_file


def compute_all_statistics_mpi(models=ALL_MODELS, force_recompute=False):
    """Stage 1: Compute statistics with MPI parallelization."""
    
    if rank == 0:
        print("="*70)
        print(f"STAGE 1: Computing density statistics with {size} MPI ranks")
        print(f"Total models: {len(models)}")
        print(f"Grid: {LP_GRID}, Box: {BOX_SIZE} Mpc/h")
        print(f"Lens planes: {N_LENS_PLANES}, Lenspots per LP: {N_LENSPOTS}")
        print("="*70)
        sys.stdout.flush()
    
    comm.Barrier()
    
    # Divide models among ranks
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
        compute_statistics_for_model(model, force_recompute=force_recompute)
    
    comm.Barrier()
    
    if rank == 0:
        print("\n" + "="*70)
        print("Stage 1 complete!")
        print("="*70)
        sys.stdout.flush()


# =============================================================================
# Stage 2: Response analysis
# =============================================================================

def load_cached_statistics(model_name):
    """Load pre-computed statistics for a model."""
    cache_file = STATS_CACHE_DIR / f'{model_name}_density_stats.h5'
    
    if not cache_file.exists():
        raise FileNotFoundError(f"Statistics not cached for {model_name}. Run Stage 1 first.")
    
    with h5py.File(cache_file, 'r') as f:
        return {
            'k': f['k'][:],
            'pdf_bins': f['pdf_bins'][:],
            'Pk': f['Pk'][:],
            'pdf': f['pdf'][:],
            'variance': f['variance'][:],
            'skewness': f['skewness'][:],
            'kurtosis': f['kurtosis'][:],
            'LP_ids': f['LP_ids'][:],
            'lenspot_ids': f['lenspot_ids'][:],
            'snapshot_ids': f['snapshot_ids'][:],
            'n_realizations': f.attrs['n_realizations'],
        }


def parse_model_name(model_name):
    """Parse model name to extract mass and radius info."""
    if not model_name.startswith('hydro_replace_'):
        return None
    
    parts = model_name.replace('hydro_replace_', '').split('_')
    
    try:
        M_lo = float(parts[1])
        M_hi_str = parts[3]
        M_hi = np.inf if M_hi_str == 'inf' else float(M_hi_str)
        R_factor = float(parts[5])
        
        return {
            'M_lo': M_lo,
            'M_hi': M_hi,
            'R_factor': R_factor,
            'is_cumulative': M_hi == np.inf,
        }
    except (IndexError, ValueError):
        return None


def compute_response_fraction(S_dmo, S_hydro, S_replace):
    """Compute response fraction F = (S_R - S_D) / (S_H - S_D)."""
    denom = S_hydro - S_dmo
    
    # Handle near-zero denominator
    with np.errstate(divide='ignore', invalid='ignore'):
        F = (S_replace - S_dmo) / denom
        F = np.where(np.abs(denom) < 1e-15 * np.abs(S_dmo), 0.0, F)
    
    return F


def run_response_analysis(stat_key='Pk'):
    """Run response analysis for a specific statistic.
    
    Parameters
    ----------
    stat_key : str
        Statistic to analyze: 'Pk', 'pdf', 'variance', 'skewness', 'kurtosis'
    """
    print(f"\n{'='*70}")
    print(f"Response analysis for: {stat_key}")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    # Load DMO and Hydro statistics
    try:
        dmo_stats = load_cached_statistics('dmo')
        hydro_stats = load_cached_statistics('hydro')
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return None
    
    # Get reference arrays
    if stat_key == 'Pk':
        x_axis = dmo_stats['k']
        x_label = 'k [h/Mpc]'
    elif stat_key == 'pdf':
        x_axis = dmo_stats['pdf_bins']
        x_label = 'log10(1 + delta)'
    else:
        x_axis = None
        x_label = None
    
    # Compute mean statistics for DMO and Hydro
    if stat_key in ['Pk', 'pdf']:
        S_dmo_mean = np.mean(dmo_stats[stat_key], axis=0)
        S_dmo_std = np.std(dmo_stats[stat_key], axis=0)
        S_hydro_mean = np.mean(hydro_stats[stat_key], axis=0)
        S_hydro_std = np.std(hydro_stats[stat_key], axis=0)
    else:
        S_dmo_mean = np.mean(dmo_stats[stat_key])
        S_dmo_std = np.std(dmo_stats[stat_key])
        S_hydro_mean = np.mean(hydro_stats[stat_key])
        S_hydro_std = np.std(hydro_stats[stat_key])
    
    print(f"\nDMO: {dmo_stats['n_realizations']} realizations")
    print(f"Hydro: {hydro_stats['n_realizations']} realizations")
    
    # Compute responses for all Replace models
    cumulative_responses = {}
    tile_responses = {}
    tile_responses_err = {}
    
    for model in CUMULATIVE_MODELS + DIFFERENTIAL_MODELS:
        try:
            model_stats = load_cached_statistics(model)
        except FileNotFoundError:
            print(f"  Skipping {model} (not cached)")
            continue
        
        model_info = parse_model_name(model)
        if model_info is None:
            continue
        
        # Compute mean for this model
        if stat_key in ['Pk', 'pdf']:
            S_model_mean = np.mean(model_stats[stat_key], axis=0)
            S_model_std = np.std(model_stats[stat_key], axis=0)
        else:
            S_model_mean = np.mean(model_stats[stat_key])
            S_model_std = np.std(model_stats[stat_key])
        
        # Compute response fraction
        F = compute_response_fraction(S_dmo_mean, S_hydro_mean, S_model_mean)
        
        # Propagate errors (simplified)
        if stat_key in ['Pk', 'pdf']:
            n_eff = min(dmo_stats['n_realizations'], hydro_stats['n_realizations'], 
                       model_stats['n_realizations'])
            F_err = np.sqrt(S_model_std**2 + S_dmo_std**2) / np.abs(S_hydro_mean - S_dmo_mean + 1e-15)
            F_err /= np.sqrt(n_eff)
        else:
            F_err = 0.0
        
        if model_info['is_cumulative']:
            cumulative_responses[model] = {
                'F': F,
                'F_err': F_err,
                'M_lo': model_info['M_lo'],
                'R_factor': model_info['R_factor'],
            }
        else:
            tile_responses[model] = F
            tile_responses_err[model] = F_err
    
    # Save results
    results = {
        'stat_key': stat_key,
        'x_axis': x_axis,
        'x_label': x_label,
        'S_dmo_mean': S_dmo_mean,
        'S_dmo_std': S_dmo_std,
        'S_hydro_mean': S_hydro_mean,
        'S_hydro_std': S_hydro_std,
        'cumulative_responses': cumulative_responses,
        'tile_responses': tile_responses,
        'tile_responses_err': tile_responses_err,
        'mass_bin_edges': MASS_BIN_EDGES,
        'alpha_edges': ALPHA_EDGES,
    }
    
    # Save to file
    save_path = OUTPUT_DIR / f'density_response_{stat_key}.npz'
    
    # Flatten nested dicts for npz
    flat_results = {
        'stat_key': stat_key,
        'S_dmo_mean': S_dmo_mean,
        'S_dmo_std': S_dmo_std,
        'S_hydro_mean': S_hydro_mean,
        'S_hydro_std': S_hydro_std,
        'mass_bin_edges': MASS_BIN_EDGES,
        'alpha_edges': ALPHA_EDGES,
    }
    if x_axis is not None:
        flat_results['x_axis'] = x_axis
    
    np.savez(save_path, **flat_results)
    
    # Save dicts as pickle
    import pickle
    pickle_path = OUTPUT_DIR / f'density_response_{stat_key}_dicts.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'cumulative_responses': cumulative_responses,
            'tile_responses': tile_responses,
            'tile_responses_err': tile_responses_err,
        }, f)
    
    print(f"\nSaved results to {save_path}")
    print(f"Saved dicts to {pickle_path}")
    sys.stdout.flush()
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Density response analysis')
    parser.add_argument('stage', nargs='?', default='both', choices=['1', '2', 'both'],
                        help='Stage to run: 1=compute stats, 2=response analysis, both=all')
    parser.add_argument('--force-recompute', action='store_true',
                        help='Force recomputation of cached statistics')
    args = parser.parse_args()
    stage = args.stage
    
    # Stage 1: Compute statistics (MPI parallel)
    if stage in ['1', 'both']:
        compute_all_statistics_mpi(
            models=ALL_MODELS,
            force_recompute=args.force_recompute
        )
    
    comm.Barrier()
    
    # Stage 2: Response analysis (rank 0 only)
    if stage in ['2', 'both']:
        if rank == 0:
            for stat_key in ['Pk', 'pdf', 'variance', 'skewness', 'kurtosis']:
                try:
                    results = run_response_analysis(stat_key=stat_key)
                    print(f"\n{stat_key} analysis complete!")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"\nERROR in {stat_key}: {e}")
                    sys.stdout.flush()
                    import traceback
                    traceback.print_exc()
    
    comm.Barrier()
    
    if rank == 0:
        print("\n" + "="*70)
        print("All density response analysis complete!")
        print("="*70)
        sys.stdout.flush()
