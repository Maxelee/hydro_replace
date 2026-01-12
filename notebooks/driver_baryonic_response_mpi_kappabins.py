"""
driver_baryonic_response_mpi_kappabins.py

MPI-parallelized analysis with KAPPA BINS (not SNR bins).
- Peaks: kappa bins from -0.05 to 0.2
- Minima: kappa bins from -0.07 to 0.07
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
STATS_CACHE_DIR = Path('./statistics_cache_kappabins')
if rank == 0:
    STATS_CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path('./response_analysis_output_kappabins')
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

# ============================================================================
# KAPPA BINS (not SNR bins!)
# ============================================================================
# Peaks: kappa from -0.05 to 0.2 (high convergence regions = clusters)
# Minima: kappa from -0.07 to 0.07 (void-like regions)

PEAK_KAPPA_BINS = np.linspace(-0.05, 0.2, 26)     # 25 bins
MINIMA_KAPPA_BINS = np.linspace(-0.07, 0.07, 29)  # 28 bins

PEAK_KAPPA_MID = 0.5 * (PEAK_KAPPA_BINS[:-1] + PEAK_KAPPA_BINS[1:])
MINIMA_KAPPA_MID = 0.5 * (MINIMA_KAPPA_BINS[:-1] + MINIMA_KAPPA_BINS[1:])

# PDF bins - covers the full range of kappa values
PDF_KAPPA_BINS = np.linspace(-0.1, 0.25, 71)  # 70 bins
PDF_KAPPA_MID = 0.5 * (PDF_KAPPA_BINS[:-1] + PDF_KAPPA_BINS[1:])

# Minkowski functional thresholds - same as PDF bins for consistency
MF_THRESHOLDS = np.linspace(-0.08, 0.15, 47)  # 46 thresholds

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

def compute_peaks_single_map(kappa, smoothing_pix=SMOOTHING_PIX, kappa_bins=PEAK_KAPPA_BINS):
    """Compute peak counts using KAPPA bins (not SNR).
    
    Peaks are local maxima in the smoothed kappa field.
    We bin by the actual kappa value, not SNR.
    """
    try:
        kappa_smooth = ndimage.gaussian_filter(kappa, smoothing_pix)
        
        data_max = ndimage.maximum_filter(kappa_smooth, size=3)
        peaks_mask = (kappa_smooth == data_max) & (kappa_smooth > kappa_bins[0])
        
        # Exclude edges
        peaks_mask[:2, :] = False
        peaks_mask[-2:, :] = False
        peaks_mask[:, :2] = False
        peaks_mask[:, -2:] = False
        
        if np.sum(peaks_mask) == 0:
            return np.zeros(len(kappa_bins) - 1)
        
        # Bin by kappa value directly
        peak_values = kappa_smooth[peaks_mask]
        peak_counts, _ = np.histogram(peak_values, bins=kappa_bins)
        
        return peak_counts
    except Exception as e:
        return np.zeros(len(kappa_bins) - 1)

def compute_minima_single_map(kappa, smoothing_pix=SMOOTHING_PIX, kappa_bins=MINIMA_KAPPA_BINS):
    """Compute minima counts using KAPPA bins (not SNR).
    
    Minima are local minima in the smoothed kappa field.
    We bin by the actual kappa value, not SNR.
    """
    try:
        kappa_smooth = ndimage.gaussian_filter(kappa, smoothing_pix)
        
        data_min = ndimage.minimum_filter(kappa_smooth, size=3)
        # Minima within the bin range
        minima_mask = (kappa_smooth == data_min)
        minima_mask &= (kappa_smooth >= kappa_bins[0]) & (kappa_smooth <= kappa_bins[-1])
        
        # Exclude edges
        minima_mask[:2, :] = False
        minima_mask[-2:, :] = False
        minima_mask[:, :2] = False
        minima_mask[:, -2:] = False
        
        if np.sum(minima_mask) == 0:
            return np.zeros(len(kappa_bins) - 1)
        
        # Bin by kappa value directly
        minima_values = kappa_smooth[minima_mask]
        minima_counts, _ = np.histogram(minima_values, bins=kappa_bins)
        
        return minima_counts
    except Exception as e:
        return np.zeros(len(kappa_bins) - 1)

def compute_pdf_single_map(kappa, smoothing_pix=SMOOTHING_PIX, kappa_bins=PDF_KAPPA_BINS):
    """Compute probability distribution function (PDF) of kappa.
    
    The PDF is the histogram of pixel values in the smoothed kappa field,
    normalized to integrate to 1.
    """
    try:
        kappa_smooth = ndimage.gaussian_filter(kappa, smoothing_pix)
        
        # Exclude edge pixels
        kappa_interior = kappa_smooth[5:-5, 5:-5].flatten()
        
        # Compute histogram (counts)
        pdf_counts, _ = np.histogram(kappa_interior, bins=kappa_bins)
        
        # Normalize to PDF (probability density)
        bin_width = kappa_bins[1] - kappa_bins[0]
        pdf = pdf_counts / (len(kappa_interior) * bin_width)
        
        return pdf
    except Exception as e:
        return np.zeros(len(kappa_bins) - 1)

def compute_minkowski_functionals_single_map(kappa, smoothing_pix=SMOOTHING_PIX, thresholds=MF_THRESHOLDS):
    """Compute Minkowski functionals V0, V1, V2 as a function of threshold.
    
    For a 2D field, the three Minkowski functionals are:
    - V0: Area (fraction of pixels above threshold)
    - V1: Perimeter (boundary length per unit area)
    - V2: Euler characteristic (# connected regions - # holes)
    
    These are powerful non-Gaussian statistics that capture morphological information.
    """
    try:
        kappa_smooth = ndimage.gaussian_filter(kappa, smoothing_pix)
        
        # Exclude edges
        kappa_interior = kappa_smooth[5:-5, 5:-5]
        n_pix = kappa_interior.size
        
        V0 = np.zeros(len(thresholds))
        V1 = np.zeros(len(thresholds))
        V2 = np.zeros(len(thresholds))
        
        for i, nu in enumerate(thresholds):
            # Binary field: 1 where kappa > threshold
            binary = (kappa_interior > nu).astype(np.int32)
            
            # V0: Area fraction
            V0[i] = np.sum(binary) / n_pix
            
            # V1: Perimeter - count boundary pixels
            # Use Sobel gradient magnitude as proxy for boundary
            grad_x = np.abs(np.diff(binary, axis=1))
            grad_y = np.abs(np.diff(binary, axis=0))
            perimeter = np.sum(grad_x) + np.sum(grad_y)
            V1[i] = perimeter / n_pix  # Normalize by area
            
            # V2: Euler characteristic using the formula for 2D binary images
            # χ = N_vertices - N_edges + N_faces (for connected components)
            # Simpler approximation: count connected components - holes
            from scipy import ndimage as ndi
            
            # Label connected regions above threshold
            labeled_above, n_above = ndi.label(binary)
            
            # Label connected regions below threshold (holes)
            labeled_below, n_below = ndi.label(1 - binary)
            
            # Euler characteristic: regions above - (regions below - 1)
            # The -1 accounts for the background
            V2[i] = n_above - (n_below - 1) if n_below > 0 else n_above
            V2[i] /= n_pix  # Normalize by area
        
        return V0, V1, V2
    except Exception as e:
        n_thresh = len(thresholds)
        return np.zeros(n_thresh), np.zeros(n_thresh), np.zeros(n_thresh)

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
    all_pdf = []
    all_V0 = []
    all_V1 = []
    all_V2 = []
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
            
            # Compute peaks and minima (using kappa bins)
            peaks = compute_peaks_single_map(kappa)
            minima = compute_minima_single_map(kappa)
            
            # Compute PDF
            pdf = compute_pdf_single_map(kappa, SMOOTHING_PIX, PDF_KAPPA_BINS)
            
            # Compute Minkowski functionals
            V0, V1, V2 = compute_minkowski_functionals_single_map(kappa, SMOOTHING_PIX, MF_THRESHOLDS)
            
            # Store successful computation
            all_C_ell.append(C_ell_map)
            all_peaks.append(peaks)
            all_minima.append(minima)
            all_pdf.append(pdf)
            all_V0.append(V0)
            all_V1.append(V1)
            all_V2.append(V2)
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
    all_pdf = np.array(all_pdf)
    all_V0 = np.array(all_V0)
    all_V1 = np.array(all_V1)
    all_V2 = np.array(all_V2)
    all_LP_ids = np.array(all_LP_ids)
    all_run_ids = np.array(all_run_ids)
    
    # Save to HDF5
    try:
        with h5py.File(cache_file, 'w') as f:
            f.create_dataset('ell', data=ell)
            f.create_dataset('peak_kappa_bins', data=PEAK_KAPPA_BINS)
            f.create_dataset('minima_kappa_bins', data=MINIMA_KAPPA_BINS)
            f.create_dataset('pdf_kappa_bins', data=PDF_KAPPA_BINS)
            f.create_dataset('mf_thresholds', data=MF_THRESHOLDS)
            f.create_dataset('C_ell', data=all_C_ell)
            f.create_dataset('peaks', data=all_peaks)
            f.create_dataset('minima', data=all_minima)
            f.create_dataset('pdf', data=all_pdf)
            f.create_dataset('V0', data=all_V0)
            f.create_dataset('V1', data=all_V1)
            f.create_dataset('V2', data=all_V2)
            f.create_dataset('LP_ids', data=all_LP_ids)
            f.create_dataset('run_ids', data=all_run_ids)
            
            f.attrs['model_name'] = model_name
            f.attrs['z'] = z
            f.attrs['n_realizations'] = n_success
            f.attrs['n_failed'] = n_fail
            f.attrs['binning_type'] = 'kappa'  # Flag for kappa bins
        
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
        print(f"Using KAPPA BINS (not SNR)")
        print(f"  Peaks: {PEAK_KAPPA_BINS[0]:.3f} to {PEAK_KAPPA_BINS[-1]:.3f} ({len(PEAK_KAPPA_BINS)-1} bins)")
        print(f"  Minima: {MINIMA_KAPPA_BINS[0]:.3f} to {MINIMA_KAPPA_BINS[-1]:.3f} ({len(MINIMA_KAPPA_BINS)-1} bins)")
        print(f"  PDF: {PDF_KAPPA_BINS[0]:.3f} to {PDF_KAPPA_BINS[-1]:.3f} ({len(PDF_KAPPA_BINS)-1} bins)")
        print(f"  MF thresholds: {MF_THRESHOLDS[0]:.3f} to {MF_THRESHOLDS[-1]:.3f} ({len(MF_THRESHOLDS)} values)")
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
        # Load kappa bins
        if 'peak_kappa_bins' in f:
            result['peak_kappa_bins'] = f['peak_kappa_bins'][:]
            result['minima_kappa_bins'] = f['minima_kappa_bins'][:]
        # Load PDF and Minkowski functionals
        if 'pdf' in f:
            result['pdf'] = f['pdf'][:]
            result['pdf_kappa_bins'] = f['pdf_kappa_bins'][:]
        if 'V0' in f:
            result['V0'] = f['V0'][:]
            result['V1'] = f['V1'][:]
            result['V2'] = f['V2'][:]
            result['mf_thresholds'] = f['mf_thresholds'][:]
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
    Delta_S_err = np.sqrt(S_D_err**2 + S_H_err**2)
    
    # -------------------------------------------------------------------------
    # Part 1: Differential tile responses
    # -------------------------------------------------------------------------
    print("\n--- Part 1: Differential (tile) responses ---")
    sys.stdout.flush()
    
    n_mass_bins = len(MASS_BIN_EDGES) - 1
    n_alpha_bins = len(ALPHA_EDGES) - 1
    n_stat_bins = len(S_D)
    
    delta_S_tiles = {}
    delta_S_tiles_err = {}
    
    for model_name in DIFFERENTIAL_MODELS:
        key = model_to_tile_key(model_name, MASS_BIN_EDGES, ALPHA_EDGES)
        if key is None:
            continue
        
        try:
            model_stats = load_cached_statistics(model_name, z=z)
            S_R_all = model_stats[stat_key]
            S_R = np.mean(S_R_all, axis=0)
            S_R_err = np.std(S_R_all, axis=0) / np.sqrt(len(S_R_all))
            
            delta_S = S_R - S_D
            delta_S_err = np.sqrt(S_R_err**2 + S_D_err**2)
            
            delta_S_tiles[key] = delta_S
            delta_S_tiles_err[key] = delta_S_err
            
            print(f"  {model_name}: tile {key}, mean delta = {np.nanmean(delta_S):.2e}")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"  WARNING: Could not load {model_name}: {e}")
            sys.stdout.flush()
    
    # Compute tile response fractions
    tile_responses = {}
    tile_responses_err = {}
    
    for key, delta_S in delta_S_tiles.items():
        F_S = delta_S / np.where(np.abs(Delta_S) > 1e-15, Delta_S, np.nan)
        
        delta_S_err = delta_S_tiles_err[key]
        F_S_err = np.abs(F_S) * np.sqrt(
            (delta_S_err / np.where(np.abs(delta_S) > 1e-15, delta_S, np.inf))**2 +
            (Delta_S_err / np.where(np.abs(Delta_S) > 1e-15, Delta_S, np.inf))**2
        )
        
        tile_responses[key] = F_S
        tile_responses_err[key] = F_S_err
    
    # -------------------------------------------------------------------------
    # Part 2: Cumulative responses
    # -------------------------------------------------------------------------
    print("\n--- Part 2: Cumulative responses F(M>M_min, α) ---")
    sys.stdout.flush()
    
    cumulative_responses = {}
    
    for model_name in CUMULATIVE_MODELS:
        config = model_to_cumulative_config(model_name)
        if config is None:
            continue
        
        M_min, alpha_max = config
        
        try:
            model_stats = load_cached_statistics(model_name, z=z)
            S_R_all = model_stats[stat_key]
            S_R = np.mean(S_R_all, axis=0)
            S_R_err = np.std(S_R_all, axis=0) / np.sqrt(len(S_R_all))
            
            delta_S_R = S_R - S_D
            delta_S_R_err = np.sqrt(S_R_err**2 + S_D_err**2)
            
            F_S = delta_S_R / np.where(np.abs(Delta_S) > 1e-15, Delta_S, np.nan)
            F_S_err = np.abs(F_S) * np.sqrt(
                (delta_S_R_err / np.where(np.abs(delta_S_R) > 1e-15, delta_S_R, np.inf))**2 +
                (Delta_S_err / np.where(np.abs(Delta_S) > 1e-15, Delta_S, np.inf))**2
            )
            
            key = f'M{M_min:.1e}_a{alpha_max}'
            cumulative_responses[key] = {
                'M_min': M_min,
                'alpha_max': alpha_max,
                'S_R': S_R,
                'S_R_err': S_R_err,
                'delta_S_R': delta_S_R,
                'F_S': F_S,
                'F_S_err': F_S_err,
            }
            
            print(f"  {model_name}: F_mean = {np.nanmean(F_S[np.isfinite(F_S)]):.2f}")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"  WARNING: Could not load {model_name}: {e}")
            sys.stdout.flush()
    
    # -------------------------------------------------------------------------
    # Part 3: Additivity test
    # -------------------------------------------------------------------------
    print("\n--- Part 3: Additivity test ---")
    sys.stdout.flush()
    
    additivity_results = {}
    
    for alpha_max in [0.5, 1.0, 3.0, 5.0]:
        alpha_idx = int(np.argmin(np.abs(np.array([0.5, 1.0, 3.0, 5.0]) - alpha_max)))
        
        sum_delta_S = np.zeros(n_stat_bins)
        
        for mass_idx in range(n_mass_bins):
            key = (mass_idx, alpha_idx)
            if key in delta_S_tiles:
                sum_delta_S += delta_S_tiles[key]
        
        direct_key = f'M{MASS_BIN_EDGES[0]:.1e}_a{alpha_max}'
        if direct_key in cumulative_responses:
            direct_delta_S = cumulative_responses[direct_key]['delta_S_R']
            
            additivity_results[alpha_max] = {
                'sum_tiles': sum_delta_S,
                'direct': direct_delta_S,
                'ratio': np.nanmean(sum_delta_S / np.where(np.abs(direct_delta_S) > 1e-15, direct_delta_S, np.nan)),
            }
            
            print(f"  α={alpha_max}: Σ(tiles)/direct = {additivity_results[alpha_max]['ratio']:.2f}")
            sys.stdout.flush()
    
    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    results = {
        'ell': dmo_stats['ell'] if stat_key == 'C_ell' else None,
        'S_D': S_D,
        'S_H': S_H,
        'Delta_S': Delta_S,
        'S_D_err': S_D_err,
        'S_H_err': S_H_err,
        'Delta_S_err': Delta_S_err,
    }
    
    # Add bin information
    if stat_key == 'peaks':
        results['kappa_bins'] = PEAK_KAPPA_BINS
        results['kappa_mid'] = PEAK_KAPPA_MID
    elif stat_key == 'minima':
        results['kappa_bins'] = MINIMA_KAPPA_BINS
        results['kappa_mid'] = MINIMA_KAPPA_MID
    elif stat_key == 'pdf':
        results['kappa_bins'] = PDF_KAPPA_BINS
        results['kappa_mid'] = PDF_KAPPA_MID
    elif stat_key in ['V0', 'V1', 'V2']:
        results['mf_thresholds'] = MF_THRESHOLDS
    
    save_path = OUTPUT_DIR / f'response_{stat_key}_z{z:02d}.npz'
    np.savez(save_path, **results)
    
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
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MPI-parallelized baryonic response analysis with KAPPA bins')
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
        print("Baryonic Response Analysis (KAPPA BINS)")
        print(f"Stage: {stage}")
        print(f"z snapshot: {Z_SNAP}")
        print(f"Force recompute: {force_recompute}")
        print(f"PEAK_KAPPA_BINS: {PEAK_KAPPA_BINS[0]:.3f} to {PEAK_KAPPA_BINS[-1]:.3f} ({len(PEAK_KAPPA_BINS)-1} bins)")
        print(f"MINIMA_KAPPA_BINS: {MINIMA_KAPPA_BINS[0]:.3f} to {MINIMA_KAPPA_BINS[-1]:.3f} ({len(MINIMA_KAPPA_BINS)-1} bins)")
        print(f"PDF_KAPPA_BINS: {PDF_KAPPA_BINS[0]:.3f} to {PDF_KAPPA_BINS[-1]:.3f} ({len(PDF_KAPPA_BINS)-1} bins)")
        print(f"MF_THRESHOLDS: {MF_THRESHOLDS[0]:.3f} to {MF_THRESHOLDS[-1]:.3f} ({len(MF_THRESHOLDS)} values)")
        print("="*70)
        sys.stdout.flush()
    
    if stage in ['1', 'both']:
        compute_all_statistics_all_models_mpi(
            models=ALL_MODELS,
            z=Z_SNAP,
            force_recompute=force_recompute
        )
    
    comm.Barrier()
    
    # Stage 2: Only rank 0
    if stage in ['2', 'both']:
        if rank == 0:
            for stat_key in ['C_ell', 'peaks', 'minima', 'pdf', 'V0', 'V1', 'V2']:
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
