#!/usr/bin/env python
"""
Unified statistics pipeline: compute all statistics for a model.

Computes density (P(k)) and convergence (C_ℓ, peaks, minima, PDF) statistics
for a single model and saves to a unified HDF5 file.

Usage:
    mpirun -np 64 python compute_all_stats.py --model MODEL_NAME
    mpirun -np 64 python compute_all_stats.py --model dmo
    mpirun -np 64 python compute_all_stats.py --model hydro
    
    # Disable specific statistics:
    mpirun -np 64 python compute_all_stats.py --model dmo --no-bispectrum
    mpirun -np 64 python compute_all_stats.py --model dmo --no-wst --no-bispectrum
    mpirun -np 64 python compute_all_stats.py --model dmo --no-density  # skip P(k)
"""

import os
import sys
import argparse
import numpy as np
import h5py
from mpi4py import MPI

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import (
    LP_BASE, RT_BASE, STATS_BASE, N_LP, N_RUNS, N_LENSPLANES,
    SNAPSHOT_ORDER, SNAPSHOT_REDSHIFTS, KAPPA_TARGETS, KAPPA_REDSHIFTS,
    SN_BINS, SMOOTHING_ARCMIN, PIXEL_SCALE_ARCMIN, BOX_SIZE, FOV_DEG,
    WST_J, WST_Q, WST_MOMENTS, BISPEC_ELL_BINS, BISPEC_CONFIGS
)
from stats_utils import (
    load_kappa, read_lensplane, compute_Pk_2d, compute_Cl,
    smooth_kappa, compute_peaks, compute_minima, compute_pdf,
    compute_minkowski_functionals, compute_wavelet_scattering_2d,
    compute_all_bispectra
)

# Global flags for which statistics to compute (set by argparse)
COMPUTE_FLAGS = {
    'density': True,
    'Cl': True,
    'peaks': True,
    'minima': True,
    'pdf': True,
    'minkowski': True,
    'wst': True,
    'bispectrum': True,
}


def load_dmo_rms():
    """Load pre-computed DMO RMS values."""
    rms_path = os.path.join(STATS_BASE, 'dmo_rms.h5')
    if not os.path.exists(rms_path):
        raise FileNotFoundError(
            f"DMO RMS file not found: {rms_path}\n"
            "Please run compute_dmo_rms.py first."
        )
    
    with h5py.File(rms_path, 'r') as f:
        rms = f['rms'][:]
        return rms


def get_lensplane_path(model, plane_idx, lp):
    """Get path to lensplane file.
    
    Lensplanes are organized as: /LP_BASE/model/LP_XX/lenspotYY.dat
    where YY is the plane index (0-39, 40 total lensplanes per LP)
    """
    lp_dir = os.path.join(LP_BASE, model, f'LP_{lp:02d}')
    lenspot_file = f'lenspot{plane_idx:02d}.dat'
    
    return os.path.join(lp_dir, lenspot_file)


def get_kappa_path(model, lp, run, kappa_num):
    """Get path to convergence map."""
    lp_str = f"LP_{lp:02d}"
    run_str = f"run{run:03d}"
    kappa_file = f"kappa{kappa_num:02d}.dat"
    
    return os.path.join(RT_BASE, model, lp_str, run_str, kappa_file)


def compute_density_stats(model, comm):
    """
    Compute density statistics (P(k)) from lensplanes.
    
    Returns
    -------
    Pk_data : np.ndarray or None
        Shape (N_LP, N_snap, N_k), only on rank 0
    k_bins : np.ndarray or None
        Wavenumber bins, only on rank 0
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Computing density statistics (P(k))...", flush=True)
    
    # Build task list: (lp, plane_idx)
    # There are 40 lensplanes per LP (lenspot00.dat to lenspot39.dat)
    tasks = []
    for lp in range(N_LP):
        for plane_idx in range(N_LENSPLANES):
            tasks.append((lp, plane_idx))
    
    # Distribute tasks
    my_tasks = tasks[rank::size]
    
    # Storage for local results
    local_Pk_list = []
    local_k_list = []
    
    # Process tasks
    for task_idx, (lp, plane_idx) in enumerate(my_tasks):
        if task_idx % 10 == 0 and task_idx > 0 and rank == 0:
            print(f"  Density: {task_idx}/{len(my_tasks)} tasks", flush=True)
        
        lp_path = get_lensplane_path(model, plane_idx, lp)
        
        if not os.path.exists(lp_path):
            if rank == 0 and task_idx == 0:
                print(f"  WARNING: Missing {lp_path}", flush=True)
            local_Pk_list.append((lp, plane_idx, None, None))
            continue
        
        try:
            # Load lensplane (binary format, not npz)
            field = read_lensplane(lp_path)
            
            # Compute P(k)
            k, Pk = compute_Pk_2d(field, BOX_SIZE)
            
            local_Pk_list.append((lp, plane_idx, k, Pk))
            
        except Exception as e:
            if rank == 0:
                print(f"  ERROR: {lp_path}: {e}", flush=True)
            local_Pk_list.append((lp, plane_idx, None, None))
    
    # Gather results
    all_results = comm.gather(local_Pk_list, root=0)
    
    if rank == 0:
        # Combine results
        # First pass: determine k bins (should be same for all)
        k_bins = None
        for results in all_results:
            for lp, plane_idx, k, Pk in results:
                if k is not None:
                    k_bins = k
                    break
            if k_bins is not None:
                break
        
        if k_bins is None:
            print("  ERROR: No valid P(k) computed!", flush=True)
            return None, None
        
        # Initialize output array
        n_k = len(k_bins)
        Pk_data = np.zeros((N_LP, N_LENSPLANES, n_k), dtype=np.float32)
        Pk_data[:] = np.nan
        
        # Fill in results
        for results in all_results:
            for lp, plane_idx, k, Pk in results:
                if Pk is not None:
                    Pk_data[lp, plane_idx, :] = Pk
        
        print(f"  Density stats complete. Shape: {Pk_data.shape}", flush=True)
        return Pk_data, k_bins
    
    return None, None


def compute_convergence_stats(model, dmo_rms, comm):
    """
    Compute convergence statistics (C_ℓ, peaks, minima, PDF, MF, WST, bispectrum) from kappa maps.
    
    Respects COMPUTE_FLAGS to selectively compute statistics.
    
    Parameters
    ----------
    dmo_rms : np.ndarray
        Pre-computed DMO RMS values, shape (N_LP, N_RUNS, N_z)
    
    Returns
    -------
    results : dict or None
        Dictionary with computed statistics. Only on rank 0.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        enabled = [k for k in ['Cl', 'peaks', 'minima', 'pdf', 'minkowski', 'wst', 'bispectrum'] 
                   if COMPUTE_FLAGS.get(k, True)]
        print(f"Computing convergence statistics: {', '.join(enabled)}...", flush=True)
    
    # Build task list: (lp, run, z_idx, kappa_num)
    tasks = []
    z_names = sorted(KAPPA_TARGETS.keys())
    for lp in range(N_LP):
        for run in range(1, N_RUNS + 1):
            for z_idx, z_name in enumerate(z_names):
                kappa_num = KAPPA_TARGETS[z_name]
                tasks.append((lp, run, z_idx, kappa_num))
    
    # Distribute tasks
    my_tasks = tasks[rank::size]
    
    # Storage for local results
    local_results = []
    
    # Process tasks
    for task_idx, (lp, run, z_idx, kappa_num) in enumerate(my_tasks):
        if task_idx % 100 == 0 and task_idx > 0 and rank == 0:
            print(f"  Convergence: {task_idx}/{len(my_tasks)} tasks", flush=True)
        
        kappa_path = get_kappa_path(model, lp, run, kappa_num)
        
        if not os.path.exists(kappa_path):
            local_results.append((lp, run, z_idx, None, None, None, None, None, None, None, None, None, None, None))
            continue
        
        try:
            # Load convergence map
            kappa = load_kappa(kappa_path)
            
            # Get DMO RMS for this realization
            rms = dmo_rms[lp, run - 1, z_idx]
            if np.isnan(rms):
                print(f"  WARNING: NaN RMS for LP{lp:02d} run{run:03d} z{z_idx}", flush=True)
                local_results.append((lp, run, z_idx, None, None, None, None, None, None, None, None, None, None, None))
                continue
            
            # Compute C_ℓ (if enabled)
            ell, Cl = (None, None)
            if COMPUTE_FLAGS.get('Cl', True):
                ell, Cl = compute_Cl(kappa, FOV_DEG)
            
            # Smooth for other statistics (needed if any threshold-based stat is enabled)
            need_smooth = any(COMPUTE_FLAGS.get(k, True) for k in ['peaks', 'minima', 'pdf', 'minkowski', 'wst'])
            kappa_smooth = smooth_kappa(kappa, SMOOTHING_ARCMIN, PIXEL_SCALE_ARCMIN) if need_smooth else None
            
            # Compute WL statistics (if enabled)
            peaks = compute_peaks(kappa_smooth, rms, SN_BINS) if COMPUTE_FLAGS.get('peaks', True) and kappa_smooth is not None else None
            minima = compute_minima(kappa_smooth, rms, SN_BINS) if COMPUTE_FLAGS.get('minima', True) and kappa_smooth is not None else None
            pdf = compute_pdf(kappa_smooth, rms, SN_BINS) if COMPUTE_FLAGS.get('pdf', True) and kappa_smooth is not None else None
            
            # Compute Minkowski functionals (if enabled)
            V0, V1, V2 = (None, None, None)
            if COMPUTE_FLAGS.get('minkowski', True) and kappa_smooth is not None:
                mf_thresholds = np.linspace(-5, 10, 51)  # 51 thresholds from -5σ to 10σ
                V0, V1, V2 = compute_minkowski_functionals(kappa_smooth, rms, mf_thresholds)
            
            # Compute Wavelet Scattering Transform (if enabled)
            S0, S1 = (None, None)
            if COMPUTE_FLAGS.get('wst', True) and kappa_smooth is not None:
                kappa_norm = kappa_smooth / rms  # Normalize by DMO RMS for consistency
                S0, S1, scales = compute_wavelet_scattering_2d(kappa_norm, J=WST_J, Q=WST_Q, moments=WST_MOMENTS)
            
            # Compute bispectra (if enabled) - this is the expensive one
            bispectra = None
            if COMPUTE_FLAGS.get('bispectrum', True):
                bispec_ell, bispectra = compute_all_bispectra(kappa, FOV_DEG, BISPEC_ELL_BINS)
            
            local_results.append((lp, run, z_idx, ell, Cl, peaks, minima, pdf, V0, V1, V2, S0, S1, bispectra))
            
        except Exception as e:
            if rank == 0 and task_idx < 5:
                print(f"  ERROR: {kappa_path}: {e}", flush=True)
            local_results.append((lp, run, z_idx, None, None, None, None, None, None, None, None, None, None, None))
    
    # Gather results
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        # Determine dimensions from constants (don't rely on first valid result)
        n_z = len(z_names)
        n_real = N_LP * N_RUNS
        n_sn = len(SN_BINS) - 1
        mf_thresholds = np.linspace(-5, 10, 51)
        n_mf = len(mf_thresholds)
        n_wst_scales = WST_J * WST_Q
        n_moments = len(WST_MOMENTS)
        wst_scales = 2.0 ** (np.arange(WST_J * WST_Q) / WST_Q)
        bispec_ell = np.sqrt(BISPEC_ELL_BINS[:-1] * BISPEC_ELL_BINS[1:])
        n_bispec_ell = len(bispec_ell)
        
        # Find ell_bins from first valid C_ℓ result (if computed)
        ell_bins = None
        if COMPUTE_FLAGS.get('Cl', True):
            for results in all_results:
                for lp, run, z_idx, ell, Cl, *rest in results:
                    if ell is not None:
                        ell_bins = ell
                        break
                if ell_bins is not None:
                    break
            if ell_bins is None:
                print("  WARNING: No valid C_ℓ computed!", flush=True)
                # Create dummy ell_bins
                ell_bins = np.arange(100, 10001, 50)
        
        n_ell = len(ell_bins) if ell_bins is not None else 0
        
        # Initialize output arrays only for enabled statistics
        result_dict = {
            'ell_bins': ell_bins,
            'mf_thresholds': mf_thresholds,
            'wst_scales': wst_scales,
            'wst_moments': np.array(WST_MOMENTS),
            'bispec_ell': bispec_ell,
        }
        
        if COMPUTE_FLAGS.get('Cl', True):
            Cl_data = np.full((n_real, n_z, n_ell), np.nan, dtype=np.float32)
            result_dict['Cl'] = Cl_data
        
        if COMPUTE_FLAGS.get('peaks', True):
            peaks_data = np.full((n_real, n_z, n_sn), np.nan, dtype=np.float32)
            result_dict['peaks'] = peaks_data
        
        if COMPUTE_FLAGS.get('minima', True):
            minima_data = np.full((n_real, n_z, n_sn), np.nan, dtype=np.float32)
            result_dict['minima'] = minima_data
        
        if COMPUTE_FLAGS.get('pdf', True):
            pdf_data = np.full((n_real, n_z, n_sn), np.nan, dtype=np.float32)
            result_dict['pdf'] = pdf_data
        
        if COMPUTE_FLAGS.get('minkowski', True):
            V0_data = np.full((n_real, n_z, n_mf), np.nan, dtype=np.float32)
            V1_data = np.full((n_real, n_z, n_mf), np.nan, dtype=np.float32)
            V2_data = np.full((n_real, n_z, n_mf), np.nan, dtype=np.float32)
            result_dict['V0'] = V0_data
            result_dict['V1'] = V1_data
            result_dict['V2'] = V2_data
        
        if COMPUTE_FLAGS.get('wst', True):
            S0_data = np.full((n_real, n_z, n_moments), np.nan, dtype=np.float32)
            S1_data = np.full((n_real, n_z, n_wst_scales, n_moments), np.nan, dtype=np.float32)
            result_dict['S0'] = S0_data
            result_dict['S1'] = S1_data
        
        if COMPUTE_FLAGS.get('bispectrum', True):
            for config in BISPEC_CONFIGS:
                result_dict[f'bispec_{config}'] = np.full((n_real, n_z, n_bispec_ell), np.nan, dtype=np.float32)
        
        # Fill in results
        for results in all_results:
            for lp, run, z_idx, ell, Cl, peaks, minima, pdf, V0, V1, V2, S0, S1, bispectra in results:
                real_idx = lp * N_RUNS + (run - 1)
                
                if COMPUTE_FLAGS.get('Cl', True) and Cl is not None:
                    result_dict['Cl'][real_idx, z_idx, :] = Cl
                
                if COMPUTE_FLAGS.get('peaks', True) and peaks is not None:
                    result_dict['peaks'][real_idx, z_idx, :] = peaks
                
                if COMPUTE_FLAGS.get('minima', True) and minima is not None:
                    result_dict['minima'][real_idx, z_idx, :] = minima
                
                if COMPUTE_FLAGS.get('pdf', True) and pdf is not None:
                    result_dict['pdf'][real_idx, z_idx, :] = pdf
                
                if COMPUTE_FLAGS.get('minkowski', True) and V0 is not None:
                    result_dict['V0'][real_idx, z_idx, :] = V0
                    result_dict['V1'][real_idx, z_idx, :] = V1
                    result_dict['V2'][real_idx, z_idx, :] = V2
                
                if COMPUTE_FLAGS.get('wst', True) and S0 is not None:
                    result_dict['S0'][real_idx, z_idx, :] = S0
                    result_dict['S1'][real_idx, z_idx, :, :] = S1
                
                if COMPUTE_FLAGS.get('bispectrum', True) and bispectra is not None:
                    for config in BISPEC_CONFIGS:
                        result_dict[f'bispec_{config}'][real_idx, z_idx, :] = bispectra[config]
        
        # Print summary
        computed = []
        if COMPUTE_FLAGS.get('Cl', True) and 'Cl' in result_dict:
            computed.append(f"Cl={result_dict['Cl'].shape}")
        if COMPUTE_FLAGS.get('peaks', True) and 'peaks' in result_dict:
            computed.append(f"peaks={result_dict['peaks'].shape}")
        if COMPUTE_FLAGS.get('minkowski', True) and 'V0' in result_dict:
            computed.append(f"MF={result_dict['V0'].shape}")
        if COMPUTE_FLAGS.get('wst', True) and 'S1' in result_dict:
            computed.append(f"WST_S1={result_dict['S1'].shape}")
        if COMPUTE_FLAGS.get('bispectrum', True):
            computed.append(f"bispec={n_bispec_ell} ell bins")
        
        print(f"  Convergence stats complete. Shapes: {', '.join(computed)}", flush=True)
        
        return result_dict
    
    return None


def save_results(model, Pk_data, k_bins, conv_results):
    """Save all statistics to HDF5. Only saves computed statistics."""
    output_dir = os.path.join(STATS_BASE, model)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'stats.h5')
    
    with h5py.File(output_path, 'w') as f:
        # Density statistics (if computed)
        if Pk_data is not None and k_bins is not None:
            f.create_dataset('Pk', data=Pk_data, dtype='float32', compression='gzip')
            f.create_dataset('k_bins', data=k_bins, dtype='float32')
            f.attrs['snapshot_order'] = SNAPSHOT_ORDER
            f.attrs['snapshot_redshifts'] = SNAPSHOT_REDSHIFTS
        
        # Convergence statistics (only save what was computed)
        if conv_results is not None:
            # C_ℓ
            if 'Cl' in conv_results:
                f.create_dataset('Cl', data=conv_results['Cl'], dtype='float32', compression='gzip')
                f.create_dataset('ell_bins', data=conv_results['ell_bins'], dtype='float32')
            
            # Peaks, minima, PDF
            if 'peaks' in conv_results:
                f.create_dataset('peaks', data=conv_results['peaks'], dtype='float32', compression='gzip')
            if 'minima' in conv_results:
                f.create_dataset('minima', data=conv_results['minima'], dtype='float32', compression='gzip')
            if 'pdf' in conv_results:
                f.create_dataset('pdf', data=conv_results['pdf'], dtype='float32', compression='gzip')
            
            # S/N bin edges (if any threshold-based stat was computed)
            if any(k in conv_results for k in ['peaks', 'minima', 'pdf']):
                f.create_dataset('sn_bin_edges', data=SN_BINS, dtype='float32')
            
            # Minkowski functionals
            if 'V0' in conv_results:
                f.create_dataset('V0', data=conv_results['V0'], dtype='float32', compression='gzip')
                f.create_dataset('V1', data=conv_results['V1'], dtype='float32', compression='gzip')
                f.create_dataset('V2', data=conv_results['V2'], dtype='float32', compression='gzip')
                f.create_dataset('mf_thresholds', data=conv_results['mf_thresholds'], dtype='float32')
            
            # Wavelet Scattering Transform
            if 'S0' in conv_results:
                f.create_dataset('S0', data=conv_results['S0'], dtype='float32', compression='gzip')
                f.create_dataset('S1', data=conv_results['S1'], dtype='float32', compression='gzip')
                f.create_dataset('wst_scales', data=conv_results['wst_scales'], dtype='float32')
                f.create_dataset('wst_moments', data=conv_results['wst_moments'], dtype='float32')
                f.attrs['WST_J'] = WST_J
                f.attrs['WST_Q'] = WST_Q
            
            # Bispectrum (all triangle configurations, if computed)
            if f'bispec_{BISPEC_CONFIGS[0]}' in conv_results:
                f.create_dataset('bispec_ell', data=conv_results['bispec_ell'], dtype='float32')
                for config in BISPEC_CONFIGS:
                    f.create_dataset(f'bispec_{config}', data=conv_results[f'bispec_{config}'], 
                                    dtype='float32', compression='gzip')
                f.attrs['bispec_configs'] = BISPEC_CONFIGS
            
            # Source redshifts
            z_names = sorted(KAPPA_TARGETS.keys())
            source_z = [KAPPA_REDSHIFTS[KAPPA_TARGETS[z]] for z in z_names]
            f.attrs['source_redshifts'] = source_z
            f.attrs['kappa_targets'] = [KAPPA_TARGETS[z] for z in z_names]
        
        # Metadata
        f.attrs['model'] = model
        f.attrs['N_LP'] = N_LP
        f.attrs['N_RUNS'] = N_RUNS
        f.attrs['smoothing_arcmin'] = SMOOTHING_ARCMIN
        f.attrs['BOX_SIZE'] = BOX_SIZE
        f.attrs['FOV_DEG'] = FOV_DEG
        
        # Record which statistics were computed
        f.attrs['computed_density'] = COMPUTE_FLAGS['density']
        f.attrs['computed_Cl'] = COMPUTE_FLAGS['Cl']
        f.attrs['computed_peaks'] = COMPUTE_FLAGS['peaks']
        f.attrs['computed_minima'] = COMPUTE_FLAGS['minima']
        f.attrs['computed_pdf'] = COMPUTE_FLAGS['pdf']
        f.attrs['computed_minkowski'] = COMPUTE_FLAGS['minkowski']
        f.attrs['computed_wst'] = COMPUTE_FLAGS['wst']
        f.attrs['computed_bispectrum'] = COMPUTE_FLAGS['bispectrum']
    
    print(f"\nResults saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Compute all statistics for a model')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., dmo, hydro, or replace model)')
    
    # Flags to enable/disable individual statistics
    parser.add_argument('--density', action='store_true', dest='density', default=True,
                        help='Compute density P(k) (default: enabled)')
    parser.add_argument('--no-density', action='store_false', dest='density',
                        help='Skip density P(k) computation')
    
    parser.add_argument('--cl', action='store_true', dest='Cl', default=True,
                        help='Compute angular power spectrum C_ℓ (default: enabled)')
    parser.add_argument('--no-cl', action='store_false', dest='Cl',
                        help='Skip C_ℓ computation')
    
    parser.add_argument('--peaks', action='store_true', dest='peaks', default=True,
                        help='Compute peak counts (default: enabled)')
    parser.add_argument('--no-peaks', action='store_false', dest='peaks',
                        help='Skip peak counts')
    
    parser.add_argument('--minima', action='store_true', dest='minima', default=True,
                        help='Compute minima counts (default: enabled)')
    parser.add_argument('--no-minima', action='store_false', dest='minima',
                        help='Skip minima counts')
    
    parser.add_argument('--pdf', action='store_true', dest='pdf', default=True,
                        help='Compute PDF (default: enabled)')
    parser.add_argument('--no-pdf', action='store_false', dest='pdf',
                        help='Skip PDF computation')
    
    parser.add_argument('--minkowski', action='store_true', dest='minkowski', default=True,
                        help='Compute Minkowski functionals (default: enabled)')
    parser.add_argument('--no-minkowski', action='store_false', dest='minkowski',
                        help='Skip Minkowski functionals')
    
    parser.add_argument('--wst', action='store_true', dest='wst', default=True,
                        help='Compute Wavelet Scattering Transform (default: enabled)')
    parser.add_argument('--no-wst', action='store_false', dest='wst',
                        help='Skip WST computation')
    
    parser.add_argument('--bispectrum', action='store_true', dest='bispectrum', default=True,
                        help='Compute bispectrum (default: enabled)')
    parser.add_argument('--no-bispectrum', action='store_false', dest='bispectrum',
                        help='Skip bispectrum computation (saves significant time)')
    
    args = parser.parse_args()
    
    # Update global flags
    global COMPUTE_FLAGS
    COMPUTE_FLAGS['density'] = args.density
    COMPUTE_FLAGS['Cl'] = args.Cl
    COMPUTE_FLAGS['peaks'] = args.peaks
    COMPUTE_FLAGS['minima'] = args.minima
    COMPUTE_FLAGS['pdf'] = args.pdf
    COMPUTE_FLAGS['minkowski'] = args.minkowski
    COMPUTE_FLAGS['wst'] = args.wst
    COMPUTE_FLAGS['bispectrum'] = args.bispectrum
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=" * 80)
        print(f"Computing statistics for model: {args.model}")
        print("=" * 80)
        print(f"MPI size: {size}")
        print()
        # Show which statistics will be computed
        enabled = [k for k, v in COMPUTE_FLAGS.items() if v]
        disabled = [k for k, v in COMPUTE_FLAGS.items() if not v]
        print(f"Enabled statistics: {', '.join(enabled)}")
        if disabled:
            print(f"Disabled statistics: {', '.join(disabled)}")
        print()
    
    # Load DMO RMS
    if rank == 0:
        print("Loading DMO RMS...")
        dmo_rms = load_dmo_rms()
        print(f"DMO RMS loaded. Shape: {dmo_rms.shape}")
        print()
    else:
        dmo_rms = None
    
    # Broadcast DMO RMS to all ranks
    dmo_rms = comm.bcast(dmo_rms, root=0)
    
    # Compute density statistics (if enabled)
    if COMPUTE_FLAGS['density']:
        Pk_data, k_bins = compute_density_stats(args.model, comm)
    else:
        Pk_data, k_bins = None, None
        if rank == 0:
            print("Skipping density P(k) computation")
    
    # Compute convergence statistics (if any are enabled)
    conv_stats_enabled = any(COMPUTE_FLAGS[k] for k in ['Cl', 'peaks', 'minima', 'pdf', 'minkowski', 'wst', 'bispectrum'])
    if conv_stats_enabled:
        conv_results = compute_convergence_stats(args.model, dmo_rms, comm)
    else:
        conv_results = None
        if rank == 0:
            print("Skipping all convergence statistics")
    
    # Save results
    if rank == 0:
        save_results(args.model, Pk_data, k_bins, conv_results)
        print()
        print("=" * 80)
        print(f"Statistics computation complete for {args.model}!")
        print("=" * 80)


if __name__ == '__main__':
    main()
