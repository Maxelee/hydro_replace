#!/usr/bin/env python
"""
Unified statistics pipeline: compute all statistics for a model.

Computes density (P(k)) and convergence (C_ℓ, peaks, minima, PDF) statistics
for a single model and saves to a unified HDF5 file.

Usage:
    mpirun -np 64 python compute_all_stats.py --model MODEL_NAME
    mpirun -np 64 python compute_all_stats.py --model dmo
    mpirun -np 64 python compute_all_stats.py --model hydro
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
    LP_BASE, RT_BASE, STATS_BASE, N_LP, N_RUNS,
    SNAPSHOT_ORDER, SNAPSHOT_REDSHIFTS, KAPPA_TARGETS, KAPPA_REDSHIFTS,
    SN_BINS, SMOOTHING_ARCMIN, PIXEL_SCALE_ARCMIN, BOX_SIZE, FOV_DEG
)
from stats_utils import (
    load_kappa, read_lensplane, compute_Pk_2d, compute_Cl,
    smooth_kappa, compute_peaks, compute_minima, compute_pdf
)


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


def get_lensplane_path(model, snap_idx, lp):
    """Get path to lensplane file.
    
    Lensplanes are organized as: /LP_BASE/model/LP_XX/lenspotYY.dat
    where YY is the snapshot index (0-19 corresponding to SNAPSHOT_ORDER)
    """
    lp_dir = os.path.join(LP_BASE, model, f'LP_{lp:02d}')
    lenspot_file = f'lenspot{snap_idx:02d}.dat'
    
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
    
    # Build task list: (lp, snap_idx)
    tasks = []
    for lp in range(N_LP):
        for snap_idx in range(len(SNAPSHOT_ORDER)):
            tasks.append((lp, snap_idx))
    
    # Distribute tasks
    my_tasks = tasks[rank::size]
    
    # Storage for local results
    local_Pk_list = []
    local_k_list = []
    
    # Process tasks
    for task_idx, (lp, snap_idx) in enumerate(my_tasks):
        if task_idx % 10 == 0 and task_idx > 0 and rank == 0:
            print(f"  Density: {task_idx}/{len(my_tasks)} tasks", flush=True)
        
        lp_path = get_lensplane_path(model, snap_idx, lp)
        
        if not os.path.exists(lp_path):
            if rank == 0 and task_idx == 0:
                print(f"  WARNING: Missing {lp_path}", flush=True)
            local_Pk_list.append((lp, snap_idx, None, None))
            continue
        
        try:
            # Load lensplane (binary format, not npz)
            field = read_lensplane(lp_path)
            
            # Compute P(k)
            k, Pk = compute_Pk_2d(field, BOX_SIZE)
            
            local_Pk_list.append((lp, snap_idx, k, Pk))
            
        except Exception as e:
            if rank == 0:
                print(f"  ERROR: {lp_path}: {e}", flush=True)
            local_Pk_list.append((lp, snap_idx, None, None))
    
    # Gather results
    all_results = comm.gather(local_Pk_list, root=0)
    
    if rank == 0:
        # Combine results
        # First pass: determine k bins (should be same for all)
        k_bins = None
        for results in all_results:
            for lp, snap_idx, k, Pk in results:
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
        Pk_data = np.zeros((N_LP, len(SNAPSHOT_ORDER), n_k), dtype=np.float32)
        Pk_data[:] = np.nan
        
        # Fill in results
        for results in all_results:
            for lp, snap_idx, k, Pk in results:
                if Pk is not None:
                    Pk_data[lp, snap_idx, :] = Pk
        
        print(f"  Density stats complete. Shape: {Pk_data.shape}", flush=True)
        return Pk_data, k_bins
    
    return None, None


def compute_convergence_stats(model, dmo_rms, comm):
    """
    Compute convergence statistics (C_ℓ, peaks, minima, PDF) from kappa maps.
    
    Parameters
    ----------
    dmo_rms : np.ndarray
        Pre-computed DMO RMS values, shape (N_LP, N_RUNS, N_z)
    
    Returns
    -------
    results : dict or None
        Dictionary with keys: 'Cl', 'peaks', 'minima', 'pdf', 'ell_bins'
        Only on rank 0
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Computing convergence statistics...", flush=True)
    
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
            local_results.append((lp, run, z_idx, None, None, None, None, None))
            continue
        
        try:
            # Load convergence map
            kappa = load_kappa(kappa_path)
            
            # Smooth
            kappa_smooth = smooth_kappa(kappa, SMOOTHING_ARCMIN, PIXEL_SCALE_ARCMIN)
            
            # Get DMO RMS for this realization
            rms = dmo_rms[lp, run - 1, z_idx]
            if np.isnan(rms):
                print(f"  WARNING: NaN RMS for LP{lp:02d} run{run:03d} z{z_idx}", flush=True)
                local_results.append((lp, run, z_idx, None, None, None, None, None))
                continue
            
            # Compute C_ℓ
            ell, Cl = compute_Cl(kappa, FOV_DEG)
            
            # Compute WL statistics
            peaks = compute_peaks(kappa_smooth, rms, SN_BINS)
            minima = compute_minima(kappa_smooth, rms, SN_BINS)
            pdf = compute_pdf(kappa_smooth, rms, SN_BINS)
            
            local_results.append((lp, run, z_idx, ell, Cl, peaks, minima, pdf))
            
        except Exception as e:
            if rank == 0 and task_idx < 5:
                print(f"  ERROR: {kappa_path}: {e}", flush=True)
            local_results.append((lp, run, z_idx, None, None, None, None, None))
    
    # Gather results
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        # Determine dimensions
        ell_bins = None
        for results in all_results:
            for lp, run, z_idx, ell, Cl, peaks, minima, pdf in results:
                if ell is not None:
                    ell_bins = ell
                    break
            if ell_bins is not None:
                break
        
        if ell_bins is None:
            print("  ERROR: No valid convergence stats computed!", flush=True)
            return None
        
        n_z = len(z_names)
        n_real = N_LP * N_RUNS
        n_ell = len(ell_bins)
        n_sn = len(SN_BINS) - 1
        
        # Initialize output arrays
        Cl_data = np.zeros((n_real, n_z, n_ell), dtype=np.float32)
        peaks_data = np.zeros((n_real, n_z, n_sn), dtype=np.float32)
        minima_data = np.zeros((n_real, n_z, n_sn), dtype=np.float32)
        pdf_data = np.zeros((n_real, n_z, n_sn), dtype=np.float32)
        
        Cl_data[:] = np.nan
        peaks_data[:] = np.nan
        minima_data[:] = np.nan
        pdf_data[:] = np.nan
        
        # Fill in results
        for results in all_results:
            for lp, run, z_idx, ell, Cl, peaks, minima, pdf in results:
                if Cl is not None:
                    real_idx = lp * N_RUNS + (run - 1)
                    Cl_data[real_idx, z_idx, :] = Cl
                    peaks_data[real_idx, z_idx, :] = peaks
                    minima_data[real_idx, z_idx, :] = minima
                    pdf_data[real_idx, z_idx, :] = pdf
        
        print(f"  Convergence stats complete. Shapes: Cl={Cl_data.shape}, peaks={peaks_data.shape}", flush=True)
        
        return {
            'Cl': Cl_data,
            'peaks': peaks_data,
            'minima': minima_data,
            'pdf': pdf_data,
            'ell_bins': ell_bins
        }
    
    return None


def save_results(model, Pk_data, k_bins, conv_results):
    """Save all statistics to HDF5."""
    output_dir = os.path.join(STATS_BASE, model)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'stats.h5')
    
    with h5py.File(output_path, 'w') as f:
        # Density statistics
        if Pk_data is not None:
            f.create_dataset('Pk', data=Pk_data, dtype='float32', compression='gzip')
            f.create_dataset('k_bins', data=k_bins, dtype='float32')
            f.attrs['snapshot_order'] = SNAPSHOT_ORDER
            f.attrs['snapshot_redshifts'] = SNAPSHOT_REDSHIFTS
        
        # Convergence statistics
        if conv_results is not None:
            f.create_dataset('Cl', data=conv_results['Cl'], dtype='float32', compression='gzip')
            f.create_dataset('peaks', data=conv_results['peaks'], dtype='float32', compression='gzip')
            f.create_dataset('minima', data=conv_results['minima'], dtype='float32', compression='gzip')
            f.create_dataset('pdf', data=conv_results['pdf'], dtype='float32', compression='gzip')
            f.create_dataset('ell_bins', data=conv_results['ell_bins'], dtype='float32')
            f.create_dataset('sn_bin_edges', data=SN_BINS, dtype='float32')
            
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
    
    print(f"\nResults saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Compute all statistics for a model')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., dmo, hydro, or replace model)')
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=" * 80)
        print(f"Computing statistics for model: {args.model}")
        print("=" * 80)
        print(f"MPI size: {size}")
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
    
    # Compute density statistics
    Pk_data, k_bins = compute_density_stats(args.model, comm)
    
    # Compute convergence statistics
    conv_results = compute_convergence_stats(args.model, dmo_rms, comm)
    
    # Save results
    if rank == 0:
        save_results(args.model, Pk_data, k_bins, conv_results)
        print()
        print("=" * 80)
        print(f"Statistics computation complete for {args.model}!")
        print("=" * 80)


if __name__ == '__main__':
    main()
