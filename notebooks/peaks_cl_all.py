from mpi4py import MPI
import numpy as np
from scipy import ndimage
from pathlib import Path
import pickle
import h5py

# =============================================================================
# MPI initialization
# =============================================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================================
# Optional Pylians import
# =============================================================================
try:
    import Pk_library as PKL
    PYLIANS_AVAILABLE = True
    if rank == 0:
        print("Pylians loaded successfully!")
except ImportError:
    PYLIANS_AVAILABLE = False
    if rank == 0:
        print("Warning: Pylians not available. Install with: pip install Pylians")


# =============================================================================
# Paths and configuration
# =============================================================================
LENSPLANE_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG')
LUX_LP_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_LP_lux/L205n2500TNG')
RT_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG')
BCM_LP_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_LP_bcm/L205n2500TNG')
PROFILE_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_fields/L205n2500TNG/profiles')
BCM_PROFILE_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_fields_bcm/L205n2500TNG/profiles')

BOX_SIZE = 205.0   # Mpc/h
MASS_UNIT = 1e10   # Msun/h
GRID_RES = 4096
RT_GRID = 1024
N_REALIZATIONS = 10
N_RT_RUNS = 100

DISCRETE_MASS_BINS = [
    (1.00e12, 3.16e12, 'Ml_1.00e12_Mu_3.16e12'),
    (3.16e12, 1.00e13, 'Ml_3.16e12_Mu_1.00e13'),
    (1.00e13, 3.16e13, 'Ml_1.00e13_Mu_3.16e13'),
    (3.16e13, 1.00e15, 'Ml_3.16e13_Mu_1.00e15'),
]

CUMULATIVE_MASS_BINS = [
    (1.00e12, np.inf, 'Ml_1.00e12_Mu_inf'),
    (3.16e12, np.inf, 'Ml_3.16e12_Mu_inf'),
    (1.00e13, np.inf, 'Ml_1.00e13_Mu_inf'),
    (3.16e13, np.inf, 'Ml_3.16e13_Mu_inf'),
]

ALL_MASS_BINS = DISCRETE_MASS_BINS + CUMULATIVE_MASS_BINS
R_FACTORS = [0.5, 1.0, 3.0, 5.0]

SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56,
                  52, 49, 46, 43, 41, 38, 35, 33, 31, 29]

SNAP_TO_Z = {
    99: 0.00, 96: 0.02, 90: 0.10, 85: 0.18, 80: 0.27, 76: 0.35,
    71: 0.46, 67: 0.55, 63: 0.65, 59: 0.76, 56: 0.85, 52: 0.97,
    49: 1.08, 46: 1.21, 43: 1.36, 41: 1.47, 38: 1.63, 35: 1.82,
    33: 1.97, 31: 2.14, 29: 2.32
}
SNAP_STACK = {snap: (SNAP_TO_Z[snap] > 0.9) for snap in SNAPSHOT_ORDER}

def get_all_models():
    models = ['dmo', 'hydro']
    for mass_bin in ALL_MASS_BINS:
        for r_factor in R_FACTORS:
            models.append(f"hydro_replace_{mass_bin[2]}_R_{r_factor}")
    return models

SNR_BINS = np.linspace(-5, 10, 21)
SNR_MID = 0.5 * (SNR_BINS[:-1] + SNR_BINS[1:])
ALL_MODELS = get_all_models()
BCM_MODELS = ['schneider19', 'schneider25', 'arico20']

if rank == 0:
    print(f"Total models: {len(ALL_MODELS)}")
    print("Configuration loaded successfully!")
    print(f"  Lensplane base: {LENSPLANE_BASE}")
    print(f"  Profile base: {PROFILE_BASE}")
    print(f"  BCM profile base: {BCM_PROFILE_BASE}")
    print(f"  Snapshots: {len(SNAPSHOT_ORDER)} (z={SNAP_TO_Z[SNAPSHOT_ORDER[0]]:.2f} "
          f"to z={SNAP_TO_Z[SNAPSHOT_ORDER[-1]]:.2f})")
    print()
    print("Replace Model Structure:")
    print("  - DISCRETE mass bins: [M_lo, M_hi) at fixed α (4 bins × 4 α values)")
    print("  - CUMULATIVE mass bins: M >= M_min at fixed α (4 thresholds × 4 α values)")
    print("  - All models are CUMULATIVE in radius: r < α × R_200")

# Explicit list of cumulative models you want to process
cum_models = [
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

# =============================================================================
# Analysis functions
# =============================================================================
def compute_convergence_power_spectrum(kappa_maps, field_size_deg=5.0):
    n_maps, grid, _ = kappa_maps.shape
    field_size_rad = np.radians(field_size_deg)
    box_size = field_size_rad

    C_ell_all = []
    ell = None

    for kappa in kappa_maps:
        delta_kappa = (kappa - np.mean(kappa)).astype(np.float32)

        if PYLIANS_AVAILABLE:
            Pk2D = PKL.Pk_plane(delta_kappa, box_size, 'None', threads=4, verbose=False)
            ell = Pk2D.k
            C_ell = Pk2D.Pk
            C_ell_all.append(C_ell)
        else:
            kappa_k = np.fft.fft2(delta_kappa)
            Pk_2d = np.abs(kappa_k) ** 2 * (field_size_rad / grid) ** 2

            lx = np.fft.fftfreq(grid, d=field_size_rad / grid) * 2 * np.pi
            ly = np.fft.fftfreq(grid, d=field_size_rad / grid) * 2 * np.pi
            lx, ly = np.meshgrid(lx, ly)
            ell_2d = np.sqrt(lx ** 2 + ly ** 2)

            ell_bins = np.logspace(np.log10(100), np.log10(20000), 30)
            ell = 0.5 * (ell_bins[:-1] + ell_bins[1:])

            C_ell = np.zeros(len(ell))
            for i in range(len(ell)):
                mask = (ell_2d >= ell_bins[i]) & (ell_2d < ell_bins[i + 1])
                if np.sum(mask) > 0:
                    C_ell[i] = np.mean(Pk_2d[mask])

            C_ell_all.append(C_ell)

    C_ell_all = np.array(C_ell_all)
    return ell, np.mean(C_ell_all, axis=0), np.std(C_ell_all, axis=0) / np.sqrt(n_maps)

def compute_power_spectrum_response(C_ell_models, ell):
    C_D = C_ell_models['dmo'][0]
    C_H = C_ell_models['hydro'][0]
    Delta_C = C_H - C_D

    F_Cl = {}
    for model, (C_ell, _) in C_ell_models.items():
        if model not in ['dmo', 'hydro']:
            with np.errstate(divide='ignore', invalid='ignore'):
                F = np.where(np.abs(Delta_C) > 1e-30, (C_ell - C_D) / Delta_C, 0.0)
            F_Cl[model] = F
    return F_Cl

def smooth_map(kappa, smoothing_scale_pix):
    # hook to enable Gaussian smoothing if desired
    return kappa  # or ndimage.gaussian_filter(kappa, smoothing_scale_pix)

def find_peaks(kappa, snr_bins=SNR_BINS):
    data_max = ndimage.maximum_filter(kappa, size=3)
    peaks_mask = (kappa == data_max)

    peaks_mask[:2, :] = False
    peaks_mask[-2:, :] = False
    peaks_mask[:, :2] = False
    peaks_mask[:, -2:] = False

    sigma = np.std(kappa)
    peak_values = kappa[peaks_mask] / sigma
    peak_counts, _ = np.histogram(peak_values, bins=snr_bins)
    return peak_counts

def find_minima(kappa, snr_bins=SNR_BINS):
    data_min = ndimage.minimum_filter(kappa, size=3)
    minima_mask = (kappa == data_min)

    minima_mask[:2, :] = False
    minima_mask[-2:, :] = False
    minima_mask[:, :2] = False
    minima_mask[:, -2:] = False

    sigma = np.std(kappa)
    minima_values = kappa[minima_mask] / sigma
    minima_counts, _ = np.histogram(minima_values, bins=snr_bins)
    return minima_counts

def compute_peak_counts_ensemble(kappa_maps, smoothing_pix=2.0, snr_bins=SNR_BINS):
    peaks_all = []
    minima_all = []

    for kappa in kappa_maps:
        kappa_smooth = smooth_map(kappa, smoothing_pix)
        peaks_all.append(find_peaks(kappa_smooth, snr_bins))
        minima_all.append(find_minima(kappa_smooth, snr_bins))

    peaks_all = np.array(peaks_all)
    minima_all = np.array(minima_all)

    return {
        'peaks_mean': np.mean(peaks_all, axis=0),
        'peaks_std': np.std(peaks_all, axis=0) / np.sqrt(len(kappa_maps)),
        'minima_mean': np.mean(minima_all, axis=0),
        'minima_std': np.std(minima_all, axis=0) / np.sqrt(len(kappa_maps)),
    }

def compute_peak_response(peak_results, snr_mid):
    n_D_peaks = peak_results['dmo']['peaks_mean']
    n_H_peaks = peak_results['hydro']['peaks_mean']
    Delta_peaks = n_H_peaks - n_D_peaks

    n_D_min = peak_results['dmo']['minima_mean']
    n_H_min = peak_results['hydro']['minima_mean']
    Delta_min = n_H_min - n_D_min

    F_peaks = {}
    F_minima = {}

    for model in peak_results.keys():
        if model not in ['dmo', 'hydro']:
            n_R_peaks = peak_results[model]['peaks_mean']
            n_R_min = peak_results[model]['minima_mean']

            with np.errstate(divide='ignore', invalid='ignore'):
                F_peaks[model] = np.where(
                    np.abs(Delta_peaks) > 0.01,
                    (n_R_peaks - n_D_peaks) / Delta_peaks,
                    1.0,
                )
                F_minima[model] = np.where(
                    np.abs(Delta_min) > 0.01,
                    (n_R_min - n_D_min) / Delta_min,
                    1.0,
                )

    return F_peaks, F_minima

if rank == 0:
    print("Peak/minima detection functions defined.")

# =============================================================================
# IO helpers
# =============================================================================
def load_kappa(fname, ng=1024):
    with open(fname, 'rb') as f:
        _ = np.fromfile(f, dtype="int32", count=1)
        kappa = np.fromfile(f, dtype="float32", count=ng * ng)
        _ = np.fromfile(f, dtype="int32", count=1)
    return kappa.reshape(ng, ng)

def load_runs(base_path, model, z=23, LP_id=0, ng=1024):
    if z is not None:
        kappas = np.zeros((50, ng, ng), dtype=np.float32)
    else:
        kappas = np.zeros((50, 40, ng, ng), dtype=np.float32)

    for i in range(1, 51):
        path = f'{base_path}/{model}/LP_{LP_id:02d}/run{i:03d}/'
        if z is not None:
            zpath = path + f'kappa{z:02d}.dat'
            kappas[i - 1] = load_kappa(zpath, ng=ng)
        else:
            for z_idx in range(40):
                zpath = path + f'kappa{z_idx:02d}.dat'
                kappas[i - 1, z_idx] = load_kappa(zpath, ng=ng)

    return kappas

def load_all_runs(base_path, model, z=23, ng=1024):
    kappas = np.zeros((10, 50, ng, ng), dtype=np.float32)
    for LP in range(10):
        kappas[LP] = load_runs(base_path, model, z, LP_id=LP, ng=ng)
    return np.concatenate(kappas, axis=0)

# =============================================================================
# MPI-parallel workflow
# =============================================================================
base_path = str(RT_BASE)
output_dir = Path('/mnt/home/mlee1/ceph/hydro_replace_analysis/response_functions/')
output_dir.mkdir(parents=True, exist_ok=True)

z = 23

# Root loads dmo and hydro, computes reference stats
if rank == 0:
    print("Root loading reference models (dmo, hydro)...")
    dmo = load_all_runs(base_path, 'dmo', z=z)
    hydro = load_all_runs(base_path, 'hydro', z=z)

    ell, Cl_dmo, lstd_dmo = compute_convergence_power_spectrum(dmo)
    _, Cl_hydro, lstd_hydro = compute_convergence_power_spectrum(hydro)

    peak_dmo = compute_peak_counts_ensemble(dmo, smoothing_pix=5.0, snr_bins=SNR_BINS)
    peak_hydro = compute_peak_counts_ensemble(hydro, smoothing_pix=5.0, snr_bins=SNR_BINS)
else:
    ell = None
    Cl_dmo = None
    lstd_dmo = None
    Cl_hydro = None
    lstd_hydro = None
    peak_dmo = None
    peak_hydro = None

# Broadcast reference data to all ranks
ell = comm.bcast(ell, root=0)
Cl_dmo = comm.bcast(Cl_dmo, root=0)
lstd_dmo = comm.bcast(lstd_dmo, root=0)
Cl_hydro = comm.bcast(Cl_hydro, root=0)
lstd_hydro = comm.bcast(lstd_hydro, root=0)
peak_dmo = comm.bcast(peak_dmo, root=0)
peak_hydro = comm.bcast(peak_hydro, root=0)

# Work distribution across ranks
models_per_rank = len(cum_models) // size
remainder = len(cum_models) % size

if rank < remainder:
    start_idx = rank * (models_per_rank + 1)
    end_idx = start_idx + models_per_rank + 1
else:
    start_idx = rank * models_per_rank + remainder
    end_idx = start_idx + models_per_rank

my_models = cum_models[start_idx:end_idx]

if rank == 0:
    print(f"Processing {len(cum_models)} models across {size} ranks")
print(f"Rank {rank}: processing {len(my_models)} models (indices {start_idx}-{end_idx-1})")

# Local computation on each rank
local_c_ell = {}
local_peaks = {}

for model in my_models:
    try:
        print(f'Rank {rank}: working on model {model}')
        maps = load_all_runs(base_path, model, z=z)
        _, Cl_model, lstd_model = compute_convergence_power_spectrum(maps)
        local_c_ell[model] = (Cl_model, lstd_model)
        local_peaks[model] = compute_peak_counts_ensemble(
            maps, smoothing_pix=3.0, snr_bins=SNR_BINS
        )
    except FileNotFoundError:
        print(f'Rank {rank}: {model} not found')

# Gather results
all_c_ell = comm.gather(local_c_ell, root=0)
all_peaks = comm.gather(local_peaks, root=0)

# Root combines and saves
if rank == 0:
    c_ell_models = {'dmo': (Cl_dmo, lstd_dmo), 'hydro': (Cl_hydro, lstd_hydro)}
    peak_results = {'dmo': peak_dmo, 'hydro': peak_hydro}

    for rank_results in all_c_ell:
        c_ell_models.update(rank_results)
    for rank_results in all_peaks:
        peak_results.update(rank_results)

    F_cl = compute_power_spectrum_response(c_ell_models, ell)
    F_peak, F_minima = compute_peak_response(peak_results, SNR_MID)

    print(f"\nCompleted! Processed {len(c_ell_models)} models total")
    print(f"\nSaving results to {output_dir}")

    np.savez(
        output_dir / f'power_spectrum_response_z{z:02d}.npz',
        ell=ell,
        **{f'F_cl_{model}': F_cl[model] for model in F_cl}
    )

    np.savez(
        output_dir / f'peak_response_z{z:02d}.npz',
        snr_mid=SNR_MID,
        **{f'F_peak_{model}': F_peak[model] for model in F_peak}
    )

    np.savez(
        output_dir / f'minima_response_z{z:02d}.npz',
        snr_mid=SNR_MID,
        **{f'F_minima_{model}': F_minima[model] for model in F_minima}
    )

    with open(output_dir / f'response_functions_z{z:02d}.pkl', 'wb') as f:
        pickle.dump(
            {
                'F_cl': F_cl,
                'F_peak': F_peak,
                'F_minima': F_minima,
                'ell': ell,
                'snr_mid': SNR_MID,
                'metadata': {
                    'redshift': SNAP_TO_Z[z],
                    'snapshot': z,
                    'n_models': len(cum_models),
                    'n_realizations': 500,
                },
            },
            f,
        )

    print("Results saved successfully!")
