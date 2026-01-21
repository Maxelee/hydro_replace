# Unified Statistics Pipeline (Final Plan)

**TL;DR**: Single MPI script computes all statistics (P(k), C_ℓ, peaks, minima, PDF) for a model in one run. DMO rms pre-computed and cached. All bin edges determined naturally by Pylians. Output: one `{model}/stats.h5` file with both density and convergence statistics.

## Steps

### 1. Create `scripts/constants.py`

Contents:
- Paths: `LP_BASE`, `RT_BASE`, `STATS_BASE = '/mnt/home/mlee1/ceph/hydro_replace_stats'`
- `build_model_name(Ml, Mu, Ri, Ro)`, `get_all_models()` → 34 models
- Grids: `MASS_THRESHOLDS`, `RADII`, `DISCRETE_MASS_BINS`, `DISCRETE_RADIUS_BINS`
- WL params: `SMOOTHING_ARCMIN = 2.5`, `FOV_DEG = 5.0`, `NG = 1024`, `SN_BINS = np.arange(-5, 11, 1)`
- `KAPPA_TARGETS = {'z0.5': 13, 'z1.0': 23, 'z2.0': 36, 'z2.5': 40}`
- `SNAPSHOT_ORDER`, `SNAPSHOT_REDSHIFTS` (20 snapshots for density)

### 2. Create `scripts/stats_utils.py`

Contents:
- I/O: `load_kappa(path, ng=1024)`, `read_lensplane(path)`
- Density: `compute_Pk_2d(field, box_size)` → `(k, Pk)` via Pylians
- Convergence: `compute_Cl(kappa, fov_deg)` → `(ell, Cl)` via Pylians (bins from FFT naturally)
- Smoothing: `smooth_kappa(kappa, smoothing_arcmin, pixel_scale_arcmin)`
- WL stats: `compute_peaks(kappa_smooth, rms, sn_bins)`, `compute_minima(...)`, `compute_pdf(...)`

### 3. Create `scripts/response_utils.py`

Contents:
- `compute_F_S(S_R, S_D, S_H, threshold=0.02)` → F_S with NaN masking
- `compute_Delta_F(S_tile, S_D, S_H)` → tile contribution
- `compute_epsilon(F_tiles_sum, F_cumulative)` → non-additivity
- `compute_mean_F(F, x, weights=None, x_range=None)` → weighted average
- `bootstrap_F(R, D, H, n_bootstrap=1000)` → `(F_mean, F_std)`

### 4. Create `scripts/compute_dmo_rms.py`

One-time MPI script:
- Compute smoothed κ rms for DMO at each (LP, run, z) combination
- Save to `{STATS_BASE}/dmo_rms.h5` with shape `(N_LP, N_RUNS, N_z)` as float32
- Run once before main pipeline

### 5. Create `scripts/compute_all_stats.py`

Unified MPI script:
- CLI: `--model MODEL` (required for incremental) or `--all` for full run
- Load DMO rms from cached `dmo_rms.h5`
- Distribute work across ranks:
  - Density: (LP, snapshot) pairs → P(k) from lensplanes
  - Convergence: (LP, run, z) tuples → C_ℓ, peaks, minima, PDF from κ maps
- Save to `{STATS_BASE}/{model}/stats.h5` with:
  - **Density**: `Pk` shape `(N_LP, N_snap, N_k)`, attrs: `k_bins`, `snapshot_redshifts`
  - **Convergence**: `Cl`, `peaks`, `minima`, `pdf` shape `(N_LP*N_RUNS, N_z, N_bins)`, attrs: `ell_bins`, `sn_bin_edges`, `source_redshifts`, `smoothing_arcmin`
- All arrays stored as float32

### 6. Create `batch/run_compute_dmo_rms.sh`

One-time SLURM job for step 4.

### 7. Create `batch/run_compute_stats.sh`

SLURM array script:
- Array indices map to model names (0-33)
- Each job: `mpirun python compute_all_stats.py --model $MODEL`

### 8. Create `notebooks/response_analysis_template.ipynb`

Structure:
- Cell 1: Select statistic (`Pk`, `Cl`, `peaks`, `minima`, `pdf`) and source redshift
- Cell 2: Load `{model}/stats.h5` for dmo, hydro, and Replace models
- Cell 3: Compute F_S, ΔF_S, ε using `response_utils`
- Cells 4-10: Figures 1-7 (scale-dependent F, cumulative summary, tile heatmap, marginals, redshift evolution, additivity, ε)

## HDF5 File Structure

```
{STATS_BASE}/
├── dmo_rms.h5                    # Pre-computed DMO rms (N_LP, N_RUNS, N_z)
├── dmo/stats.h5
├── hydro/stats.h5
├── hydro_replace_Ml_.../stats.h5
│   ├── Pk                        # (20, 20, ~200) - LP × snapshot × k
│   ├── Cl                        # (2000, 4, ~50) - realization × z × ℓ
│   ├── peaks                     # (2000, 4, 15) - realization × z × S/N bin
│   ├── minima                    # (2000, 4, 15)
│   ├── pdf                       # (2000, 4, 15)
│   └── attrs: k_bins, ell_bins, sn_bin_edges, source_redshifts, ...
```

## Further Considerations

1. **Compute order**: Run `compute_dmo_rms.py` first (one job), then `compute_all_stats.py` for all models. The stats script should auto-check for `dmo_rms.h5` and error clearly if missing.

2. **Time estimate**: ~2000 realizations × 4 z × 5 stats per model, ~34 models. Estimate ~10-30 min per model with 64 MPI ranks. Total ~6-17 hours if run sequentially, or ~30 min wall time with array job parallelism.

## Reference: Key Code Patterns

### From `peak_counts.ipynb` - Smoothing and Peak Detection

```python
# Smoothing parameters
SMOOTHING_ARCMIN = 2.5
PIXEL_SCALE_ARCMIN = FOV_DEG * 60.0 / NG  # ~0.293 arcmin/pixel
SIGMA_PIX = SMOOTHING_ARCMIN / PIXEL_SCALE_ARCMIN  # ~8.5 pixels

# Apply Gaussian smoothing
from scipy.ndimage import gaussian_filter
kappa_smooth = gaussian_filter(kappa, sigma=SIGMA_PIX)

# Compute RMS from DMO
rms = np.std(kappa_dmo_smooth)

# S/N bins
SN_BINS = np.arange(-5, 11, 1)  # 16 edges → 15 bins

# Peak detection
from scipy.ndimage import maximum_filter
is_peak = (kappa_smooth == maximum_filter(kappa_smooth, size=3))
peak_values = kappa_smooth[is_peak]
peak_sn = peak_values / rms
peaks_hist, _ = np.histogram(peak_sn, bins=SN_BINS)
```

### From `density_final.ipynb` - Power Spectrum

```python
import Pk_library as PKL

def compute_Pk_2d(field, box_size=205.0):
    """Compute 2D power spectrum using Pylians."""
    # Convert to overdensity
    mean_val = np.mean(field)
    delta = (field - mean_val) / mean_val
    delta = delta.astype(np.float32)
    
    # Compute P(k)
    Pk2D = PKL.Pk_plane(delta, box_size, 'None', 1)
    k = Pk2D.k
    Pk = Pk2D.Pk
    return k, Pk
```

### Response Fraction

```python
def compute_F_S(S_R, S_D, S_H, threshold=0.02):
    """Compute response fraction with significance masking."""
    Delta_S = S_H - S_D
    F = (S_R - S_D) / Delta_S
    
    # Mask where baryonic effect < threshold
    significant = np.abs(Delta_S) > threshold * np.abs(S_D)
    F[~significant] = np.nan
    
    return F
```
