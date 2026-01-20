# Copilot Instructions for Hydro Replace Project

## Project Overview

This repository implements a **baryonic response formalism** for weak lensing cosmology. The goal is to quantify how baryonic physics (from hydrodynamical simulations) affects cosmological observables as a function of halo mass, radius, and redshift.

The paper being written is **"Responding to baryons with the baryonic response metric"** (see `draft.tex`).

## Key Concepts

### Response Formalism

The core idea is to construct "Replace" density fields where DMO halos are selectively replaced with their hydrodynamical counterparts:

```
ρ_R(x) = ρ_DMO(x) + Σ_halos [ρ_Hydro(x) - ρ_DMO(x)]
```

The **cumulative response fraction** measures how much baryonic effect is captured:

```
F_S(M_min, α) = (S_Replace - S_DMO) / (S_Hydro - S_DMO)
```

where S is any statistic (power spectrum, peak counts, etc.), M_min is the minimum halo mass included, and α is the radius factor in units of R₂₀₀.

### Model Configurations

- **Base models**: DMO (dark matter only), Hydro (full hydrodynamical)
- **Cumulative Replace models**: 16 total (4 mass thresholds × 4 radii)
  - Mass thresholds: 10^12, 10^12.5, 10^13, 10^13.5 M⊙/h
  - Radius factors: 0.5, 1.0, 3.0, 5.0 × R₂₀₀
- **Discrete Replace models**: 16 tiles (4 mass bins × 4 radius shells)
  - Exclusive mass-radius regions for additivity testing
- **Realizations**: 
  - For lensplanes (density analysis): 20 LP orientations (LP_00 to LP_19)
  - For ray-tracing (WL analysis): 20 LP × 100 runs = 2000 convergence maps per model

## Environment

Always activate the virtual environment before running any code:

```bash
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
```

Required modules on the cluster:
```bash
module load python openmpi python-mpi hdf5
```

## Code Structure

### Batch Scripts (`batch/`)

**Active scripts:**
- `run_dmo_hydro_lensplanes.sh` - Generate DMO and Hydro lensplanes
- `run_lux_binned_unified.sh` - Full ray-tracing pipeline (2040 array jobs)
- `run_unified_2500_binned_array.sh` - Replace field generation (200 array jobs)

### Core Scripts (`scripts/`)

**Active scripts:**
- `generate_all_unified.py` - Main MPI pipeline for generating:
  - Stacked density profiles
  - 2D projected maps
  - Lensplanes for ray-tracing
  - Supports `--mode cumulative` or `--mode binned`

- `convert_to_lensplanes.py` - FFT conversion of mass planes to lensing potential format for the `lux` ray-tracer

- `generate_all_unified_bcm.py` - BCM (baryonic correction model) variant

- `response_visualization.py` - Helper functions for response plotting (used by `07_response_visualization_test.ipynb`)

### Notebooks (`notebooks/`)

**Active notebooks for the paper:**

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| `binned_pipeline.ipynb` | Full analysis pipeline | Model grid figure, response fractions |
| `density_final.ipynb` | 3D matter power spectrum | Figs 2-4: scale-dependent response |
| `bispectrum_final.ipynb` | Bispectrum analysis | Bispectrum response kernels |
| `peak_counts.ipynb` | WL peak statistics | Peak count response vs ν |
| `07_response_visualization_test.ipynb` | Response visualization | Publication figures |
| `data_products_guide.ipynb` | Data access guide | Documentation |

## Data Paths

| Data | Path |
|------|------|
| TNG simulations | `/mnt/sdceph/users/sgenel/IllustrisTNG/` |
| Raw lensplanes | `/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG/` |
| Lux lensplanes | `/mnt/home/mlee1/ceph/hydro_replace_LP_lux/L205n2500TNG/` |
| Convergence maps | `/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG/` |
| Density fields | `/mnt/home/mlee1/ceph/hydro_replace_fields/` |

## Common Tasks

### Running the Pipeline

```bash
# Full pipeline for one snapshot
mpirun -np 64 python scripts/generate_all_unified.py \
    --snap 96 --sim-res 2500 --enable-lensplanes --mode binned

# Submit array job for all snapshots
sbatch batch/run_unified_2500_binned_array.sh
```

### Loading Lensplane Data

```python
import numpy as np

# Load a lensplane
lp_path = "/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG/snap_96/replace_M12.0_R3.0/LP_00/lensplane_00.npz"
data = np.load(lp_path)
plane = data['plane']  # 4096x4096 mass array
```

### Loading Convergence Maps

```python
import numpy as np

def load_kappa(fname, ng=1024):
    """Load single kappa map from lux binary format."""
    with open(fname, 'rb') as f:
        dummy = np.fromfile(f, dtype="int32", count=1)
        kappa = np.fromfile(f, dtype="float", count=ng*ng)
        dummy = np.fromfile(f, dtype="int32", count=1)
    return kappa.reshape(ng, ng)

# Load convergence map from ray-tracing output
RT_BASE = '/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG'
kappa_path = f"{RT_BASE}/dmo/LP_00/run001/kappa20.dat"  # kappa20 ≈ z_s ~ 1.0
kappa = load_kappa(kappa_path, ng=1024)  # 1024x1024 convergence map
# 20 LPs (LP_00 to LP_19) × 100 runs (run001 to run100) = 2000 maps per model
```

### Computing Response Fractions

```python
# Compute cumulative response fraction
F_S = (S_replace - S_dmo) / (S_hydro - S_dmo)

# Handle division by zero where hydro ≈ dmo
mask = np.abs(S_hydro - S_dmo) / S_dmo > 0.01
F_S[~mask] = np.nan
```

### Bootstrap Error Estimation for Response

For peak counts and other statistics with shared cosmic variance:

```python
def bootstrap_F(data_R, data_D, data_H, n_bootstrap=1000):
    """
    Compute F and uncertainty via bootstrap over realizations.
    data_R/D/H: arrays of shape (N_LP, N_RUNS, N_BINS)
    """
    n_lp, n_runs, n_bins = data_R.shape
    n_real = n_lp * n_runs
    
    R_flat = data_R.reshape(n_real, n_bins)
    D_flat = data_D.reshape(n_real, n_bins)
    H_flat = data_H.reshape(n_real, n_bins)
    
    rng = np.random.default_rng(42)
    F_samples = np.zeros((n_bootstrap, n_bins))
    
    for i in range(n_bootstrap):
        idx = rng.choice(n_real, size=n_real, replace=True)
        N_R = np.sum(R_flat[idx], axis=0)
        N_D = np.sum(D_flat[idx], axis=0)
        N_H = np.sum(H_flat[idx], axis=0)
        F_samples[i] = (N_R - N_D) / (N_H - N_D)
    
    return np.nanmean(F_samples, axis=0), np.nanstd(F_samples, axis=0)
```

## Model Naming Convention

```
dmo                                                    # Dark matter only
hydro                                                  # Full hydrodynamical
hydro_replace_Ml_1.00e12_Mu_1.00e15_Ri_0.0_Ro_3.0      # Cumulative: M > 10^12, r < 3R200
hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.5_Ro_1.0      # Discrete: single mass-radius tile
```

**Full model naming**: `hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}`
- `Ml` = mass lower bound (M⊙/h)
- `Mu` = mass upper bound (`1.00e15` for cumulative = all masses above Ml)
- `Ri` = radius inner bound (`0.0` for cumulative = from center)
- `Ro` = radius outer bound (in units of R₂₀₀)

**Mass values**: `1.00e12`, `3.16e12`, `1.00e13`, `3.16e13`, `1.00e15`
**Radius values**: `0.0`, `0.5`, `1.0`, `3.0`, `5.0`

**Examples**:
- Cumulative (M > 10¹², r < 3R₂₀₀): `hydro_replace_Ml_1.00e12_Mu_1.00e15_Ri_0.0_Ro_3.0`
- Discrete tile (10¹² < M < 10^12.5, 0.5 < r < 1.0 R₂₀₀): `hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.5_Ro_1.0`

## Important Constants

```python
BOX_SIZE = 205.0  # Mpc/h
GRID_RES = 4096   # Lensplane resolution
RT_GRID = 1024    # Ray-tracing output resolution
FOV_DEG = 5.0     # Field of view in degrees
N_LP = 20         # Lensplane orientations (LP_00 to LP_19)
N_RUNS = 100      # Ray-traced maps per LP (run001 to run100)

# Smoothing for peak counts
SMOOTHING_ARCMIN = 2.5  # Gaussian smoothing scale
PIXEL_SCALE_ARCMIN = FOV_DEG * 60.0 / RT_GRID  # ~0.293 arcmin/pixel

# Mass thresholds (M⊙/h)
MASS_THRESHOLDS = ['1.00e12', '3.16e12', '1.00e13', '3.16e13']

# Radius factors
ALPHA_VALUES = [0.5, 1.0, 3.0, 5.0]
```

## Snapshot-Redshift Mapping

The pipeline uses 20 TNG snapshots spanning z = 0.0 to z ≈ 2.9:

| Index | Snapshot | Redshift | Stack |
|-------|----------|----------|-------|
| 0 | 99 | 0.00 | No |
| 1 | 96 | 0.04 | No |
| 2 | 90 | 0.15 | No |
| 3 | 85 | 0.27 | No |
| 4 | 80 | 0.40 | No |
| 5 | 76 | 0.50 | No |
| 6 | 71 | 0.64 | No |
| 7 | 67 | 0.78 | No |
| 8 | 63 | 0.93 | No |
| 9 | 59 | 1.07 | No |
| 10 | 56 | 1.18 | No |
| 11 | 52 | 1.36 | Yes |
| 12 | 49 | 1.50 | Yes |
| 13 | 46 | 1.65 | Yes |
| 14 | 43 | 1.82 | Yes |
| 15 | 41 | 1.93 | Yes |
| 16 | 38 | 2.12 | Yes |
| 17 | 35 | 2.32 | Yes |
| 18 | 33 | 2.49 | Yes |
| 19 | 29 | 2.87 | Yes |

**Stack** indicates high-z snapshots that use stacked (2× box) planes.

```python
# Snapshot order for ray-tracing pipeline
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
SNAPSHOT_REDSHIFTS = [0.04, 0.15, 0.27, 0.40, 0.50, 0.64, 0.78, 0.93, 1.07, 1.18,
                     1.36, 1.50, 1.65, 1.82, 1.93, 2.12, 2.32, 2.49, 2.68, 2.87]
```

## Source/Lens Plane Mapping (lux ray-tracer)

The `kappa{NN}.dat` files correspond to source planes at different redshifts:

| File | χ (h⁻¹Mpc) | z_s | File | χ (h⁻¹Mpc) | z_s |
|------|-----------|------|------|-----------|------|
| kappa01 | 102.5 | 0.034 | kappa21 | 2152.5 | 0.914 |
| kappa02 | 205.0 | 0.070 | kappa22 | 2255.0 | 0.973 |
| kappa03 | 307.5 | 0.105 | kappa23 | 2357.5 | 1.034 |
| kappa04 | 410.0 | 0.142 | kappa24 | 2460.0 | 1.097 |
| kappa05 | 512.5 | 0.179 | kappa25 | 2562.5 | 1.163 |
| kappa06 | 615.0 | 0.216 | kappa26 | 2665.0 | 1.231 |
| kappa07 | 717.5 | 0.255 | kappa27 | 2767.5 | 1.302 |
| kappa08 | 820.0 | 0.294 | kappa28 | 2870.0 | 1.375 |
| kappa09 | 922.5 | 0.335 | kappa29 | 2972.5 | 1.452 |
| kappa10 | 1025.0 | 0.376 | kappa30 | 3075.0 | 1.532 |
| kappa11 | 1127.5 | 0.418 | kappa31 | 3177.5 | 1.615 |
| kappa12 | 1230.0 | 0.462 | kappa32 | 3280.0 | 1.703 |
| kappa13 | 1332.5 | 0.506 | kappa33 | 3382.5 | 1.794 |
| kappa14 | 1435.0 | 0.552 | kappa34 | 3485.0 | 1.889 |
| kappa15 | 1537.5 | 0.599 | kappa35 | 3587.5 | 1.989 |
| kappa16 | 1640.0 | 0.648 | kappa36 | 3690.0 | 2.094 |
| kappa17 | 1742.5 | 0.698 | kappa37 | 3792.5 | 2.203 |
| kappa18 | 1845.0 | 0.749 | kappa38 | 3895.0 | 2.319 |
| kappa19 | 1947.5 | 0.803 | kappa39 | 3997.5 | 2.440 |
| kappa20 | 2050.0 | 0.858 | kappa40 | 4100.0 | 2.568 |

**Common source planes**: `kappa20` (z_s ≈ 0.86), `kappa25` (z_s ≈ 1.16), `kappa30` (z_s ≈ 1.53)

## Code Style

- Use numpy-style docstrings
- MPI-aware code should check `rank == 0` for I/O
- Always use absolute paths for data files
- Prefer `np.savez_compressed` for large arrays

## When Helping with Code

1. **For batch scripts**: Ensure SLURM directives match cluster requirements (partition: `cca`)
2. **For MPI code**: Use `comm.Barrier()` for synchronization, distribute work by `rank`
3. **For notebooks**: Assume the hydro_replace venv is active
4. **For figures**: Save to `notebooks/figures/` with descriptive names
5. **For the paper**: Reference equations from `draft.tex` (e.g., Eq. 3 for cumulative response)

## Common Issues

1. **Memory**: TNG300 has 2500³ particles—load particles in chunks
2. **Missing data**: Check if lensplanes exist before processing
3. **MPI deadlock**: Ensure all ranks reach collective operations
4. **Slow KDTree**: Build once, query many times in `generate_all_unified.py`
