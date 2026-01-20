# Unified Statistics Pipeline

Single MPI script computes all statistics (P(k), C_ℓ, peaks, minima, PDF) for each model. DMO RMS values are pre-computed and cached. All bin edges are determined naturally by Pylians. Each model produces one `stats.h5` file with both density and convergence statistics.

## Quick Start

### 1. Compute DMO RMS (one-time setup)

```bash
sbatch batch/run_compute_dmo_rms.sh
```

This computes the smoothed convergence RMS for all DMO realizations and saves to `/mnt/home/mlee1/ceph/hydro_replace_stats/dmo_rms.h5`. Run this once before processing any models.

### 2. Compute Statistics for All Models

```bash
sbatch batch/run_compute_stats.sh
```

This array job (indices 0-33) processes all main models:
- Index 0: dmo
- Index 1: hydro  
- Indices 2-17: 16 cumulative Replace models (4 mass × 4 radii)
- Indices 18-33: 16 discrete tile Replace models (4 mass × 4 radii)

Each job runs with 64 MPI ranks and takes ~30-120 minutes depending on model.

### 3. Analyze Results

Open `notebooks/response_analysis_template.ipynb` to:
- Load statistics from `stats.h5` files
- Compute response fractions F_S
- Generate publication figures

## File Structure

```
/mnt/home/mlee1/ceph/hydro_replace_stats/
├── dmo_rms.h5                          # Pre-computed DMO RMS (N_LP, N_RUNS, N_z)
├── dmo/
│   └── stats.h5
├── hydro/
│   └── stats.h5
└── hydro_replace_Ml_1.00e+12_Mu_1.00e+15_Ri_0.0_Ro_3.0/
    └── stats.h5
```

### HDF5 Structure: `stats.h5`

Each model's `stats.h5` contains:

**Datasets:**
- `Pk`: (20 LP, 20 snapshots, ~200 k-bins) - 2D matter power spectrum from lensplanes
- `k_bins`: Wavenumber bins (h/Mpc)
- `Cl`: (2000 realizations, 4 redshifts, ~50 ℓ-bins) - Angular power spectrum  
- `ell_bins`: Multipole bins
- `peaks`: (2000, 4, 15) - Peak count histograms in S/N bins
- `minima`: (2000, 4, 15) - Minima count histograms
- `pdf`: (2000, 4, 15) - Full PDF histograms
- `sn_bin_edges`: S/N bin edges (16 edges → 15 bins)

**Attributes:**
- `snapshot_order`: [96, 90, 85, ..., 29] - TNG snapshot numbers
- `snapshot_redshifts`: [0.04, 0.15, ..., 2.87] - Corresponding redshifts
- `source_redshifts`: [0.506, 1.034, 2.094, 2.568] - Source plane redshifts
- `kappa_targets`: [13, 23, 36, 40] - Corresponding kappa file numbers
- `N_LP`: 20, `N_RUNS`: 100
- `smoothing_arcmin`: 2.5
- `BOX_SIZE`: 205.0, `FOV_DEG`: 5.0

## Scripts

### Core Scripts

- `scripts/constants.py` - Paths, parameters, model naming
- `scripts/stats_utils.py` - Statistics computation (P(k), C_ℓ, peaks, etc.)
- `scripts/response_utils.py` - Response fraction calculations
- `scripts/compute_dmo_rms.py` - One-time DMO RMS computation
- `scripts/compute_all_stats.py` - Main unified statistics pipeline

### Batch Scripts

- `batch/run_compute_dmo_rms.sh` - SLURM job for DMO RMS
- `batch/run_compute_stats.sh` - SLURM array job for all models

## Computing Statistics for a Single Model

```bash
# Activate environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
module load python openmpi python-mpi hdf5

# Run for specific model
mpirun -np 64 python scripts/compute_all_stats.py --model dmo
mpirun -np 64 python scripts/compute_all_stats.py --model hydro
mpirun -np 64 python scripts/compute_all_stats.py --model "hydro_replace_Ml_1.00e+12_Mu_1.00e+15_Ri_0.0_Ro_3.0"
```

## Response Analysis Workflow

### Loading Data

```python
import h5py
import numpy as np
from pathlib import Path

STATS_BASE = '/mnt/home/mlee1/ceph/hydro_replace_stats'

# Load a statistic
def load_statistic(model, stat_name, z_idx=None):
    path = Path(STATS_BASE) / model / 'stats.h5'
    
    with h5py.File(path, 'r') as f:
        data = f[stat_name][:]
        
        # Get x-axis
        if stat_name == 'Pk':
            x = f['k_bins'][:]
        elif stat_name == 'Cl':
            x = f['ell_bins'][:]
        else:  # peaks, minima, pdf
            x = f['sn_bin_edges'][:]
            x = 0.5 * (x[:-1] + x[1:])  # Bin centers
        
        # Extract redshift slice for convergence stats
        if stat_name != 'Pk' and z_idx is not None:
            data = data[:, z_idx, :]
    
    return data, x

# Example: Load peak counts at z~1.0
peaks_dmo, sn_bins = load_statistic('dmo', 'peaks', z_idx=1)
peaks_hydro, _ = load_statistic('hydro', 'peaks', z_idx=1)
peaks_replace, _ = load_statistic('hydro_replace_Ml_1.00e+12_Mu_1.00e+15_Ri_0.0_Ro_3.0', 'peaks', z_idx=1)
```

### Computing Response Fractions

```python
from scripts.response_utils import bootstrap_F

# Compute response with bootstrap errors
F_mean, F_std = bootstrap_F(peaks_replace, peaks_dmo, peaks_hydro, n_bootstrap=1000)

# Plot
import matplotlib.pyplot as plt
plt.errorbar(sn_bins, F_mean, yerr=F_std, marker='o')
plt.xlabel('S/N')
plt.ylabel('F_S')
plt.show()
```

## Model Naming Convention

### Cumulative Models
Format: `hydro_replace_Ml_{Ml}_Mu_1.00e+15_Ri_0.0_Ro_{Ro}`

- `Ml`: Mass threshold (1.00e+12, 3.16e+12, 1.00e+13, 3.16e+13)
- `Ro`: Radius cutoff in R₂₀₀ units (0.5, 1.0, 3.0, 5.0)

Example: `hydro_replace_Ml_1.00e+12_Mu_1.00e+15_Ri_0.0_Ro_3.0`
→ Replaces all halos with M > 10¹² M☉/h within r < 3R₂₀₀

### Discrete Tile Models
Format: `hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}`

- Mass bins: (1.00e+12, 3.16e+12), (3.16e+12, 1.00e+13), (1.00e+13, 3.16e+13), (3.16e+13, 1.00e+15)
- Radius shells: (0.0, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, 5.0) R₂₀₀

Example: `hydro_replace_Ml_1.00e+12_Mu_3.16e+12_Ri_0.5_Ro_1.0`
→ Replaces halos in 10¹² < M < 10^12.5 M☉/h and 0.5 < r < 1.0 R₂₀₀

## Statistics Definitions

### Density Statistics (from lensplanes)

**P(k)**: 2D matter power spectrum computed via Pylians `Pk_plane`
- Input: Mass surface density from lensplanes (4096² grid, 205 Mpc/h box)
- Output: ~200 k-bins from k ~ 0.1 to ~100 h/Mpc
- Shape: (20 LP, 20 snapshots, N_k)

### Convergence Statistics (from ray-tracing)

**C_ℓ**: Angular power spectrum computed via Pylians `Pk_plane`
- Input: Convergence κ from lux ray-tracer (1024² grid, 5° FOV)
- Output: ~50 ℓ-bins from ℓ ~ 100 to ~10,000
- Shape: (2000 realizations, 4 source z, N_ℓ)

**Peaks**: Counts of local maxima in S/N bins
- Smoothing: 2.5 arcmin Gaussian (σ ~ 8.5 pixels)
- Normalization: S/N = κ / σ_DMO where σ_DMO from smoothed DMO maps
- Detection: 3×3 maximum filter
- Bins: 15 bins from S/N = -5 to 10

**Minima**: Counts of local minima in S/N bins
- Same smoothing and normalization as peaks
- Detection: 3×3 minimum filter

**PDF**: Full pixel distribution in S/N bins
- All pixels contribute (not just extrema)
- Same smoothing and normalization

## Source Redshifts

The 4 source plane redshifts for convergence analysis:

| Index | kappa file | Redshift | Label |
|-------|-----------|----------|-------|
| 0 | kappa13 | z = 0.506 | z0.5 |
| 1 | kappa23 | z = 1.034 | z1.0 |
| 2 | kappa36 | z = 2.094 | z2.0 |
| 3 | kappa40 | z = 2.568 | z2.5 |

## Response Fraction Formalism

**Cumulative response:**
```
F_S(M, α) = (S_Replace - S_DMO) / (S_Hydro - S_DMO)
```

**Tile contribution:**
```
ΔF_S(M_bin, r_shell) = (S_tile - S_DMO) / (S_Hydro - S_DMO)
```

**Non-additivity:**
```
ε = F_cumulative - Σ ΔF_tiles
```

Positive ε indicates super-additivity (tiles interact constructively).

## Bootstrap Error Estimation

For statistics with cosmic variance (peaks, minima, PDF), use bootstrap resampling:

```python
from scripts.response_utils import bootstrap_F

# data shape: (N_realizations, N_bins)
F_mean, F_std = bootstrap_F(S_replace, S_dmo, S_hydro, n_bootstrap=1000, seed=42)
```

This accounts for correlations between DMO, Hydro, and Replace realizations.

## Troubleshooting

### Missing dmo_rms.h5
```
FileNotFoundError: DMO RMS file not found
```
**Solution:** Run `sbatch batch/run_compute_dmo_rms.sh` first.

### Missing lensplane or kappa files
Check that ray-tracing pipeline completed successfully:
```bash
ls /mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG/snap_96/dmo/LP_00/
ls /mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG/dmo/LP_00/run001/
```

### Memory issues
If jobs run out of memory, reduce `--ntasks-per-node` in batch scripts and increase `--mem`.

### NaN in results
Check that:
1. Input data files exist
2. DMO RMS values are not NaN
3. Baryonic effect |S_hydro - S_dmo| > threshold

## Time Estimates

- DMO RMS computation: ~2-3 hours (one-time)
- Statistics per model: ~30-120 minutes with 64 MPI ranks
- Total for 34 models (array job): ~30-120 minutes wall time

## Contact

See main project README for contact information and citation details.
