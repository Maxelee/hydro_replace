# Baryonic Response Formalism: Hydro Replace Pipeline

A comprehensive framework for quantifying how baryonic physics affects weak lensing observables, developed for the paper **"Responding to baryons with the baryonic response metric"**.

## Overview

This project implements a novel **response formalism** to systematically measure how different regions of halo parameter space (mass, radius, redshift) contribute to baryonic effects on cosmological observables. By constructing "Replace" density fields—where dark matter only (DMO) halos are selectively replaced with their hydrodynamical counterparts from IllustrisTNG—we can isolate and quantify:

- Which halo masses dominate baryonic suppression of the matter power spectrum
- How far in radius (in units of R₂₀₀) one must model correctly
- Whether these requirements differ across statistics (power spectrum, angular power spectrum, peak counts, bispectrum)
- How baryonic correction models (BCMs) compare to hydrodynamical truth

### Key Scientific Questions Addressed

1. **Attribution**: Which halo masses, radii, and redshifts dominate the baryonic impact on a given observable?
2. **Self-consistency**: How can we test whether a BCM is physically consistent across multiple observables?
3. **Survey requirements**: What precision is needed in different regions of halo parameter space for Stage-IV surveys?

## Project Structure

```
hydro_replace2/
├── batch/                      # SLURM job submission scripts
│   ├── run_dmo_hydro_lensplanes.sh    # Generate DMO/Hydro lensplanes
│   ├── run_lux_binned_unified.sh      # Ray-tracing pipeline
│   └── run_unified_2500_binned_array.sh # Replace field generation
├── scripts/                    # Core computation scripts
│   ├── generate_all_unified.py        # Main pipeline: profiles, maps, lensplanes
│   ├── convert_to_lensplanes.py       # FFT conversion for ray-tracing
│   ├── generate_all_unified_bcm.py    # BCM variant pipeline
│   └── response_visualization.py      # Response function plotting
├── notebooks/                  # Analysis and visualization
│   ├── binned_pipeline.ipynb          # Full pipeline analysis & figures
│   ├── density_final.ipynb            # Matter power spectrum response
│   ├── bispectrum_final.ipynb         # Bispectrum response analysis
│   ├── peak_counts.ipynb              # Peak counts response
│   ├── 07_response_visualization_test.ipynb # Response visualization
│   ├── data_products_guide.ipynb      # Data access guide
│   └── figures/                       # Generated figures
├── config/                     # Configuration files
│   ├── analysis_params.yaml
│   ├── bcm_params.yaml
│   ├── raytrace_config.yaml
│   ├── simulation_paths.yaml
│   └── snapshot_list.yaml
├── docs/                       # Documentation
└── archive/                    # Archived scripts, notebooks, and reports
```

## Environment Setup

### Python Environment

Activate the project virtual environment:

```bash
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
```

### Key Dependencies

- **numpy**, **scipy**: Core numerical computation
- **h5py**: HDF5 file I/O for simulation data
- **mpi4py**: MPI parallelization for large-scale jobs
- **MAS_library**: Mass assignment schemes for density fields
- **matplotlib**: Visualization
- **astropy**: Astronomical utilities

### Data Locations

| Data Type | Location |
|-----------|----------|
| TNG Simulations | `/mnt/sdceph/users/sgenel/IllustrisTNG/` |
| Lensplanes (raw) | `/mnt/home/mlee1/ceph/hydro_replace_LP/` |
| Lensplanes (lux) | `/mnt/home/mlee1/ceph/hydro_replace_LP_lux/` |
| Ray-traced maps | `/mnt/home/mlee1/ceph/hydro_replace_RT/` |
| Density fields | `/mnt/home/mlee1/ceph/hydro_replace_fields/` |

## Pipeline Overview

### Phase 1-3: Profile & Map Generation

```bash
# Generate Replace density fields for a single snapshot
mpirun -np 64 python scripts/generate_all_unified.py \
    --snap 99 --sim-res 2500 --enable-lensplanes --mode binned
```

### Phase 4: Lensplane Generation

```bash
sbatch batch/run_unified_2500_binned_array.sh
```

### Phase 5: Ray-Tracing

```bash
sbatch batch/run_lux_binned_unified.sh
```

## Core Formalism

### Cumulative Response Fraction

For a statistic $S$ (e.g., power spectrum, peak counts), the cumulative response fraction measures how much of the total baryonic effect is captured when replacing halos above mass threshold $M_{\min}$ within radius $\alpha R_{200}$:

$$F_S(M_{\min}, \alpha) = \frac{S_R(M_{\min}, \alpha) - S_D}{S_H - S_D}$$

where:
- $S_R$: Statistic from Replace field
- $S_D$: Statistic from DMO field  
- $S_H$: Statistic from Hydro field

### Discrete Tile Response

For binned mass-radius tiles:

$$\Delta F_S(M_a, M_{a+1}; \alpha_i, \alpha_{i+1}) = \frac{S_R^{\text{tile}} - S_D}{S_H - S_D}$$

## Notebooks Guide

### [binned_pipeline.ipynb](notebooks/binned_pipeline.ipynb)
**Purpose**: Main analysis notebook for the paper  
**Produces**: 
- Fig. 1: Replace model grid visualization
- Cumulative and discrete response fractions
- Non-additivity diagnostics
- Cross-statistic comparisons

### [density_final.ipynb](notebooks/density_final.ipynb)
**Purpose**: 3D matter power spectrum analysis  
**Produces**:
- Fig. 2: Scale-dependent power spectrum response
- Fig. 3: Cumulative response summary heatmaps
- Fig. 4: Redshift evolution of response

### [bispectrum_final.ipynb](notebooks/bispectrum_final.ipynb)
**Purpose**: Matter bispectrum response analysis  
**Produces**:
- Bispectrum response fractions $F_B(k; M_{\min}, \alpha)$
- Scale and configuration dependence
- Comparison with power spectrum response

### [peak_counts.ipynb](notebooks/peak_counts.ipynb)
**Purpose**: Weak lensing peak count response  
**Produces**:
- Peak count distributions for DMO/Hydro/Replace
- Response as function of peak height ν
- Mass and radius dependence of peak statistics

### [07_response_visualization_test.ipynb](notebooks/07_response_visualization_test.ipynb)
**Purpose**: Response function visualization  
**Requires**: `scripts/response_visualization.py`  
**Produces**: Publication-quality response plots

### [data_products_guide.ipynb](notebooks/data_products_guide.ipynb)
**Purpose**: Guide to accessing and using data products

## Data Products

### Replace Lensplanes

Located at `/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG/`:

```
snap_{SNAP}/
├── dmo/LP_{REALIZATION:02d}/lensplane_{PLANE:02d}.npz
├── hydro/LP_{REALIZATION:02d}/lensplane_{PLANE:02d}.npz
└── replace_M{MASS_BIN}_R{RADIUS_BIN}/LP_{REALIZATION:02d}/lensplane_{PLANE:02d}.npz
```

### Convergence Maps

Located at `/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG/`:

```
{MODEL}/LP_{LP:02d}/run{RUN:03d}/
├── kappa{ZIDX:02d}.dat   # Binary lux format (kappa20 ≈ z_s ~ 1.0)
├── gamma1{ZIDX:02d}.dat
└── gamma2{ZIDX:02d}.dat
```

Loading convergence maps:
```python
def load_kappa(fname, ng=1024):
    """Load single kappa map from lux binary format."""
    with open(fname, 'rb') as f:
        dummy = np.fromfile(f, dtype="int32", count=1)
        kappa = np.fromfile(f, dtype="float", count=ng*ng)
        dummy = np.fromfile(f, dtype="int32", count=1)
    return kappa.reshape(ng, ng)
```

### Model Naming Convention

| Model Type | Example | Description |
|------------|---------|-------------|
| DMO | `dmo` | Dark matter only |
| Hydro | `hydro` | Full hydrodynamical |
| Cumulative | `hydro_replace_Ml_1.00e12_Mu_1.00e15_Ri_0.0_Ro_3.0` | M > 10¹² M⊙/h, r < 3R₂₀₀ |
| Discrete | `hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.5_Ro_1.0` | Single mass-radius tile |

**Naming format**: `hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}`
- `Ml/Mu`: Mass lower/upper bounds (M⊙/h). Use `Mu=1.00e15` for cumulative (all masses above Ml)
- `Ri/Ro`: Radius inner/outer bounds (× R₂₀₀). Use `Ri=0.0` for cumulative (from center)

### Snapshot-Redshift Mapping

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

**Stack** indicates high-z snapshots that use stacked (2× box) planes for ray-tracing.

### Source Plane Redshifts (kappa files)

The `kappa{NN}.dat` output files correspond to source planes at different redshifts:

| File | z_s | File | z_s | File | z_s | File | z_s |
|------|------|------|------|------|------|------|------|
| kappa01 | 0.034 | kappa11 | 0.418 | kappa21 | 0.914 | kappa31 | 1.615 |
| kappa02 | 0.070 | kappa12 | 0.462 | kappa22 | 0.973 | kappa32 | 1.703 |
| kappa03 | 0.105 | kappa13 | 0.506 | kappa23 | 1.034 | kappa33 | 1.794 |
| kappa04 | 0.142 | kappa14 | 0.552 | kappa24 | 1.097 | kappa34 | 1.889 |
| kappa05 | 0.179 | kappa15 | 0.599 | kappa25 | 1.163 | kappa35 | 1.989 |
| kappa06 | 0.216 | kappa16 | 0.648 | kappa26 | 1.231 | kappa36 | 2.094 |
| kappa07 | 0.255 | kappa17 | 0.698 | kappa27 | 1.302 | kappa37 | 2.203 |
| kappa08 | 0.294 | kappa18 | 0.749 | kappa28 | 1.375 | kappa38 | 2.319 |
| kappa09 | 0.335 | kappa19 | 0.803 | kappa29 | 1.452 | kappa39 | 2.440 |
| kappa10 | 0.376 | kappa20 | 0.858 | kappa30 | 1.532 | kappa40 | 2.568 |

**Commonly used**: `kappa20` (z_s ≈ 0.86), `kappa25` (z_s ≈ 1.16), `kappa30` (z_s ≈ 1.53)

## Key Results (Summary)

1. **Power spectrum**: Replacing halos M > 10¹² h⁻¹M⊙ out to 5R₂₀₀ captures ~90% of baryonic suppression
2. **Angular power spectrum**: Less stringent—R₂₀₀ replacement already captures ~85% at z~1
3. **Peak counts**: Dominated by massive halos (M > 10¹³ h⁻¹M⊙) within ~R₂₀₀
4. **Redshift evolution**: Response peaks near cosmic noon (z~1) for low-mass halos

## Batch Job Submission

```bash
# Generate all Replace lensplanes (200 array jobs)
sbatch batch/run_unified_2500_binned_array.sh

# Generate DMO/Hydro lensplanes (20 array jobs)
sbatch batch/run_dmo_hydro_lensplanes.sh

# Run ray-tracing (2040 array jobs)
sbatch batch/run_lux_binned_unified.sh
```

## Citation

If you use this code or data products, please cite:

```bibtex
@article{author2026baryonic,
  title={Responding to baryons with the baryonic response metric},
  author={...},
  journal={...},
  year={2026}
}
```

## Contact

For questions about the code or data products, contact the repository maintainer.

## License

[Add license information]
