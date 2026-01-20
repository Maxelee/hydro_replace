# Data Products Reference

This document lists all data products created by the hydro_replace2 pipeline and their locations.

## Pipeline Scripts and Data Output Locations

### 1. `run_unified_2500_array.sh` → `generate_all_unified.py`

**Purpose**: Generate DMO, Hydro, and Replace density fields and lensplanes for L205n2500TNG

**Output Locations**:
- **Lens planes**: `/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG/`
  - Structure: `{model}/LP_{realization:02d}/lenspot{file_idx:02d}.dat`
  - Models: `dmo`, `hydro`, `hydro_replace_{mass_bin}_R_{r_factor}`
  - Realizations: LP_00 to LP_09 (10 total)
  - File indices: 0-39 (20 snapshots × 2 planes per snapshot)
  - Binary format: `int32(grid) + float64[grid×grid](Σ×Δχ) + int32(grid)`

- **Profiles** (not produced by 2500 array, but referenced):
  - `/mnt/home/mlee1/ceph/hydro_replace_fields/L205n2500TNG/profiles/`
  - Files: `profiles_snap{NNN:03d}.h5`

**Mass Configurations (8 total)**:
| Label | Mass Range [M☉/h] | Type |
|-------|-------------------|------|
| `Ml_1.00e12_Mu_3.16e12` | 10^12.0 - 10^12.5 | Discrete |
| `Ml_3.16e12_Mu_1.00e13` | 10^12.5 - 10^13.0 | Discrete |
| `Ml_1.00e13_Mu_3.16e13` | 10^13.0 - 10^13.5 | Discrete |
| `Ml_3.16e13_Mu_1.00e15` | 10^13.5 - 10^15.0 | Discrete |
| `Ml_1.00e12_Mu_inf` | > 10^12.0 | Cumulative |
| `Ml_3.16e12_Mu_inf` | > 10^12.5 | Cumulative |
| `Ml_1.00e13_Mu_inf` | > 10^13.0 | Cumulative |
| `Ml_3.16e13_Mu_inf` | > 10^13.5 | Cumulative |

**Radius Factors (4 total)**: α ∈ {0.5, 1.0, 3.0, 5.0} × R_200

**Total Replace Configurations**: 8 mass bins × 4 radius factors = 32 models + DMO + Hydro = 34 models

**Snapshots Processed (20 total)**:
| Array Index | Snapshot | Redshift |
|-------------|----------|----------|
| 0 | 96 | 0.02 |
| 1 | 90 | 0.10 |
| 2 | 85 | 0.18 |
| 3 | 80 | 0.27 |
| 4 | 76 | 0.35 |
| 5 | 71 | 0.46 |
| 6 | 67 | 0.55 |
| 7 | 63 | 0.65 |
| 8 | 59 | 0.76 |
| 9 | 56 | 0.85 |
| 10 | 52 | 0.97 |
| 11 | 49 | 1.08 |
| 12 | 46 | 1.21 |
| 13 | 43 | 1.36 |
| 14 | 41 | 1.47 |
| 15 | 38 | 1.63 |
| 16 | 35 | 1.82 |
| 17 | 33 | 1.97 |
| 18 | 31 | 2.14 |
| 19 | 29 | 2.32 |

---

### 2. `run_unified_bcm_array.sh` → `generate_all_unified_bcm.py`

**Purpose**: Generate BCM-corrected density fields and lensplanes

**Output Locations**:
- **Lens planes**: `/mnt/home/mlee1/ceph/hydro_replace_LP_bcm/L205n2500TNG/`
  - Structure: `{model}/LP_{realization:02d}/lenspot{file_idx:02d}.dat`
  - Models: `schneider19`, `schneider25`, `arico20`
  - Same binary format as main pipeline

- **Profiles**: `/mnt/home/mlee1/ceph/hydro_replace_fields_bcm/L205n2500TNG/profiles/`
  - Files: `profiles_{model}_snap{NNN:03d}.h5`

**BCM Models**:
| Model | Description | Key Parameters |
|-------|-------------|----------------|
| `schneider19` | Schneider+2019 baryonification | θ_ej=4, M_c=1.1e13, μ_β=0.55 |
| `schneider25` | Schneider+2025 updated gas | θ_c=0.3, M_c=1e15, μ=0.8 |
| `arico20` | Aricò+2020 model | M_c=1.2e14, β=0.6, θ_out=3.0 |

---

### 3. `run_lux_pipeline_array.sh` → Lux ray-tracing

**Purpose**: Perform ray-tracing through lensplanes to generate convergence maps

**Output Location**: `/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG/`
- Structure: `{model}/realization_{NNN:03d}/...`

---

## Key File Formats

### Lensplane Binary (`.dat`)
```
int32:    grid_size (e.g., 4096)
float64:  field[grid_size × grid_size] - surface density × comoving distance (Σ × Δχ)
int32:    grid_size (footer, for verification)
```

### Profile HDF5 (`.h5`)
```
Attributes:
  - radial_bins: array of radial bin edges (in units of R_200)
  - mass_bin_edges: array of log10(M) edges
  
Datasets:
  - stacked_dmo: (n_mass_bins, n_radial_bins) - DMO density profiles
  - stacked_hydro: (n_mass_bins, n_radial_bins) - Hydro density profiles
  - stacked_bcm: (n_mass_bins, n_radial_bins) - BCM density profiles (BCM files only)
  - individual_dmo_density, individual_hydro_density: (n_halos, n_radial_bins)
  - halo_log_masses, halo_radii_dmo, halo_radii_hydro
  - mass_bin_indices, n_halos_per_bin
```

---

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Box Size | 205 Mpc/h |
| Simulation | L205n2500TNG (Hydro), L205n2500TNG_DM (DMO) |
| Mass Unit | 10^10 M☉/h |
| Lens plane grid | 4096 |
| Ray-tracing grid | 1024 |
| Realizations | 10 (LP_00 to LP_09) |
| Planes per snapshot | 2 |

---

## Usage in formalism.ipynb

The notebook uses these paths:
```python
# Main lens planes
LENSPLANE_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG')

# BCM lens planes
BCM_LP_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_LP_bcm/L205n2500TNG')

# Profiles
PROFILE_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_fields/L205n2500TNG/profiles')
BCM_PROFILE_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_fields_bcm/L205n2500TNG/profiles')

# Ray-tracing output
RT_BASE = Path('/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG')
```
