# Hydro Replace Project: Technical Architecture

## Overview

This document outlines the technical architecture for the hydro replacement analysis across multiple redshifts, integration with ray-tracing (lux), and comparison with BCM models.

---

## 1. Ray-Tracing Requirements (lux)

### Snapshots Required
From `lux.ini`, the ray-tracing uses **20 snapshots** spanning z ≈ 0 to z ≈ 2:

| Snapshot | z (approx) | Stack | Notes |
|----------|------------|-------|-------|
| 96 | 0.04 | false | Individual plane |
| 90 | 0.15 | false | |
| 85 | 0.27 | false | |
| 80 | 0.40 | false | |
| 76 | 0.50 | false | |
| 71 | 0.64 | false | |
| 67 | 0.78 | false | |
| 63 | 0.93 | false | |
| 59 | 1.07 | false | |
| 56 | 1.18 | false | |
| 52 | 1.36 | true | Stacked (2× box) |
| 49 | 1.50 | true | |
| 46 | 1.65 | true | |
| 43 | 1.82 | true | |
| 41 | 1.93 | true | |
| 38 | 2.12 | true | |
| 35 | 2.32 | true | |
| 33 | 2.49 | true | |
| 31 | 2.68 | true | |
| 29 | 2.87 | true | |

### lux Data Loading
From `read_hdf.cpp`, lux loads:
- **PartType0** (gas): Coordinates, Masses
- **PartType1** (DM): Coordinates, MassTable[1]
- **PartType4** (stars): Coordinates, Masses

For DMO runs (`IllustrisTNG-Dark`): Only PartType1

### Lens Plane Configuration
- **Grid**: 4096 × 4096
- **Planes per snapshot**: 2
- **Total planes**: 40
- **Field of view**: 5 × 5 deg²
- **Output**: Lens potential φ(θ) → κ(θ) convergence maps

---

## 2. Multi-Snapshot Replacement Strategy

### Challenge
For ray-tracing, we need replaced snapshots for **all 20 redshifts**, not just z=0.

### Approach Options

#### Option A: Full Snapshot Replacement (Most Accurate)
For each of 20 snapshots:
1. Load halo catalogs at that redshift
2. Match halos between hydro and DMO (bijective matching)
3. Extract particles within 5×R_200c from hydro halos
4. Replace DMO particles with hydro particles
5. Save modified snapshot files (HDF5)

**Pros**: Most accurate, works directly with lux  
**Cons**: ~100 TB storage, very slow, complex

#### Option B: Lens Plane Replacement (Recommended)
1. Run lux to generate lens planes from DMO simulation
2. Run lux to generate lens planes from hydro simulation
3. For each snapshot/plane, compute replacement at 2D projected level
4. Modify lens plane files directly

**Pros**: Much smaller storage (~10 GB), faster iteration  
**Cons**: Projection effects, less accurate at small scales

#### Option C: Hybrid Approach
1. Generate 3D pixelized density grids per snapshot (like existing data)
2. Feed grids to modified lux (requires lux code changes)
3. Combine DMO/hydro/replaced at grid level

**Pros**: Leverages existing infrastructure  
**Cons**: Requires lux modification

### REVISED: Option D (Direct lux Modification)

**Decision**: After reviewing existing data products, lens plane replacement is NOT the best approach. Instead, we will:

1. **Modify lux** to read pre-baryonified snapshot data (HDF5 format)
2. **Create branch**: `git checkout -b hydro_replace` in `/mnt/home/mlee1/lux/`
3. **Leverage existing halo-level BCM**: 519 halos already processed with Arico20

**Why this approach?**
- We already have per-halo BCM outputs (`/mnt/home/mlee1/ceph/baryonification_output/halos/`)
- Full 3D coordinates preserved (no projection artifacts)
- Can extend to multiple BCM models without re-running lux
- Clean separation: BCM processing → snapshot replacement → ray-tracing

**Implementation**:
1. Write `create_replaced_snapshot.py` that combines BCM halo outputs into full snapshot
2. Modify `lux/read_hdf.cpp` to read our custom HDF5 format (or convert to TNG-like format)
3. Run lux on replaced snapshots for each configuration

---

## 3. Redshift Evolution Analysis

### Key Science Questions
1. How does P(k) suppression evolve with z?
2. Does replacement radius dependence change with z?
3. What mass scale dominates at different z?

### Implementation
```python
# Pseudocode for redshift evolution
snapshots = [99, 96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
redshifts = [0.0, 0.04, 0.15, 0.27, 0.40, 0.50, 0.64, 0.78, 0.93, 1.07, 1.18, 
             1.36, 1.50, 1.65, 1.82, 1.93, 2.12, 2.32, 2.49, 2.68, 2.87]

for snap, z in zip(snapshots, redshifts):
    # 1. Load halo catalog at this z
    # 2. Compute density grids (DMO, hydro, replaced)
    # 3. Compute P(k) for each
    # 4. Store S(k, z) = P_hydro/P_DMO
```

---

## 4. BCM Comparison (BaryonForge)

### Available Models
From BaryonForge repository:

| Model | Reference | Key Features |
|-------|-----------|--------------|
| `Schneider19` | Schneider+2015, 2019 | Original BCM, gas ejection + stars |
| `Arico20` | Arico+2020, 2021 | TNG-calibrated, improved gas model |
| `Mead20` | Mead+2020 | HMcode parametrization |
| `Schneider25` | Schneider+2025 | Latest, hot/inner gas separation |

### Profile Components
Each model provides:
- `DarkMatter`: NFW profile (modified concentration)
- `Gas`: Hot/ejected gas distribution
- `Stars`: Central + satellite stellar profiles
- `CollisionlessMatter`: Combined DM + baryons after relaxation
- `DarkMatterBaryon`: Full profile (1-halo + 2-halo)

### Integration Plan
```python
import BaryonForge as bfg

# Initialize models
bcm_schneider = bfg.Profiles.Schneider19.DarkMatterBaryon(M_c=1e14, theta_ej=4)
bcm_arico = bfg.Profiles.Arico20.DarkMatterBaryon(M_c=1e14)
bcm_mead = bfg.Profiles.Mead20.DarkMatterBaryon(M_0=1e14)

# Compute profiles at halo positions
for model in [bcm_schneider, bcm_arico, bcm_mead]:
    rho_bcm = model.real(cosmo, r, M_200c, a)
    
# Apply BCM to DMO halos
# Compare: ρ_BCM(r) vs ρ_hydro(r) vs ρ_replaced(r)
```

---

## 5. Data Management Strategy

### The Challenge
- TNG-300 box: 2500³ particles × 20 snapshots = massive I/O
- Each snapshot: ~600 files × ~10 GB = ~6 TB
- Total: ~120 TB for all snapshots

### Strategy: Incremental Processing

#### Level 1: Halo Catalogs (Small, ~1 GB total)
```
/mnt/home/mlee1/ceph/hydro_replace/catalogs/
├── halo_catalog_snap{NN}_hydro.h5
├── halo_catalog_snap{NN}_dmo.h5
└── matched_catalog_snap{NN}.h5
```

#### Level 2: Extracted Particles (Medium, ~100 GB per snapshot)
Only for halos M > 10^12 M☉, within 5×R_200c
```
/mnt/home/mlee1/ceph/hydro_replace/extracted/
├── snap{NN}/
│   ├── halo_{ID}_hydro.h5
│   └── halo_{ID}_dmo.h5
```

#### Level 3: Pixelized Grids (Large, ~8 GB per config)
```
/mnt/home/mlee1/ceph/hydro_replace/pixelized/
├── snap{NN}/
│   ├── density_dmo_4096.npz
│   ├── density_hydro_4096.npz
│   └── density_replaced_rad{R}_mass{M}.npz
```

#### Level 4: Power Spectra (Small, ~1 MB per config)
```
/mnt/home/mlee1/ceph/hydro_replace/power_spectra/
├── Pk_snap{NN}_dmo.npz
├── Pk_snap{NN}_hydro.npz
└── Pk_snap{NN}_replaced_rad{R}_mass{M}.npz
```

### Parallelization Strategy

#### File-Level Parallelism (within snapshot)
- TNG-300 has 600 files per snapshot
- Distribute files across MPI ranks
- Each rank builds partial density grid
- MPI_Reduce to combine grids

```python
# MPI pattern for file distribution
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_files = 600
files_per_rank = n_files // size
my_files = range(rank * files_per_rank, (rank + 1) * files_per_rank)

for file_idx in my_files:
    load_and_grid_particles(file_idx)

# Reduce grids
comm.Reduce(local_grid, global_grid, op=MPI.SUM, root=0)
```

#### Snapshot-Level Parallelism
- For independent snapshots, submit separate SLURM jobs
- Use job arrays: `sbatch --array=0-19 process_snapshot.sh`

---

## 6. Implementation Phases

### Phase 1: Single Snapshot Validation (Week 1)
- [ ] Verify existing z=0 data products
- [ ] Test power spectrum computation on pixelized maps
- [ ] Compare with known TNG P(k) from literature

### Phase 2: Multi-Snapshot Extension (Weeks 2-3)
- [ ] Extend pipeline to arbitrary snapshot
- [ ] Process 5 key snapshots: z = 0, 0.5, 1.0, 1.5, 2.0
- [ ] Analyze S(k, z) evolution

### Phase 3: BCM Comparison (Weeks 3-4)
- [ ] Install/configure BaryonForge
- [ ] Compute BCM profiles for matched halos
- [ ] Compare profiles: hydro vs BCM vs replaced

### Phase 4: Ray-Tracing Integration (Weeks 4-6)
- [ ] Generate lens planes from replaced snapshots
- [ ] Run lux ray-tracing
- [ ] Compute convergence maps and statistics

### Phase 5: Paper-Ready Analysis (Weeks 6-8)
- [ ] Peak statistics comparison
- [ ] PDF analysis
- [ ] Systematic uncertainties

---

## 7. Key Files to Create/Modify

### New Scripts
1. `scripts/09_multi_snapshot_processing.py` - Process multiple z
2. `scripts/10_bcm_comparison.py` - BaryonForge integration
3. `scripts/11_lensplane_replacement.py` - Modify lens planes
4. `scripts/12_redshift_evolution.py` - S(k, z) analysis

### New Batch Scripts
1. `batch/process_all_snapshots.sh` - SLURM array job
2. `batch/run_bcm.sh` - BCM computation
3. `batch/lux_replaced.sh` - Ray-tracing on replaced

### Configuration Updates
1. `config/snapshot_list.yaml` - All 20 snapshots with z
2. `config/bcm_params.yaml` - BCM model parameters
3. `config/raytrace_config.yaml` - lux configuration

---

## 8. Dependencies

### Python Packages
```
numpy
scipy
h5py
mpi4py
pyyaml
matplotlib
illustris_python
MAS_library (Pylians3)
pyccl  # For BaryonForge
BaryonForge  # BCM models
```

### System Requirements
- MPI (OpenMPI or MPICH)
- HDF5 with parallel I/O
- FFTW3 (for lux)
- GSL (for lux)
- Boost MPI + serialization (for lux)

---

## References

1. Osato, Liu & Haiman 2021 (κTNG): Ray-tracing methodology
2. Miller+2025 (arXiv:2511.10634): Mass redistribution validation
3. Arico+2020, 2021: BCM calibration on TNG
4. Schneider+2015, 2019: Original BCM model
5. Mead+2020: HMcode BCM parametrization
