# Full Pipeline Documentation

## Methods: Comparing Baryonic Effects on Weak Lensing Observables

This document describes the complete computational pipeline for generating weak lensing convergence maps from cosmological simulations, comparing different methods for incorporating baryonic physics effects.

---

## 1. Overview

### 1.1 Scientific Motivation

Weak gravitational lensing is sensitive to the total matter distribution along the line of sight. While dark matter dominates the mass budget, baryonic processes (gas cooling, star formation, AGN feedback) redistribute matter on scales relevant to cosmic shear measurements. This pipeline compares four approaches:

1. **DMO (Dark Matter Only)**: Collisionless N-body simulation
2. **Hydro**: Full hydrodynamic simulation (IllustrisTNG physics)
3. **Replace**: Hybrid method replacing DMO halos with matched Hydro counterparts
4. **BCM (Baryonic Correction Model)**: Analytical prescription applied to DMO

### 1.2 Pipeline Stages

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Matching    │ ──> │  2. 2D Maps     │ ──> │  3. Lens Planes │ ──> │  4. Ray-Tracing │
│  (DMO ↔ Hydro)  │     │  (Projected ρ)  │     │  (δ × dz)       │     │  (κ maps)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 1.3 Simulation Data

| Simulation | Resolution | Particles | Box Size | Mass Resolution |
|------------|------------|-----------|----------|-----------------|
| L205n2500TNG | 2500³ | 15.6 billion | 205 Mpc/h | 4.7×10⁻³ (10¹⁰ M☉/h) |
| L205n2500TNG_DM | 2500³ | 15.6 billion | 205 Mpc/h | 4.7×10⁻³ (10¹⁰ M☉/h) |

Data location: `/mnt/sdceph/users/sgenel/IllustrisTNG/`

---

## 2. Stage 1: Halo Matching

### 2.1 Purpose

Establish a bijective mapping between halos in the DMO and Hydro simulations. This enables:
- Direct comparison of the same structures across simulations
- The "Replace" method which swaps DMO halos with their Hydro counterparts

### 2.2 Algorithm

**Script**: `scripts/generate_matches_fast.py`

The matching algorithm uses a two-step approach:

#### Step 1: Spatial Candidate Selection
For each DMO halo, identify Hydro halos within a search radius:
```
r_search = max(3 × R_200,crit, 1.0 Mpc/h)
```

This uses a KD-tree for efficient O(N log N) spatial queries with periodic boundary handling.

#### Step 2: Bijective Matching via Mass Ranking

For each DMO halo (sorted by mass, descending):
1. Find all unmatched Hydro candidates within `r_search`
2. Compute mass ratio: `η = M_hydro / M_dmo`
3. Accept match if `0.3 < η < 3.0` (factor of 3 tolerance)
4. Select the Hydro halo with mass ratio closest to unity
5. Mark both halos as matched (bijective constraint)

The mass-descending order ensures massive halos (which are rarer and more important for lensing) get optimal matches first.

### 2.3 Output Format

**File**: `matches/matches_snap{NNN}.npz`

```python
{
    'dmo_indices': np.array,      # Indices into DMO halo catalog
    'hydro_indices': np.array,    # Corresponding Hydro halo indices
    'dmo_masses': np.array,       # M_200,crit [10^10 M☉/h]
    'hydro_masses': np.array,     # M_200,crit [10^10 M☉/h]
    'dmo_positions': np.array,    # [Mpc/h], shape (N, 3)
    'hydro_positions': np.array,  # [Mpc/h], shape (N, 3)
}
```

### 2.4 Validation Metrics

- **Mass ratio distribution**: Should peak near 1.0 with scatter ~0.1-0.2 dex
- **Position offset**: Median < 100 kpc/h for well-matched pairs
- **Completeness**: >95% for M > 10^13 M☉/h

---

## 3. Stage 2: 2D Projected Density Maps

### 3.1 Purpose

Generate 2D projected surface density fields (Σ) for each model variant. These serve as:
- Intermediate validation products
- Input for lens plane generation

### 3.2 Physics

The surface density is computed by projecting particles along the z-axis:

$$\Sigma(x, y) = \int_{-L/2}^{L/2} \rho(x, y, z) \, dz$$

For discrete particles, this becomes a mass assignment to a 2D grid.

### 3.3 Mass Assignment: Triangular Shaped Cloud (TSC)

**Script**: `scripts/generate_all.py`

We use TSC (second-order) mass assignment for consistency with the lux ray-tracing code:

$$W_{TSC}(x) = \begin{cases}
\frac{3}{4} - x^2 & |x| \leq \frac{1}{2} \\
\frac{1}{2}\left(\frac{3}{2} - |x|\right)^2 & \frac{1}{2} < |x| \leq \frac{3}{2} \\
0 & |x| > \frac{3}{2}
\end{cases}$$

Each particle contributes to a 3×3 grid cell neighborhood (in 2D) with weights:
```python
W(x, y) = W_TSC(x - x_grid) × W_TSC(y - y_grid)
```

Implementation uses `MAS_library` from Pylians:
```python
MASL.MA(positions, field, BoxSize, MAS='TSC', W=masses)
```

### 3.4 Model Variants

#### 3.4.1 DMO
Direct projection of all dark matter particles:
```python
Σ_DMO(x, y) = Σ_i m_i × W(x - x_i, y - y_i)
```

#### 3.4.2 Hydro
Projection of dark matter + gas + stars:
```python
Σ_Hydro = Σ_DM + Σ_gas + Σ_stars
```

#### 3.4.3 Replace

The Replace method creates a hybrid density field:

1. **Identify replacement regions**: Spheres of radius `R_replace = 5 × R_200` around matched halos above mass threshold
2. **Remove DMO particles** within replacement regions
3. **Insert Hydro particles** (DM + gas + stars) from matched halos

Mathematically:
$$\Sigma_{Replace} = \Sigma_{DMO} \times (1 - M_{halo}) + \Sigma_{Hydro} \times M_{halo}$$

where M_halo is a binary mask for halo regions.

**Mass threshold variants**: M > 10^{12.5}, 10^{13}, 10^{13.5}, 10^{14} M☉/h

#### 3.4.4 BCM (Baryonic Correction Model)

BCM analytically modifies the DMO density profile around halos using physically-motivated prescriptions.

**Implementation**: Uses `BaryonForge` package with three calibrations:
- **Arico20**: Calibrated to hydrodynamic simulations (Aricò et al. 2020)
- **Schneider19**: Original BCM (Schneider & Teyssier 2015, updated 2019)
- **Schneider25**: Updated calibration (Schneider et al. 2025)

The BCM modifies the halo density profile:
$$\rho_{BCM}(r) = \rho_{DMO}(r) \times \frac{\rho_{model}(r)}{\rho_{NFW}(r)}$$

where ρ_model includes:
- **Bound gas**: Hot halo gas in hydrostatic equilibrium
- **Ejected gas**: Gas expelled by feedback to large radii
- **Central galaxy**: Stellar component following a Hernquist profile
- **Adiabatic relaxation**: Response of dark matter to baryon concentration

### 3.5 Output Format

**Directory**: `L205n{RES}TNG/snap{NNN}/projected/`

**Files**:
- `dmo.npz`: DMO surface density
- `hydro.npz`: Hydro surface density  
- `replace_Mgt{X.X}.npz`: Replace with mass threshold
- `bcm_{Model}_Mgt{X.X}.npz`: BCM variant

**Structure**:
```python
{
    'field': np.array,  # Shape (N_grid, N_grid), units: M☉/h / (Mpc/h)²
}
```

### 3.6 MPI Parallelization

Particles are distributed across MPI ranks by spatial decomposition:
1. Each rank loads particles from assigned snapshot files
2. Local density field computed on each rank
3. Global reduction via `MPI.Reduce` with `MPI.SUM`
4. Rank 0 writes final output

---

## 4. Stage 3: Lens Plane Generation

### 4.1 Purpose

Generate lens planes for ray-tracing by applying random rotations and translations to the 3D particle distribution, projecting to 2D, and converting to lens plane format.

**Important**: This stage works directly with 3D particle positions from the simulation, NOT with the 2D density maps from Stage 2. This is necessary because:
- Each random realization requires a unique rotation/translation
- These transformations must be applied in 3D before projection
- Different realizations cannot share the same 2D maps

The outputs are:
- Density contrast fields δ
- Scaled by comoving distance intervals dz
- Binary format compatible with lux ray-tracing code

### 4.2 Physics

The lens plane stores the dimensionless quantity:
$$\delta \times \chi \times \Delta\chi$$

where:
- δ = (ρ - ρ̄)/ρ̄ is the density contrast
- χ is the comoving distance to the lens plane
- Δχ is the comoving thickness of the lens plane

For weak lensing, the convergence κ is:
$$\kappa(\vec{\theta}) = \frac{3H_0^2\Omega_m}{2c^2} \int_0^{\chi_s} \frac{\chi(\chi_s - \chi)}{\chi_s} \frac{\delta(\chi\vec{\theta}, \chi)}{a(\chi)} d\chi$$

### 4.3 Snapshot Configuration

**Script**: `scripts/generate_lensplanes.py`

20 snapshots spanning z = 0.04 to z = 2.87:

| Snapshot | Redshift | Stack | Comoving Distance |
|----------|----------|-------|-------------------|
| 96 | 0.04 | No | ~120 Mpc/h |
| 90 | 0.15 | No | ~430 Mpc/h |
| ... | ... | ... | ... |
| 29 | 2.87 | Yes | ~2800 Mpc/h |

**Stacking**: At high redshift (z > 1.2), the box is doubled in the transverse direction to cover the larger angular diameter distance.

### 4.4 Algorithm

For each snapshot and model, the lens plane generation must be performed at the particle level to allow independent random transformations for each realization. This is NOT done post-hoc on the 2D maps from Stage 2.

#### Randomization Strategy

The transformation matches lux's behavior exactly:
1. **Random displacement**: Uniform translation vector in [0, L_box)³
2. **Random flip**: Independent sign flip (negate) for each axis
3. **Random projection direction**: x, y, or z axis

This provides 3 × 2³ = 24 discrete orientations plus continuous translations.

**Note**: This is NOT a full rotation matrix - it's discrete flips plus translation, which is sufficient for decorrelating lens planes while being computationally simpler.

#### For each realization (seed):

1. **Generate random transformation parameters**:
   ```python
   class RandomizationState:
       proj_dir = rng.integers(0, 3)           # Projection axis (x=0, y=1, z=2)
       displacement = rng.uniform(0, L_box, 3)  # Translation vector
       flip = rng.integers(0, 2, 3).astype(bool)  # Sign flip per axis
   ```
   
2. **Load 3D particle positions and masses**:
   - Load particles from simulation snapshots
   - For Replace: Use matched halo replacements (as in Stage 2)
   - For BCM: Apply density profile modifications

3. **Apply transformation in 3D** (matching lux order):
   ```python
   # Step 1: Add displacement
   x_transformed = x_particle + displacement
   
   # Step 2: Apply flips (negate coordinates)
   for axis in range(3):
       if flip[axis]:
           x_transformed[:, axis] = -x_transformed[:, axis]
   
   # Step 3: Apply periodic boundary conditions
   x_transformed = np.mod(x_transformed, L_box)
   ```

4. **Project to 2D** along the chosen projection axis using TSC mass assignment:
   ```python
   # proj_dir determines which axis to project along
   # proj_dir=2 (z): keep (x, y) -> most common
   Σ(x, y) = Σ_i m_i × W_TSC(x - x_i) × W_TSC(y - y_i)
   ```

5. **Compute density contrast**:
   ```python
   Σ_mean = Ω_m × ρ_crit × L_box  # Expected mean surface density
   δ = (Σ - Σ_mean) / Σ_mean
   ```

6. **Scale by comoving distance interval**:
   ```python
   Δχ = L_box / planes_per_snapshot  # Comoving thickness in Mpc/h
   output = δ × Δχ
   ```

7. **Write binary format** compatible with lux
8. **Repeat** for each random seed/realization

### 4.5 Output Format

**Directory**: `/mnt/home/mlee1/ceph/hydro_replace_lensplanes/L205n{RES}TNG/{model}/`

**Structure**: One directory per model variant (dmo, hydro, replace_Mgt{X.X}, bcm_Mgt{X.X})

**Contents**: For each realization (multiple random seeds):
- `lens_plane_snap{NNN}_seed{SEED}_plane{P}.bin`: Binary lens plane file

**Binary format** (for lux PreProjected mode):
```
[int32: grid_size]
[float64[N×N]: δ × Δχ values, row-major]
[int32: grid_size]  # Footer for validation
```

where:
- `grid_size`: 4096 (LP_grid parameter)
- Each element: density contrast × comoving distance interval
- Row-major ordering: fastest-varying index is x (column index)
- Periodic boundary conditions applied during projection

---

## 5. Stage 4: Ray-Tracing

### 5.1 Purpose

Trace light rays through the lens plane stack to generate convergence (κ) maps. This is the final observable that can be compared to weak lensing data.

### 5.2 The Lux Code

**Location**: `/mnt/home/mlee1/lux/`

Lux performs ray-tracing in two phases:

#### Phase 1: Lens Potential (lenspot)
Solve the 2D Poisson equation on each lens plane:
$$\nabla^2 \psi = 2\kappa = \frac{3H_0^2\Omega_m}{c^2} \frac{\chi}{a} \delta \times \Delta\chi$$

Uses FFT-based Poisson solver:
$$\hat{\psi}(\vec{k}) = -\frac{2\hat{\kappa}(\vec{k})}{|\vec{k}|^2}$$

#### Phase 2: Ray-Tracing
Propagate rays through the lens plane stack using the multi-plane lens equation:

$$\vec{\theta}_{i+1} = \vec{\theta}_i - \frac{D_{i+1,i}}{D_{i+1}} \nabla\psi_i(\vec{\theta}_i)$$

where D_{ij} are angular diameter distances.

The convergence is accumulated:
$$\kappa_{total} = \sum_i \kappa_i \times \frac{D_{is} D_i}{D_s}$$

### 5.3 Configuration

**Script**: `batch/run_lux_all.sh`

Key parameters:
```
LP_grid = 4096        # Lens potential grid resolution
RT_grid = 1024        # Ray-tracing output resolution
planes_per_snapshot = 2
RT_randomization = 100  # Number of convergence maps per realization
angle = 5.0           # Field of view in degrees
```

### 5.4 Output

**Directory**: `/mnt/home/mlee1/ceph/lux_out/L205n{RES}TNG/{model}/`

**Files**:
- `lenspot/`: Lens potential maps (intermediate)
- `raytracing/kappa_{NNN}.fits`: Convergence maps

Each model produces 100 κ maps (from RT_randomization) per random realization of rotations/translations.

---

## 6. Batch Job Dependencies

### 6.1 SLURM Job Chain

```bash
# Stage 1: Matches (serial, 1 node per snapshot)
sbatch run_all_matches.sh  # Job A

# Stage 2: Maps (parallel, 4 nodes per snapshot)
sbatch --dependency=aftercorr:A run_maps_pending.sh   # Job B (waits on matches)
sbatch run_maps_completed.sh                           # Job C (no dependency)

# Stage 3: Lens Planes (parallel, 4 nodes per model)
sbatch --dependency=afterok:B:C run_lensplanes_all.sh  # Job D

# Stage 4: Ray-Tracing (parallel, 4 nodes per model)
sbatch --dependency=aftercorr:D run_lux_all.sh         # Job E
```

### 6.2 Dependency Types

- `afterok`: Wait for ALL tasks to complete successfully
- `aftercorr`: Wait for CORRESPONDING array task (enables pipeline parallelism)

---

## 7. File Locations Summary

| Data Product | Location |
|--------------|----------|
| Simulation data | `/mnt/sdceph/users/sgenel/IllustrisTNG/` |
| Matches | `/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{RES}TNG/matches/` |
| 2D Maps | `/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{RES}TNG/snap{NNN}/projected/` |
| Lens Planes | `/mnt/home/mlee1/ceph/hydro_replace_lensplanes/L205n{RES}TNG/{model}/` |
| Ray-Tracing Output | `/mnt/home/mlee1/ceph/lux_out/L205n{RES}TNG/{model}/` |

---

## 8. Validation Checkpoints

### 8.1 Matches
- Mass ratio histogram should peak at ~0.95-1.0
- Position offsets should be < 200 kpc/h for 90% of pairs
- Completeness > 95% for M > 10^13 M☉/h

### 8.2 2D Maps
- Visual inspection: Hydro should show reduced power on small scales
- Power spectrum ratio: P_Hydro/P_DMO < 1 at k > 1 h/Mpc

### 8.3 Lens Planes
- Mean should be ~0 (density contrast)
- RMS should be ~0.1-1 depending on redshift

### 8.4 Convergence Maps
- PDF should follow expected skewed distribution
- Power spectrum should match theoretical predictions

---

## 9. References

1. **IllustrisTNG**: Pillepich et al. (2018), Nelson et al. (2019)
2. **BCM - Arico20**: Aricò et al. (2020), MNRAS 495, 4800
3. **BCM - Schneider**: Schneider & Teyssier (2015), JCAP 12, 049
4. **BaryonForge**: https://github.com/DhayaaAnbajagane/BaryonForge
5. **Lux**: Custom ray-tracing code (internal)
6. **Pylians**: https://pylians3.readthedocs.io/

---

## 10. Code Repository

**Location**: `/mnt/home/mlee1/hydro_replace2/`

```
scripts/
├── generate_matches_fast.py   # Stage 1: Halo matching
├── generate_all.py            # Stage 2: 2D maps
├── generate_lensplanes.py     # Stage 3: Lens planes
└── generate_lux_configs.py    # Stage 4: Lux configuration

batch/
├── run_all_matches.sh         # SLURM: Matching jobs
├── run_maps_completed.sh      # SLURM: Maps for existing matches
├── run_maps_pending.sh        # SLURM: Maps waiting on matches
├── run_lensplanes_all.sh      # SLURM: Lens plane generation
└── run_lux_all.sh             # SLURM: Ray-tracing
```
