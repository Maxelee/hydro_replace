# Unified Pipeline with Multi-Configuration Lensplane Generation

## Overview

This document describes the unified pipeline that generates:
1. Radial density profiles around halos
2. Halo statistics (baryon fractions, mass conservation)
3. 2D projected density maps
4. **Lensplanes for ray-tracing** with multiple (mass_bin, R_factor) configurations

The key innovation is that **KDTree queries are reused** across many configurations, enabling efficient generation of lensplanes for different mass selections and excision radii without rebuilding expensive data structures.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Load Particles & Build KDTrees (ONE TIME)            │
│  - Load DMO particles (pos, mass)                               │
│  - Load Hydro particles (pos, mass, type)                       │
│  - Build KDTree_dmo, KDTree_hydro                               │
│  - Load halo catalog (positions, R200, M200)                    │
│  Time: ~30-60s | Memory: particles + 2 KDTrees                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Compute Profiles & Statistics (ONE TIME)             │
│  - Query KDTree_hydro at 30 radial bins (up to 5×R200)          │
│  - Compute mass profiles (DM, gas, stars, total)                │
│  - Compute baryon fractions at 6 apertures                      │
│  - Compute mass conservation ratios                             │
│  Output: profiles.h5, halo_statistics.h5                        │
│  Time: ~60-120s                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: Generate 2D Density Maps (ONE TIME)                  │
│  - DMO map: project all DMO particles                           │
│  - Hydro map: project all Hydro particles                       │
│  - Replace map: DMO_background + Hydro_halos (default config)   │
│  Output: density_dmo.npz, density_hydro.npz, density_replace.npz│
│  Time: ~30-60s                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: Generate DMO & Hydro Lensplanes (ONE TIME)           │
│  - Pre-generate N×pps random transforms (N=10, pps=2 → 20)      │
│  - For each plane: transform → slice → project → write          │
│  Output: dmo/LP_0..LP_19, hydro/LP_0..LP_19                     │
│  Time: ~2-5 min                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5: Generate Replace Lensplanes (LOOP over configs)      │
│  For each (mass_bin, R_factor) configuration:                   │
│    1. Select halos in mass bin                                  │
│    2. Query KDTrees at R_factor × R200 (FAST - tree in memory)  │
│    3. Build Replace = DMO_bg + Hydro_halos                      │
│    4. Generate 20 lensplanes with same transforms               │
│    5. Write to config-specific directory                        │
│  Output: hydro_replace_Ml_{}_Mu_{}_R_{}/LP_0..LP_19             │
│  Time: ~1-2 min per configuration                               │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Parameters

### Lensplane Generation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_realizations` | 10 | Number of random rotation/translation realizations |
| `planes_per_snapshot` | 2 | Number of depth slices per realization |
| `total_planes` | 20 | N × pps = 10 × 2 |
| `grid_res` | 4096 | Grid resolution (configurable) |
| `seed` | 2020 | Random seed for reproducibility |
| `box_size` | 205.0 | Mpc/h |

### Mass Bins

#### Exclusive Bins (non-overlapping)
| Label | M_lo (M☉/h) | M_hi (M☉/h) | Description |
|-------|-------------|-------------|-------------|
| `M12.0-12.5` | 10^12.0 | 10^12.5 | Low-mass groups |
| `M12.5-13.0` | 10^12.5 | 10^13.0 | Intermediate groups |
| `M13.0-13.5` | 10^13.0 | 10^13.5 | Massive groups |
| `M13.5-15.0` | 10^13.5 | 10^15.0 | Clusters |

#### Cumulative Bins (for studying mass thresholds)
| Label | M_lo (M☉/h) | M_hi (M☉/h) | Description |
|-------|-------------|-------------|-------------|
| `M12.0+` | 10^12.0 | ∞ | All halos above 10^12 |
| `M12.5+` | 10^12.5 | ∞ | All halos above 10^12.5 |
| `M13.0+` | 10^13.0 | ∞ | All halos above 10^13 |
| `M13.5+` | 10^13.5 | ∞ | All halos above 10^13.5 |

### Excision Radius Factors

| R_factor | Excision Radius | Description |
|----------|-----------------|-------------|
| 0.5 | 0.5 × R200 | Core only |
| 1.0 | 1.0 × R200 | Virial radius (default) |
| 3.0 | 3.0 × R200 | Extended halo |
| 5.0 | 5.0 × R200 | Including outskirts |

### Total Configurations

- **Mass bins**: 8 (4 exclusive + 4 cumulative)
- **R factors**: 4
- **Total Replace configs**: 8 × 4 = **32**
- **Lensplanes per config**: 20
- **Total Replace lensplanes**: 32 × 20 = **640**

Plus DMO (20) + Hydro (20) = **680 total lensplanes per snapshot**

## Output Directory Structure

```
/mnt/home/mlee1/ceph/hydro_replace_LP/
├── L205n625TNG/
│   ├── snap099/
│   │   ├── dmo/
│   │   │   ├── LP_00/
│   │   │   │   └── density_plane_000.bin
│   │   │   ├── LP_01/
│   │   │   └── ... LP_19/
│   │   │
│   │   ├── hydro/
│   │   │   ├── LP_00/
│   │   │   └── ... LP_19/
│   │   │
│   │   ├── hydro_replace_Ml_1.00e+12_Mu_inf_R_1.0/
│   │   │   ├── LP_00/
│   │   │   └── ... LP_19/
│   │   │
│   │   ├── hydro_replace_Ml_1.00e+12_Mu_3.16e+12_R_0.5/
│   │   │   └── ...
│   │   │
│   │   └── ... (32 replace directories)
│   │
│   ├── snap090/
│   │   └── ...
│   │
│   └── transforms.h5  # Saved transform parameters for reproducibility
│
└── L205n2500TNG/
    └── ...
```

## File Formats

### Lensplane Binary Format (lux compatible)

```
[int32: grid_size]
[float64[grid_size × grid_size]: delta × dz]
[int32: grid_size]
```

Where:
- `grid_size` = 4096
- `delta` = overdensity field (dimensionless)
- `dz` = depth of slice in comoving Mpc/h

### Transform Parameters (transforms.h5)

```python
{
    'n_realizations': 10,
    'planes_per_snapshot': 2,
    'seed': 2020,
    'box_size': 205.0,
    'proj_dirs': int[20],          # Projection axis (0, 1, or 2)
    'displacements': float[20, 3], # Random displacement vectors (absolute units, Mpc/h)
    'flips': bool[20],             # Whether to flip coordinates
}
```

## Command-Line Interface

### Full Pipeline

```bash
# Run all phases (1-5)
mpirun -n 64 python scripts/generate_all_unified.py \
    --snap 99 \
    --sim-res 625 \
    --grid 625 \
    --lensplane-grid 4096 \
    --enable-lensplanes
```

### Phase 5 Only (Lensplanes for New Configs)

```bash
# Run only Phase 5 (loads particles, builds trees, generates Replace lensplanes)
mpirun -n 64 python scripts/generate_all_unified.py \
    --snap 99 \
    --sim-res 625 \
    --phase5-only
```

### Incremental Generation

```bash
# Generate only missing configurations (checks existing directories)
mpirun -n 64 python scripts/generate_all_unified.py \
    --snap 99 \
    --sim-res 625 \
    --phase5-only \
    --incremental
```

## Implementation Details

### Random Transform Generation

```python
class TransformGenerator:
    """Generate reproducible random transforms for lensplanes."""
    
    def __init__(self, n_realizations=10, pps=2, seed=2020, box_size=205.0):
        self.n_realizations = n_realizations
        self.pps = pps
        self.seed = seed
        self.box_size = box_size
        self.n_total = n_realizations * pps
        
        # Pre-generate all transforms
        rng = np.random.RandomState(seed)
        self.proj_dirs = rng.randint(0, 3, self.n_total)
        self.displacements = rng.uniform(0, box_size, (self.n_total, 3))  # Absolute units
        self.flips = rng.choice([True, False], self.n_total)
    
    def get_transform(self, plane_idx):
        """Get transform parameters for a specific plane."""
        return {
            'proj_dir': self.proj_dirs[plane_idx],
            'displacement': self.displacements[plane_idx],
            'flip': self.flips[plane_idx],
            'plane_within_real': plane_idx % self.pps,  # Which depth slice
        }
    
    def save(self, filepath):
        """Save transforms for reproducibility."""
        with h5py.File(filepath, 'w') as f:
            f.attrs['n_realizations'] = self.n_realizations
            f.attrs['pps'] = self.pps
            f.attrs['seed'] = self.seed
            f.attrs['box_size'] = self.box_size
            f.create_dataset('proj_dirs', data=self.proj_dirs)
            f.create_dataset('displacements', data=self.displacements)
            f.create_dataset('flips', data=self.flips)
```

### Efficient KDTree Query for Multiple Configs

```python
def generate_replace_lensplanes(config, transforms, kdtree_dmo, kdtree_hydro,
                                 pos_dmo, mass_dmo, pos_hydro, mass_hydro,
                                 halos, comm, output_dir):
    """Generate Replace lensplanes for one (mass_bin, R_factor) config."""
    rank, size = comm.rank, comm.size
    M_lo, M_hi, R_factor = config
    
    # 1. Select halos in mass bin (cheap - all ranks do this)
    halo_mask = (halos['M200'] >= M_lo) & (halos['M200'] < M_hi)
    selected_halos = halos[halo_mask]
    n_selected = len(selected_halos)
    
    if rank == 0:
        print(f"  Config M=[{M_lo:.1e}, {M_hi:.1e}], R={R_factor}: {n_selected} halos")
    
    # 2. Query LOCAL particles against selected halos
    #    Each rank has its own local particles
    excision_radii = selected_halos['R200'] * R_factor
    
    # Query which local DMO particles are in halos
    local_dmo_in_halo = np.zeros(len(local_pos_dmo), dtype=bool)
    for i, (center, radius) in enumerate(zip(selected_halos['pos'], excision_radii)):
        idx = kdtree_dmo.query_ball_point(center, radius)
        local_dmo_in_halo[idx] = True
    
    # Query which local Hydro particles are in halos
    local_hydro_in_halo = np.zeros(len(local_pos_hydro), dtype=bool)
    for i, (center, radius) in enumerate(zip(selected_halos['pos'], excision_radii)):
        idx = kdtree_hydro.query_ball_point(center, radius)
        local_hydro_in_halo[idx] = True
    
    # 3. Build local Replace arrays
    local_pos_replace = np.concatenate([
        local_pos_dmo[~local_dmo_in_halo],      # DMO background
        local_pos_hydro[local_hydro_in_halo]    # Hydro halos
    ])
    local_mass_replace = np.concatenate([
        local_mass_dmo[~local_dmo_in_halo],
        local_mass_hydro[local_hydro_in_halo]
    ])
    
    # 4. Generate all lensplanes for this config
    for plane_idx in range(transforms.n_total):
        t = transforms.get_transform(plane_idx)
        
        # Transform and project local particles
        local_delta = project_with_transform(
            local_pos_replace, local_mass_replace,
            t, grid_res, box_size, plane_idx % transforms.pps
        )
        
        # Reduce across MPI ranks
        global_delta = np.zeros_like(local_delta) if rank == 0 else None
        comm.Reduce(local_delta, global_delta, op=MPI.SUM, root=0)
        
        # Write (rank 0 only)
        if rank == 0:
            write_lensplane(output_dir, plane_idx, global_delta)
```

### Projection with Transform

```python
def project_with_transform(pos, mass, transform, grid_res, box_size, depth_slice_idx):
    """Apply transform and project to 2D lensplane.
    
    Args:
        pos: (N, 3) particle positions in original coordinates
        mass: (N,) particle masses
        transform: dict with 'proj_dir', 'displacement', 'flip'
        grid_res: output grid resolution
        box_size: simulation box size (Mpc/h)
        depth_slice_idx: which depth slice (0 or 1 for pps=2)
    
    Returns:
        (grid_res, grid_res) density field
    """
    # Apply displacement (in box units, then scale)
    pos_t = pos + transform['displacement'] * box_size
    
    # Apply flip
    if transform['flip']:
        pos_t = box_size - pos_t
    
    # Periodic boundary conditions
    pos_t = pos_t % box_size
    
    # Get projection axes
    proj_dir = transform['proj_dir']
    depth_axis = proj_dir
    plane_axes = [i for i in range(3) if i != depth_axis]
    
    # Slice by depth (for pps=2, each slice is half the box)
    pps = 2
    depth = pos_t[:, depth_axis]
    depth_min = depth_slice_idx * box_size / pps
    depth_max = (depth_slice_idx + 1) * box_size / pps
    in_slice = (depth >= depth_min) & (depth < depth_max)
    
    # Extract 2D coordinates
    pos_2d = pos_t[in_slice][:, plane_axes].astype(np.float32)
    mass_slice = mass[in_slice].astype(np.float32)
    
    # TSC mass assignment
    delta = np.zeros((grid_res, grid_res), dtype=np.float64)
    MASL.MA(pos_2d, delta, box_size, 'TSC', W=mass_slice)
    
    return delta
```

## Performance Estimates

### L205n625TNG (test resolution)

| Phase | Time | Notes |
|-------|------|-------|
| Phase 1 | ~30s | Load + KDTree build |
| Phase 2 | ~60s | Profiles + statistics |
| Phase 3 | ~30s | 2D maps |
| Phase 4 | ~2 min | DMO + Hydro lensplanes (40 total) |
| Phase 5 | ~32 min | 32 configs × ~1 min each |
| **Total** | **~35 min** | Single snapshot |

### L205n2500TNG (production resolution)

| Phase | Time | Notes |
|-------|------|-------|
| Phase 1 | ~5 min | Much larger particle count |
| Phase 2 | ~10 min | More particles per halo |
| Phase 3 | ~5 min | Larger grids |
| Phase 4 | ~15 min | 4096² grids |
| Phase 5 | ~3-5 hours | 32 configs, larger data |
| **Total** | **~4-6 hours** | Single snapshot |

### Storage Requirements

Per snapshot (4096² grid, float64):
- Each lensplane: ~134 MB
- DMO (20 planes): ~2.7 GB
- Hydro (20 planes): ~2.7 GB
- Replace (32 configs × 20 planes): ~86 GB
- **Total per snapshot**: ~91 GB

For 20 snapshots: **~1.8 TB per simulation**

## Usage Examples

### Example 1: Full Pipeline Run

```bash
#!/bin/bash
#SBATCH --job-name=unified_full
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=cca

module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

cd /mnt/home/mlee1/hydro_replace2

srun python3 -u scripts/generate_all_unified.py \
    --snap 99 \
    --sim-res 625 \
    --mass-min 12.5 \
    --radius-mult 5.0 \
    --grid 625 \
    --enable-lensplanes \
    --lensplane-grid 4096
```

### Example 2: Phase 5 Only (Add New Configurations)

```bash
# After initial run, generate only Replace lensplanes
srun python3 -u scripts/generate_all_unified.py \
    --snap 99 \
    --sim-res 625 \
    --phase5-only \
    --incremental
```

### Example 3: Test with Small Grid

```bash
# Quick test with smaller lensplane grid (1024 instead of 4096)
srun python3 -u scripts/generate_all_unified.py \
    --snap 99 \
    --sim-res 625 \
    --mass-min 12.5 \
    --grid 625 \
    --enable-lensplanes \
    --lensplane-grid 1024 \
    --output-suffix "_lp_test"
```

## Validation

### Lensplane Consistency Checks

1. **Mass conservation**: Sum of lensplane should equal total mass × (1/pps)
2. **Transform reproducibility**: Same seed produces identical transforms
3. **Replace correctness**: DMO_lensplane - DMO_halo + Hydro_halo ≈ Replace_lensplane

### Comparison with Original Pipeline

The unified pipeline should produce:
- **Identical** profiles and statistics to the cached pipeline
- **Identical** 2D maps to the cached pipeline
- **Identical** DMO/Hydro lensplanes to generate_lensplanes.py (with same seed)

## Future Extensions

1. **BCM lensplanes**: Add baryonic correction model lensplanes
2. **Lightcone construction**: Stack lensplanes across snapshots
3. **Power spectrum analysis**: Compute P(k) from lensplanes
4. **Convergence maps**: Apply ray-tracing to generate κ maps
