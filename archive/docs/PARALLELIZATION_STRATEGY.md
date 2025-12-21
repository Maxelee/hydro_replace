# Parallelization Strategy for Hydro Replace Pipeline

## Overview

The pipeline has several computationally intensive steps that need different parallelization approaches:

| Step | Operation | Bottleneck | Parallelization |
|------|-----------|------------|-----------------|
| 1 | Halo Matching | Particle ID lookups in 244M+ particles | **SKIP** - use spatial matching |
| 2 | Particle Loading | I/O | Chunked HDF5 reads |
| 3 | Replace/BCM Transform | Per-halo operations | Serial (fast with KDTree) |
| 4 | Mass Assignment | Grid operations | OpenMP (MAS_library) |
| 5 | Halo Profiles | Per-halo radial binning | Serial (fast with KDTree) |
| 6 | Output | I/O | Single writer |

## Key Insight: Skip Particle-Based Matching

The BijectiveMatcher uses particle IDs to match halos - this is **extremely slow** (O(N) lookup per halo in 244M particles).

**Better approach**: Use **spatial matching** based on halo positions:
- Match halos by proximity (within some tolerance)
- Much faster: O(N_halos * log(N_halos)) with KDTree
- Already have mass-matching criteria

## Revised Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: SPATIAL MATCHING                      │
│  Input: DMO + Hydro halo catalogs                                │
│  Output: matched_halos.h5                                        │
│  Method: KDTree spatial match + mass ratio filter                │
│  Time: ~seconds                                                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STEP 2: PARTICLE TRANSFORM (MPI)                 │
│  For each mode: dmo, hydro, replace, bcm-*                       │
│                                                                  │
│  Substeps:                                                       │
│   2a. Load particles (chunked, all ranks read same data)         │
│   2b. Distribute halos across ranks                              │
│   2c. Each rank transforms particles for its halos               │
│   2d. Reduce/gather transformed positions                        │
│                                                                  │
│  Time: ~1-5 min with 16 ranks                                    │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: MASS ASSIGNMENT + OUTPUT                    │
│  Input: Transformed particle positions + masses                  │
│  Output: 2D density maps, power spectra, lens planes            │
│  Method: MAS_library (OpenMP parallel internally)                │
│  Time: ~30 sec                                                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STEP 4: HALO PROFILES                          │
│  Input: Transformed particles + matched halo catalog             │
│  Output: Radial density profiles ρ(r/R_200) for each halo       │
│  Method: KDTree query + radial binning                          │
│  Time: ~1-2 min for ~3k halos                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Step 1: Fast Spatial Matching

```python
def spatial_match_halos(dmo_cat, hydro_cat, max_separation=0.5):
    """
    Match halos by spatial proximity and mass.
    
    Parameters
    ----------
    max_separation : float
        Maximum separation in units of R_200 (DMO)
    
    Returns matched pairs where:
    - Separation < max_separation * R_200
    - Mass ratio within factor of 3
    """
    from scipy.spatial import cKDTree
    
    # Build tree of hydro positions
    tree = cKDTree(hydro_cat.positions)
    
    # Query for each DMO halo
    matches = []
    for i, (pos, r200, mass) in enumerate(zip(
        dmo_cat.positions, dmo_cat.radii, dmo_cat.masses
    )):
        # Find hydro halos within search radius
        candidates = tree.query_ball_point(pos, r=max_separation * r200)
        
        # Filter by mass ratio
        for j in candidates:
            mass_ratio = hydro_cat.masses[j] / mass
            if 0.33 < mass_ratio < 3.0:
                matches.append((i, j))
                break
    
    return matches
```

### Step 2: MPI-Parallel Transform

The key is to have each rank handle a subset of halos:

```python
# Pseudocode for MPI transform
def transform_parallel(particles, halos, mode, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Each rank gets a subset of halos
    my_halos = halos[rank::size]
    
    # Build KDTree once (all ranks have same particles)
    tree = cKDTree(particles['pos'])
    
    # Each rank computes displacements for its halos
    my_displacements = []
    for halo in my_halos:
        if mode == 'replace':
            disp = compute_replacement(halo, particles, tree)
        elif mode.startswith('bcm'):
            disp = compute_bcm_displacement(halo, particles, tree)
        my_displacements.append(disp)
    
    # Gather all displacements to rank 0
    all_displacements = comm.gather(my_displacements, root=0)
    
    if rank == 0:
        # Apply all displacements
        return apply_displacements(particles, all_displacements)
```

### Step 3: Efficient Output

Use existing MAS_library for projection - it's already optimized.

## File Structure

```
scripts/
├── 01_spatial_match.py      # Fast KDTree matching (~seconds)
├── 02_transform_mpi.py      # MPI parallel transform (~minutes)  
├── 03_project_output.py     # Mass assignment + save (~30s)
└── run_pipeline.sh          # Master orchestration script

batch/
├── run_full_pipeline.sh     # SLURM submission for full pipeline
└── run_single_mode.sh       # Run one mode (for testing)
```

## Expected Performance

| Resolution | Particles | Halos (>10^13) | Matching | Transform | Project | Total |
|------------|-----------|----------------|----------|-----------|---------|-------|
| 625^3 | 244M | ~3k | 5s | 2min | 30s | ~3min |
| 1250^3 | 1.9B | ~25k | 10s | 10min | 2min | ~15min |
| 2500^3 | 15.6B | ~200k | 30s | 60min | 10min | ~75min |

With 16 MPI ranks, transform time scales ~linearly.

## Memory Considerations

- 625^3: ~8 GB for particles
- 1250^3: ~64 GB for particles  
- 2500^3: ~500 GB for particles (need distributed memory)

For 2500^3, we need chunked processing or distributed particles across nodes.
