# Copilot Instructions for hydro_replace2

## Project Overview

This project generates 2D projected density maps and ray-tracing outputs comparing different cosmological simulation methods:
- **DMO**: Dark Matter Only simulations
- **Hydro**: Full hydrodynamic simulations (IllustrisTNG)
- **Replace**: Hybrid method replacing DMO halos with matched Hydro counterparts
- **BCM**: Baryonic Correction Models (Arico+20, Schneider+19, Schneider+25)

The pipeline supports weak lensing analysis via integration with the `lux` ray-tracing code.

## Key Directories

- `scripts/` - Main Python scripts (MPI-parallel)
- `batch/` - SLURM job submission scripts
- `config/` - YAML configuration files
- `notebooks/` - Jupyter notebooks for analysis
- `logs/` - SLURM output logs
- `archive/` - Old/deprecated code for reference

## Important Scripts

### Core Pipeline
- `scripts/generate_all.py` - Main MPI pipeline for density maps
- `scripts/generate_matches_fast.py` - Bijective halo matching (DMO ↔ Hydro)
- `scripts/generate_profiles.py` - Radial density profiles around halos
- `scripts/generate_lensplanes.py` - Generate lens planes for ray-tracing

### Ray-Tracing Integration
- `scripts/generate_lux_configs.py` - Generate lux configuration files
- `batch/run_raytracing_pipeline.sh` - Full ray-tracing orchestration

## External Dependencies

### Simulation Data
- Location: `/mnt/sdceph/users/sgenel/IllustrisTNG/`
- Simulations: L205n625TNG, L205n1250TNG, L205n2500TNG (and _DM variants)
- Access via `illustris_python` package

### Lux Ray-Tracing Code
- Location: `/mnt/home/mlee1/lux/`
- Config format: Simple key-value pairs (NO section headers like `[path]`)
- Valid parameters: `input_dir`, `LP_output_dir`, `RT_output_dir`, `simulation_format`, `LP_grid`, `RT_grid`, `planes_per_snapshot`, `projection_direction`, `translation_rotation`, `LP_random_seed`, `RT_random_seed`, `RT_randomization`, `angle`, `verbose`, `snapshot_list`, `snapshot_stack`
- `snapshot_list`: comma-separated integers
- `snapshot_stack`: comma-separated `true`/`false` (NOT 0/1)
- Has two phases: lenspot (lens potential) and raytracing

### Output Locations
- Density fields: `/mnt/home/mlee1/ceph/hydro_replace_fields/`
- Lens planes: `/mnt/home/mlee1/ceph/hydro_replace_lensplanes/`
- Lux output: `/mnt/home/mlee1/ceph/lux_out/`

## Coding Conventions

### MPI Usage
- All parallel scripts use `mpi4py`
- Rank 0 handles I/O and coordination
- Use `comm.Reduce` with contiguous arrays (`.copy()` if needed)
- Always barrier before collective operations

### Mass Assignment
- Use TSC (Triangular Shaped Cloud) for density fields to match lux
- Grid wrapping handled with modulo operations

### File Formats
- `.npz` for 2D density maps (key: `field`)
- `.h5` for profiles and complex data structures
- Binary for lux: `int32(grid_size) + float64[N×N](delta×dz) + int32(grid_size)`

### Units
- Positions: Mpc/h (or kpc/h, check context)
- Masses: M_sun/h (usually stored as 10^10 M_sun/h, multiply by 1e10)
- Box size: 205 Mpc/h for L205n* simulations

## SLURM Job Submission

### Environment Setup
```bash
module load python openmpi hdf5 fftw gsl boost/mpi-1.84.0
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
```

### Typical Resources
- 625 resolution: 4 nodes, 64 tasks, 4 hours
- 1250 resolution: 8 nodes, 128 tasks, 8 hours
- 2500 resolution: 16 nodes, 256 tasks, 12 hours

### Important: Never Run MPI Jobs on Login Nodes
Always use `sbatch` for MPI jobs. Login nodes are shared resources.

## Common Issues

### Grid Size Configuration in Lux
- `LP_grid` (lens potential): Should be 4096 for high-resolution lens planes
- `RT_grid` (ray-tracing): Should be 1024 for ray-tracing output
- Density files are written with grid size in header (int32)

### Missing config.dat for Lux
- Lux needs `config.dat` from lens plane generation
- Copy from DMO directory if missing for other models

### NaN Values in BCM
- BCM can produce NaN at box corners (periodic boundary issues)
- Replace with 0 before writing density files

## Ray-Tracing Snapshot List (20 snapshots)

The standard snapshot list for ray-tracing uses 20 snapshots from z≈2 to z=0:

```
snapshot_list = 96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29
snapshot_stack = false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true
```

| Snapshot | Redshift | Stack |
|----------|----------|-------|
| 96 | 0.02 | false |
| 90 | 0.10 | false |
| 85 | 0.18 | false |
| 80 | 0.27 | false |
| 76 | 0.35 | false |
| 71 | 0.46 | false |
| 67 | 0.55 | false |
| 63 | 0.65 | false |
| 59 | 0.76 | false |
| 56 | 0.85 | false |
| 52 | 0.97 | true |
| 49 | 1.08 | true |
| 46 | 1.21 | true |
| 43 | 1.36 | true |
| 41 | 1.47 | true |
| 38 | 1.63 | true |
| 35 | 1.82 | true |
| 33 | 1.97 | true |
| 31 | 2.14 | true |
| 29 | 2.32 | true |

## Testing

For quick tests, use L205n625TNG with:
- `TEST=1` environment variable
- Single snapshot (e.g., snap 96)
- Default grid sizes: LP_GRID=4096, RT_GRID=1024
