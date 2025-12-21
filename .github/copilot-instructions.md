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

### Grid Size Mismatch in Lux
- Ensure `LP_grid` and `RT_grid` in lux config match density file grid size
- Density files are written with grid size in header (int32)

### Missing config.dat for Lux
- Lux needs `config.dat` from lens plane generation
- Copy from DMO directory if missing for other models

### NaN Values in BCM
- BCM can produce NaN at box corners (periodic boundary issues)
- Replace with 0 before writing density files

## Key Snapshots for TNG

| Snapshot | Redshift | Scale Factor |
|----------|----------|--------------|
| 99 | 0.00 | 1.000 |
| 91 | 0.10 | 0.909 |
| 84 | 0.20 | 0.833 |
| 78 | 0.30 | 0.769 |
| 72 | 0.40 | 0.714 |
| 67 | 0.50 | 0.667 |
| 59 | 0.70 | 0.588 |
| 50 | 1.00 | 0.500 |
| 40 | 1.50 | 0.400 |
| 33 | 2.00 | 0.333 |

## Testing

For quick tests, use L205n625TNG with:
- `TEST=1` environment variable
- Single snapshot (e.g., snap 99)
- `GRID_RES=1024` (reduced from 4096)
