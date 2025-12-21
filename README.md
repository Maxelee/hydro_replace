# Hydro Replace

Generate 2D projected density maps and weak lensing ray-tracing outputs comparing DMO, Hydro, Replace, and BCM methods for IllustrisTNG simulations.

## Overview

This pipeline compares different methods for modeling baryonic effects on matter distribution:
- **DMO**: Dark Matter Only simulations
- **Hydro**: Full hydrodynamic simulations (IllustrisTNG)  
- **Replace**: Hybrid method replacing DMO halos with matched Hydro counterparts
- **BCM**: Baryonic Correction Models (Arico+20, Schneider+19, Schneider+25)

Includes integration with the **lux** ray-tracing code for weak lensing analysis.

## Quick Start

```bash
# Activate environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Generate density maps for snapshot 99
sbatch --export=SNAP=99,SIM_RES=2500 batch/run_full_pipeline.sh

# Full ray-tracing pipeline (lens planes + lux)
TEST=1 ./batch/run_raytracing_pipeline.sh 625  # Test with low-res
./batch/run_raytracing_pipeline.sh 2500        # Production
```

## Structure

```
hydro_replace2/
├── batch/
│   ├── run_full_pipeline.sh       # Density map generation
│   ├── run_raytracing_pipeline.sh # Full ray-tracing orchestration
│   ├── run_lensplanes.sh          # Lens plane generation only
│   └── run_profiles.sh            # Radial profile generation
├── scripts/
│   ├── generate_all.py            # Main density map pipeline (MPI)
│   ├── generate_matches_fast.py   # Bijective halo matching (DMO ↔ Hydro)
│   ├── generate_lensplanes.py     # Lens planes for ray-tracing (MPI)
│   ├── generate_profiles.py       # Radial density profiles (MPI)
│   └── generate_lux_configs.py    # Lux configuration generator
├── notebooks/
│   ├── analyze_results.ipynb      # Power spectra & visualization
│   └── power_by_mass.ipynb        # Mass-binned analysis
├── config/                        # YAML configuration files
├── logs/                          # SLURM output logs
└── archive/                       # Old/deprecated code (for reference)
```

## Usage

### Generate Maps

```bash
# Basic usage (625 resolution, snap 99)
sbatch --export=SNAP=99,SIM_RES=625 batch/run_full_pipeline.sh

# Full resolution (2500), multiple nodes for memory
sbatch --export=SNAP=99,SIM_RES=2500 -N 8 -n 64 batch/run_full_pipeline.sh

# Only BCM (skip DMO/Hydro/Replace if they exist)
sbatch --export=SNAP=99,SIM_RES=2500,ONLY_BCM=true,SKIP_EXISTING=true batch/run_full_pipeline.sh

# Specific mass range
sbatch --export=SNAP=99,SIM_RES=2500,MASS_MIN=13.0,MASS_MAX=14.0 batch/run_full_pipeline.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SNAP` | 99 | Snapshot number |
| `SIM_RES` | 625 | Simulation resolution (625, 1250, 2500) |
| `GRID_RES` | 1024 | 2D map resolution |
| `MASS_MIN` | 12.5 | log10(M_min) for halo selection |
| `MASS_MAX` | - | log10(M_max) for mass bins |
| `SKIP_EXISTING` | false | Skip if output exists |
| `ONLY_BCM` | false | Only generate BCM maps |
| `BCM_MODELS` | "Arico20 Schneider19 Schneider25" | Which BCM models to run |

## Output

### Density Maps
Located at `/mnt/home/mlee1/ceph/hydro_replace_fields/`:

```
L205n{RES}TNG/
├── matches/
│   └── matches_snap{NN}.npz    # Halo matching results
└── snap{NN}/
    ├── projected/
    │   ├── dmo.npz             # DMO density map
    │   ├── hydro.npz           # Hydro density map
    │   ├── replace.npz         # Replacement method map
    │   ├── bcm_arico20.npz     # BCM Arico+20 map
    │   ├── bcm_schneider19.npz # BCM Schneider+19 map
    │   └── bcm_schneider25.npz # BCM Schneider+25 map
    └── profiles_Mgt*.h5        # Radial density profiles
```

### Ray-Tracing Output
Located at `/mnt/home/mlee1/ceph/lux_out/`:

```
L205n{RES}TNG/
├── dmo/
│   ├── lux_dmo.ini             # Lux configuration
│   ├── config.dat              # Ray-tracing config
│   ├── lenspot01.dat - 40.dat  # Lens potential maps
│   └── run001-100/             # Ray-tracing realizations
│       ├── kappa*.dat          # Convergence maps
│       └── gamma*.dat          # Shear maps
├── hydro/
├── replace_Mgt12.0/
├── bcm_arico20/
└── ...
```

## Key Snapshots

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

## Memory Requirements

| Resolution | Nodes | Tasks | ~RAM/task |
|------------|-------|-------|-----------|
| 625 | 1 | 16 | ~8 GB |
| 1250 | 2 | 32 | ~16 GB |
| 2500 | 8 | 64 | ~30 GB |
