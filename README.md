# Hydro Replace

Generate 2D projected density maps comparing DMO, Hydro, Replace, and BCM methods for TNG simulations.

## Quick Start

```bash
# Activate environment
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate

# Submit a job for snapshot 99
cd batch
sbatch --export=SNAP=99,SIM_RES=2500 run_full_pipeline.sh
```

## Structure

```
hydro_replace2/
├── batch/
│   └── run_full_pipeline.sh    # SLURM job submission
├── scripts/
│   ├── generate_all.py         # Main pipeline (MPI)
│   └── generate_matches_fast.py # Halo matching
├── notebooks/
│   ├── analyze_results.ipynb   # Power spectra & visualization
│   └── power_by_mass.ipynb     # Mass-binned analysis
├── config/                     # YAML configuration files
├── logs/                       # SLURM output logs
└── archive/                    # Old/unused code (for reference)
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
    └── profiles.h5             # Halo info
```

## Key Snapshots

| Snapshot | Redshift |
|----------|----------|
| 99 | 0.00 |
| 76 | 0.50 |
| 59 | 1.07 |
| 49 | 1.50 |
| 40 | 1.93 |

## Memory Requirements

| Resolution | Nodes | Tasks | ~RAM/task |
|------------|-------|-------|-----------|
| 625 | 1 | 16 | ~8 GB |
| 1250 | 2 | 32 | ~16 GB |
| 2500 | 8 | 64 | ~30 GB |
