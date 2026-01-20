# Copilot Instructions for hydro_replace2

## Project Overview

This project implements a **Mass-Radius-Redshift Response Formalism** for quantifying how baryonic feedback modifies cosmological weak lensing observables. The core idea is to construct "Replace" density fields that interpolate between DMO and Hydro simulations by selectively replacing halos based on mass and radius criteria.

### Scientific Goal
Measure the **response kernel** $K_S(M_a, M_b; \alpha_i; z_k)$ which quantifies what fraction of the baryonic effect on observable $S$ comes from halos in mass bin $[M_a, M_b)$ at radius factor $\alpha$ and redshift $z$.

### Simulation Types
- **DMO**: Dark Matter Only simulations (baseline)
- **Hydro**: Full hydrodynamic simulations (IllustrisTNG) with baryonic physics
- **Replace**: Hybrid fields where DMO halos are replaced with matched Hydro counterparts within $\alpha \times R_{200}$
- **BCM**: Baryonic Correction Models (Arico+20, Schneider+19, Schneider+25) applied to DMO particles

### Key Equations
The Replace field is defined as:
$$\rho_{\rm R}(\mathbf{x}) = \rho_{\rm D}(\mathbf{x}) + \sum_{i \in \mathcal{H}(M_{\min})} \left[\rho_{\rm H,halo}^{(i)} - \rho_{\rm D,halo}^{(i)}\right]$$

The response fraction measures how much of the Hydro-DMO difference is captured:
$$F_S(M_{\min}, \alpha) = \frac{S_{\rm R} - S_{\rm D}}{S_{\rm H} - S_{\rm D}}$$

## Key Directories

- `scripts/` - Main Python scripts (MPI-parallel, only 3 active scripts)
- `batch/` - SLURM job submission scripts (only 3 active scripts)
- `config/` - YAML configuration files
- `notebooks/` - Jupyter notebooks for analysis and visualization
- `logs/` - SLURM output logs
- `archive/` - Old/deprecated code for reference

## Active Scripts

### Pipeline Scripts (scripts/)
- `generate_all_unified.py` - Unified pipeline for DMO/Hydro/Replace maps and lensplanes
- `generate_all_unified_bcm.py` - BCM pipeline (Schneider19, Schneider25, Arico20)
- `convert_to_lensplanes.py` - Convert mass planes to lux lensing potential format

### Batch Scripts (batch/)
- `run_unified_2500_array.sh` - Array job for L205n2500TNG (20 snapshots)
- `run_unified_bcm_array.sh` - Array job for BCM models
- `run_lux_pipeline_array.sh` - Array job for lux ray-tracing (34 models × 10 realizations)

## Replace Configurations

The pipeline generates Replace fields for combinations of:

### Mass Bins (M_lo to M_hi in M_sun/h)
- `Ml_1.00e12_Mu_3.16e12` - 10^12.0 to 10^12.5
- `Ml_1.00e12_Mu_inf` - 10^12.0+ (cumulative)
- `Ml_3.16e12_Mu_1.00e13` - 10^12.5 to 10^13.0
- `Ml_3.16e12_Mu_inf` - 10^12.5+ (cumulative)
- `Ml_1.00e13_Mu_3.16e13` - 10^13.0 to 10^13.5
- `Ml_1.00e13_Mu_inf` - 10^13.0+ (cumulative)
- `Ml_3.16e13_Mu_1.00e15` - 10^13.5 to 10^15.0
- `Ml_3.16e13_Mu_inf` - 10^13.5+ (cumulative)

### Radius Factors (α × R_200)
- `R_0.5` - Core region only
- `R_1.0` - Within virial radius
- `R_3.0` - Extended halo
- `R_5.0` - Full splash-back region

Total: 8 mass configs × 4 R factors = 32 Replace configurations + DMO + Hydro = 34 models

## External Dependencies

### Simulation Data
- Location: `/mnt/sdceph/users/sgenel/IllustrisTNG/`
- Simulations: L205n625TNG, L205n1250TNG, L205n2500TNG (and _DM variants)
- Access via `illustris_python` package

### Lux Ray-Tracing Code
- Location: `/mnt/home/mlee1/lux/`
- Config format: Simple key-value pairs (NO section headers like `[path]`)
- Valid parameters: `LP_output_dir`, `RT_output_dir`, `LP_grid`, `RT_grid`, `planes_per_snapshot`, `angle`, `RT_random_seed`, `RT_randomization`, `snapshot_list`, `snapshot_stack`, `verbose`
- `snapshot_list`: comma-separated integers
- `snapshot_stack`: comma-separated `true`/`false` (NOT 0/1)

### Output Locations
- Lens planes: `/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG/`
- Lux format: `/mnt/home/mlee1/ceph/hydro_replace_LP_lux/L205n2500TNG/`
- Ray-tracing: `/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG/`

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
module purge
module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
```

### Typical Resources (L205n2500TNG)
- Unified pipeline: 8 nodes, 16 tasks, 12 hours per snapshot
- BCM pipeline: 8 nodes, 32 tasks, 12 hours per snapshot
- Lux ray-tracing: 1 node, 40 tasks, 4 hours per model/realization

### Important: Never Run MPI Jobs on Login Nodes
Always use `sbatch` for MPI jobs. Login nodes are shared resources.

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

## Common Issues

### Grid Size Configuration
- `LP_grid` (lens potential): 4096 for high-resolution lens planes
- `RT_grid` (ray-tracing): 1024 for ray-tracing output
- Map grid: 1024 for 2D density maps

### NaN Values in BCM
- BCM can produce NaN at box corners (periodic boundary issues)
- Replace with 0 before writing density files
