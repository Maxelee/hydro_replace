# Hydro Replace Project - Copilot Instructions

## Project Overview

This is a PhD research project analyzing baryonic effects on weak lensing observables using hydrodynamic simulation replacement techniques. The project uses TNG-300 simulations (hydro and dark-matter-only) from IllustrisTNG and will extend to CAMELS multi-simulation suites.

## Cluster Environment

### Virtual Environment
**Always activate before running scripts:**
```bash
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
```

### SLURM Job Submission
Located on the Flatiron CCA cluster. Example batch script template:
```bash
#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=icelake
#SBATCH -J job_name
#SBATCH -n 16
#SBATCH -N 10
#SBATCH --exclusive
#SBATCH -o OUTPUT.o%j
#SBATCH -e OUTPUT.e%j
#SBATCH --mail-user=mlee@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH -t 06-23:15:00

module load python openmpi python-mpi hdf5
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
srun -n16 python3 -u script.py
```

### Ray-Tracing (lux)
C++ ray-tracing code located at `/mnt/home/mlee1/lux/`
- **Executable**: `/mnt/home/mlee1/lux/lux`
- **Config files**: `*.ini` files in the lux directory
- **Output**: `/mnt/home/mlee1/ceph/lux_out/`
- **Dependencies**: MPI, HDF5, FFTW3, GSL, Boost (MPI + serialization)

## Key Technical Context

### Simulations
- **TNG-300 Hydro**: `/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output`
- **TNG-300 DMO**: `/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output`
- **Box size**: 205 Mpc/h
- **Snapshot 99**: z = 0 (primary analysis snapshot)
- **DM particle mass (DMO)**: 0.0047271638660809 × 10¹⁰ M☉/h
- **DM particle mass (Hydro)**: 0.00398342749867548 × 10¹⁰ M☉/h

### Existing Data Products
Pre-computed pixelized replacement maps are at: `/mnt/home/mlee1/ceph/pixelized/`

Files follow naming: `pixelized_maps_res{RES}_axis{AXIS}_{MODE}_rad{RADIUS}_mass{MASS_LABEL}.npz`
- Resolutions: 4096
- Modes: `normal` (replace DMO with hydro), `inverse` (reverse)
- Radii: 1, 3, 5 (× R_200c)
- Mass bins: 10.0-12.5, 12.5-13.0, 13.0-13.5, 13.5-14.0, gt14.0, cumulative variants

### Key Libraries
- `illustris_python`: TNG data loading
- `MAS_library` / `Pylians3`: Mass assignment and power spectra
- `mpi4py`: Parallel processing
- `scipy.spatial.cKDTree`: Fast spatial queries
- `BaryonForge`: BCM implementation (Arico+2020)
- `h5py`: HDF5 file handling

## Code Style Guidelines

### Python
- Use type hints for function parameters and return values
- Prefer numpy operations over loops
- Use dataclasses for configuration objects
- Follow MPI patterns: rank 0 does I/O, broadcast/gather for communication
- Log with rank-aware logging (only rank 0 prints by default)

### File Organization
- `src/hydro_replace/`: Core package modules
- `scripts/`: Pipeline scripts (numbered 01-08)
- `config/`: YAML configuration files
- `notebooks/`: Exploratory Jupyter notebooks
- `tests/`: Unit tests

### Configuration
Use YAML config files in `config/`:
- `simulation_paths.yaml`: File paths and cosmology
- `analysis_params.yaml`: Mass bins, radii, output directories
- `raytrace_config.yaml`: Ray-tracing parameters

## Common Tasks

### Loading TNG Data
```python
from hydro_replace.data.load_simulations import SimulationData
from hydro_replace.data.halo_catalogs import HaloCatalog

sim = SimulationData('/path/to/simulation')
catalog = HaloCatalog(sim, snapshot=99)
```

### Running Pipeline Scripts
```bash
# Sequential (for testing)
python scripts/01_prepare_data.py

# MPI parallel (for production)
mpirun -np 16 python scripts/01_prepare_data.py
```

### Computing Power Spectra
```python
from hydro_replace.analysis.power_spectrum import compute_power_spectrum

k, Pk = compute_power_spectrum(density_field, box_size=205.0)
```

## Important Conventions

### Units
- Masses: M☉/h (stored as 10¹⁰ M☉/h in simulation files)
- Distances: Mpc/h (positions in simulation files are kpc/h)
- Power spectra: (Mpc/h)³

### Mass Bins (log₁₀ M☉/h)
- Regular: 12.0-12.5, 12.5-13.0, 13.0-13.5, 13.5-14.0, 14.0-14.5, >14.5
- Cumulative: >10¹², >10¹³, >10¹⁴

### Replacement Radii
- 1× R_200c: Halo interior
- 3× R_200c: Includes 1-halo term  
- 5× R_200c: Full profile + 2-halo contribution

## Debugging Tips

1. **Memory issues**: Use chunked loading, reduce grid resolution for testing
2. **MPI deadlocks**: Check all ranks reach barriers/collectives
3. **Wrong units**: Always verify factor of 1e3 for kpc↔Mpc, 1e10 for mass
4. **Periodic boundaries**: Use `periodic_distance` utilities

## References

- Miller+2025 (arXiv:2511.10634): Mass redistribution validation
- Arico+2020, 2021: BCM model
- Lee+2023 (MNRAS 519:573): Peak statistics methodology
- Villaescusa-Navarro+2021: CAMELS overview

## Project Timeline

- Paper 1: Power spectrum + peaks (TNG-300 replacement)
- Paper 2: Multi-BCM comparison  
- Paper 3: CAMELS parameter emulator
