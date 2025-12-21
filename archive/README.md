# Hydro Replace

A modular pipeline for analyzing baryonic correction models using halo replacement techniques.

## Overview

This project provides a comprehensive toolkit for:
1. **Bijective Matching**: Match halos between hydrodynamic and dark matter-only (DMO) simulations
2. **Particle Extraction**: Extract particles around matched halos at various radii
3. **Halo Replacement**: Replace DMO particles with hydrodynamic particles
4. **Profile Analysis**: Compute stacked density profiles
5. **Power Spectrum**: Calculate matter power spectra and suppression ratios
6. **Ray-Tracing**: Generate weak lensing convergence maps
7. **Peak Finding**: Analyze peak statistics in convergence maps

## Installation

### Requirements

- Python 3.9+
- MPI (OpenMPI or MPICH) for parallel processing
- HDF5 library

### Setup

```bash
# Clone the repository
cd /mnt/home/mlee1/hydro_replace2

# Create conda environment
conda env create -f environment.yml
conda activate hydro_replace

# Install package in development mode
pip install -e .
```

### Dependencies

Core dependencies:
- numpy, scipy, h5py
- mpi4py (for parallel processing)
- astropy (for cosmology)
- pyyaml (for configuration)

Optional dependencies:
- Pylians3 (for power spectra): `pip install Pylians`
- BaryonForge (for BCM): `pip install BaryonForge`
- illustris_python (for TNG data): Clone from [illustris_python](https://github.com/illustristng/illustris_python)

## Project Structure

```
hydro_replace2/
├── config/                     # Configuration files
│   ├── simulation_paths.yaml   # Simulation paths and cosmology
│   ├── analysis_params.yaml    # Analysis parameters
│   └── raytrace_config.yaml    # Ray-tracing settings
├── src/hydro_replace/          # Main package
│   ├── data/                   # Data loading and matching
│   │   ├── load_simulations.py
│   │   ├── halo_catalogs.py
│   │   ├── bijective_matching.py
│   │   └── particle_extraction.py
│   ├── replacement/            # Halo replacement
│   │   ├── replace_core.py
│   │   ├── mass_bins.py
│   │   └── validation.py
│   ├── analysis/               # Analysis tools
│   │   ├── profiles.py
│   │   ├── power_spectrum.py
│   │   ├── mass_conservation.py
│   │   └── statistics.py
│   ├── raytrace/               # Ray-tracing
│   │   ├── raytrace_engine.py
│   │   ├── convergence_maps.py
│   │   └── peak_finding.py
│   ├── bcm/                    # Baryon correction models
│   │   └── arico_bcm.py
│   └── utils/                  # Utilities
│       ├── logging_setup.py
│       ├── periodic_boundary.py
│       ├── parallel.py
│       └── io_helpers.py
├── scripts/                    # Pipeline scripts
│   ├── 01_prepare_data.py
│   ├── 02_extract_particles.py
│   ├── 03_halo_replacement.py
│   ├── 05_compute_profiles.py
│   └── 06_compute_power.py
└── tests/                      # Unit tests
```

## Quick Start

### 1. Configure Paths

Edit `config/simulation_paths.yaml` to set your simulation paths:

```yaml
simulations:
  tng300:
    hydro: /path/to/TNG300/output
    dmo: /path/to/TNG300-Dark/output
    box_size: 205.0  # Mpc/h
```

### 2. Run Pipeline

```bash
# Stage 1: Prepare data and match halos
mpirun -np 16 python scripts/01_prepare_data.py

# Stage 2: Extract particles
mpirun -np 16 python scripts/02_extract_particles.py

# Stage 3: Perform halo replacement
mpirun -np 16 python scripts/03_halo_replacement.py

# Stage 5: Compute profiles
mpirun -np 16 python scripts/05_compute_profiles.py

# Stage 6: Compute power spectra
python scripts/06_compute_power.py --threads 8
```

### 3. Analyze Results

```python
from hydro_replace.utils import load_hdf5
from hydro_replace.analysis import PowerSpectrum

# Load power spectrum
pk_data = load_hdf5('output/power_spectra/pk_hydro.h5')
pk = PowerSpectrum(**pk_data)

# Load profiles
profiles = load_hdf5('output/profiles/profiles_M3.h5')
rho_mean = profiles['rho_hydro_mean']
```

## Usage Examples

### Bijective Matching

```python
from hydro_replace.data import BijectiveMatcher, HaloCatalog

# Load catalogs
cat_hydro = HaloCatalog.from_illustris('/path/to/hydro', snapshot=99)
cat_dmo = HaloCatalog.from_illustris('/path/to/dmo', snapshot=99)

# Perform matching
matcher = BijectiveMatcher(box_size=205.0)
matched = matcher.match(
    halo_ids_hydro=cat_hydro.halo_ids,
    mbp_ids_hydro=cat_hydro.most_bound_ids,
    halo_ids_dmo=cat_dmo.halo_ids,
    mbp_ids_dmo=cat_dmo.most_bound_ids,
)

print(f"Matched {matched.n_matched} halo pairs")
```

### Power Spectrum Analysis

```python
from hydro_replace.analysis import compute_power_spectrum, compute_suppression

# Compute power spectra
pk_hydro = compute_power_spectrum(coords_hydro, masses_hydro, box_size=205.0)
pk_dmo = compute_power_spectrum(coords_dmo, masses_dmo, box_size=205.0)

# Compute suppression
k, S = compute_suppression(pk_hydro, pk_dmo)
print(f"Suppression at k=1 h/Mpc: {S[np.argmin(np.abs(k-1))]:.3f}")
```

### Profile Stacking

```python
from hydro_replace.analysis import compute_density_profile, stack_profiles

# Compute profiles for multiple halos
profiles = []
for halo in halos:
    profile = compute_density_profile(
        coords, masses, halo['center'], r_bins, box_size
    )
    profiles.append(profile.density)

# Stack
stacked, scatter = stack_profiles(profiles, method='median')
```

## Configuration

### Mass Bins

Configure mass bins in `config/analysis_params.yaml`:

```yaml
mass_bins:
  regular:
    - 1.0e12   # 10^12 Msun/h
    - 3.16e12  # 10^12.5
    - 1.0e13   # 10^13
    - 3.16e13  # 10^13.5
    - 1.0e14   # 10^14
    - 3.16e14  # 10^14.5
    - 1.0e15   # 10^15
```

### Radius Multipliers

```yaml
extraction:
  radius_multipliers: [1.0, 3.0, 5.0]  # Multiples of R200c
```

## SLURM Job Submission

Example SLURM script for running on a cluster:

```bash
#!/bin/bash
#SBATCH --job-name=hydro_replace
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --partition=gen

module load python mpi

source activate hydro_replace

mpirun python scripts/03_halo_replacement.py --config config/analysis_params.yaml
```

## Output Format

All outputs are saved in HDF5 format with self-documenting attributes:

```python
# Example HDF5 structure
/power_spectra/pk_hydro.h5
├── k [dataset]           # Wavenumber array
├── pk [dataset]          # Power spectrum
├── n_modes [dataset]     # Number of modes per bin
└── attrs
    ├── box_size
    ├── grid_size
    └── mas
```

## Citation

If you use this code, please cite:

```bibtex
@article{hydro_replace,
  author = {Your Name},
  title = {Hydro Replace: ...},
  year = {2024}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
