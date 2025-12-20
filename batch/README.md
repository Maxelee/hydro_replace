# Batch Scripts for SLURM Job Submission

This directory contains SLURM batch scripts for running the hydro_replace pipeline on the Flatiron CCA cluster.

## Quick Start

```bash
# First, run the test job to verify data access
sbatch batch/test_data_access.sh

# Check job status
squeue -u $USER

# View output
cat logs/test_data_access.o*
```

## Pipeline Order

Run these in sequence (each depends on outputs from previous stages):

| Script | Description | Time | Resources |
|--------|-------------|------|-----------|
| `test_data_access.sh` | Verify data access & imports | 30 min | 1 node, 32GB |
| `01_prepare_data.sh` | Load catalogs, bijective matching | 1-2 hr | 1 node, 32GB |
| `02_extract_particles.sh` | Extract particles within 5R_vir | 6-12 hr | 4 nodes, MPI |
| `03_halo_replacement.sh` | Replace DMO with hydro particles | 12-24 hr | 4 nodes, MPI |
| `06_compute_power.sh` | Compute power spectra P(k) | 2-6 hr | 1 node, 128GB |
| `07_raytrace.sh` | Ray-tracing with lux | 1-2 days | 2 nodes, MPI |

## Submitting Jobs

```bash
cd /mnt/home/mlee1/hydro_replace2

# Submit single job
sbatch batch/01_prepare_data.sh

# Submit with dependency (wait for previous job to complete)
JOB1=$(sbatch --parsable batch/01_prepare_data.sh)
sbatch --dependency=afterok:$JOB1 batch/02_extract_particles.sh
```

## Monitoring

```bash
# Check queue status
squeue -u $USER

# Check job details
scontrol show job <JOBID>

# View live output
tail -f logs/01_prepare_data.o<JOBID>

# Cancel job
scancel <JOBID>
```

## Output Locations

- **Logs**: `logs/` (relative to project root)
- **Matched catalogs**: `/mnt/home/mlee1/ceph/hydro_replace/matching/`
- **Extracted particles**: `/mnt/home/mlee1/ceph/hydro_replace/extracted_halos/`
- **Replaced snapshots**: `/mnt/home/mlee1/ceph/hydro_replace/replaced_snapshots/`
- **Power spectra**: `/mnt/home/mlee1/ceph/hydro_replace/power_spectra/`
- **Ray-tracing output**: `/mnt/home/mlee1/ceph/lux_out/`

## Pre-computed Data

The pixelized replacement maps have already been computed and are available at:
```
/mnt/home/mlee1/ceph/pixelized/
```

These can be used directly for power spectrum analysis without running stages 1-3.

## Environment

All scripts activate the virtual environment:
```bash
source /mnt/home/mlee1/venvs/hydro_replace/bin/activate
```

Required modules:
```bash
module load python openmpi python-mpi hdf5
```

For lux ray-tracing:
```bash
module load openmpi hdf5 fftw gsl boost
```
