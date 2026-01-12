# Data Recovery Plan for Missing Pipeline Products

## Executive Summary

This document outlines the recovery plan for missing data products in the hydro_replace pipeline.

### Current Status (as of diagnosis)

| Stage | Complete | Missing | Percentage |
|-------|----------|---------|------------|
| Lensplanes (LP) | 13,384 | 216 | 98.4% |
| LUX Format | 13,384 | 216 | 98.4% |
| Ray-Tracing | Needs re-run for 79 combinations | - | - |

### Root Cause

Missing data is concentrated in 16 snapshots across 13 Replace model configurations. The failures appear to be sporadic, affecting different realizations (mostly indices 4-9) for different models.

### Affected Snapshots

| Snapshot | Array Index | # Missing Planes | # Models Affected |
|----------|-------------|------------------|-------------------|
| 29 | 19 | 21 | 2 |
| 33 | 17 | 28 | 2 |
| 35 | 16 | 12 | 1 |
| 38 | 15 | 13 | 1 |
| 41 | 14 | 7 | 2 |
| 46 | 12 | 18 | 1 |
| 49 | 11 | 16 | 1 |
| 52 | 10 | 28 | 2 |
| 56 | 9 | 13 | 1 |
| 59 | 8 | 10 | 1 |
| 67 | 6 | 11 | 1 |
| 76 | 4 | 12 | 1 |
| 80 | 3 | 7 | 1 |
| 85 | 2 | 11 | 1 |
| 90 | 1 | 4 | 1 |
| 96 | 0 | 5 | 1 |

## Recovery Pipeline

### Step 1: Regenerate Missing Lensplanes

**Script:** `batch/run_lensplane_recovery.sh`

This runs the unified pipeline's Phase 5 (Replace lensplanes only) for the 16 snapshots with missing data. The `--incremental` flag ensures only missing files are regenerated.

```bash
sbatch batch/run_lensplane_recovery.sh
```

**Resources:** 8 nodes × 16 tasks = 128 cores per snapshot, 6 hours
**Total:** 16 array tasks

### Step 2: Convert to LUX Format

**Script:** `batch/run_lux_convert_recovery.sh`

After lensplanes are regenerated, this converts them to the LUX binary format required for ray-tracing. Only processes the 79 model/realization combinations that were affected.

```bash
sbatch --dependency=afterok:$LP_JOB_ID batch/run_lux_convert_recovery.sh
```

**Resources:** 1 node × 40 tasks per combination, 2 hours
**Total:** 79 array tasks

### Step 3: Run Ray-Tracing

**Script:** `batch/run_rt_recovery.sh`

Finally, run ray-tracing for the recovered lensplanes. Each combination generates 100 convergence maps.

```bash
sbatch --dependency=afterok:$LUX_JOB_ID batch/run_rt_recovery.sh
```

**Resources:** 1 node × 40 tasks per combination, 24 hours
**Total:** 79 array tasks

## Using the Master Script

The easiest way to run recovery is via the master script:

```bash
# 1. First, diagnose what's missing
./recovery_master.sh diagnose

# 2. Submit all jobs with proper dependencies
./recovery_master.sh submit

# 3. After completion, verify everything is complete
./recovery_master.sh verify
```

## Verification

After recovery completes, verify with:

```bash
python3 scripts/verify_pipeline_completeness.py
```

This will check:
- All 34 models have complete lensplanes (40 × 10 = 400 files)
- All models have LUX format files
- All models have at least 500 convergence maps (goal)

## File Locations

| Product | Location |
|---------|----------|
| Lensplanes | `/mnt/home/mlee1/ceph/hydro_replace_LP/L205n2500TNG/{model}/LP_{00-09}/` |
| LUX Format | `/mnt/home/mlee1/ceph/hydro_replace_LP_lux/L205n2500TNG/{model}/LP_{00-09}/` |
| Ray-Tracing | `/mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG/{model}/LP_{00-09}/run{001-100}/` |

## Expected Final Output

| Metric | Expected |
|--------|----------|
| Total Models | 34 (2 base + 32 replace) |
| Realizations per Model | 10 |
| Lensplanes per Realization | 40 |
| RT Runs per Realization | 100 |
| **Total Kappa Maps per Model** | **1,000** (10 × 100) |
| **Goal** | **≥500 per model** |

## Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/find_missing_lensplanes.py` | Diagnose missing LP files |
| `scripts/verify_pipeline_completeness.py` | Verify all products complete |
| `batch/run_lensplane_recovery.sh` | SLURM array job for LP recovery |
| `batch/run_lux_convert_recovery.sh` | SLURM array job for LUX conversion |
| `batch/run_rt_recovery.sh` | SLURM array job for ray-tracing |
| `recovery_master.sh` | Master orchestration script |

## Monitoring

```bash
# Check queue status
squeue -u $USER

# Check job output
tail -f logs/lp_recovery_*.o
tail -f logs/lux_convert_recovery_*.o
tail -f logs/rt_recovery_*.o
```

## Estimated Time

| Step | Wall Time | Queue Time |
|------|-----------|------------|
| LP Recovery | ~6 hours | depends on queue |
| LUX Convert | ~2 hours | ~30 min after LP |
| Ray-Tracing | ~24 hours | ~30 min after LUX |
| **Total** | **~32 hours** | plus queue wait |
