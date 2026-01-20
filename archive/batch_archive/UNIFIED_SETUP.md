# Lux Ray-Tracing Pipeline - Unified Setup

## What Changed

### Before (Sequential 4-Part Approach)
```
run_lux_binned_part1.sh  (array 0-509)    ─┐
run_lux_binned_part2.sh  (array 510-1019) ──┼─→ Must run sequentially
run_lux_binned_part3.sh  (array 1020-1529) ─┤   (submit one after another)
run_lux_binned_part4.sh  (array 1530-2039) ─┘
```

**Problem**: Only one script's array jobs run at a time. Total wall clock time ≈ 4× longer.

### After (Unified Parallel Approach)
```
run_lux_binned_unified.sh (array 0-2039) ──→ All 2040 jobs run in parallel
```

**Benefit**: All 2040 (model, realization) combinations process simultaneously. Total wall clock time ≈ 1× (same as processing one part).

## How to Use

### Submit the Unified Job
```bash
cd /mnt/home/mlee1/hydro_replace2/batch
sbatch run_lux_binned_unified.sh
```

This will:
- Submit 2040 parallel array tasks (one per (model, realization))
- Each task processes 1 model + 1 realization
- Within each task:
  - Step 1: Convert 40 lensplanes in parallel (srun -n 40)
  - Step 2: Run 100 ray-tracing realizations in parallel (srun -n 40)

### Monitor Progress
```bash
# Watch all jobs
watch squeue -u $USER -j <job_id>

# Or count completed tasks
cd logs/
ls lux_binned_unified_* | wc -l  # Should reach 2040
```

### Check Completion
```bash
# Count successful output files
find /mnt/home/mlee1/ceph/hydro_replace_RT/L205n2500TNG -type f -name "kappa_40.dat" | wc -l

# Should be 102 models × 20 realizations = 2040
```

## Job Layout

| Parameter | Value |
|-----------|-------|
| Nodes | 1 |
| Tasks | 40 (for MPI parallelization within each job) |
| Time | 24 hours (per array task) |
| Memory | 180G |
| Queue | cca |
| Array Range | 0-2039 (2040 total jobs) |

## Task Mapping

Each `SLURM_ARRAY_TASK_ID` maps to:
```bash
MODEL_IDX = SLURM_ARRAY_TASK_ID / 20
REALIZATION = SLURM_ARRAY_TASK_ID % 20
```

Example:
- Task 0: Model 0 (dmo), Realization 0
- Task 19: Model 0 (dmo), Realization 19
- Task 20: Model 1 (hydro), Realization 0
- Task 2039: Model 101 (last Replace config), Realization 19

## Cleanup of Old Scripts (Optional)

The 4-part scripts are now redundant. You can archive them:
```bash
mkdir -p /mnt/home/mlee1/hydro_replace2/batch/archive
mv run_lux_binned_part[1-4].sh archive/
```

But keep them if you want to reference them later.

## Performance Comparison

### Before (4 Sequential Parts)
- Part 1: ~6 hours (510 jobs, max parallelism 510)
- Part 2: ~6 hours (510 jobs, max parallelism 510)
- Part 3: ~6 hours (510 jobs, max parallelism 510)
- Part 4: ~6 hours (510 jobs, max parallelism 510)
- **Total wall-clock time: ~24 hours**

### After (Unified)
- All 2040 jobs: ~6 hours (max parallelism 2040 simultaneously)
- **Total wall-clock time: ~6 hours** ✅

**Speedup: 4×** (subject to queue scheduling and available resources)

## Important Notes

1. **Resource Limits**: Your cluster must support 2040 simultaneous jobs. If the queue has a limit, jobs will queue and run when resources free up. This is fine—the scheduler will manage it.

2. **Cache Checking**: Each task checks if output already exists:
   - Conversion: checks for `config.dat` + `lenspot40.dat`
   - Ray-tracing: checks for `run100/kappa_40.dat`
   - Skips if already complete (efficient for retries)

3. **MPI Within Tasks**: Each array task still uses `srun -n 40` for intra-task parallelism. This doesn't conflict with array-level parallelism.

4. **Fallback**: If the unified job fails or hits queue limits, you can still run the 4-part scripts as a backup (though less efficient).
