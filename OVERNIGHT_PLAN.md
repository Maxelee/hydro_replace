# Overnight Pipeline Plan - December 21-22, 2025

## Overview of Requests

### Priority 1: 2D Projected Maps
- Models: DMO, Hydro, Replace, BCM (using best params from parameters_best.md)
- Snapshots: 20 ray-tracing snapshots + snap 99 (21 total)
- Mass thresholds (for Replace & BCM): M > 10^12.5, 10^13, 10^13.5, 10^14
- This creates: DMO (1) + Hydro (1) + Replace (4) + BCM×3×4 = 18 variants per snapshot
- Total: 21 snapshots × 18 variants = 378 map files

### Priority 2: Profiles (CORRECTED CODE)
- All matched halos, all models, all 21 snapshots
- Uses fixed generate_profiles.py (streams through all files per halo)
- Mass thresholds: Same 4 thresholds as above
- **⚠️ This is SLOW** - new code processes each halo serially through all snapshot files

### Priority 3: Density Fields for Lux (Lens Planes)
- 20 random rotations/translations per model
- All models and mass thresholds
- Grid resolution: 4096 for lens potential generation
- Binary format for lux PreProjected mode

### Priority 4: Ray-Tracing
- 20 random realizations × 100 convergence maps = 2000 maps per model
- Grid resolution: 1024 for ray-tracing output
- Uses lux with lenspot + raytracing phases

---

## CURRENT STATUS (Updated: December 21, 2025 ~5:30 PM)

### Jobs Submitted and Running
| Job ID | Name | Status | Purpose |
|--------|------|--------|---------|
| 2362760 | gen_matches[0-16] | PD (Resources) | Generate matches for 17 missing snapshots |
| 2362761 | gen_maps[0-20] | PD (Dependency) | Generate 2D maps for 21 snapshots (waits on matches) |
| 2362559-62 | lensplanes | R (~1.5hr) | Lens plane generation for BCM variants |
| 2362751-52 | lux_dmo/hydro | R (~30min) | Ray-tracing for DMO and Hydro |
| 2361824 | spawner-jupyter | R (~22hr) | Jupyter notebook server |

### Existing Data (L205n2500TNG)
- **Matches**: snaps 40, 49, 59, 76, 99 ✅ (generating 17 more)
- **2D Maps**: snaps 49, 59, 76, 99 (partial - old naming convention)
- **Profiles**: snap 99 only (BROKEN - needs regeneration with fixed code)
- **Lens planes**: In progress for DMO/Hydro + BCM variants

### Pipeline Dependencies
```
Matches (job 2362760) ──┬──> 2D Maps (job 2362761)
                        │
                        └──> Profiles (not yet submitted)
                        
Current lensplanes ─────> Lux raytracing
```

---

## Batch Scripts Created

### 1. `batch/run_all_matches.sh` - SUBMITTED (job 2362760)
- Array job for 17 missing snapshots
- Serial (1 task per snapshot), ~30-60 min each
- Snapshots: 29, 31, 33, 35, 38, 41, 43, 46, 52, 56, 63, 67, 71, 80, 85, 90, 96

### 2. `batch/run_all_maps.sh` - SUBMITTED (job 2362761, waiting on 2362760)
- Array job for 21 snapshots
- Parallel (64 tasks), ~4-6 hours each
- For each snapshot:
  - Step 1: DMO + Hydro (skip Replace/BCM)
  - Step 2: Loop over 4 mass thresholds for Replace + BCM

### 3. `batch/run_all_profiles.sh` - CREATED (not yet submitted)
- Array job for 21 snapshots
- Parallel (128 tasks), ~12 hours each (TBD)
- Uses FIXED generate_profiles.py
- Default: M > 10^13 halos only (for speed)

---

## Script Updates Made

### `scripts/generate_all.py`
Added new command-line flags:
- `--skip-bcm`: Skip BCM generation
- `--skip-replace`: Skip Replace generation  
- `--skip-dmo-hydro`: Skip DMO/Hydro generation (for mass-threshold-specific runs)
- `--output-suffix`: Suffix for output files (not currently used - mass_label handles this)

Logic updated:
- BCM loop now respects `--skip-bcm` flag
- Replace generation now respects `--skip-replace` flag
- DMO/Hydro generation now respects `--skip-dmo-hydro` flag

---

## Expected Timeline

### Tonight (Dec 21)
- **~5:30 PM**: Matches job starts (17 snapshots × ~45 min = 12+ hrs if sequential)
- **~6:00 PM**: Lens plane jobs finish (started 1.5 hrs ago)
- **~6:00 PM**: Lux raytracing jobs finish (started 30 min ago)

### Tomorrow Morning (Dec 22)
- **~6:00 AM**: Matches should be done (if array job gets resources)
- **~6:00 AM**: Maps job starts (depends on matches)
- Maps will take ~4-6 hours per snapshot (21 snapshots × 4-6 hrs = 84-126 hrs if sequential)
  - BUT it's an array job, so parallelizable!

### Key Check Points
1. Check matches completed: `ls /mnt/home/mlee1/ceph/hydro_replace_fields/L205n2500TNG/matches/`
2. Check maps progress: `ls /mnt/home/mlee1/ceph/hydro_replace_fields/L205n2500TNG/snap*/projected/`
3. Check job status: `squeue -u mlee1`

---

## To Submit Tomorrow (After Maps)

### Profiles Job
```bash
# Submit after maps complete (or with dependency)
sbatch --dependency=afterok:2362761 batch/run_all_profiles.sh
```

### Additional Lens Planes (with rotations)
Need to create/modify batch script for 20 random realizations

### Additional Ray-Tracing
Need to create batch script for 20 realizations × multiple models

---

## Notes

- The `--skip-existing` flag in maps job will skip files that exist
- Old files use naming like `replace.npz`, new files use `replace_Mgt12.5.npz`
- Profiles are SLOW with fixed code - consider running only on M > 10^13.5 halos
