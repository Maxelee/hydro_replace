# Hydro Replace Pipeline - Clear Plan

## Goal

Generate projected density maps for ray-tracing across multiple snapshots.

### Required Outputs (per snapshot):
1. **DMO** - Full dark matter only field
2. **Hydro** - Full hydrodynamic field (gas + DM + stars)
3. **Replace** - DMO with halo regions replaced by hydro
4. **BCM** - DMO with baryonic correction model applied

### Additional Outputs:
- **Density Profiles** - ρ(r/R200) for DMO, Hydro, BCM for matched halos

## Working Code Reference

All working code exists in `/mnt/home/mlee1/Hydro_replacement/`:

| Script | Purpose | Status |
|--------|---------|--------|
| `bijective2.py` | Fast halo matching (MPI, histogram method) | ✅ Works |
| `dmo_hydro_extract_full.py` | Full-box DMO & Hydro pixelization | ✅ Works |
| `extract_and_pixelize_full3D.py` | Replacement fields (by mass bin) | ✅ Works |
| `BCM.py` | BCM profiles with BaryonForge | ✅ Works |

## Simulation Parameters

### TNG300-1 (High Resolution)
- **DMO**: `/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output`
- **Hydro**: `/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output`
- **Box**: 205 Mpc/h
- **DM mass (DMO)**: 0.0047271638660809 × 10¹⁰ M☉/h
- **DM mass (Hydro)**: 0.00398342749867548 × 10¹⁰ M☉/h

### Target Snapshots
Key redshifts for lensing:
- snap 99: z = 0.0
- snap 91: z = 0.1
- snap 84: z = 0.2
- snap 78: z = 0.3
- snap 72: z = 0.4
- snap 67: z = 0.5
- snap 59: z = 0.7
- snap 50: z = 1.0

## New Pipeline Structure

```
/mnt/home/mlee1/hydro_replace2/
├── scripts/
│   ├── run_full_pipeline.py      # Master script (loops over snapshots)
│   ├── step1_match_halos.py      # From bijective2.py
│   ├── step2_pixelize_fields.py  # DMO, Hydro, Replace
│   ├── step3_bcm_fields.py       # BCM application
│   └── step4_compute_profiles.py # Density profiles
│
├── batch/
│   └── submit_pipeline.sh        # SLURM submission
│
└── output/  (on ceph)
    └── L205n2500TNG/
        ├── snap099/
        │   ├── matches.npz       # Halo matches
        │   ├── dmo.npz           # 3D pixelized DMO
        │   ├── hydro.npz         # 3D pixelized Hydro
        │   ├── replace_gt13.npz  # Replacement (M > 10^13)
        │   ├── bcm_arico20.npz   # BCM field
        │   └── profiles.h5       # Density profiles
        ├── snap091/
        │   └── ...
        └── ...
```

## Output Format for Lux

Lux needs 2D projected lens planes. Two options:

### Option A: Save 3D fields, project in lux
- Save 3D density cubes (res³)
- Lux projects along each axis when reading

### Option B: Pre-project to 2D
- Save 2D projected maps (res²) × 3 axes
- Smaller files, faster I/O

**Recommendation**: Option A (3D fields) is more flexible for different projection depths.

## Execution Plan

### Phase 1: Halo Matching (if not already done)
```bash
# Only need to run once per simulation
mpirun -np 32 python step1_match_halos.py
```
- Already have matches at `/mnt/home/mlee1/ceph/halo_matches.npz`

### Phase 2: Generate Fields (per snapshot)
```bash
# For each snapshot
for SNAP in 99 91 84 78 72 67 59 50; do
    mpirun -np 64 python step2_pixelize_fields.py --snap $SNAP
    mpirun -np 16 python step3_bcm_fields.py --snap $SNAP
done
```

### Phase 3: Compute Profiles
```bash
python step4_compute_profiles.py --snap 99
```

## Timeline Estimate

| Step | Time per Snapshot | Total (8 snaps) |
|------|-------------------|-----------------|
| Matching | 10 min (once) | 10 min |
| DMO/Hydro fields | 20 min | 2.5 hr |
| Replace fields | 30 min | 4 hr |
| BCM fields | 40 min | 5.5 hr |
| Profiles | 10 min | 1.3 hr |
| **Total** | | **~14 hr** |

## Current Status

### New Scripts Created

| Script | Purpose | Based On |
|--------|---------|----------|
| `generate_fields.py` | DMO, Hydro, Replace fields | extract_and_pixelize_full3D.py |
| `generate_bcm_field.py` | BCM fields (3 models) | BCM.py |
| `compute_profiles.py` | Density profiles | BCM.py |
| `run_full_pipeline.sh` | Master SLURM submission | - |

### Existing Data

- **Halo matches**: `/mnt/home/mlee1/ceph/halo_matches.npz`
  - 519 matched halos (M > 10^12.8)
  - Snapshot 99 only
  - From bijective particle-ID matching

- **3D pixelized fields**: `/mnt/home/mlee1/ceph/pixelized_3D/`
  - DMO and Hydro full-box (res 1024)
  - Replace fields (normal/inverse, various mass bins and radii)

## Next Steps

1. **Test single snapshot**:
   ```bash
   cd /mnt/home/mlee1/hydro_replace2
   sbatch batch/run_full_pipeline.sh
   ```

2. **Verify outputs** for snap 99

3. **Run bijective matching** for other snapshots (if needed)

4. **Full pipeline run** across all snapshots
