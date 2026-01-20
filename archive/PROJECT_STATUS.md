# Hydro Replace Pipeline - Project Organization

## 🎯 Project Goal

Compare weak lensing signals from different cosmological modeling approaches:
- **DMO**: Dark Matter Only simulations (baseline)
- **Hydro**: Full hydrodynamic simulations (IllustrisTNG - "truth")
- **Replace**: Hybrid method replacing DMO halos with matched Hydro counterparts
- **BCM**: Baryonic Correction Models (Arico20, Schneider19, Schneider25)

## 📊 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PER-SNAPSHOT PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │ 1. MATCHES   │───▶│ 2. PARTICLE  │───▶│ 3. ANALYSIS              │  │
│  │              │    │    CACHE     │    │    - Profiles            │  │
│  │ DMO ↔ Hydro  │    │              │    │    - Baryon fractions    │  │
│  │ bijective    │    │ IDs within   │    │    - Mass conservation   │  │
│  │              │    │ 5×R200       │    │                          │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│         │                   │                                           │
│         │                   ▼                                           │
│         │            ┌──────────────────────────────────┐              │
│         └───────────▶│ 4. 2D DENSITY MAPS               │              │
│                      │    - DMO, Hydro                  │              │
│                      │    - Replace (various M_min)     │              │
│                      │    - BCM × 3 models × M_min      │              │
│                      └──────────────────────────────────┘              │
│                                     │                                   │
└─────────────────────────────────────┼───────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAY-TRACING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ 5. LENS      │───▶│ 6. RAY       │───▶│ 7. ANALYSIS  │              │
│  │    PLANES    │    │    TRACING   │    │              │              │
│  │              │    │              │    │ Power spectra│              │
│  │ 20 seeds ×   │    │ lux code     │    │ Peak counts  │              │
│  │ 20 snaps     │    │ κ maps       │    │ Comparisons  │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📁 Current Status

### Data Products (L205n2500TNG)

| Step | Product | Status | Location |
|------|---------|--------|----------|
| 1. Matches | `matches_snap{XXX}.npz` | ✅ 21/21 snaps | `matches/` |
| 2. Particle Cache | `cache_snap{XXX}.h5` | 🔄 1/21 (running) | `particle_cache/` |
| 3a. Profiles | `profiles_snap{XXX}.h5` | ❌ Not started | `profiles/` |
| 3b. Halo Stats | `halo_statistics_snap{XXX}.h5` | ❌ Not started | `analysis/` |
| 4. 2D Maps | `field_*.npz` | ✅ 21 snap dirs | `snap{XXX}/` |
| 5. Lens Planes | Binary files | 🔄 Partial | `lensplanes/` |
| 6. Ray Tracing | κ maps | 🔄 Partial | `lux_out/` |

### Currently Running Jobs

| Job ID | Name | Purpose | Started |
|--------|------|---------|---------|
| 2363517 | cache_test | Particle cache for snap 99 (2500) | ~2h ago |
| 2363449 | gen_matches | Matches for remaining snaps | ~3.5h ago |

### Test Pipeline (L205n625TNG)

| Step | Status |
|------|--------|
| Matches | ✅ snap 99 done |
| Particle Cache | ⏳ Job queued (test_625_full) |
| Halo Statistics | ⏳ Part of test_625_full |

## 📂 Code Organization

### Core Scripts (`scripts/`)

| Script | Purpose | Status |
|--------|---------|--------|
| `generate_matches_fast.py` | Bijective DMO↔Hydro matching | ✅ Production |
| `generate_particle_cache.py` | Cache particle IDs for halos | ✅ Production |
| `generate_all.py` | 2D density map generation | ✅ Production |
| `generate_profiles.py` | Radial density profiles | ✅ Production |
| `generate_lensplanes.py` | Lens plane generation | ✅ Production |
| `generate_lux_configs.py` | Lux ray-tracing configs | ✅ Production |
| `run_full_raytracing.py` | Ray-tracing orchestration | ✅ Production |
| `particle_access.py` | Particle access library | 🆕 New |
| `particle_analysis.py` | Analysis functions | 🆕 New |
| `compute_halo_statistics.py` | Baryon fractions, mass conservation | 🆕 New |
| `example_halo_analysis.py` | Example usage | 🆕 New |

### Batch Scripts (`batch/`)

**Production Scripts:**
| Script | Purpose |
|--------|---------|
| `run_all_matches.sh` | Generate matches for all snapshots |
| `run_all_maps.sh` | Generate 2D density maps |
| `run_all_profiles.sh` | Generate radial profiles |
| `run_all_lensplanes.sh` | Generate lens planes |
| `run_halo_statistics.sh` | Compute baryon fractions |
| `run_lux_all.sh` | Run lux ray-tracing |
| `run_lux_2500.sh` | Lux for 2500 resolution |
| `run_raytracing_pipeline.sh` | Full ray-tracing orchestration |
| `submit_full_pipeline.sh` | Master pipeline script |
| `run_full_pipeline.sh` | Alternative pipeline script |

**Test Scripts:**
| Script | Purpose |
|--------|---------|
| `test_625_full.sh` | Full test on 625 resolution |
| `test_cache_single.sh` | Single snapshot cache test |
| `test_lensplane_single.sh` | Single lens plane test |

**Archived** (`batch/archive/`): Old/redundant scripts moved for reference

## 🔧 Immediate Action Items

### Priority 1: Validate New Code (Today)
1. [ ] Wait for `test_625_full` job to complete
2. [ ] Check particle cache structure (new format with hydro_at_dmo, hydro_at_hydro)
3. [ ] Verify halo statistics output
4. [ ] Fix any bugs

### Priority 2: Complete Particle Caches (This Week)
1. [ ] Finish 2500 cache for snap 99
2. [ ] Generate caches for all 21 snapshots
3. [ ] Run halo statistics on all snapshots

### Priority 3: Science Analysis
1. [ ] Baryon fraction vs halo mass
2. [ ] Mass conservation DMO↔Hydro
3. [ ] Density profiles comparison
4. [ ] Lens plane validation

## 📋 Git Organization Suggestion

Keep everything in `master` but organize with clear commit messages:

```bash
# Current state - commit the new particle access code
git add scripts/particle_access.py scripts/particle_analysis.py scripts/compute_halo_statistics.py
git add scripts/generate_particle_cache.py  # Updated version
git add batch/test_625_full.sh batch/run_halo_statistics.sh
git commit -m "Add particle access library and halo statistics pipeline"

# Add docs
git add docs/PARTICLE_ACCESS_DESIGN.md
git commit -m "Add particle access design documentation"
```

Alternative: Use feature branches if you want to experiment:
- `feature/particle-cache` - New caching system
- `feature/halo-analysis` - Baryon fractions, profiles
- `production/lensplanes` - Stable lens plane code

## 🗂️ File Locations Quick Reference

```
/mnt/home/mlee1/
├── hydro_replace2/           # This repo
│   ├── scripts/              # Python code
│   ├── batch/                # SLURM scripts
│   ├── notebooks/            # Analysis notebooks
│   └── logs/                 # Job outputs
│
├── ceph/hydro_replace_fields/
│   └── L205n{RES}TNG/
│       ├── matches/          # Halo matching results
│       ├── particle_cache/   # Particle ID caches
│       ├── analysis/         # Baryon fractions, profiles
│       └── snap{XXX}/        # 2D density maps
│
├── ceph/hydro_replace_lensplanes/
│   └── L205n2500TNG/
│       └── seed{XXXX}/       # Lens planes per random seed
│
└── ceph/lux_out/             # Ray-tracing outputs
```

## 🚦 What To Do Right Now

1. **Wait** for test_625_full job to start/complete
2. **Monitor** with: `tail -f logs/test_625_full_*.o`
3. **Once validated**, scale up to 2500 resolution
4. **Commit** the new code to git

## 📝 Parameters to Vary

| Parameter | Values | Purpose |
|-----------|--------|---------|
| Mass threshold | 10^12.5, 10^13, 10^13.5, 10^14 | Which halos to replace |
| Radius factor | 3, 5 × R200 | Replacement aperture |
| BCM model | Arico20, Schneider19, Schneider25 | Baryon correction |
| Random seed | 2020-2039 | Lens plane realizations |

---

*Last updated: December 22, 2025*
