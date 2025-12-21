# Efficient Path Forward: 3 Papers Strategy

> **Key Insight**: Build a unified pipeline that handles replacement/BCM during ray-tracing for all 20 snapshots.

---

## Dream Pipeline Summary

```
┌──────────────────────────────────────────────────────────────────┐
│  FOR EACH SNAPSHOT (20 total, z=0 to z≈3):                       │
│                                                                  │
│  1. Load TNG data (DMO + Hydro + Halo catalogs)                 │
│  2. Apply modification (based on mode flag):                     │
│     - DMO: baseline                                              │
│     - HYDRO: truth                                               │
│     - REPLACE: swap DMO halos with hydro particles               │
│     - BCM: apply BaryonForge model to DMO halos                  │
│  3. Save intermediate products:                                  │
│     - 2D projected maps                                          │
│     - P(k) power spectra                                         │
│     - Halo profiles (DMO, hydro, BCM) out to 5×R_vir            │
│  4. Feed modified particles to lux ray-tracing                   │
│  5. Save κ(θ) convergence maps                                   │
│  6. Peak count analysis                                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Current Assets Summary

| Asset | Location | Status | Papers |
|-------|----------|--------|--------|
| P(k) z=0 (66 configs) | `/mnt/home/mlee1/ceph/power_spectra/` | ✅ Ready | 1, 2 |
| Pixelized maps z=0 | `/mnt/home/mlee1/ceph/pixelized/` | ✅ Ready | 1 |
| BCM Arico20 halos (519) | `/mnt/home/mlee1/ceph/baryonification_output/halos/` | ✅ Ready | 2 |
| Halo matching | `/mnt/home/mlee1/ceph/halo_matches.npz` | ✅ Ready | All |
| lux ray-tracer | `/mnt/home/mlee1/lux/` | ✅ Ready | 1 |
| BCM_lensing code | GitHub: Maxelee/BCM_lensing | ✅ Available | 2 |

---

## Paper 1: Peaks + P(k) (TNG-300)

### What We Need to Build
| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| **Unified pipeline script** | HIGH | 1 week | Core infrastructure for all papers |
| Process all 20 snapshots | HIGH | 2-3 days compute | After pipeline built |
| BCM on all snapshots | HIGH | 2-3 days compute | Using BaryonForge Arico20 |
| lux ray-tracing (4 modes) | HIGH | 1 week compute | DMO, Hydro, Replace, BCM |
| Peak detection + statistics | HIGH | 2 days | Reuse 2023 code |

### Runs Needed
| Mode | Description | Priority |
|------|-------------|----------|
| DMO | Baseline, no baryons | HIGH |
| Hydro | Truth from TNG-300 | HIGH |
| Replace (5×R_200, M>10¹²) | Main result | HIGH |
| BCM Arico20 | BCM baseline | HIGH |
| Replace (3×R_200) | Radius test | MEDIUM |
| Replace (M>10¹³) | Mass threshold test | MEDIUM |

### Existing Data to Leverage
- ✅ 519 halos already BCM'd at z=0 (validation)
- ✅ P(k) computed for z=0 replacement (66 configs)
- ✅ Halo matching code working

---

## Paper 2: Multi-BCM Comparison

### What We Need
| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Add BCM models to pipeline | HIGH | 2 days | Schneider19, Mead20 via BaryonForge |
| Run pipeline with each BCM | HIGH | 3 days compute | Reuse Paper 1 infrastructure |
| Profile comparison analysis | HIGH | 2 days | BCM vs Hydro vs Replaced |
| Peak comparison | HIGH | 1 day | Which BCM matches peaks best? |

### BCM Models to Test
| Model | Reference | Notes |
|-------|-----------|-------|
| Arico20 | Arico+2020, 2021 | TNG-calibrated (baseline) |
| Schneider19 | Schneider+2015, 2019 | Original BCM |
| Mead20 | Mead+2020 | HMcode parametrization |
| Schneider25 | Schneider+2025 | Latest, if available |

---

## Paper 3: CAMELS Emulator

### What We Have
1. ✅ Working matching + extraction pipeline
2. ✅ BCM infrastructure

### What We Need
| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| CAMELS data access | HIGH | 1 day | Request access |
| Adapt pipeline for CAMELS | MEDIUM | 1 week | Smaller boxes (50 Mpc) |
| Mass conservation measurements | HIGH | 2 weeks | 81 sims × matching |
| GP emulator training | HIGH | 1 week | Standard scikit-learn |

### Efficient Workflow
```
Can start after Paper 1, Paper 2 work done
Week 1-2: CAMELS data access and pipeline adaptation
Week 3-4: Mass conservation measurements
Week 5-6: Emulator development
Week 7-8: Write Paper 3
```

---

## 5-Snapshot Pilot Plan

For multi-redshift validation before full ray-tracing:

| Snapshot | z | What to Compute | Priority |
|----------|---|-----------------|----------|
| 99 | 0.0 | ✅ Already done | N/A |
| 76 | 0.5 | P(k), profiles | HIGH |
| 63 | 0.93 | P(k), profiles | HIGH |
| 49 | 1.5 | P(k) only | MEDIUM |
| 35 | 2.32 | P(k) only | LOW |

### Implementation
1. Extract halo catalogs for snap 76, 63, 49, 35
2. Run BCM on matched halos
3. Compute P(k) for replaced snapshots
4. Plot S(k, z) evolution

---

## Critical Path (Minimum Viable Papers)

### Paper 1 MVP
1. ✅ Use existing P(k) for suppression analysis
2. Run lux on DMO + Hydro + Replaced (3 configurations)
3. Peak counts comparison

### Paper 2 MVP
1. ✅ Use existing Arico20 profiles
2. Add one additional BCM model (Schneider19)
3. Direct profile comparison

### Paper 3 MVP
1. Mass conservation on CAMELS-CV set (27 sims)
2. Simple GP emulator for ΔM(r, θ)

---

## Immediate Next Steps

### This Week: Build the Pipeline Core
1. **Create `HydroReplacePipeline` class** - Main Python orchestrator
2. **Test on single snapshot** (snap 99, z=0) with all 4 modes
3. **Validate outputs** against existing z=0 P(k) data

### Next Week: Scale to All Snapshots
1. Process all 20 snapshots with pipeline
2. Run lux ray-tracing for each mode
3. Generate convergence maps

### Week 3+: Analysis & Papers
1. Peak detection and comparison
2. Profile analysis (BCM vs Hydro vs Replace)
3. Write Paper 1

---

## Implementation Files

```
Priority 1: Pipeline Core
├── scripts/hydro_replace_pipeline.py    # Main orchestrator class
├── src/hydro_replace/replacement.py     # Particle replacement logic
├── src/hydro_replace/bcm_wrapper.py     # BaryonForge interface
├── src/hydro_replace/profiles.py        # Profile computation & saving
└── src/hydro_replace/power_spectrum.py  # P(k) computation

Priority 2: lux Integration  
├── lux branch: hydro_replace            # Minimal modifications if needed
├── batch/run_pipeline_single.sh         # Test single snapshot
└── batch/run_pipeline_full.sh           # Full 20-snapshot run

Priority 3: Analysis
├── scripts/analyze_peaks.py             # Peak detection on κ maps
├── scripts/compare_bcm_models.py        # Multi-BCM comparison
└── notebooks/paper1_figures.ipynb       # Publication plots
```

---

## Data Flow

```
TNG-300 Snapshots (20)
       │
       ▼
┌─────────────────────────────────────────┐
│     HydroReplacePipeline                │
│                                         │
│  mode = 'dmo' | 'hydro' | 'replace' |   │
│          'bcm:Arico20' | 'bcm:Schneider'│
└─────────────────────────────────────────┘
       │
       ├──► 2D Maps (Σ)     ──► Compare projections
       ├──► 3D P(k)         ──► Suppression S(k,z)
       ├──► Profiles ρ(r)   ──► BCM validation
       │
       ▼
   lux ray-tracing
       │
       ▼
   κ(θ) maps ──► Peak counts n(ν) ──► Paper 1
```
