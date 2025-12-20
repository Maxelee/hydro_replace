# Efficient Path Forward: 3 Papers Strategy

> **Key Insight**: We have substantial pre-computed data. The efficient path leverages existing products and fills gaps strategically.

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

### What We Have
1. ✅ Full P(k) suppression analysis ready (66 configurations)
2. ✅ Per-halo BCM profiles vs hydro profiles (519 halos)
3. ✅ Ray-tracing code (lux)

### What We Need
| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Visualize existing P(k) | HIGH | 1 day | Create suppression ratio plots |
| Profile comparison analysis | HIGH | 2 days | Already have data in .h5 files |
| Run lux on DMO/Hydro | HIGH | 1 week | Standard runs |
| Create replaced snapshots | MEDIUM | 3 days | Combine BCM halo outputs |
| Run lux on replaced | MEDIUM | 1 week | After snapshot creation |
| Peak detection + statistics | HIGH | 3 days | Reuse 2023 code |

### Efficient Workflow
```
Week 1: Visualize P(k), analyze BCM profiles
Week 2: Run lux on DMO + Hydro (parallel with analysis)
Week 3: Create replaced snapshots, run lux
Week 4: Peak detection, statistical comparison
Week 5: Write Paper 1
```

---

## Paper 2: Multi-BCM Comparison

### What We Have
1. ✅ Arico20 BCM applied to 519 halos
2. ✅ Per-halo density profiles: BCM, DMO, Hydro all stored
3. ✅ BCM_lensing code for additional models

### What We Need
| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Additional BCM models | HIGH | 3 days | Schneider19, Mead20 via BaryonForge |
| Profile comparison plots | HIGH | 2 days | Use existing .h5 data |
| P(k) from BCM snapshots | MEDIUM | 2 days | Grid baryonified_coords |
| Peak comparison | LOW | 3 days | After Paper 1 pipeline |

### Efficient Workflow
```
Can start immediately using existing BCM outputs
Week 1: Analyze existing Arico20 profiles (519 halos)
Week 2: Add Schneider19 + Mead20 models via BaryonForge
Week 3: Comparative analysis, plots
Week 4: Write Paper 2 (short paper ~8 pages)
```

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

## Immediate Next Actions

### Today/This Week
1. **Create P(k) suppression plots** from existing `/mnt/home/mlee1/ceph/power_spectra/`
2. **Analyze BCM profile quality** using `/mnt/home/mlee1/ceph/baryonification_output/halos/`
3. **Test lux** on a single snapshot

### Next Week
1. Run full lux on DMO + Hydro
2. Create replaced snapshot pipeline
3. Start peak analysis

---

## lux Modification Strategy

**Branch**: `hydro_replace` in `/mnt/home/mlee1/lux/`

### Option A: Convert to TNG-like Format (Recommended)
- Write `create_replaced_snapshot.py` that outputs HDF5 in TNG format
- Lux reads it as-is, no code changes needed
- Easy to validate

### Option B: Modify read_hdf.cpp
- Add flag for "replaced" mode
- Read from our custom format
- More flexible but requires C++ changes

**Decision**: Start with Option A for speed, consider Option B if performance issues.

---

## Data Dependencies Graph

```
halo_matches.npz
       │
       ├──► BCM.py ──► baryonification_output/halos/*.h5
       │                      │
       │                      ├──► profile_comparison (Paper 2)
       │                      │
       │                      └──► create_replaced_snapshot.py ──► replaced_snap.hdf5
       │                                                                  │
       │                                                                  ▼
       │                                                          lux ray-tracing
       │                                                                  │
       ├──► pixelized/ ──► power_spectra/ ──► suppression plots          │
       │                                              │                   │
       │                                              ▼                   ▼
       │                                         Paper 1 P(k)        Paper 1 Peaks
       │
       └──► CAMELS (future) ──► mass_conservation ──► GP emulator ──► Paper 3
```
