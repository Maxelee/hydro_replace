# Simulation Specifications

## Overview

This document provides detailed specifications for all simulations used in the Hydro Replace project.

---

## TNG-300 Suite

### TNG-300 Hydrodynamic (Full Physics)

| Property | Value | Notes |
|----------|-------|-------|
| **Box Size** | 205 Mpc/h (≈302.6 cMpc) | Comoving |
| **Particles (DM)** | 2500³ = 15,625,000,000 | Type 1 |
| **Particles (Gas)** | 2500³ initially | Type 0 |
| **DM Mass Resolution** | 3.98 × 10⁷ M☉/h | 0.00398342749867548 × 10¹⁰ M☉/h |
| **Gas Mass Resolution** | 7.44 × 10⁶ M☉/h | Initial; varies due to feedback |
| **Softening (z=0)** | 1.0 kpc/h | Comoving for DM/stars |
| **Snapshots** | 100 (z=20 to z=0) | Snapshot 99 = z=0 |
| **Path** | `/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output` | |

### TNG-300-Dark (DMO)

| Property | Value | Notes |
|----------|-------|-------|
| **Box Size** | 205 Mpc/h (≈302.6 cMpc) | Same as hydro |
| **Particles (DM)** | 2500³ = 15,625,000,000 | Type 1 only |
| **DM Mass Resolution** | 4.73 × 10⁷ M☉/h | 0.0047271638660809 × 10¹⁰ M☉/h |
| **Softening (z=0)** | 1.0 kpc/h | |
| **Snapshots** | 100 | |
| **Path** | `/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output` | |

### Key Differences: Hydro vs DMO

- **DM particle mass differs** because in hydro runs, the total matter budget is split between DM and baryons
- Mass ratio: M_DM(DMO) / M_DM(Hydro) = 1.186 (accounting for Ω_b / Ω_m)
- When comparing halos, normalize by total matter enclosed, not particle count

---

## Cosmological Parameters

### TNG-300 Cosmology (Planck 2015)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **h** | 0.6774 | Hubble parameter |
| **Ω_m** | 0.3089 | Total matter density |
| **Ω_b** | 0.0486 | Baryon density |
| **Ω_Λ** | 0.6911 | Dark energy density |
| **σ_8** | 0.8159 | Amplitude of fluctuations |
| **n_s** | 0.9667 | Spectral index |

---

## Halo Definitions

### FoF (Friends-of-Friends)

| Property | Value |
|----------|-------|
| **Linking length** | b = 0.2 |
| **Minimum particles** | 32 |

### Spherical Overdensity

| Definition | Overdensity | Use Case |
|------------|-------------|----------|
| **M_200c** | 200 × ρ_crit | Primary halo mass definition |
| **R_200c** | Corresponding radius | Primary halo size |
| **M_500c** | 500 × ρ_crit | Cluster comparisons |
| **M_vir** | Bryan & Norman (1998) | Alternative |

---

## Mass Bins for Analysis

### Regular Bins (6 bins)

| Bin | log₁₀(M_200c [M☉/h]) | Expected Halos (TNG-300) |
|-----|----------------------|--------------------------|
| 1 | 12.0 - 12.5 | ~10,000 |
| 2 | 12.5 - 13.0 | ~3,000 |
| 3 | 13.0 - 13.5 | ~800 |
| 4 | 13.5 - 14.0 | ~200 |
| 5 | 14.0 - 14.5 | ~50 |
| 6 | > 14.5 | ~10 |

### Cumulative Bins

| Bin | Mass Cut | Description |
|-----|----------|-------------|
| 1 | M > 10¹² M☉ | All resolved halos |
| 2 | M > 10¹³ M☉ | Groups and clusters |
| 3 | M > 10¹⁴ M☉ | Massive clusters only |

---

## Replacement Radii

| Multiplier | Physical Meaning |
|------------|-----------------|
| 1× R_200c | Halo interior only |
| 3× R_200c | Includes 1-halo term |
| 5× R_200c | Full profile + 2-halo |

---

## CAMELS Simulations (Paper 3)

### SIMBA Suite

| Property | Value |
|----------|-------|
| **Box Size** | 25 Mpc/h |
| **Particles** | 256³ DM + 256³ gas |
| **Variations** | LH (1000), CV (27), 1P |
| **Parameters** | Ω_m, σ_8, A_SN1, A_SN2, A_AGN1, A_AGN2 |
| **Path** | `/mnt/sdceph/users/.../CAMELS/Sims/SIMBA/` |

### IllustrisTNG Suite  

| Property | Value |
|----------|-------|
| **Box Size** | 25 Mpc/h |
| **Particles** | 256³ DM + 256³ gas |
| **Variations** | LH (1000), CV (27), 1P |
| **Path** | `/mnt/sdceph/users/.../CAMELS/Sims/IllustrisTNG/` |

### Astrid Suite

| Property | Value |
|----------|-------|
| **Box Size** | 25 Mpc/h |
| **Particles** | 256³ DM + 256³ gas |
| **Variations** | LH (1000), CV (27), 1P |
| **Path** | `/mnt/sdceph/users/.../CAMELS/Sims/Astrid/` |

### CAMELS Parameter Ranges

| Parameter | Min | Fiducial | Max | Physical Meaning |
|-----------|-----|----------|-----|-----------------|
| **Ω_m** | 0.1 | 0.3 | 0.5 | Matter density |
| **σ_8** | 0.6 | 0.8 | 1.0 | Fluctuation amplitude |
| **A_SN1** | 0.25 | 1.0 | 4.0 | SN energy per unit SFR |
| **A_SN2** | 0.5 | 1.0 | 2.0 | SN wind speed scaling |
| **A_AGN1** | 0.25 | 1.0 | 4.0 | AGN feedback efficiency |
| **A_AGN2** | 0.5 | 1.0 | 2.0 | AGN feedback mode |

---

## Ray-Tracing Specifications

### Light Cone Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Field of View** | 5° × 5° | 25 deg² |
| **Pixel Scale** | 0.5 arcmin | 600 × 600 pixels |
| **LP Grid** | 4096 | Light propagation |
| **RT Grid** | 1024 | Ray-tracing |
| **Source n(z)** | LSST-like | z_med ≈ 0.9 |
| **Source Density** | 27 arcmin⁻² | After cuts |
| **Shape Noise** | σ_ε = 0.26 | Per component |

### Lens Planes

| Property | Value |
|----------|-------|
| **Number** | 50 |
| **z Range** | 0 to 2 |
| **Spacing** | Δz ≈ 0.04 |

---

## Data Products Summary

| File | Size Estimate | Contents |
|------|---------------|----------|
| `TNG300_halos_z0.h5` | ~100 MB | Halo catalogs |
| `matched_halos_TNG300.csv` | ~10 MB | Matched halo pairs |
| `hydro_halo_particles_5Rvir.h5` | ~100 GB | Extracted particles |
| `power_spectra_TNG300.h5` | ~50 MB | All P(k) measurements |
| `halo_profiles_TNG300.h5` | ~1 GB | Radial profiles |
| `convergence_maps_*.fits` | ~500 MB each | κ maps |
| `peak_counts.csv` | ~1 MB | Peak statistics |

---

## References

- **TNG-300**: Springel+2018, Nelson+2018, Pillepich+2018
- **CAMELS**: Villaescusa-Navarro+2021, 2023
- **Cosmology**: Planck Collaboration 2015
- **Halo Mass Function**: Tinker+2008
