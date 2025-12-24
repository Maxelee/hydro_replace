# Pipeline Computations Reference

This document details the mathematical computations performed at each step of the hydro_replace pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Step 0: Halo Matching](#step-0-halo-matching)
3. [Step 1: Particle Cache Generation](#step-1-particle-cache-generation)
4. [Step 2: Halo Statistics](#step-2-halo-statistics)
5. [Step 3: Density Profiles](#step-3-density-profiles)
6. [Step 4: 2D Density Maps](#step-4-2d-density-maps)
7. [Step 5: Lens Planes](#step-5-lens-planes)
8. [Step 6: Power Spectrum Analysis](#step-6-power-spectrum-analysis)

---

## Overview

The pipeline compares three cosmological simulation approaches:

- **DMO (Dark Matter Only)**: Pure N-body simulation with only dark matter particles
- **Hydro (Hydrodynamic)**: Full physics simulation including gas, stars, and baryonic feedback
- **Replace**: Hybrid method that replaces DMO halos with matched Hydro halos

### Simulation Parameters

| Resolution | Box Size | DM Mass (DMO) | DM Mass (Hydro) | Baryon Mass |
|------------|----------|---------------|-----------------|-------------|
| L205n625   | 205 Mpc/h | $3.03 \times 10^9\ M_\odot/h$ | $2.55 \times 10^9\ M_\odot/h$ | $4.77 \times 10^8\ M_\odot/h$ |
| L205n1250  | 205 Mpc/h | $3.78 \times 10^8\ M_\odot/h$ | $3.19 \times 10^8\ M_\odot/h$ | $5.96 \times 10^7\ M_\odot/h$ |
| L205n2500  | 205 Mpc/h | $4.73 \times 10^7\ M_\odot/h$ | $3.98 \times 10^7\ M_\odot/h$ | $7.45 \times 10^6\ M_\odot/h$ |

---

## Step 0: Halo Matching

**Script**: `generate_matches_fast.py`

### Purpose
Establish bijective (one-to-one) correspondence between DMO and Hydro halos.

### Key Insight
DMO and Hydro simulations share identical initial conditions, so dark matter particles have the same IDs in both simulations. This enables matching via particle ID overlap rather than spatial proximity.

### Algorithm

1. **Build Hydro Particle Map**: For each massive Hydro halo ($M_{200} > 10^{12}\ M_\odot/h$), store mapping:
   $$\text{particle\_ID} \rightarrow \text{halo\_index}$$

2. **Forward Matching**: For each DMO halo $i$, find Hydro halo $j$ with maximum particle overlap:
   $$j = \arg\max_k \left| \mathcal{P}_i^{\text{DMO}} \cap \mathcal{P}_k^{\text{Hydro}} \right|$$
   
   where $\mathcal{P}$ denotes the set of particle IDs.

3. **Compute Forward Overlap Fraction**:
   $$f_{\text{forward}} = \frac{\left| \mathcal{P}_i^{\text{DMO}} \cap \mathcal{P}_j^{\text{Hydro}} \right|}{N_i^{\text{DMO}}}$$

4. **Compute Reverse Overlap Fraction**:
   $$f_{\text{reverse}} = \frac{\left| \mathcal{P}_i^{\text{DMO}} \cap \mathcal{P}_j^{\text{Hydro}} \right|}{N_j^{\text{Hydro}}}$$

5. **Combined Overlap**: Use geometric mean to ensure good match in both directions:
   $$f_{\text{overlap}} = \sqrt{f_{\text{forward}} \times f_{\text{reverse}}}$$

6. **Bijective Filtering**: If multiple DMO halos match the same Hydro halo, keep only the match with highest $f_{\text{overlap}}$.

### Selection Criteria
- Minimum overlap: $f_{\text{overlap}} \geq 0.5$
- Minimum particles: $N \geq 100$
- Minimum mass: $M_{200} \geq 10^{12}\ M_\odot/h$

### Output
- `matches_snap{XXX}.npz`: Arrays of matched DMO/Hydro indices, positions, masses, radii

---

## Step 1: Particle Cache Generation

**Script**: `generate_particle_cache.py`

### Purpose
Pre-compute and store particle IDs within a fixed radius of each halo center for fast subsequent queries.

### Method

For each matched halo pair $(i_{\text{DMO}}, i_{\text{Hydro}})$:

1. **Query DMO particles** around DMO halo center:
   $$\mathcal{P}_{\text{DMO}}^{(i)} = \left\{ p : \left| \mathbf{r}_p - \mathbf{r}_{\text{DMO}}^{(i)} \right| \leq R_{\text{mult}} \times R_{200}^{\text{DMO}} \right\}$$

2. **Query Hydro particles** around both centers:
   - `hydro_at_dmo`: Hydro particles around DMO center (for Replace maps)
   - `hydro_at_hydro`: Hydro particles around Hydro center (for proper Hydro profiles)

3. **Periodic Boundary Handling**: For halos near box edges, wrap coordinates:
   $$\Delta \mathbf{r} = \begin{cases}
   \Delta \mathbf{r} - L_{\text{box}} & \text{if } \Delta r > L_{\text{box}}/2 \\
   \Delta \mathbf{r} + L_{\text{box}} & \text{if } \Delta r < -L_{\text{box}}/2
   \end{cases}$$

### Parameters
- $R_{\text{mult}} = 5.0$ (cache particles within $5 \times R_{200}$)
- $L_{\text{box}} = 205\ \text{Mpc}/h$

### Output
- `cache_snap{XXX}.h5`: HDF5 file with particle IDs per halo and halo metadata

---

## Step 2: Halo Statistics

**Script**: `compute_halo_statistics_lowmem.py`

### Purpose
Compute baryon fractions and mass conservation diagnostics for matched halo pairs.

### Baryon Fraction

The baryon fraction within radius $R$ is:

$$f_b(<R) = \frac{M_{\text{gas}}(<R) + M_{\star}(<R)}{M_{\text{total}}(<R)}$$

where:
- $M_{\text{gas}}$ = mass of gas particles (PartType 0)
- $M_{\star}$ = mass of star particles (PartType 4)  
- $M_{\text{total}}$ = total mass (gas + stars + DM)

#### Component Fractions

$$f_{\text{gas}}(<R) = \frac{M_{\text{gas}}(<R)}{M_{\text{total}}(<R)}$$

$$f_{\star}(<R) = \frac{M_{\star}(<R)}{M_{\text{total}}(<R)}$$

#### Cosmic Baryon Fraction (Reference)

$$f_b^{\text{cosmic}} = \frac{\Omega_b}{\Omega_m} = \frac{0.0486}{0.3089} \approx 0.157$$

### Mass Conservation Ratio

Compares total mass within $R$ between DMO and Hydro:

$$\text{ratio}_{\text{total}} = \frac{M_{\text{Hydro,total}}(<R)}{M_{\text{DMO}}(<R)}$$

$$\text{ratio}_{\text{DM}} = \frac{M_{\text{Hydro,DM}}(<R)}{M_{\text{DMO}}(<R)}$$

### Radii Evaluated

Statistics are computed at 6 apertures:
$$R/R_{200} \in \{0.5, 1.0, 2.0, 3.0, 4.0, 5.0\}$$

### Centering Convention

- **DMO particles**: Centered on DMO halo center, radii normalized by $R_{200}^{\text{DMO}}$
- **Hydro particles**: Centered on Hydro halo center, radii normalized by $R_{200}^{\text{Hydro}}$

This "apples-to-apples" comparison ensures each profile is relative to its own halo definition.

### Output
- `halo_statistics_snap{XXX}.h5`: Per-halo arrays of $f_b$, $f_{\text{gas}}$, $f_\star$, mass ratios at each radius

---

## Step 3: Density Profiles

**Script**: `generate_profiles_cached_new.py`

### Purpose
Compute spherically-averaged density profiles $\rho(r)$ for stacking analysis.

### Spherical Density Profile

For particles in radial bin $[r_i, r_{i+1}]$:

$$\rho_i = \frac{\sum_{r_i \leq r_j < r_{i+1}} m_j}{V_{\text{shell}}(r_i, r_{i+1})}$$

where the shell volume (in units of $R_{200}^3$) is:

$$V_{\text{shell}} = \frac{4\pi}{3}\left(r_{i+1}^3 - r_i^3\right)$$

### Radial Binning

Using logarithmic bins from $0.01\ R_{200}$ to $5.0\ R_{200}$:

$$r_i = 10^{\log_{10}(0.01) + i \cdot \Delta\log r}$$

where $\Delta\log r = \frac{\log_{10}(5.0) - \log_{10}(0.01)}{N_{\text{bins}}}$

### Stacking by Mass

Profiles are stacked in mass bins:
$$\log_{10}(M_{200}/[M_\odot/h]) \in \{[12.0, 12.5], [12.5, 13.0], ..., [14.5, 16.0]\}$$

Within each bin, profiles are averaged:

$$\langle \rho(r) \rangle = \frac{1}{N_{\text{halos}}} \sum_{k=1}^{N_{\text{halos}}} \rho_k(r)$$

### Output
- `profiles_snap{XXX}.h5`: Stacked $\rho_{\text{DMO}}(r)$, $\rho_{\text{Hydro}}(r)$ per mass bin

---

## Step 4: 2D Density Maps

**Script**: `generate_maps_cached_lowmem.py`

### Purpose
Generate projected 2D surface density maps for power spectrum analysis.

### Projection

Project 3D particle distribution along axis $\hat{z}$:

$$\Sigma(\mathbf{x}_\perp) = \int \rho(\mathbf{x}_\perp, z)\, dz$$

In practice, particles are assigned to a 2D grid using mass assignment.

### Mass Assignment Scheme: TSC (Triangular Shaped Cloud)

Each particle's mass is distributed over a $3 \times 3$ cell region:

$$W_{\text{TSC}}(\Delta x) = \begin{cases}
\frac{3}{4} - \Delta x^2 & |\Delta x| \leq 0.5 \\
\frac{1}{2}\left(\frac{3}{2} - |\Delta x|\right)^2 & 0.5 < |\Delta x| \leq 1.5 \\
0 & \text{otherwise}
\end{cases}$$

The 2D weight is:
$$W_{2D}(\Delta x, \Delta y) = W_{\text{TSC}}(\Delta x) \times W_{\text{TSC}}(\Delta y)$$

### Replace Map Construction

The Replace map combines:

1. **Background**: DMO particles *excluding* those within $5 R_{200}$ of matched halos
2. **Halos**: Hydro particles within $5 R_{200}$ of matched halo centers

$$\Sigma_{\text{Replace}} = \Sigma_{\text{DMO,background}} + \Sigma_{\text{Hydro,halos}}$$

This is implemented by:
1. Removing DMO particles with IDs in the cache
2. Adding Hydro particles with IDs in the cache (centered on DMO halo positions)

### Output
- `snap{XXX}/projected/{dmo,hydro,replace_M12p5}.npz`: 4096² surface density maps

---

## Step 5: Lens Planes

**Script**: `generate_lensplanes.py`

### Purpose
Generate density contrast fields for weak lensing ray-tracing with `lux`.

### Density Contrast

The dimensionless density contrast is:

$$\delta = \frac{\rho}{\bar{\rho}} - 1 = \frac{\Sigma}{\bar{\Sigma}} - 1$$

### Lens Plane Format

For ray-tracing, we output $\delta \times \Delta z$ where $\Delta z$ is the comoving slice thickness:

$$\Delta z = \frac{L_{\text{box}}}{N_{\text{planes}}}$$

For stacking mode (high-$z$), the effective thickness is doubled by tiling.

### Binary Output Format

```
[int32: grid_size]
[float64[N×N]: δ × Δz data]
[int32: grid_size (footer)]
```

### Grid Resolution
- `LP_grid = 4096`: Lens plane resolution
- `RT_grid = 1024`: Ray-tracing output resolution

---

## Step 6: Power Spectrum Analysis

**Script**: Analysis performed in notebooks using `Pk_library`

### 2D Power Spectrum

The 2D power spectrum of the projected density field:

$$P_{2D}(k) = \left\langle |\tilde{\delta}(\mathbf{k})|^2 \right\rangle_{|\mathbf{k}|=k}$$

where $\tilde{\delta}(\mathbf{k})$ is the Fourier transform of the overdensity field.

### Computation Steps

1. **Overdensity**: $\delta = \Sigma/\bar{\Sigma} - 1$
2. **FFT**: $\tilde{\delta}(\mathbf{k}) = \mathcal{F}[\delta(\mathbf{x})]$
3. **Azimuthal Average**: Average $|\tilde{\delta}|^2$ in annular $k$-bins

### Window Function Correction

The TSC mass assignment introduces a window function. For mode $\mathbf{k}$:

$$W_{\text{TSC}}(\mathbf{k}) = \prod_i \left[\frac{\sin(k_i \Delta x / 2)}{k_i \Delta x / 2}\right]^3$$

`Pk_library` applies appropriate deconvolution.

### Key Diagnostics

1. **Hydro/DMO Ratio**: Measures baryonic suppression
   $$R_{\text{H/D}}(k) = \frac{P_{\text{Hydro}}(k)}{P_{\text{DMO}}(k)}$$

2. **Replace/DMO Ratio**: Validates Replace method
   $$R_{\text{R/D}}(k) = \frac{P_{\text{Replace}}(k)}{P_{\text{DMO}}(k)}$$

3. **Replace/Hydro Ratio**: Measures Replace accuracy
   $$R_{\text{R/H}}(k) = \frac{P_{\text{Replace}}(k)}{P_{\text{Hydro}}(k)}$$

Ideally $R_{\text{R/H}} \approx 1$ at all scales.

---

## Units Summary

| Quantity | Simulation Units | Physical Units |
|----------|------------------|----------------|
| Position | kpc/h | Mpc/h (divide by 1000) |
| Mass | $10^{10}\ M_\odot/h$ | $M_\odot/h$ (multiply by $10^{10}$) |
| Velocity | km/s | km/s |
| Density | $M_\odot/h\ /\ (\text{Mpc}/h)^3$ | - |
| $k$ | $h$/Mpc | - |
| $P(k)$ | $(\text{Mpc}/h)^2$ for 2D | - |

---

## File Locations

| Output Type | Path |
|-------------|------|
| Matches | `/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{RES}TNG/matches/` |
| Cache | `/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{RES}TNG/particle_cache/` |
| Statistics | `/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{RES}TNG/analysis/` |
| Profiles | `/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{RES}TNG/profiles/` |
| Maps | `/mnt/home/mlee1/ceph/hydro_replace_fields/L205n{RES}TNG/snap{XXX}/projected/` |
| Lens Planes | `/mnt/home/mlee1/ceph/hydro_replace_lensplanes/L205n{RES}TNG/` |
| Lux Output | `/mnt/home/mlee1/ceph/lux_out/L205n{RES}TNG/` |
