
---
tags: hydro-replace, todo, master-plan, phd-projects
date: 2025-12-20
status: active
---

# Hydro Replace Project Suite: Master TODO List
**Chronological & Logical Organization**

> **Philosophy**: Group tasks by type and dependencies, not by paper. Do foundational work once; extend for all papers as needed.

---

## EXISTING DATA PRODUCTS (Pre-computed)

**Location**: `/mnt/home/mlee1/ceph/pixelized/`  
**Total Files**: 61 files (~8.7 GB)

### Pixelized Replacement Maps
Generated via `extract_and_pixelize_full3D.py` using MPI-parallel cKDTree masking + CIC gridding.

**File Naming**: `pixelized_maps_res{RES}_axis{AXIS}_{MODE}_rad{RADIUS}_mass{MASS_LABEL}.npz`

| Parameter | Values |
|-----------|--------|
| Resolution | 4096³ |
| Axes | 0, 1, 2 |
| Modes | `normal` (DMO→Hydro), `inverse` (Hydro→DMO) |
| Radii | 1×, 3×, 5× R_200c |
| Mass bins (regular) | 10.0-12.5, 12.5-13.0, 13.0-13.5, 13.5-14.0, gt14.0 |
| Mass bins (cumulative) | gt10.0, gt12.5, gt13.0, gt13.5, gt14.0 |

**Contents per .npz file**:
- `hydro_density`: 2D projected hydro matter density
- `dmo_density`: 2D projected DMO matter density  
- `replaced_density`: DMO with hydro particles swapped in replacement regions

> **Note**: These data products cover a comprehensive parameter scan and can be used directly for power spectrum analysis without re-running extraction.

---

## PHASE 0: Foundation & Setup (Week 1: Dec 23-29, 2025)

### Literature & Context Building
*Do this first to inform all decisions*

- [ ] **LIT-0.1**: Read Miller et al. 2025 (arXiv:2511.10634) in full—this validates your approach
- [ ] **LIT-0.2**: Extract Miller's Table 1 values and Fig 1 for direct comparison
- [ ] **LIT-0.3**: Re-read your 2023 peaks paper (Lee+2023 MNRAS 519:573)—extract exact methodology
- [ ] **LIT-0.4**: Screenshot key figures from your 2023 paper for replication
- [ ] **LIT-0.5**: Read Schneider & Teyssier 2015 (2-halo BCM)
- [ ] **LIT-0.6**: Read Arico+2020 (A20-BCM original)
- [ ] **LIT-0.7**: Read Anbajagane+2024 map-level BCM (arXiv:2505.07949)
- [ ] **LIT-0.8**: Create `Literature_Notes.md` with 1-paragraph summaries of each
- [ ] **LIT-0.9**: Create master BibTeX file: `hydro_replace_references.bib`
- [ ] **LIT-0.10**: Read CAMELS overview (Villaescusa-Navarro+2021) for Paper 3 context
- [ ] **LIT-0.11**: Quick-read your CARPoolGP paper (Lee+2024) to recall GP methods

**→ Deliverable**: `Literature_Notes.md` + `hydro_replace_references.bib`

### Workspace Organization

- [ ] **SETUP-0.1**: Create Obsidian project structure:
		Hydro_Replace_Project/  
		├── Literature_Notes.md  
		├── Paper1_Peaks_Plan.md  
		├── Paper2_MultiBC_Plan.md  
		├── Paper3_Emulator_Plan.md  
		├── Master_TODO.md (this file)  
		├── Weekly_Progress/  
		├── Figures/  
		└── Data_Products/
- [x] **SETUP-0.2**: Create code repository structure:
		hydro_replace/  
		├── data/  
		├── analysis/  
		├── plotting/  
		├── paper1_peaks/  
		├── paper2_bcm_comparison/  
		├── paper3_emulator/  
		└── utils/
- [x] **SETUP-0.3**: Create `simulation_specifications.md` documenting all sim details
- [x] **SETUP-0.4**: Set up version control: `git init`, `.gitignore` for large data
- [x] **SETUP-0.5**: Create `README.md` with project overview
- [x] **SETUP-0.6**: Create `.github/copilot-instructions.md` for AI assistant context
- [x] **SETUP-0.7**: Create `notebooks/power_by_mass.ipynb` for interactive power spectrum analysis

**→ Deliverable**: Organized workspace ready for analysis

---

## PHASE 1: Core Data Infrastructure (Week 2-3: Dec 30-Jan 12)

### TNG-300 Data Access & Catalogs
*Build once, use for Papers 1 & 2*

- [ ] **DATA-1.1**: Verify access to TNG-300 z=0 snapshot (hydro)
- [ ] **DATA-1.2**: Verify access to TNG-300-Dark z=0 snapshot (DMO)
- [ ] **DATA-1.3**: Download/locate SubLink merger trees
- [ ] **DATA-1.4**: Document specs: box size (205 Mpc), resolution, particle masses
- [ ] **DATA-1.5**: Load FoF halo catalog: TNG-300 (M > 10^12 M☉)
- [ ] **DATA-1.6**: Load FoF halo catalog: TNG-300-Dark (same mass range)
- [ ] **DATA-1.7**: Extract halo properties: M_200c, R_200c, position, velocity
- [ ] **DATA-1.8**: Count halos per mass bin (10-12, 12-12.5, 12.5-13, 13-13.5, 13.5-14, 14+)
- [ ] **DATA-1.9**: Save as HDF5: `TNG300_halos_z0.h5` and `TNG300Dark_halos_z0.h5`

**→ Deliverable**: `TNG300_halos_z0.h5` (both hydro & DMO)

### Bijective Halo Matching
*Critical for all papers—do this perfectly once*

- [x] **MATCH-1.1**: Implement most-bound-particle matching algorithm
- [ ] **MATCH-1.2**: Match all halos M > 10^12 M☉ between DMO and hydro
- [ ] **MATCH-1.3**: Compute matching success rate (target: >95%)
- [ ] **MATCH-1.4**: For failed matches (<5%), document reasons (recent mergers, etc.)
- [ ] **MATCH-1.5**: Verify M_200c agreement between matched pairs (<10% scatter expected)
- [ ] **MATCH-1.6**: Visual inspection: 10 random halos in 3D projection
- [ ] **MATCH-1.7**: Save matched catalog: `matched_halos_TNG300.csv`
- [ ] **MATCH-1.8**: Write documentation: `Matching_Methodology.md`

**→ Deliverable**: `matched_halos_TNG300.csv` + validation plots

### Particle Extraction (Hydro Halos)
*For Papers 1 & 2 replacement experiments*

- [x] **EXTRACT-1.1**: For each hydro halo, extract particles within 5R_vir
- [x] **EXTRACT-1.2**: Separate by type: DM (type 1), gas (type 0), stars (type 4)
- [x] **EXTRACT-1.3**: Store: positions, velocities, masses, IDs
- [x] **EXTRACT-1.4**: Verify total mass within 5R_vir matches expected
- [x] **EXTRACT-1.5**: Handle periodic boundary conditions (flag edge halos)
- [ ] **EXTRACT-1.6**: Test on 5 halos: visual + mass check
- [ ] **EXTRACT-1.7**: Save: `hydro_halo_particles_5Rvir.h5` (large file, ~100 GB?)
- [ ] **EXTRACT-1.8**: Document storage location and access method

**→ Deliverable**: `hydro_halo_particles_5Rvir.h5`

---

## PHASE 2: Hydro Replacement Implementation (Week 4: Jan 13-19)

### Replacement Algorithm
*Core method for Papers 1 & 2*

- [x] **REPLACE-2.1**: Write `replace_halo()` function:
- Input: DMO snapshot, hydro halo particles, halo center, radius
- Output: modified snapshot with hydro particles inserted
- [x] **REPLACE-2.2**: Implement particle ID tracking (avoid duplicates)
- [x] **REPLACE-2.3**: Coordinate transformation: comoving ↔ physical
- [ ] **REPLACE-2.4**: Test on single halo: before/after particle count
- [ ] **REPLACE-2.5**: Verify momentum conservation within replaced region
- [ ] **REPLACE-2.6**: Create visualization: density slice before/after
- [ ] **REPLACE-2.7**: Profile 1 halo: does replaced ρ(r) match hydro? (sanity check)

**→ Deliverable**: Working `replace_halo()` function + validation

### Replacement Strategies
*Different radii & mass bins*

- [x] **REPLACE-2.8**: Implement replacement at R_200c (inner only)
- [x] **REPLACE-2.9**: Implement replacement at 3×R_200c (moderate)
- [x] **REPLACE-2.10**: Implement replacement at 5×R_200c (full, main analysis)
- [x] **REPLACE-2.11**: Document why these radii (Miller comparison + physics)
- [x] **REPLACE-2.12**: Mass bin strategy: define 6 bins (as above)
- [x] **REPLACE-2.13**: Create replacement runs:  *(See `/mnt/home/mlee1/ceph/pixelized/` - 61 files covering all mass bins × 3 radii × 2 modes)*
- Cumulative: all halos M > 10^12
- By mass bin: 6 separate runs
- By radius: 3 choices × 6 mass bins = 18 runs (for Paper 1 Fig 7-8)

**→ Deliverable**: 18+ replaced snapshot configurations

### Validation
*Critical QA step*

- [ ] **REPLACE-2.14**: Total particle count: before vs after (document difference)
- [ ] **REPLACE-2.15**: Check for particle overlaps at halo boundaries
- [ ] **REPLACE-2.16**: Visualize 3D density: DMO vs Replaced (3 projections: xy, xz, yz)
- [ ] **REPLACE-2.17**: Compute ρ(r) for 10 random replaced halos: match hydro profiles?
- [ ] **REPLACE-2.18**: Create validation report: `Replacement_Validation.md`

**→ Deliverable**: Validated replacement method + QA report

---

## PHASE 3: Mass Conservation Analysis (Week 5: Jan 20-26)

### Mass Deficit Measurements (TNG-300)
*For Paper 1; foundation for Paper 3*

- [ ] **MASS-3.1**: For each matched halo, compute M_hydro(<r) at r = [0.1, 0.5, 1, 2, 3, 5] × R_200c
- [ ] **MASS-3.2**: For same halos, compute M_DMO(<r) at same radii
- [ ] **MASS-3.3**: Calculate ΔM(r, M) = (M_hydro - M_DMO) / M_DMO
- [ ] **MASS-3.4**: Bin by halo mass (6 bins)
- [ ] **MASS-3.5**: Compute median ΔM(r, M) + 16-84th percentile scatter
- [ ] **MASS-3.6**: Save: `mass_deficit_TNG300.csv`

**→ Deliverable**: `mass_deficit_TNG300.csv`

### Miller et al. Comparison

- [ ] **MASS-3.7**: Extract Miller Table 1 values (α = 0.5, 1.0, 2.0 R_200)
- [ ] **MASS-3.8**: Reproduce Miller's measurements on your TNG data
- [ ] **MASS-3.9**: Compare: your ΔM vs Miller's (should agree within ~10%)
- [ ] **MASS-3.10**: If discrepant: diagnose (snapshot version? matching method?)
- [ ] **MASS-3.11**: Document comparison in `Miller_Comparison.md`

**→ Deliverable**: Validation that you match Miller's findings

### Plotting (Paper 1 Figure 1)

- [ ] **MASS-3.12**: **FIGURE: Mass deficit vs radius by mass bin**
- X: r/R_200c (log scale, 0.1 to 5)
- Y: (M_hydro - M_DMO) / M_DMO
- 6 curves for 6 mass bins, shaded 16-84 percentiles
- Overplot Miller+2025 points for validation
- Caption: "Mass deficit increases with radius for massive halos (M > 10^13.5), but low-mass halos show flattening/reversal beyond 2R_200, consistent with stellar feedback operating at shorter scales (Miller+2025)."
- [ ] **MASS-3.13**: Export as PDF: `Fig1_MassDeficit.pdf`

**→ Deliverable**: Paper 1 Figure 1

---

## PHASE 4: Power Spectrum Analysis (Week 6-7: Jan 27-Feb 9)

### Power Spectrum Computation
*For Papers 1 & 2*

- [x] **POWER-4.1**: Install/verify Pylians3 for 3D P(k)
- [ ] **POWER-4.2**: Grid TNG-300-Dark to 1024³ → compute P_DMO(k)
- [ ] **POWER-4.3**: Grid TNG-300 hydro (total matter) to 1024³ → compute P_hydro(k)
- [ ] **POWER-4.4**: For each of 18 replacement runs, grid → compute P_replaced(k)
- [ ] **POWER-4.5**: k-range: 0.1 to 10 h/Mpc (check Nyquist)
- [ ] **POWER-4.6**: Save all: `power_spectra_TNG300.h5`

**→ Deliverable**: `power_spectra_TNG300.h5`

### Suppression Calculation

- [ ] **POWER-4.7**: Compute S(k) = P_hydro(k) / P_DMO(k) (true baryonic suppression)
- [ ] **POWER-4.8**: Compute S_replaced(k) = P_replaced(k) / P_DMO(k) for each run
- [ ] **POWER-4.9**: Compute cumulative S(k): as halos added by mass (Fig 7 style)
- [ ] **POWER-4.10**: Compute by mass bin: 6 bins (Fig 8 style)

**→ Deliverable**: All suppression curves

### Validation

- [ ] **POWER-4.11**: Compare your S(k) to published TNG-300 (Springel+2018)
- [ ] **POWER-4.12**: Should agree to few percent; if not, diagnose
- [ ] **POWER-4.13**: Compare low-mass boost to Miller Fig 1 (k ~ 2-4 h/Mpc)
- [ ] **POWER-4.14**: Document: where does replacement match true hydro P(k)?

**→ Deliverable**: Validation report

### Plotting (Paper 1 Figures 2-3)

- [ ] **POWER-4.15**: **FIGURE 2: Cumulative P(k) suppression by mass**
- Layout: 6 panels (one per mass bin)
- Each panel: S(k) for R = 1×, 3×, 5× R_200
- Dashed: true hydro S(k)
- X: k [h/Mpc], Y: S(k)
- Caption: "Progressive suppression as replacement radius increases..."

- [ ] **POWER-4.16**: **FIGURE 3: Regular mass bins P(k)**
- Layout: 5 mass bins × 3 radii (5 columns, 3 rows or similar)
- Highlight low-mass power boost at k ~ 2-4 h/Mpc
- Caption: "Low-mass halos show power enhancement..."

- [ ] **POWER-4.17**: **FIGURE: Direct S(k) comparison**
- Single panel: S_replaced (5R_vir, all masses) vs S_hydro
- Show agreement/disagreement by k-range

**→ Deliverable**: Paper 1 Figures 2, 3, and S(k) comparison

---

## PHASE 5: Halo Profile Analysis (Week 8: Feb 10-16)

### Profile Extraction (TNG-300)
*For Papers 1 & 2*

- [x] **PROFILE-5.1**: For all matched halos, compute ρ_hydro(r) (50 radial bins, 0.01 to 5 R_200c)
- [x] **PROFILE-5.2**: For same halos, compute ρ_DMO(r)
- [x] **PROFILE-5.3**: Save: `halo_profiles_TNG300.h5` (indexed by halo ID)

**→ Deliverable**: `halo_profiles_TNG300.h5`

### BCM Application (Arico+2020)
*For Papers 1 & 2*

- [x] **BCM-5.1**: Obtain/re-implement Arico+2020 BCM code
- [x] **BCM-5.2**: Verify parameters: M_c, β, μ, θ_ej (from paper)
- [x] **BCM-5.3**: Apply to all TNG-300-Dark halos
- [x] **BCM-5.4**: Compute ρ_BCM(r) for all halos
- [x] **BCM-5.5**: Grid BCM-modified particles → P_BCM(k)
- [x] **BCM-5.6**: Validate: P_BCM should match P_hydro to ~1% (per Arico's claim)
- [x] **BCM-5.7**: Save: `halo_profiles_BCM.h5`

**→ Deliverable**: BCM-modified simulation + profiles

### Profile Residuals (Paper 1 Analysis)

- [ ] **PROFILE-5.8**: Compute (ρ_BCM - ρ_hydro) / ρ_hydro for each halo
- [ ] **PROFILE-5.9**: Bin by mass (4 bins: 13-13.5, 13.5-14, 14-14.5, 14.5-15)
- [ ] **PROFILE-5.10**: Compute median residual per bin
- [ ] **PROFILE-5.11**: Identify over-collapse radius: where BCM > hydro (typically r < 0.1 R_200)
- [ ] **PROFILE-5.12**: Identify expulsion radius: where BCM < hydro (typically r > R_200)
- [ ] **PROFILE-5.13**: Compute mean fractional error by mass bin
- [ ] **PROFILE-5.14**: Test correlation: does max|residual| increase with M? (yes, expected)

**→ Deliverable**: Profile residual statistics

### Plotting (Paper 1 Figure 4-5)

- [ ] **PROFILE-5.15**: **FIGURE 4: BCM vs Hydro profiles (reproduce committee Fig 9)**
- 4 panels for 4 mass bins
- X: r/R_200c (log), Y: (ρ_BCM/ρ_hydro) - 1
- Shade 16-84 percentiles
- Annotate: "over-collapse" zone (r < 0.1), "expulsion" zone (r > 1)

- [ ] **PROFILE-5.16**: **FIGURE 5: Profile error vs halo mass**
- X: log M_200c, Y: max|residual| within R_200c
- Show increasing trend
- Caption: "BCM profile errors increase with halo mass..."

**→ Deliverable**: Paper 1 Figures 4-5

---

## PHASE 6: Convergence Maps & Peak Counts (Week 9-10: Feb 17-Mar 2)

### Ray-Tracing Setup
*Reuse your 2023 methodology*

- [x] **RAYTRACE-6.1**: Verify/install ray-tracing code (your 2023 code)
- [x] **RAYTRACE-6.2**: Define source redshift: n(z) for LSST (n_gal = 27 arcmin⁻², z_med = 0.9)
- [x] **RAYTRACE-6.3**: Lens planes: 50 planes, z = 0 to 2
- [x] **RAYTRACE-6.4**: Map size: 5×5 degrees (match 2023)
- [x] **RAYTRACE-6.5**: Pixel resolution: 0.5 arcmin
- [x] **RAYTRACE-6.6**: Document in `Methods_RayTracing.md`

**→ Deliverable**: Ray-tracing pipeline ready

### Map Generation (4 Sets × 10 Realizations)
*For Paper 1*

- [ ] **RAYTRACE-6.7**: Generate κ_DMO maps (10 realizations, different projections)
- [ ] **RAYTRACE-6.8**: Generate κ_Hydro maps (10 realizations, true TNG-300)
- [ ] **RAYTRACE-6.9**: Generate κ_BCM maps (10 realizations, A20-BCM modified)
- [ ] **RAYTRACE-6.10**: Generate κ_Replace maps (10 realizations, Hydro Replace 5R_vir all masses)
- [ ] **RAYTRACE-6.11**: Add shape noise: σ_ε = 0.26 per galaxy
- [ ] **RAYTRACE-6.12**: Save as FITS: `convergence_maps_[DMO|Hydro|BCM|Replace]_r[00-09].fits`
- [ ] **RAYTRACE-6.13**: Total: 40 maps

**→ Deliverable**: 40 convergence maps

### Map Validation

- [ ] **RAYTRACE-6.14**: Compute statistics: mean, variance, skewness for each set
- [ ] **RAYTRACE-6.15**: Compare κ_Hydro variance to your 2023 paper (should match)
- [ ] **RAYTRACE-6.16**: Visual inspection: 5 random maps per set
- [ ] **RAYTRACE-6.17**: Check for artifacts (edge effects, pixelation, NaNs)

**→ Deliverable**: QA validated maps

### Peak Finding
*Paper 1 core analysis*

- [x] **PEAK-6.1**: Apply 1 arcmin Gaussian smoothing to all 40 maps
- [x] **PEAK-6.2**: Implement peak finder: local maxima detection
- [x] **PEAK-6.3**: Compute S/N: ν = κ_peak / σ_κ
- [x] **PEAK-6.4**: Define bins: ν = [0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6+]
- [x] **PEAK-6.5**: Count peaks per bin for all 40 maps
- [x] **PEAK-6.6**: Average over 10 realizations per set
- [x] **PEAK-6.7**: Compute errors: std dev across realizations
- [x] **PEAK-6.8**: Save: `peak_counts.csv`

**→ Deliverable**: `peak_counts.csv`

### Statistical Analysis

- [ ] **PEAK-6.9**: Compute n(ν) for κ_Hydro (truth baseline)
- [ ] **PEAK-6.10**: Compute n(ν) for κ_DMO, κ_BCM, κ_Replace
- [ ] **PEAK-6.11**: Fractional difference: Δn/n = (n_model - n_Hydro) / n_Hydro
- [ ] **PEAK-6.12**: χ² for each model vs Hydro: Σ[(n_model - n_Hydro)² / σ²]
- [ ] **PEAK-6.13**: Identify dominant χ² bins (expect ν > 4 for BCM)
- [ ] **PEAK-6.14**: Measure improvement: % reduction in χ² (Replace vs BCM)
- [ ] **PEAK-6.15**: Cumulative deficit for ν > 5: ∫[n_Hydro - n_model]dν

**→ Deliverable**: Peak statistics table

### Plotting (Paper 1 Figures 6-8)

- [ ] **PEAK-6.16**: **FIGURE 6: Peak counts n(ν)**
- X: S/N (ν), Y: n(ν) [peaks/deg²]
- 4 curves: Hydro (black), DMO (gray), BCM (red), Replace (blue)
- Error bars, inset: fractional difference

- [ ] **PEAK-6.17**: **FIGURE 7: Fractional difference per bin**
- X: S/N, Y: Δn/n
- 2 curves: BCM, Replace
- Shade ν > 4 (BCM failure region)

- [ ] **PEAK-6.18**: **FIGURE 8: χ² contribution per bin**
- X: S/N, Y: χ²_i
- 2 bars per bin: BCM vs Replace

**→ Deliverable**: Paper 1 Figures 6-8 (KEY RESULTS)

---

## PHASE 7: Peak-Halo Connection (Week 11: Mar 3-9)

### Peak-Halo Matching

- [ ] **PEAKHALO-7.1**: Extract top 100 peaks from κ_Hydro maps (highest S/N)
- [ ] **PEAKHALO-7.2**: Record positions (RA, Dec or x, y pixel)
- [ ] **PEAKHALO-7.3**: Match each peak to nearest halo (within 1 Mpc projected)
- [ ] **PEAKHALO-7.4**: Record halo mass M_200c for each peak
- [ ] **PEAKHALO-7.5**: Repeat for κ_BCM and κ_Replace maps (same peak locations)
- [ ] **PEAKHALO-7.6**: Create table: peak_id, position, κ_peak, M_halo, model
- [ ] **PEAKHALO-7.7**: Save: `peak_halo_matches.csv`

**→ Deliverable**: `peak_halo_matches.csv`

### Residual Analysis

- [ ] **PEAKHALO-7.8**: For each peak: Δκ = κ_model - κ_Hydro
- [ ] **PEAKHALO-7.9**: Bin by halo mass (3 bins: 13-13.5, 13.5-14, 14+)
- [ ] **PEAKHALO-7.10**: Compute mean |Δκ| per mass bin for BCM and Replace
- [ ] **PEAKHALO-7.11**: Test: does BCM residual correlate with mass? (expect yes)
- [ ] **PEAKHALO-7.12**: Test: does Replace residual correlate with mass? (expect no/weaker)

**→ Deliverable**: Peak-halo residual statistics

### Conditional Peak Statistics (Optional but Strong)

- [ ] **PEAKHALO-7.13**: Select peaks associated with M > 10^14 halos only
- [ ] **PEAKHALO-7.14**: Compute n(ν) for high-mass subsample
- [ ] **PEAKHALO-7.15**: Compare BCM vs Replace vs Hydro (expect BCM worse for massive halos)

**→ Deliverable**: Conditional statistics

### Plotting (Paper 1 Figures 9-10)

- [ ] **PEAKHALO-7.16**: **FIGURE 9: Peak residual vs halo mass**
- X: log M_200c, Y: Δκ / κ_Hydro
- Red points: BCM residuals (show mass trend)
- Blue points: Replace residuals (flat, smaller)
- Caption: "BCM profile errors produce mass-dependent peak biases..."

- [ ] **PEAKHALO-7.17**: **FIGURE 10: Profile-to-peak connection (two-panel)**
- Left: Profile comparison (Fig 4, repeated for context)
- Right: Peak amplitude vs mass (this analysis)
- Caption: "BCM core over-collapse produces overly concentrated halos, causing systematic peak amplitude errors that increase with halo mass. Hydro Replace corrects profiles and thereby eliminates peak biases."

**→ Deliverable**: Paper 1 Figures 9-10 (CONCEPTUAL KEY)

---

## PHASE 8: Paper 1 Writing (Week 12-14: Mar 10-30)

### Drafting (All Sections)

- [ ] **WRITE-8.1**: Draft Abstract (250 words, include quantitative improvement: "X% χ² reduction")
- [ ] **WRITE-8.2**: Draft Introduction (3 pages, 9 paragraphs as outlined in master list)
- [ ] **WRITE-8.3**: Draft Methods §2.1-2.3 (TNG sims, matching, Hydro Replace procedure)
- [ ] **WRITE-8.4**: Draft Methods §2.4-2.7 (BCM, P(k), ray-tracing, peaks)
- [ ] **WRITE-8.5**: Draft Results I: Profiles (§3.1-3.2, 3 pages)
- [ ] **WRITE-8.6**: Draft Results II: Power Spectrum (§3.3-3.5, 2 pages)
- [ ] **WRITE-8.7**: Draft Results III: Peaks (§3.6-3.8, 3 pages) **[MOST IMPORTANT]**
- [ ] **WRITE-8.8**: Draft Discussion (§4.1-4.5, 3 pages, include softened BCM claims)
- [ ] **WRITE-8.9**: Draft Conclusions (1 page, 4 bullet takeaways)

**→ Deliverable**: Full draft

### Figures & Tables

- [ ] **WRITE-8.10**: Create schematic: Hydro Replace procedure (Methods figure)
- [ ] **WRITE-8.11**: Finalize all 10 figures (Figs 1-10 from above)
- [ ] **WRITE-8.12**: Write figure captions (2-3 sentences each, interpretive not descriptive)
- [ ] **WRITE-8.13**: Create Table 1: Halo counts per mass bin
- [ ] **WRITE-8.14**: Create Table 2: Peak count χ² comparison (BCM vs Replace)
- [ ] **WRITE-8.15**: Create Table 3: Profile error by mass bin

**→ Deliverable**: All figures + tables

### Finalization

- [ ] **WRITE-8.16**: Compile bibliography from `hydro_replace_references.bib`
- [ ] **WRITE-8.17**: Check all citations formatted correctly (natbib/bibtex)
- [ ] **WRITE-8.18**: Check all figure/table references in text
- [ ] **WRITE-8.19**: Check all equation numbers
- [ ] **WRITE-8.20**: Spell check + grammar check (Grammarly?)
- [ ] **WRITE-8.21**: Read full draft aloud to catch awkward phrasing
- [ ] **WRITE-8.22**: Write acknowledgments (funding: NSF?, sims: TNG, collaborators, Shy)

**→ Deliverable**: Complete Paper 1 draft

### Advisor Review

- [ ] **WRITE-8.23**: Send draft to Shy with cover email: "Here's the Hydro Replace + Peaks paper. Key result: profiles matter for peaks."
- [ ] **WRITE-8.24**: Incorporate Shy's feedback (1 week turnaround expected)
- [ ] **WRITE-8.25**: Send revised draft to all committee members for comments
- [ ] **WRITE-8.26**: Final revisions based on committee feedback

**→ Deliverable**: Submission-ready Paper 1 by March 30, 2026

**→ SUBMIT PAPER 1 TO MNRAS by April 1, 2026**

---

## PHASE 9: Paper 2 Setup (Week 15-16: Mar 31-Apr 13)
*Multi-BCM comparison; short paper*

### Additional BCM Implementations

- [ ] **BCM-9.1**: Contact Schneider for baryonification code or re-implement
- [ ] **BCM-9.2**: Obtain HMCode with baryons (Mead code)
- [ ] **BCM-9.3**: Check BaryonForge for accessible implementations
- [ ] **BCM-9.4**: Calibrate Schneider BCM to TNG-300 P(k) (target <1% for k < 1 h/Mpc)
- [ ] **BCM-9.5**: Calibrate HMCode to TNG-300 P(k)
- [ ] **BCM-9.6**: Document fitted parameters in `BCM_Comparison_Table.md`

**→ Deliverable**: 3-4 calibrated BCMs ready

### Comparative Profile Analysis

- [ ] **PROFILE-9.1**: For 1000 halos, compute ρ_Schneider(r) (reuse hydro/DMO from Phase 5)
- [ ] **PROFILE-9.2**: For same halos, compute ρ_HMCode(r)
- [ ] **PROFILE-9.3**: Save: `bcm_comparison_profiles.h5`
- [ ] **PROFILE-9.4**: Compute mean fractional error vs r for each BCM
- [ ] **PROFILE-9.5**: Compute RMS error integrated over r
- [ ] **PROFILE-9.6**: Bin by mass; compute metrics per bin
- [ ] **PROFILE-9.7**: Create ranking table: which BCM best matches hydro profiles?
- [ ] **PROFILE-9.8**: **Key test: Does Schneider (with 2-halo) improve profiles vs Arico?**

**→ Deliverable**: Comparative profile statistics

### Comparative Peak Analysis

- [ ] **PEAK-9.1**: Generate κ_Schneider maps (10 realizations, reuse ray-tracing pipeline)
- [ ] **PEAK-9.2**: Generate κ_HMCode maps (10 realizations)
- [ ] **PEAK-9.3**: Compute n(ν) for all BCMs
- [ ] **PEAK-9.4**: Compute χ² for each BCM vs Hydro
- [ ] **PEAK-9.5**: Rank BCMs by peak accuracy
- [ ] **PEAK-9.6**: Test hypothesis: better profile → better peaks?

**→ Deliverable**: Multi-BCM peak statistics

### Plotting (Paper 2: 6 Figures Total)

- [ ] **PLOT-9.1**: **FIGURE 1: Multi-BCM profile comparison (4 panels by mass bin)**
- [ ] **PLOT-9.2**: **FIGURE 2: Profile error vs mass (3-4 curves for BCMs)**
- [ ] **PLOT-9.3**: **FIGURE 3: Multi-BCM peak counts n(ν)**
- [ ] **PLOT-9.4**: **FIGURE 4: Profile RMS error vs Peak χ² (scatter plot, points = BCMs)**
- [ ] **PLOT-9.5**: **FIGURE 5: 2-halo term contribution (Schneider only, beyond 2R_200)**
- [ ] **PLOT-9.6**: **FIGURE 6: BCM performance summary (bar chart: ranks by metric)**

**→ Deliverable**: All Paper 2 figures

---

## PHASE 10: Paper 2 Writing (Week 17-18: Apr 14-27)

### Drafting (Shorter Format: ~8 Pages)

- [ ] **WRITE-10.1**: Abstract: "We compare 3-4 BCMs on profiles and peaks"
- [ ] **WRITE-10.2**: Intro (1.5 pages): Motivation from Paper 1, why compare BCMs?
- [ ] **WRITE-10.3**: Methods (2 pages): BCMs tested, calibration details, analysis pipeline
- [ ] **WRITE-10.4**: Results (3 pages): Profile comparison, peak comparison, rankings, 2-halo test
- [ ] **WRITE-10.5**: Discussion (1.5 pages): Which BCM features matter? Is 2-halo sufficient?
- [ ] **WRITE-10.6**: Conclusions: Recommendations for BCM users and developers

**→ Deliverable**: Full Paper 2 draft

### Finalization

- [ ] **WRITE-10.7**: Finalize all 6 figures with interpretive captions
- [ ] **WRITE-10.8**: Create Table: BCM parameter values and calibration data
- [ ] **WRITE-10.9**: Spell/grammar check
- [ ] **WRITE-10.10**: Send to Shy for feedback
- [ ] **WRITE-10.11**: Revise and submit to MNRAS

**→ SUBMIT PAPER 2 by May 1, 2026**

---

## PHASE 11: CAMELS Data & Mass Conservation (Week 19-21: Apr 28-May 18)
*Foundation for Paper 3 emulator*

### CAMELS Access

- [ ] **CAMELS-11.1**: Access CAMELS-SB suite (27 sims)
- [ ] **CAMELS-11.2**: Access CAMELS-TNG suite (27 sims)
- [ ] **CAMELS-11.3**: Access CAMELS-Astrid suite (27 sims)
- [ ] **CAMELS-11.4**: Total: 81 sims spanning (Ω_m, σ_8, A_SN1, A_SN2, A_AGN1, A_AGN2)
- [ ] **CAMELS-11.5**: Download z=0 snapshots or use 50 Mpc/h boxes
- [ ] **CAMELS-11.6**: Create `camels_parameter_table.csv` with all θ values

**→ Deliverable**: CAMELS sim access + parameter table

### Halo Catalogs (Reuse Matching Method from Phase 1)

- [ ] **CAMELS-11.7**: Load halo catalogs for all 81 hydro sims (M > 10^12)
- [ ] **CAMELS-11.8**: Load halo catalogs for all 81 DMO sims
- [ ] **CAMELS-11.9**: Bijectively match hydro-DMO halos (reuse Phase 1 code)
- [ ] **CAMELS-11.10**: Verify ~10-50 matched halos per sim
- [ ] **CAMELS-11.11**: Save: `camels_matched_halos.csv`

**→ Deliverable**: CAMELS halo database

### Mass Conservation Measurements (Extends Phase 3)

- [ ] **CAMELS-11.12**: For each halo in 81 hydro sims, compute M_hydro(<r) at r = [0.5, 1, 2, 3, 5] × R_200c
- [ ] **CAMELS-11.13**: For matched DMO halos, compute M_DMO(<r)
- [ ] **CAMELS-11.14**: Calculate ΔM(r, M, θ) = (M_hydro - M_DMO) / M_DMO
- [ ] **CAMELS-11.15**: Store: halo_id, M_200c, θ (6D vector), ΔM at 5 radii
- [ ] **CAMELS-11.16**: Total data: ~81 × 30 × 5 = ~12,000 points
- [ ] **CAMELS-11.17**: Save: `camels_mass_conservation.h5`

**→ Deliverable**: `camels_mass_conservation.h5`

### Mass Binning

- [ ] **CAMELS-11.18**: Bin halos: 12-12.5, 12.5-13, 13-13.5, 13.5-14 (log M☉)
- [ ] **CAMELS-11.19**: For each bin, compute ΔM(r, θ)
- [ ] **CAMELS-11.20**: Ensure ≥5 halos per (mass bin, radius, θ combination)

**→ Deliverable**: Binned mass conservation data

---

## PHASE 12: Parameter Dependence Analysis (Week 22-23: May 19-Jun 1)

### Visual Exploration

- [ ] **PARAM-12.1**: Plot ΔM(R_200c) vs A_AGN1 (fix other params)
- [ ] **PARAM-12.2**: Plot ΔM(R_200c) vs A_AGN2
- [ ] **PARAM-12.3**: Plot ΔM(R_200c) vs A_SN1
- [ ] **PARAM-12.4**: Plot ΔM(R_200c) vs A_SN2
- [ ] **PARAM-12.5**: Plot ΔM(R_200c) vs Ω_m
- [ ] **PARAM-12.6**: Plot ΔM(R_200c) vs σ_8
- [ ] **PARAM-12.7**: Identify which parameters dominate at which radii

**→ Deliverable**: Exploratory parameter plots

### Correlation Analysis

- [ ] **PARAM-12.8**: Compute Spearman correlation: ρ(ΔM, A_AGN1) for each radius
- [ ] **PARAM-12.9**: Repeat for all 6 parameters
- [ ] **PARAM-12.10**: Create heatmap: 5 radii × 6 parameters (color = correlation strength)
- [ ] **PARAM-12.11**: Interpret: "At r > 2R_200, A_AGN1 dominates; at r < R_200, A_SN dominates"

**→ Deliverable**: Parameter sensitivity table + heatmap

### Plotting (Paper 3 Figures 1-2)

- [ ] **PARAM-12.12**: **FIGURE 1: ΔM vs feedback parameters (6 panels)**
- X: parameter value, Y: ΔM at R_200c
- Color by mass bin, show trends

- [ ] **PARAM-12.13**: **FIGURE 2: Correlation heatmap**
- Rows: 5 radii, Columns: 6 parameters
- Show AGN dominates large r, SN small r

**→ Deliverable**: Paper 3 Figures 1-2

---

## PHASE 13: GP Emulator Development (Week 24-26: Jun 2-22)

### Feature Engineering

- [ ] **EMULATOR-13.1**: Define inputs: X = (Ω_m, σ_8, A_SN1, A_SN2, A_AGN1, A_AGN2, log M, r/R_200)
- [ ] **EMULATOR-13.2**: Define target: y = ΔM(r, M, θ)
- [ ] **EMULATOR-13.3**: Normalize inputs: StandardScaler (mean=0, std=1)
- [ ] **EMULATOR-13.4**: Check for skewness in y; apply log-transform if needed

**→ Deliverable**: Feature matrix X, target vector y

### Train/Test Split

- [ ] **EMULATOR-13.5**: Use 64 CAMELS sims for training (80%)
- [ ] **EMULATOR-13.6**: Reserve 17 sims for testing (20%)
- [ ] **EMULATOR-13.7**: Stratified split: ensure parameter space coverage
- [ ] **EMULATOR-13.8**: Training points: ~10,000
- [ ] **EMULATOR-13.9**: Testing points: ~2,500

**→ Deliverable**: Train/test split

### GP Training

- [ ] **EMULATOR-13.10**: Use scikit-learn GaussianProcessRegressor or GPy
- [ ] **EMULATOR-13.11**: Kernel: RBF with length scales per dimension
- [ ] **EMULATOR-13.12**: Test: single GP with r as input vs separate GP per radius (compare)
- [ ] **EMULATOR-13.13**: Hyperparameter tuning: maximize marginal likelihood
- [ ] **EMULATOR-13.14**: Train on full training set (~1-2 hours)
- [ ] **EMULATOR-13.15**: Save: `mass_redistribution_gp.pkl`

**→ Deliverable**: Trained GP emulator

### Validation

- [ ] **EMULATOR-13.16**: Predict ΔM for all test set halos
- [ ] **EMULATOR-13.17**: Compute MAE (target: <5%)
- [ ] **EMULATOR-13.18**: Compute RMSE
- [ ] **EMULATOR-13.19**: Compute R² (target: >0.95)
- [ ] **EMULATOR-13.20**: If poor: diagnose (more data? different kernel? outliers?)
- [ ] **EMULATOR-13.21**: Test on CAMELS-1P (single-param variations, extrapolation test)
- [ ] **EMULATOR-13.22**: Test on CAMELS-CV (cosmology variations)

**→ Deliverable**: Validation report

### Physical Consistency Checks

- [ ] **EMULATOR-13.23**: Check: ΔM → 0 as A_AGN, A_SN → 0 (no feedback limit)
- [ ] **EMULATOR-13.24**: Check: ΔM(r) monotonic with r
- [ ] **EMULATOR-13.25**: Check: ΔM increases with A_AGN at large r

**→ Deliverable**: Physics validation

### Plotting (Paper 3 Figures 3-5)

- [ ] **EMULATOR-13.26**: **FIGURE 3: Predicted vs True ΔM**
- X: True, Y: Predicted
- Color by radius, 1:1 line

- [ ] **EMULATOR-13.27**: **FIGURE 4: Residual vs parameters (6 panels)**
- Show no systematic bias

- [ ] **EMULATOR-13.28**: **FIGURE 5: GP uncertainty**
- X: r/R_200, Y: ΔM ± 2σ

**→ Deliverable**: Paper 3 Figures 3-5

---

## PHASE 14: Analytic Fitting & Physics (Week 27-28: Jun 23-Jul 6)

### Functional Form

- [ ] **FIT-14.1**: Propose analytic form:
\[
\Delta M = f_{\text{core}}(M, A_{\text{SN}}) + f_{\text{expel}}(M, A_{\text{AGN}}) \cdot g(r/R_{200})
\]
- [ ] **FIT-14.2**: Try f_core: power-law in M and A_SN
- [ ] **FIT-14.3**: Try f_expel: exponential in r, power-law in A_AGN
- [ ] **FIT-14.4**: Fit using scipy.optimize.curve_fit
- [ ] **FIT-14.5**: Compare analytic to GP (compute R²)
- [ ] **FIT-14.6**: If R² > 0.9: analytic form is interpretable alternative

**→ Deliverable**: Analytic model

### Parameter Derivatives

- [ ] **FIT-14.7**: Use GP to compute ∂ΔM/∂A_AGN1 across parameter space
- [ ] **FIT-14.8**: Compute ∂ΔM/∂A_SN1
- [ ] **FIT-14.9**: Plot derivatives vs radius (shows where each param matters)
- [ ] **FIT-14.10**: Interpret: quantify at what radius AGN effects dominate

**→ Deliverable**: Parameter derivatives + interpretation

### Plotting (Paper 3 Figures 6-7)

- [ ] **FIT-14.11**: **FIGURE 6: Analytic model vs data**
- Several panels: ΔM(r) for different θ
- Solid: data, dashed: analytic fit

- [ ] **FIT-14.12**: **FIGURE 7: Parameter derivatives**
- X: r/R_200, Y: ∂ΔM/∂A_i
- 6 curves for 6 parameters

**→ Deliverable**: Paper 3 Figures 6-7

---

## PHASE 15: Applications & Paper 3 Writing (Week 29-32: Jul 7-Aug 3)

### Application 1: Mass-Conserving BCM

- [ ] **APP-15.1**: Use emulator to predict ΔM(r, M, θ) for TNG-300-Dark halos
- [ ] **APP-15.2**: Implement redistribution: remove ΔM from core, add to outskirts
- [ ] **APP-15.3**: Compute P(k) from mass-conserving BCM
- [ ] **APP-15.4**: Compare to Arico BCM and true hydro
- [ ] **APP-15.5**: Check: improvement in P(k) or peaks?

**→ Deliverable**: Proof-of-concept mass-conserving BCM

### Application 2: Constraining Feedback (Conceptual)

- [ ] **APP-15.6**: Forward model: given θ, predict n(ν) using emulator
- [ ] **APP-15.7**: Create n(ν, θ) surface over parameter space
- [ ] **APP-15.8**: Inverse: simplified Fisher forecast for constraining θ from LSST peaks
- [ ] **APP-15.9**: Demonstrate: "LSST could constrain A_AGN to X%"

**→ Deliverable**: Forecasted constraints figure

### Paper 3 Writing (Full Paper: ~15 Pages)

- [ ] **WRITE-15.1**: Abstract: "GP emulator for ΔM(r, M, θ) using CAMELS"
- [ ] **WRITE-15.2**: Intro (3 pages): Papers 1+2 motivation, CAMELS, emulators, feedback
- [ ] **WRITE-15.3**: Methods (3 pages): CAMELS sims, measurements, GP training, validation
- [ ] **WRITE-15.4**: Results I (3 pages): Parameter dependence, correlations
- [ ] **WRITE-15.5**: Results II (2 pages): Emulator performance, analytic fit
- [ ] **WRITE-15.6**: Applications (2 pages): Mass-conserving BCM, observational constraints
- [ ] **WRITE-15.7**: Discussion (2 pages): Limitations, extensions
- [ ] **WRITE-15.8**: Conclusions: Community tool, feedback inference

**→ Deliverable**: Full Paper 3 draft

### Code Release

- [ ] **WRITE-15.9**: Create GitHub repo: `CAMELS_MassRedistribution_Emulator`
- [ ] **WRITE-15.10**: Include: trained GP (.pkl), inference code, tutorial Jupyter notebook
- [ ] **WRITE-15.11**: Write README: installation, usage, citation
- [ ] **WRITE-15.12**: Add to paper: "Code: github.com/..."

**→ Deliverable**: Public code

### Finalization

- [ ] **WRITE-15.13**: Finalize all 12-15 figures with captions
- [ ] **WRITE-15.14**: Create tables: CAMELS parameter ranges, emulator performance metrics
- [ ] **WRITE-15.15**: Spell/grammar check
- [ ] **WRITE-15.16**: Send to Shy + collaborators (consider adding Villaescusa-Navarro?)
- [ ] **WRITE-15.17**: Revise based on feedback

**→ SUBMIT PAPER 3 by September 1, 2026**

---

## PHASE 16: Conference Presentations & Job Market (Ongoing: May-Oct 2026)

### Conferences

- [ ] **CONF-16.1**: Submit Paper 1 results to summer conference (June-Aug: AAS, Cosmo21)
- [ ] **CONF-16.2**: Prepare 15-min talk for Paper 1 (practice 5+ times)
- [ ] **CONF-16.3**: Submit Paper 3 to fall conference (Sep-Nov: DPF, AAS)
- [ ] **CONF-16.4**: Prepare poster version of Paper 2 (optional)

### Job Market Prep (Fall 2026 → Fall 2027 Applications)

- [ ] **JOB-16.1**: Update CV: 3 first-author papers, 2 co-author
- [ ] **JOB-16.2**: Draft research statement highlighting Hydro Replace contributions
- [ ] **JOB-16.3**: Prepare job talk (45 min) based on Papers 1+3
- [ ] **JOB-16.4**: Request letters: Shy, Zoltan, Greg, Colin
- [ ] **JOB-16.5**: Identify postdoc positions compatible with Erin's career (media hubs: NYC, LA, Bay Area, Chicago, DC, etc.)

---

## ONGOING: Weekly & Monthly Management

### Weekly Reviews (Every Friday)

- [ ] **WEEKLY-1**: Review completed tasks from all phases
- [ ] **WEEKLY-2**: Update `Master_TODO.md` with checkmarks
- [ ] **WEEKLY-3**: Update `Projects.md` in Obsidian
- [ ] **WEEKLY-4**: Compute finish/start ratio for the week (target: 80/20)
- [ ] **WEEKLY-5**: Identify blockers for next week
- [ ] **WEEKLY-6**: Plan top 3 priorities for next week
- [ ] **WEEKLY-7**: Update Shy via email if major milestone

### Bi-Weekly Advisor Meetings

- [ ] **ADVISOR-1**: Prepare 2-3 slides showing recent figures
- [ ] **ADVISOR-2**: Discuss blockers and get guidance
- [ ] **ADVISOR-3**: Share draft sections as they're completed

### Monthly Committee Updates

- [ ] **COMMITTEE-1**: Send 1-paragraph progress email to full committee
- [ ] **COMMITTEE-2**: Highlight: papers submitted, figures completed, conferences

---

## KEY REFERENCES (Organized Alphabetically)

### Baryonic Correction Models
- Anbajagane+2024: Map-level baryonification (arXiv:2505.07949)
- Arico+2020, 2021: A20-BCM
- Mead+2021: HMCode with baryons
- Schneider & Teyssier 2015: 2-halo baryonification
- Schneider+2019: Extended 2-halo model

### CAMELS & Emulators
- Villaescusa-Navarro+2021, 2023: CAMELS overview
- Anbajagane+2023: CAMELS emulators
- Ni+2023: Baryonification emulator
- Your CARPoolGP: Lee+2024

### Weak Lensing & Peaks
- Your peaks paper: Lee+2023 MNRAS 519:573
- Dietrich & Hartlap: Peak theory
- Liu & Haiman: Peak systematics
- DES Y3 moments: Abbott+2020

### Hydro Simulations
- TNG-300: Springel+2018, Nelson+2018, Pillepich+2018
- IllustrisTNG papers: Genel, Bryan, Vogelsberger
- Miller+2025: Mass-dependent baryonic feedback (arXiv:2511.10634)

### Stage IV Surveys
- LSST Science Book
- Euclid Red Book
- Roman Space Telescope overview

---

## SUMMARY TIMELINE

| Phase | Dates | Deliverable |
|-------|-------|-------------|
| 0: Foundation | Dec 23-29 | Literature + workspace setup |
| 1: Core Data | Dec 30-Jan 12 | TNG halos + matching |
| 2: Replacement | Jan 13-19 | Working replacement code |
| 3: Mass Conservation | Jan 20-26 | Mass deficit analysis |
| 4: Power Spectrum | Jan 27-Feb 9 | All P(k) + suppression |
| 5: Profiles | Feb 10-16 | Profile analysis + BCM |
| 6: Convergence Maps | Feb 17-Mar 2 | 40 maps + peak counts |
| 7: Peak-Halo | Mar 3-9 | Peak-halo connection |
| **8: Paper 1 Writing** | **Mar 10-30** | **Complete Paper 1 draft** |
| **→ SUBMIT PAPER 1** | **Apr 1** | **MNRAS submission** |
| 9: Paper 2 Setup | Mar 31-Apr 13 | Multi-BCM profiles |
| **10: Paper 2 Writing** | **Apr 14-27** | **Complete Paper 2 draft** |
| **→ SUBMIT PAPER 2** | **May 1** | **MNRAS submission** |
| 11: CAMELS Data | Apr 28-May 18 | CAMELS mass conservation |
| 12: Parameter Analysis | May 19-Jun 1 | Parameter sensitivity |
| 13: GP Emulator | Jun 2-22 | Trained emulator |
| 14: Analytic Fitting | Jun 23-Jul 6 | Physics interpretation |
| **15: Paper 3 Writing** | **Jul 7-Aug 3** | **Complete Paper 3 draft** |
| **→ SUBMIT PAPER 3** | **Sep 1** | **MNRAS submission** |
| 16: Conferences/Jobs | May-Oct | Talks + job prep |

---

## NOTES

### When You Feel Scattered
- Return to this document
- Check off completed tasks (dopamine hit!)
- Identify next 3 tasks only
- Don't jump phases unless blocked
- Use weekly review to stay on track

### When You're Blocked
- Document the blocker in `Blockers.md`
- Email Shy immediately
- Work on parallel task from different phase (e.g., writing while waiting for data)
- Don't let one blocker stop all progress

### When You Have a New Idea
- Write 1 paragraph in `Future_Ideas.md`
- Set 2-month calendar reminder
- Return to current phase tasks
- Do NOT start new analysis until current paper is submitted

---

**Total Tasks: ~450 (atomic, checkable, organized by logic not paper)**

**Start Date: December 23, 2025**  
**Paper 1 Submission Target: April 1, 2026**  
**Paper 2 Submission Target: May 1, 2026**  
**Paper 3 Submission Target: September 1, 2026**

🚀 **Let's finish these papers!**
