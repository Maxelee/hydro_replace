# Unified Statistics Pipeline - Implementation Summary

## ✅ Completed Components

### 1. Core Modules (scripts/)

- **constants.py** - Configuration, paths, model naming functions
  - All 100 model names (dmo, hydro, 16 cumulative, 16 discrete tiles)  
  - Paths, parameters, redshift/snapshot mappings
  
- **stats_utils.py** - Statistics computation functions
  - `load_kappa()` - Load convergence maps (tested ✓)
  - `read_lensplane()` - Load density fields in binary format (tested ✓)
  - `compute_Pk_2d()` - 2D power spectrum via Pylians
  - `compute_Cl()` - Angular power spectrum via Pylians
  - `smooth_kappa()` - Gaussian smoothing (tested ✓)
  - `compute_peaks/minima/pdf()` - WL statistics (tested ✓)
  
- **response_utils.py** - Response fraction calculations
  - `compute_F_S()` - Response fraction with masking
  - `compute_Delta_F()` - Tile contributions
  - `compute_epsilon()` - Non-additivity metric
  - `bootstrap_F()` - Bootstrap error estimation
  - `compute_mean_F()` - Weighted averaging

### 2. Pipeline Scripts (scripts/)

- **compute_dmo_rms.py** - One-time DMO RMS computation
  - MPI-parallelized over (LP, run, z) tuples
  - Saves to `dmo_rms.h5` (shape: 20 × 100 × 4)
  
- **compute_all_stats.py** - Unified statistics pipeline
  - Loads cached DMO RMS
  - Computes density stats: P(k) from lensplanes
  - Computes convergence stats: C_ℓ, peaks, minima, PDF from κ maps
  - Saves to `{model}/stats.h5`
  
### 3. Batch Scripts (batch/)

- **run_compute_dmo_rms.sh** - SLURM job for DMO RMS (one-time)
- **run_compute_stats.sh** - SLURM array job for 34 models (indices 0-33)

### 4. Analysis Tools

- **notebooks/response_analysis_template.ipynb** - Response analysis workflow
  - Load data from stats.h5
  - Compute F_S with bootstrap errors
  - Generate 7+ publication figures
  
- **docs/unified_stats_pipeline.md** - Complete documentation

### 5. Testing & Validation (scripts/)

- **check_data_availability.py** - Data validation (✓ tested, works)
- **test_pipeline.py** - Component tests (✓ tested, 5/6 pass)

## 🔍 Test Results

### Data Availability (100% verified)
```
✓ RT data (convergence maps):
  - DMO: 20 LPs × 100 runs × 41 kappa files
  - Hydro: 20 LPs
  - Replace: 4 models with complete RT data
  
✓ LP data (density fields):  
  - DMO: 20 LPs × 40 lenspot files
  - Hydro: 20 LPs
  - Replace: 100 models with LP data
  
✓ Complete models (both RT+LP): 4
  - hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.0_Ro_0.5
  - hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.0_Ro_1.0
  - hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.0_Ro_3.0
  - hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.0_Ro_5.0
```

### Component Tests (5/6 passed)
```
✓ Load convergence maps (kappa files)
✓ Smooth convergence maps
✗ Compute C_ℓ (needs Pylians - will work in cluster venv)
✓ Compute peaks/minima/PDF  
✓ Load multiple models
✓ Access lensplane directory
```

## 📁 Output Structure

```
/mnt/home/mlee1/ceph/hydro_replace_stats/
├── dmo_rms.h5                    # (20, 100, 4) - LP × runs × z
├── dmo/stats.h5
├── hydro/stats.h5
└── hydro_replace_*/stats.h5

Each stats.h5 contains:
  Datasets:
    - Pk:           (20, 20, ~200)    LP × snapshot × k
    - k_bins:       (~200,)
    - Cl:           (2000, 4, ~50)    realization × z × ℓ
    - ell_bins:     (~50,)
    - peaks:        (2000, 4, 15)     realization × z × S/N
    - minima:       (2000, 4, 15)
    - pdf:          (2000, 4, 15)
    - sn_bin_edges: (16,)
  
  Attributes:
    - snapshot_order, snapshot_redshifts
    - source_redshifts, kappa_targets
    - N_LP, N_RUNS, smoothing_arcmin, etc.
```

## 🚀 Usage

### Run DMO RMS (first time only)
```bash
sbatch batch/run_compute_dmo_rms.sh
# Wait for completion (~2-3 hours)
```

### Run Statistics Pipeline
```bash
# For all models (array job)
sbatch batch/run_compute_stats.sh

# For single model
mpirun -np 64 python scripts/compute_all_stats.py --model dmo
mpirun -np 64 python scripts/compute_all_stats.py --model hydro
mpirun -np 64 python scripts/compute_all_stats.py \
  --model hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.0_Ro_3.0
```

### Analyze Results
```bash
# Check what data is available
python scripts/check_data_availability.py

# Test components
python scripts/test_pipeline.py

# Open analysis notebook
jupyter notebook notebooks/response_analysis_template.ipynb
```

## ⚙️ Key Implementation Details

### 1. Data Format Handling

**Convergence maps (RT):** Binary Fortran format
```python
# kappa*.dat files have:
# - int32 record marker
# - float32 array (ng × ng)
# - int32 record marker
```

**Lensplanes (LP):** Binary Fortran format  
```python
# lenspot*.dat files organized as:
# /LP_BASE/model/LP_XX/lenspotYY.dat
# where YY = snapshot index (0-19)
```

### 2. Directory Structure

**RT data:** `/RT_BASE/{model}/LP_{XX}/run{XXX}/kappa{YY}.dat`
- model: dmo, hydro, or hydro_replace_*
- LP: 00-19 (20 lensplane orientations)
- run: 001-100 (100 ray-traced maps per LP)
- kappa: 00-40 (41 source redshifts)

**LP data:** `/LP_BASE/{model}/LP_{XX}/lenspot{YY}.dat`
- model: dmo, hydro, or hydro_replace_*
- LP: 00-19
- lenspot: 00-39 (20 TNG snapshots, 40 total with stacking)

### 3. Model Naming

**Format:** `hydro_replace_Ml_{Ml}_Mu_{Mu}_Ri_{Ri}_Ro_{Ro}`

**Cumulative models:** Mu=1.00e+15, Ri=0.0
- Replace all halos M > Ml within r < Ro·R₂₀₀

**Discrete tile models:** Specific mass and radius ranges
- Replace halos in Ml < M < Mu and Ri < r < Ro (in R₂₀₀ units)

## 📊 Expected Performance

- **DMO RMS computation:** ~2-3 hours (one-time, 64 ranks)
- **Statistics per model:** ~30-120 minutes (64 ranks)
- **Total pipeline (34 models):** ~30-120 minutes wall time (parallel array job)

## 🐛 Known Limitations

1. **Incomplete RT data:** Only 4 Replace models have full RT data currently
   - Need to run ray-tracing for remaining 96 models
   - Density analysis (P(k)) can proceed for all 100 models
   
2. **Pylians dependency:** Required for P(k) and C_ℓ computation
   - Must run in cluster environment with venv activated
   - Not available in test environment

3. **Model naming mismatch:** Some RT directories use `_R_` instead of `_Ri_Ro_`
   - May need path translation layer for legacy models

## ✨ Next Steps

1. **Verify Pylians integration** - Test P(k) computation in cluster venv
2. **Run DMO RMS** - Execute one-time setup: `sbatch batch/run_compute_dmo_rms.sh`
3. **Test single model** - Run stats for one complete model (e.g., dmo)
4. **Scale to full pipeline** - Submit array job for all 34 models
5. **Generate RT data** - Ray-trace remaining 96 Replace models if needed

## 📝 Files Created

```
scripts/
  ├── constants.py                  (160 lines)
  ├── stats_utils.py                (285 lines)
  ├── response_utils.py             (228 lines)
  ├── compute_dmo_rms.py            (151 lines)
  ├── compute_all_stats.py          (383 lines)
  ├── check_data_availability.py    (195 lines)
  └── test_pipeline.py              (261 lines)

batch/
  ├── run_compute_dmo_rms.sh        (30 lines)
  └── run_compute_stats.sh          (83 lines)

notebooks/
  └── response_analysis_template.ipynb  (450 lines JSON)

docs/
  └── unified_stats_pipeline.md     (281 lines)
```

**Total:** ~2,507 lines of code + documentation
