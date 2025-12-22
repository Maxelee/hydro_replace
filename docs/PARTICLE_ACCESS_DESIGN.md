# Particle Access Library Design

## Overview

You've built an excellent particle caching system. Now we need a clean API layer that makes it trivial to:
1. Load matched halos for a snapshot
2. Access particles for each halo efficiently
3. Perform analyses (profiles, mass conservation, baryon fractions) across halo mass and radius

## Design Pattern: `MatchedHaloSnapshot` Class

A single unified interface for all particle access operations:

```python
from particle_access import MatchedHaloSnapshot

# Load and work with matched halos at a snapshot
mh = MatchedHaloSnapshot(
    snapshot=99, 
    sim_res=2500,
    cache_dir='/mnt/home/mlee1/ceph/hydro_replace_fields'
)

# Basic access
matches = mh.get_matches()  # dict of dmo_idx -> hydro_idx
halo = mh.get_halo(dmo_idx=100)  # HaloInfo object

# Particle access (the core feature)
dmo_particles = mh.get_particles('dmo', dmo_idx=100, radius_mult=5.0)
hydro_particles = mh.get_particles('hydro', dmo_idx=100, radius_mult=3.0)

# Bulk operations
for halo_idx, halo_info in mh.iter_matched_halos(mass_range=[12.5, 13.5]):
    dmo_pids = mh.get_particles('dmo', halo_idx, radius_mult=5.0)
    hydro_pids = mh.get_particles('hydro', halo_idx, radius_mult=5.0)
    
    # Analyze halo...
    result = analyze_baryon_fraction(dmo_pids, hydro_pids, halo_info)
```

---

## Proposed Components

### 1. **Core Data Structure: `HaloInfo`**

```python
class HaloInfo:
    """Lightweight halo metadata."""
    dmo_idx: int
    hydro_idx: int
    mass_msun_h: float  # log10 or linear?
    r200_mpc_h: float
    position_mpc_h: np.ndarray  # [x, y, z]
    
    def match_id(self) -> str:
        """Unique identifier for this matched pair."""
        return f"dmo{self.dmo_idx}_hydro{self.hydro_idx}"
```

**Question**: Should mass be stored as log10 or linear? Linear is more useful for cuts, but log10 is what you're likely filtering on.

---

### 2. **Particle Access Layer: `ParticleLoader`**

Handle the actual HDF5 I/O and caching:

```python
class ParticleLoader:
    """Manages particle ID cache file I/O."""
    
    def __init__(self, cache_file: str):
        """Open cache file (or lazy-load on first access)."""
        
    def get_particle_ids(self, mode: str, halo_idx: int) -> np.ndarray:
        """
        Fetch particle IDs for a halo.
        
        Args:
            mode: 'dmo' or 'hydro'
            halo_idx: halo index from matching
        
        Returns:
            Sorted array of particle IDs
        """
        
    def get_multiple_halos(self, mode: str, halo_indices: list) -> dict:
        """Batch fetch multiple halos efficiently."""
```

**Benefits**:
- Lazy loading: Don't read until needed
- Caching: Keep frequently accessed halos in memory
- Batch operations: Minimize file I/O for loops

---

### 3. **Multi-Radius Interface**

Since you want to analyze at different radius multiples (out to 5R200):

```python
class RadialShell:
    """Particles in a radial range."""
    
    def __init__(self, inner_r200_mult, outer_r200_mult, particle_ids, 
                 particle_coords, particle_types):
        self.particles_ids = particle_ids
        self.inner_mult = inner_r200_mult
        self.outer_mult = outer_r200_mult
        
    @property
    def n_particles(self):
        return len(self.particle_ids)
    
    @property
    def total_mass(self, mass_per_particle):
        """Compute integrated mass."""
        return self.n_particles * mass_per_particle
    
    def by_type(self, ptype):
        """Filter particles by type (gas, DM, stars)."""
```

**Challenge**: Your cache stores particle IDs, but to compute masses and baryon fractions you need:
- Particle masses (from snapshots)
- Particle types (gas, DM, stars)
- Particle coordinates (for 3D analysis)

**Options**:

#### Option A: Minimal Cache + On-Demand Loading
- Cache only stores **particle IDs**
- On query, load full particle data from snapshots
- Pro: Small cache files, simple structure
- Con: Slow for repeated access to same halo

#### Option B: Extended Cache
- Cache stores particle IDs + **minimal properties** (mass, type)
- Still smaller than full snapshot but enables fast analysis
- Pro: Fast queries, reasonable cache size
- Con: More complex cache generation

#### Option C: Hybrid Approach (Recommended)
- Cache always has: **particle IDs**
- Optional extended cache: **particle masses + types + 2-3 coordinates**
- At runtime, choose which to load based on analysis needs

```python
# Minimal (fast)
particles = mh.get_particles('dmo', halo_idx)  # Just IDs

# Extended (slower, more flexible)
particles = mh.get_particles('dmo', halo_idx, include_masses=True, 
                             include_coords=True)
```

---

### 4. **Analysis Functions**

Build analysis utilities on top of particle access:

```python
# profiles.py - Reuse your existing logic
def compute_radial_profile(halo_info, dmo_particles, hydro_particles):
    """Binned density profile."""
    
def compute_baryon_fraction(halo_info, dmo_particles, hydro_particles):
    """f_b = M_hydro_gas / M_dmo."""
    
def compute_mass_conservation(dmo_pid_set, hydro_pids):
    """What fraction of DMO mass is traced in Hydro?"""

# Operators: can chain these
class HaloAnalyzer:
    def __init__(self, mh: MatchedHaloSnapshot):
        self.mh = mh
    
    def compute_all(self, dmo_idx, radii=[1.0, 2.0, 5.0]):
        """Compute profiles, f_b, conservation for multiple radii."""
        results = {}
        for r in radii:
            dmo_pids = self.mh.get_particles('dmo', dmo_idx, radius_mult=r)
            hydro_pids = self.mh.get_particles('hydro', dmo_idx, radius_mult=r)
            
            results[r] = {
                'profile': compute_radial_profile(...),
                'f_b': compute_baryon_fraction(...),
                'conservation': compute_mass_conservation(...),
            }
        return results
```

---

## File Structure

```
scripts/
├── particle_access.py          # Core library
│   ├── MatchedHaloSnapshot
│   ├── ParticleLoader
│   ├── HaloInfo
│   └── RadialShell
│
├── particle_analysis.py        # Analysis functions
│   ├── compute_radial_profile()
│   ├── compute_baryon_fraction()
│   ├── compute_mass_conservation()
│   └── HaloAnalyzer
│
├── test_particle_access.py     # Unit tests
│
└── examples/
    ├── baryon_fractions.py     # Example: mass conservation
    ├── profiles_batch.py       # Example: compute profiles for all halos
    └── systematic_analysis.py  # Example: vary radius, halo mass
```

---

## Usage Examples

### Example 1: Quick Baryon Fraction Check

```python
from particle_access import MatchedHaloSnapshot
from particle_analysis import compute_baryon_fraction

mh = MatchedHaloSnapshot(snapshot=99, sim_res=2500)

for dmo_idx, hydro_idx in mh.get_matches().items():
    halo = mh.get_halo(dmo_idx)
    
    dmo_ids = mh.get_particles('dmo', dmo_idx)
    hydro_ids = mh.get_particles('hydro', dmo_idx)
    
    f_b = compute_baryon_fraction(dmo_ids, hydro_ids)
    print(f"Halo {dmo_idx} (M={halo.mass_msun_h:.1e}): f_b = {f_b:.3f}")
```

### Example 2: Halo Mass Dependence

```python
mh = MatchedHaloSnapshot(snapshot=99, sim_res=2500)
analyzer = HaloAnalyzer(mh)

# Restrict to massive halos
halos = mh.iter_matched_halos(mass_range=[13.0, 15.0])

results_by_mass = {}
for halo in halos:
    results = analyzer.compute_all(halo.dmo_idx, radii=[1, 2, 3, 5])
    results_by_mass[halo.dmo_idx] = results
```

### Example 3: Radius Dependence

```python
mh = MatchedHaloSnapshot(snapshot=99, sim_res=2500)
analyzer = HaloAnalyzer(mh)

# Pick one halo
halo_idx = 1000
results = analyzer.compute_all(halo_idx, radii=np.linspace(0.5, 5.0, 20))

# Plot as function of radius
import matplotlib.pyplot as plt
radii = list(results.keys())
f_b_values = [results[r]['f_b'] for r in radii]
plt.plot(radii, f_b_values, 'o-')
plt.xlabel('Radius / R200')
plt.ylabel('Baryon Fraction')
plt.show()
```

---

## API Design Decisions

### **Decision 1: Particle IDs vs Full Data**

**Current**: Cache stores only particle IDs
- ✅ Small cache files
- ✅ Simple generation
- ❌ Requires snapshot access for mass/type queries
- ❌ Slower for repeated analysis

**Recommendation**: Start minimal, extend cache if bottleneck identified

---

### **Decision 2: Matching Dictionary Structure**

Currently you have match files with arrays. Should we wrap in a structured interface?

```python
# Current
matches = np.load('matches.npz')
dmo_idx = matches['dmo_indices'][i]

# Proposed
mh = MatchedHaloSnapshot(...)
matches = mh.get_matches()  # OrderedDict or DataFrame
matches[dmo_idx]  # -> hydro_idx

# Or iterable interface
for halo in mh.iter_matched_halos():
    print(halo.dmo_idx, halo.hydro_idx, halo.mass_msun_h)
```

DataFrame approach is clean but adds pandas dependency.

---

### **Decision 3: Radial Multiplier Semantics**

How should sub-R200 queries work?

**Option A**: Store full 5R200, filter at query time
```python
mh.get_particles('dmo', idx, radius_mult=2.0)  # Subset of cached
```

**Option B**: Pre-compute multiple radii during cache generation
```python
# Cache stores separate datasets for each radius: halo_1R200, halo_2R200, halo_5R200
# Faster but larger files
```

**Recommendation**: Option A is simpler and matches your current cache structure

---

## Implementation Priority

### **Phase 1: Minimal Viable Access Layer** (1-2 hours)
- `MatchedHaloSnapshot` class
- `ParticleLoader` wrapper around HDF5
- Basic iteration and filtering
- **Goal**: Make particle loading easy

### **Phase 2: Analysis Utilities** (2-3 hours)
- `HaloInfo` structure
- `compute_baryon_fraction()`, `compute_mass_conservation()`
- `HaloAnalyzer` for batch operations
- **Goal**: Enable science-ready analysis

### **Phase 3: Extended Cache** (if needed, 4+ hours)
- Include masses + types in cache
- Reduce snapshot I/O bottleneck
- May require regenerating all cache files
- **Goal**: Fast analysis for large samples

---

## Questions for You

1. **Particle data needs**: For baryon fractions, do you need:
   - Just counts (N_hydro / N_dmo)?
   - Actual masses (requires snapshot access)?
   - Mass-weighted by type (gas, stars)?

2. **Cache scope**: Should we expand the current cache (ID-only) or keep minimal and load on-demand?

3. **Analysis scope**: What metrics beyond baryon fraction matter?
   - Density profiles (radial binning)?
   - Phase space (position + velocity)?
   - Substructure (separate halos within R200)?

4. **Performance targets**: What's acceptable?
   - Load single halo in <1s?
   - Analyze 100 halos in <1min?
   - Full sample (1000+ halos) overnight?

---

## Suggested Next Steps

1. **Create `particle_access.py`** with core `MatchedHaloSnapshot` class
2. **Write test script** on snapshot 99 to validate design
3. **Implement quick analysis** (e.g., baryon fractions for all halos)
4. **Iterate based on performance** - then decide on cache expansion

Would you like me to start implementing Phase 1?
