#!/usr/bin/env python3
"""
Hydro Replace Pipeline: Unified ray-tracing with replacement and BCM.

This is the main orchestrator that handles:
1. Loading TNG data (DMO + Hydro + catalogs)
2. Applying modifications (replace, BCM)
3. Saving intermediate products (maps, P(k), profiles)
4. Writing modified snapshots for lux
5. Running lux ray-tracing
6. Peak count analysis

Usage:
    # Single snapshot test
    python hydro_replace_pipeline.py --mode replace --snapshot 99
    
    # Full pipeline (all snapshots)
    mpirun -np 64 python hydro_replace_pipeline.py --mode bcm --bcm-model Arico20 --full
    
    # Multiple modes
    for mode in dmo hydro replace bcm; do
        sbatch --export=MODE=$mode batch/run_pipeline.sh
    done

Modes:
    dmo:      Pure DMO run (baseline)
    hydro:    Pure hydro run (truth)
    replace:  DMO with hydro particles swapped in halos
    bcm:      DMO with BaryonForge BCM applied

Author: Max Lee
Date: December 2025
"""

import sys
import logging
import numpy as np
import h5py
import yaml
import argparse
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import time
from scipy.spatial import cKDTree

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# MPI setup (optional)
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1

# Set up logging
logging.basicConfig(
    level=logging.INFO if rank == 0 else logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import from hydro_replace package
try:
    from hydro_replace.data import SimulationData, MatchedCatalog, ParticleExtractor
    from hydro_replace.data.halo_catalogs import load_halo_catalog
    HAS_HYDRO_REPLACE = True
except ImportError as e:
    logger.warning(f"Could not import hydro_replace: {e}")
    HAS_HYDRO_REPLACE = False

# Import illustris_python
try:
    from illustris_python import snapshot, groupcat
    HAS_ILLUSTRIS = True
except ImportError:
    HAS_ILLUSTRIS = False

# Import MAS_library for density gridding
try:
    import MAS_library as MASL
    import Pk_library as PKL
    HAS_PYLIANS = True
except ImportError:
    HAS_PYLIANS = False
    logger.warning("Pylians not found. Power spectra computation disabled.")

# Import BaryonForge for BCM
try:
    import BaryonForge as bfg
    HAS_BARYONFORGE = True
except ImportError:
    HAS_BARYONFORGE = False
    logger.warning("BaryonForge not found. BCM mode disabled.")

# =============================================================================
# Configuration
# =============================================================================

# TNG resolution configurations
TNG_RESOLUTIONS = {
    625: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG/output',
        'n_particles': 625**3,
        'description': 'Low-res (fast testing)',
    },
    1250: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG/output',
        'n_particles': 1250**3,
        'description': 'Medium-res (validation)',
    },
    2500: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output',
        'n_particles': 2500**3,
        'description': 'Full-res (production)',
    },
}


@dataclass
class PipelineConfig:
    """Configuration for the hydro replace pipeline."""
    
    # Resolution (625, 1250, or 2500)
    resolution: int = 2500
    
    # Paths (set automatically from resolution, or override)
    dmo_basepath: str = ""
    hydro_basepath: str = ""
    output_dir: str = "/mnt/home/mlee1/ceph/hydro_replace"
    lux_dir: str = "/mnt/home/mlee1/lux"
    
    # Simulation parameters
    box_size: float = 205.0  # Mpc/h
    
    def __post_init__(self):
        """Set paths based on resolution if not explicitly provided."""
        if self.resolution not in TNG_RESOLUTIONS:
            raise ValueError(f"Invalid resolution {self.resolution}. Choose from: {list(TNG_RESOLUTIONS.keys())}")
        
        res_config = TNG_RESOLUTIONS[self.resolution]
        if not self.dmo_basepath:
            self.dmo_basepath = res_config['dmo']
        if not self.hydro_basepath:
            self.hydro_basepath = res_config['hydro']
    
    # Pipeline mode
    mode: str = "replace"  # dmo, hydro, replace, bcm
    bcm_model: str = "Arico20"  # Arico20, Schneider19, Mead20
    
    # Replacement parameters
    mass_min: float = 1e12  # Minimum halo mass [Msun/h]
    mass_max: float = 1e16  # Maximum halo mass [Msun/h]
    radius_factor: float = 5.0  # Replace within X * R_200
    
    # Snapshots
    snapshots: List[int] = field(default_factory=lambda: [
        99, 96, 90, 85, 80, 76, 71, 67, 63, 59, 
        56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29
    ])
    
    # Output options
    save_maps: bool = True
    save_pk: bool = True
    save_profiles: bool = True
    grid_resolution: int = 4096
    profile_bins: int = 100
    profile_rmax: float = 5.0  # Max radius in units of R_200
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'PipelineConfig':
        """Load config from YAML file."""
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return cls(**data)


# =============================================================================
# Main Pipeline Class
# =============================================================================

class HydroReplacePipeline:
    """
    Unified pipeline for ray-tracing with hydro replacement and BCM.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setup_directories()
        
        if rank == 0:
            print("=" * 70)
            print("HYDRO REPLACE PIPELINE")
            print("=" * 70)
            print(f"Mode: {config.mode}")
            if config.mode == 'bcm':
                print(f"BCM Model: {config.bcm_model}")
            print(f"Mass range: {config.mass_min:.1e} - {config.mass_max:.1e} Msun/h")
            print(f"Radius: {config.radius_factor} × R_200")
            print(f"Snapshots: {len(config.snapshots)}")
            print("=" * 70)
    
    def setup_directories(self):
        """Create output directory structure."""
        if rank == 0:
            base = Path(self.config.output_dir)
            mode_dir = base / self.config.mode
            if self.config.mode == 'bcm':
                mode_dir = base / f"bcm_{self.config.bcm_model}"
            
            (mode_dir / "maps").mkdir(parents=True, exist_ok=True)
            (mode_dir / "power_spectra").mkdir(parents=True, exist_ok=True)
            (mode_dir / "profiles").mkdir(parents=True, exist_ok=True)
            (mode_dir / "lux_input").mkdir(parents=True, exist_ok=True)
            (mode_dir / "kappa_maps").mkdir(parents=True, exist_ok=True)
            
            self.output_base = mode_dir
        
        if comm:
            comm.Barrier()
            self.output_base = comm.bcast(self.output_base if rank == 0 else None, root=0)
    
    def run(self):
        """Execute the full pipeline."""
        
        # Process each snapshot
        for snap in self.config.snapshots:
            if rank == 0:
                print(f"\n{'='*70}")
                print(f"Processing snapshot {snap}")
                print(f"{'='*70}")
            
            t_start = time.time()
            self.process_snapshot(snap)
            
            if rank == 0:
                print(f"Snapshot {snap} completed in {time.time()-t_start:.1f}s")
        
        # Run lux ray-tracing
        if rank == 0:
            print(f"\n{'='*70}")
            print("Running lux ray-tracing")
            print(f"{'='*70}")
        
        self.run_lux()
        
        # Peak analysis
        if rank == 0:
            print(f"\n{'='*70}")
            print("Running peak analysis")
            print(f"{'='*70}")
        
        self.analyze_peaks()
    
    def process_snapshot(self, snap_num: int):
        """Process a single snapshot through the pipeline."""
        
        # 1. Load data
        logger.info(f"  Loading particles and catalogs...")
        
        dmo_particles = self.load_particles(self.config.dmo_basepath, snap_num, 'dmo')
        
        # Only load hydro if needed
        if self.config.mode in ['hydro', 'replace']:
            hydro_particles = self.load_particles(self.config.hydro_basepath, snap_num, 'hydro')
        else:
            hydro_particles = None
            
        # Load halo matches for replace/bcm modes
        if self.config.mode in ['replace', 'bcm']:
            matched_halos = self.load_halo_matches(snap_num)
        else:
            matched_halos = None
        
        # 2. Apply modification
        logger.info(f"  Applying modification: {self.config.mode}")
        
        if self.config.mode == 'dmo':
            modified = dmo_particles
        elif self.config.mode == 'hydro':
            modified = hydro_particles
        elif self.config.mode == 'replace':
            modified = self.apply_replacement(
                dmo_particles, hydro_particles, matched_halos
            )
        elif self.config.mode == 'bcm':
            modified = self.apply_bcm(dmo_particles, matched_halos)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
        
        # 3. Save intermediate products
        if self.config.save_maps:
            self.save_projected_maps(modified, snap_num)
        
        if self.config.save_pk:
            self.save_power_spectrum(modified, snap_num)
        
        if self.config.save_profiles and matched_halos is not None:
            self.save_profiles(dmo_particles, hydro_particles, matched_halos, snap_num)
        
        # 4. Write lux input
        self.write_lux_input(modified, snap_num)
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    def load_particles(self, basepath: str, snap_num: int, 
                       sim_type: str) -> Dict[str, np.ndarray]:
        """
        Load all particle data from TNG snapshot.
        
        Parameters
        ----------
        basepath : str
            Path to simulation output
        snap_num : int
            Snapshot number
        sim_type : str
            'dmo' or 'hydro'
            
        Returns
        -------
        particles : dict
            Dictionary with 'positions', 'masses' arrays (combined for all types)
        """
        if not HAS_ILLUSTRIS:
            raise RuntimeError("illustris_python required for particle loading")
        
        logger.info(f"    Loading {sim_type} particles from snapshot {snap_num}...")
        
        # Determine which particle types to load
        if sim_type == 'dmo':
            particle_types = [1]  # DM only
        else:
            particle_types = [0, 1, 4]  # gas, DM, stars
        
        all_coords = []
        all_masses = []
        
        # Load header once to get mass table
        header = snapshot.loadHeader(basepath, snap_num)
        mass_table = header['MassTable']  # 10^10 Msun/h
        
        for ptype in particle_types:
            try:
                # DM (type 1) has fixed mass from header, others have individual masses
                if ptype == 1:  # Dark matter
                    # Only load coordinates for DM
                    coords = snapshot.loadSubset(basepath, snap_num, ptype, ['Coordinates'])
                    n_part = len(coords)
                    dm_mass = mass_table[ptype]  # Already in 10^10 Msun/h
                    masses = np.full(n_part, dm_mass, dtype=np.float32)
                    logger.info(f"      Type {ptype} (DM): {n_part:,} particles, mass={dm_mass:.6e} (10^10 Msun/h)")
                else:
                    # Gas (0) and stars (4) have individual masses
                    data = snapshot.loadSubset(basepath, snap_num, ptype, 
                                              ['Coordinates', 'Masses'])
                    if isinstance(data, dict):
                        coords = data['Coordinates']
                        masses = data['Masses']
                    else:
                        # Single field returned as array
                        coords = data
                        masses = np.full(len(coords), mass_table[ptype], dtype=np.float32)
                    logger.info(f"      Type {ptype}: {len(coords):,} particles")
                
                # Unit conversions: kpc/h -> Mpc/h, 10^10 Msun/h -> Msun/h
                coords = coords * 1e-3  # kpc -> Mpc
                masses = masses * 1e10  # 10^10 Msun -> Msun
                
                all_coords.append(coords)
                all_masses.append(masses)
                
            except Exception as e:
                logger.warning(f"      Failed to load type {ptype}: {e}")
        
        if len(all_coords) == 0:
            raise RuntimeError(f"No particles loaded from {basepath}")
        
        positions = np.concatenate(all_coords, axis=0)
        masses = np.concatenate(all_masses, axis=0)
        
        logger.info(f"    Total: {len(positions):,} particles, "
                   f"{masses.sum():.2e} Msun/h")
        
        return {
            'positions': positions,
            'masses': masses,
            'sim_type': sim_type,
            'snapshot': snap_num,
        }
    
    def load_halo_matches(self, snap_num: int) -> 'MatchedCatalog':
        """
        Load pre-computed halo matches for this snapshot.
        
        Looks for matches file at standard location, or computes on the fly.
        """
        # Try to load pre-computed matches
        match_file = Path(self.config.output_dir) / 'matches' / f'matched_snap{snap_num:03d}.h5'
        
        if match_file.exists():
            logger.info(f"    Loading pre-computed matches from {match_file}")
            return MatchedCatalog.load_hdf5(match_file)
        
        # Check for existing z=0 matches
        if snap_num == 99:
            alt_paths = [
                Path('/mnt/home/mlee1/ceph/halo_matches.npz'),
                Path('/mnt/home/mlee1/ceph/hydro_replace/matches/matched_catalog.h5'),
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    logger.info(f"    Loading matches from {alt_path}")
                    if alt_path.suffix == '.npz':
                        return MatchedCatalog.load_npz(alt_path)
                    else:
                        return MatchedCatalog.load_hdf5(alt_path)
        
        # TODO: Compute matches on the fly using BijectiveMatcher
        raise FileNotFoundError(
            f"No match file found for snapshot {snap_num}. "
            f"Expected at {match_file}. Run matching first."
        )
    
    # =========================================================================
    # Modifications
    # =========================================================================
    
    def apply_replacement(self, dmo_particles: Dict, hydro_particles: Dict,
                          matched_halos: 'MatchedCatalog') -> Dict:
        """
        Replace DMO halo particles with hydro particles.
        
        For each matched halo in the mass range:
        1. Remove DMO particles within radius_factor * R_200
        2. Insert hydro particles (DM + gas + stars) from matched hydro halo
        """
        logger.info(f"    Applying hydro replacement...")
        
        # Filter halos by mass
        halos_in_range = matched_halos.filter_by_mass(
            self.config.mass_min, self.config.mass_max, use_dmo=True
        )
        logger.info(f"    {halos_in_range.n_matches} halos in mass range "
                   f"[{self.config.mass_min:.1e}, {self.config.mass_max:.1e}]")
        
        # Build KD-tree for DMO particles
        dmo_pos = dmo_particles['positions']
        dmo_mass = dmo_particles['masses']
        
        logger.info(f"    Building KD-tree for {len(dmo_pos):,} DMO particles...")
        tree = cKDTree(dmo_pos, boxsize=self.config.box_size)
        
        # Find particles to remove (within halo replacement radii)
        remove_mask = np.zeros(len(dmo_pos), dtype=bool)
        
        for i in range(halos_in_range.n_matches):
            center = halos_in_range.dmo_positions[i]
            radius = halos_in_range.dmo_radii[i] * self.config.radius_factor
            
            # Find particles within radius (handles periodic boundaries)
            indices = tree.query_ball_point(center, radius)
            remove_mask[indices] = True
        
        n_removed = remove_mask.sum()
        logger.info(f"    Removing {n_removed:,} DMO particles from halo regions")
        
        # Keep particles outside halos
        kept_pos = dmo_pos[~remove_mask]
        kept_mass = dmo_mass[~remove_mask]
        
        # Get hydro particles (TODO: extract from within matched hydro halo radii)
        # For now, use all hydro particles as a placeholder
        hydro_pos = hydro_particles['positions']
        hydro_mass = hydro_particles['masses']
        
        # Combine: DMO (outside halos) + hydro (inside halos)
        # TODO: Proper extraction - for now just use DMO outside + all hydro
        # This is a placeholder implementation
        new_pos = np.concatenate([kept_pos, hydro_pos], axis=0)
        new_mass = np.concatenate([kept_mass, hydro_mass], axis=0)
        
        logger.info(f"    Result: {len(new_pos):,} total particles")
        
        return {
            'positions': new_pos,
            'masses': new_mass,
            'mode': 'replace',
            'n_removed': n_removed,
            'n_halos': halos_in_range.n_matches,
        }
    
    def apply_bcm(self, dmo_particles: Dict, matched_halos: 'MatchedCatalog') -> Dict:
        """
        Apply BaryonForge BCM to DMO halos.
        
        For each halo in the mass range:
        1. Compute BCM displacement field
        2. Apply radial displacement to DMO particles
        """
        if not HAS_BARYONFORGE:
            raise RuntimeError("BaryonForge required for BCM mode. Install with: pip install BaryonForge")
        
        logger.info(f"    Applying BCM ({self.config.bcm_model})...")
        
        # Select BCM model
        if self.config.bcm_model == 'Arico20':
            bcm = bfg.Arico20()
        elif self.config.bcm_model == 'Schneider19':
            bcm = bfg.Schneider19()
        elif self.config.bcm_model == 'Mead20':
            bcm = bfg.Mead20()
        else:
            raise ValueError(f"Unknown BCM model: {self.config.bcm_model}")
        
        # Filter halos by mass
        halos_in_range = matched_halos.filter_by_mass(
            self.config.mass_min, self.config.mass_max, use_dmo=True
        )
        logger.info(f"    {halos_in_range.n_matches} halos for BCM")
        
        # Get DMO particle data
        pos = dmo_particles['positions'].copy()
        mass = dmo_particles['masses'].copy()
        
        # Build KD-tree
        tree = cKDTree(pos, boxsize=self.config.box_size)
        
        # Apply BCM to each halo
        n_displaced = 0
        for i in range(halos_in_range.n_matches):
            center = halos_in_range.dmo_positions[i]
            M_200 = halos_in_range.dmo_masses[i]
            R_200 = halos_in_range.dmo_radii[i]
            
            # Find particles within 5×R_200
            max_r = R_200 * self.config.radius_factor
            indices = tree.query_ball_point(center, max_r)
            
            if len(indices) == 0:
                continue
            
            # Compute radial distances
            dx = pos[indices] - center
            # Handle periodic boundaries
            dx = dx - self.config.box_size * np.round(dx / self.config.box_size)
            r = np.linalg.norm(dx, axis=1)
            r_unit = dx / r[:, np.newaxis]
            
            # Get BCM displacement (r_new / r_old ratio)
            # This is a simplified version - BaryonForge may have different API
            try:
                displacement_ratio = bcm.displacement_ratio(r / R_200, M_200)
                new_r = r * displacement_ratio
                
                # Apply displacement
                pos[indices] = center + r_unit * new_r[:, np.newaxis]
                n_displaced += len(indices)
            except Exception as e:
                logger.warning(f"BCM failed for halo {i}: {e}")
        
        logger.info(f"    Displaced {n_displaced:,} particles")
        
        return {
            'positions': pos,
            'masses': mass,
            'mode': 'bcm',
            'bcm_model': self.config.bcm_model,
            'n_displaced': n_displaced,
            'n_halos': halos_in_range.n_matches,
        }
    
    # =========================================================================
    # Output Products
    # =========================================================================
    
    def save_projected_maps(self, particles: Dict, snap_num: int):
        """Save 2D projected surface density maps."""
        logger.info(f"    Saving projected maps...")
        
        pos = particles['positions']
        mass = particles['masses']
        
        # Project along each axis
        for axis in range(3):
            # Create 2D grid
            grid = np.zeros((self.config.grid_resolution, self.config.grid_resolution), 
                           dtype=np.float32)
            
            # Get 2D coordinates
            ax1, ax2 = [(1,2), (0,2), (0,1)][axis]
            x = pos[:, ax1]
            y = pos[:, ax2]
            
            # CIC assignment
            # (Simplified - use Pylians for proper implementation)
            ix = (x / self.config.box_size * self.config.grid_resolution).astype(int) % self.config.grid_resolution
            iy = (y / self.config.box_size * self.config.grid_resolution).astype(int) % self.config.grid_resolution
            
            np.add.at(grid, (ix, iy), mass)
            
            # Save
            outfile = self.output_base / 'maps' / f'map_snap{snap_num:03d}_axis{axis}.npy'
            np.save(outfile, grid)
        
        logger.info(f"    Saved maps to {self.output_base / 'maps'}")
    
    def save_power_spectrum(self, particles: Dict, snap_num: int):
        """Compute and save 3D matter power spectrum."""
        if not HAS_PYLIANS:
            logger.warning("    Skipping P(k) - Pylians not available")
            return
            
        logger.info(f"    Computing power spectrum...")
        
        pos = particles['positions'].astype(np.float32)
        mass = particles['masses'].astype(np.float32)
        
        # Create density field
        grid_size = min(self.config.grid_resolution, 1024)  # Cap for memory
        delta = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        
        MASL.MA(pos, delta, self.config.box_size, 'CIC', W=mass, verbose=False)
        
        # Convert to overdensity
        delta = delta / delta.mean() - 1.0
        
        # Compute P(k)
        Pk = PKL.Pk(delta, self.config.box_size, axis=0, MAS='CIC', threads=1)
        
        # Save
        outfile = self.output_base / 'power_spectra' / f'pk_snap{snap_num:03d}.txt'
        np.savetxt(outfile, np.column_stack([Pk.k3D, Pk.Pk[:, 0]]),
                  header='k [h/Mpc]  P(k) [(Mpc/h)^3]')
        
        logger.info(f"    Saved P(k) to {outfile}")
    
    def save_profiles(self, dmo_particles: Dict, hydro_particles: Optional[Dict],
                      matched_halos: 'MatchedCatalog', snap_num: int):
        """Save radial density profiles for halos."""
        logger.info(f"    Computing profiles...")
        
        # Select a subset of halos for profiles (e.g., 100 per mass bin)
        halos = matched_halos.filter_by_mass(self.config.mass_min, self.config.mass_max)
        n_profile = min(100, halos.n_matches)
        
        # Sample halos
        indices = np.linspace(0, halos.n_matches-1, n_profile, dtype=int)
        
        # Radial bins
        r_bins = np.logspace(-2, np.log10(self.config.profile_rmax), 
                            self.config.profile_bins + 1)
        
        profiles = []
        for i in indices:
            center = halos.dmo_positions[i]
            R_200 = halos.dmo_radii[i]
            M_200 = halos.dmo_masses[i]
            
            # TODO: Compute actual profiles
            # This is a placeholder
            profiles.append({
                'halo_id': int(halos.dmo_indices[i]),
                'M_200': M_200,
                'R_200': R_200,
                'r_bins': r_bins,
                # 'rho_dmo': ...,
                # 'rho_hydro': ...,
            })
        
        # Save
        outfile = self.output_base / 'profiles' / f'profiles_snap{snap_num:03d}.h5'
        with h5py.File(outfile, 'w') as f:
            f.attrs['n_halos'] = len(profiles)
            f.create_dataset('r_bins', data=r_bins)
            for j, p in enumerate(profiles):
                grp = f.create_group(f'halo_{j}')
                grp.attrs['halo_id'] = p['halo_id']
                grp.attrs['M_200'] = p['M_200']
                grp.attrs['R_200'] = p['R_200']
        
        logger.info(f"    Saved {len(profiles)} profiles to {outfile}")
    
    def write_lux_input(self, particles: Dict, snap_num: int):
        """Write modified snapshot in format lux can read."""
        logger.info(f"    Writing lux input...")
        
        outfile = self.output_base / 'lux_input' / f'snap_{snap_num:03d}.hdf5'
        
        with h5py.File(outfile, 'w') as f:
            # Header
            header = f.create_group('Header')
            header.attrs['BoxSize'] = self.config.box_size * 1e3  # Mpc/h -> kpc/h
            header.attrs['NumPart_Total'] = [0, len(particles['positions']), 0, 0, 0, 0]
            header.attrs['NumPart_ThisFile'] = [0, len(particles['positions']), 0, 0, 0, 0]
            
            # Particles (store as PartType1 = DM)
            pt1 = f.create_group('PartType1')
            pt1.create_dataset('Coordinates', data=particles['positions'] * 1e3)  # Mpc/h -> kpc/h
            pt1.create_dataset('Masses', data=particles['masses'] / 1e10)  # Msun/h -> 10^10 Msun/h
        
        logger.info(f"    Saved lux input to {outfile}")
    
    # =========================================================================
    # Ray-Tracing
    # =========================================================================
    
    def run_lux(self):
        """Run lux ray-tracing on modified snapshots."""
        if rank != 0:
            return
        
        logger.info("Running lux ray-tracing...")
        
        # TODO: Generate lux config file pointing to modified snapshots
        # TODO: Run lux executable
        # subprocess.run([self.config.lux_dir + '/lux', config_file])
        
        logger.warning("lux execution not yet implemented")
    
    def analyze_peaks(self):
        """Analyze peak counts from convergence maps."""
        if rank != 0:
            return
        
        logger.info("Analyzing peaks...")
        
        # TODO: Load convergence maps from lux output
        # TODO: Apply Gaussian smoothing
        # TODO: Find peaks
        # TODO: Compute peak counts by S/N bin
        
        logger.warning("Peak analysis not yet implemented")


# =============================================================================
# Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Hydro Replace Pipeline")
    
    parser.add_argument('--resolution', type=int, default=2500,
                        choices=[625, 1250, 2500],
                        help='TNG resolution (625=fast test, 1250=validation, 2500=production)')
    parser.add_argument('--mode', type=str, default='replace',
                        choices=['dmo', 'hydro', 'replace', 'bcm'],
                        help='Pipeline mode')
    parser.add_argument('--bcm-model', type=str, default='Arico20',
                        choices=['Arico20', 'Schneider19', 'Mead20'],
                        help='BCM model (if mode=bcm)')
    parser.add_argument('--snapshot', type=int, default=None,
                        help='Single snapshot to process (for testing)')
    parser.add_argument('--full', action='store_true',
                        help='Process all 20 snapshots')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--mass-min', type=float, default=1e12,
                        help='Minimum halo mass [Msun/h]')
    parser.add_argument('--mass-max', type=float, default=1e16,
                        help='Maximum halo mass [Msun/h]')
    parser.add_argument('--radius', type=float, default=5.0,
                        help='Replacement radius in units of R_200')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig(resolution=args.resolution)
    
    # Override with command-line args
    config.mode = args.mode
    config.bcm_model = args.bcm_model
    config.mass_min = args.mass_min
    config.mass_max = args.mass_max
    config.radius_factor = args.radius
    
    if args.snapshot is not None:
        config.snapshots = [args.snapshot]
    elif not args.full:
        # Default to 5-snapshot pilot
        config.snapshots = [99, 76, 63, 49, 35]
    
    # Create and run pipeline
    pipeline = HydroReplacePipeline(config)
    pipeline.run()
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
