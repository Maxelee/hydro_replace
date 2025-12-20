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
    import pyccl as ccl
    HAS_BARYONFORGE = True
except ImportError:
    HAS_BARYONFORGE = False
    bfg = None
    ccl = None
    logger.warning("BaryonForge or pyccl not found. BCM mode disabled.")

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
    bcm_model: str = "Arico20"  # Arico20, Schneider19, Schneider25
    
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
# Helper Functions
# =============================================================================

def load_snapshot_header(basepath: str, snap_num: int) -> dict:
    """
    Load header attributes from TNG snapshot using h5py directly.
    
    illustris_python doesn't have loadHeader, so we read directly.
    """
    snap_path = snapshot.snapPath(basepath, snap_num)
    header = {}
    with h5py.File(snap_path, 'r') as f:
        for key in f['Header'].attrs.keys():
            header[key] = f['Header'].attrs[key]
    return header


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
            # For bcm/replace modes, compute profiles from modified particles
            self.save_halo_profiles(modified, matched_halos, snap_num)
        
        # 4. Write lux input (modified snapshot in TNG-compatible format)
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
        
        # Load header from HDF5 file directly
        snap_path = snapshot.snapPath(basepath, snap_num)
        with h5py.File(snap_path, 'r') as f:
            mass_table = f['Header'].attrs['MassTable']  # 10^10 Msun/h
        
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
        Apply BaryonForge BCM to DMO particles.
        
        Uses the BaryonForge library to compute displacement functions and
        apply them to particles around halos. This is the proper way to use
        BCM - it computes M_enc(r) for DMO and DMB profiles, then finds the
        displacement needed to transform one to the other.
        
        BaryonForge workflow:
        1. Define DMO and DMB (dark matter + baryons) profiles
        2. Create Baryonification3D model from these profiles
        3. Setup interpolator for fast displacement lookup
        4. Create ParticleSnapshot and HaloNDCatalog objects
        5. Run BaryonifySnapshot to apply displacements
        """
        if not HAS_BARYONFORGE:
            raise RuntimeError(
                "BaryonForge required for BCM mode. Install with: pip install BaryonForge pyccl"
            )
        
        logger.info(f"    Applying BCM ({self.config.bcm_model}) using BaryonForge...")
        
        # Get snapshot redshift for scale factor
        snap_num = dmo_particles.get('snapshot', 99)
        header = load_snapshot_header(self.config.dmo_basepath, snap_num)
        redshift = header['Redshift']
        a = 1.0 / (1.0 + redshift)
        
        logger.info(f"    Snapshot {snap_num}: z = {redshift:.3f}, a = {a:.4f}")
        
        # Setup CCL cosmology (required by BaryonForge)
        # TNG-300 cosmology
        cosmo = ccl.Cosmology(
            Omega_c=0.3089 - 0.0486,  # Omega_m - Omega_b
            Omega_b=0.0486,
            h=0.6774,
            sigma8=0.8159,
            n_s=0.9667,
            matter_power_spectrum='linear'
        )
        cosmo.compute_sigma()
        
        # BCM model parameters (can be customized)
        # These are reasonable defaults from BaryonForge
        if self.config.bcm_model == 'Schneider19':
            bpar = dict(
                theta_ej=4, theta_co=0.1, M_c=1e14/0.6774, mu_beta=0.4,
                eta=0.3, eta_delta=0.3, tau=-1.5, tau_delta=0,
                epsilon_max=self.config.radius_factor,
            )
            DMO = bfg.Profiles.Schneider19.DarkMatterOnly(**bpar)
            DMB = bfg.Profiles.Schneider19.DarkMatterBaryon(**bpar)
            
        elif self.config.bcm_model == 'Schneider25':
            bpar = dict(
                theta_ej=4, theta_co=0.1, M_c=1e14/0.6774, mu_beta=0.6,
                epsilon0=0.25, epsilon1=0,
                epsilon_max=self.config.radius_factor,
            )
            DMO = bfg.Profiles.Schneider25.DarkMatterOnly(**bpar)
            DMB = bfg.Profiles.Schneider25.DarkMatterBaryon(**bpar)
            
        elif self.config.bcm_model == 'Arico20':
            bpar = dict(
                M_c=1e14/0.6774, mu=0.4, theta_inn=0.4, theta_out=0.25,
                M_inn=1e13/0.6774, epsilon_max=self.config.radius_factor,
            )
            DMO = bfg.Profiles.Arico20.DarkMatterOnly(**bpar)
            DMB = bfg.Profiles.Arico20.DarkMatterBaryon(**bpar)
            
        else:
            raise ValueError(f"Unknown BCM model: {self.config.bcm_model}. "
                           f"Choose from: Schneider19, Schneider25, Arico20")
        
        # Create 3D Baryonification model
        logger.info(f"    Setting up Baryonification3D interpolator...")
        Baryons = bfg.Profiles.Baryonification3D(DMO, DMB, cosmo=cosmo, 
                                                  epsilon_max=self.config.radius_factor)
        
        # Setup interpolator for fast evaluation
        # This precomputes displacement(r, M, z) on a grid
        Baryons.setup_interpolator(
            z_min=max(0.01, redshift - 0.1),
            z_max=redshift + 0.1,
            N_samples_z=3,
            M_min=self.config.mass_min,
            M_max=self.config.mass_max,
            N_samples_Mass=30,
            R_min=1e-3,
            R_max=self.config.radius_factor * 10,  # Go beyond epsilon_max
            N_samples_R=200,
            verbose=(rank == 0),
        )
        
        # Filter halos by mass
        halos_in_range = matched_halos.filter_by_mass(
            self.config.mass_min, self.config.mass_max, use_dmo=True
        )
        logger.info(f"    {halos_in_range.n_matches} halos for BCM")
        
        # Create BaryonForge data structures
        # Cosmology dict for BaryonForge
        cosmo_dict = {
            'Omega_m': 0.3089,
            'Omega_b': 0.0486,
            'h': 0.6774,
            'sigma8': 0.8159,
            'n_s': 0.9667,
            'w0': -1.0,
        }
        
        # Create HaloNDCatalog (halo positions and masses)
        halo_cat = bfg.utils.HaloNDCatalog(
            x=halos_in_range.dmo_positions[:, 0],
            y=halos_in_range.dmo_positions[:, 1],
            z=halos_in_range.dmo_positions[:, 2],
            M=halos_in_range.dmo_masses,
            redshift=redshift,
            cosmo=cosmo_dict,
        )
        
        # Create ParticleSnapshot
        pos = dmo_particles['positions']
        masses = dmo_particles['masses']
        
        particle_snap = bfg.utils.ParticleSnapshot(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            M=masses,
            L=self.config.box_size,
            redshift=redshift,
            cosmo=cosmo_dict,
        )
        
        logger.info(f"    Running BaryonifySnapshot on {len(pos):,} particles...")
        
        # Run baryonification
        Runner = bfg.Runners.BaryonifySnapshot(
            halo_cat, particle_snap,
            epsilon_max=self.config.radius_factor,
            model=Baryons,
            verbose=(rank == 0),
        )
        
        new_cat = Runner.process()
        
        # Extract new positions
        new_pos = np.column_stack([new_cat['x'], new_cat['y'], new_cat['z']])
        
        logger.info(f"    BCM applied successfully")
        
        return {
            'positions': new_pos,
            'masses': masses,
            'mode': 'bcm',
            'bcm_model': self.config.bcm_model,
            'n_halos': halos_in_range.n_matches,
            'snapshot': snap_num,
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
    
    def save_halo_profiles(self, particles: Dict, matched_halos: 'MatchedCatalog', 
                            snap_num: int):
        """
        Save radial density profiles for all halos in mass range.
        
        Computes 3D spherical density profiles out to 5×R_200 for each halo
        from the modified particle distribution (BCM or replacement).
        
        Output: HDF5 file with profiles for each halo.
        """
        logger.info(f"    Computing halo profiles to {self.config.profile_rmax}×R_200...")
        
        # Filter halos by mass
        halos = matched_halos.filter_by_mass(self.config.mass_min, self.config.mass_max, 
                                              use_dmo=True)
        n_halos = halos.n_matches
        logger.info(f"    Computing profiles for {n_halos} halos")
        
        # Radial bins (in units of R_200)
        n_bins = self.config.profile_bins
        r_bins_R200 = np.logspace(-2, np.log10(self.config.profile_rmax), n_bins + 1)
        r_centers_R200 = np.sqrt(r_bins_R200[:-1] * r_bins_R200[1:])  # Geometric mean
        
        # Get particle positions and masses
        pos = particles['positions']
        mass = particles['masses']
        box = self.config.box_size
        
        # Build KD-tree for fast neighbor queries
        tree = cKDTree(pos, boxsize=box)
        
        # Storage for all profiles
        all_profiles = np.zeros((n_halos, n_bins), dtype=np.float64)
        all_enclosed_mass = np.zeros((n_halos, n_bins), dtype=np.float64)
        halo_ids = np.zeros(n_halos, dtype=np.int64)
        halo_masses = np.zeros(n_halos, dtype=np.float64)
        halo_radii = np.zeros(n_halos, dtype=np.float64)
        
        for i in range(n_halos):
            center = halos.dmo_positions[i]
            R_200 = halos.dmo_radii[i]  # in Mpc/h
            M_200 = halos.dmo_masses[i]  # in Msun/h
            
            # Store halo info
            halo_ids[i] = halos.dmo_indices[i]
            halo_masses[i] = M_200
            halo_radii[i] = R_200
            
            # Physical radii for bins
            r_bins_phys = r_bins_R200 * R_200  # Mpc/h
            r_max = r_bins_phys[-1]
            
            # Find all particles within max radius
            indices = tree.query_ball_point(center, r_max)
            
            if len(indices) == 0:
                continue
            
            # Get particle positions relative to halo center
            dx = pos[indices] - center
            # Handle periodic boundaries
            dx = dx - box * np.round(dx / box)
            r = np.linalg.norm(dx, axis=1)  # Radial distance
            
            # Get particle masses
            m = mass[indices] if isinstance(mass, np.ndarray) and len(mass) > 1 else np.full(len(indices), mass)
            if not isinstance(m, np.ndarray):
                m = np.full(len(indices), m)
            
            # Bin particles by radius
            bin_idx = np.digitize(r, r_bins_phys) - 1
            
            for j in range(n_bins):
                in_bin = (bin_idx == j)
                # Shell volume
                V_shell = (4.0/3.0) * np.pi * (r_bins_phys[j+1]**3 - r_bins_phys[j]**3)
                # Density in Msun/h / (Mpc/h)^3
                all_profiles[i, j] = np.sum(m[in_bin]) / V_shell
                # Enclosed mass (cumulative)
                in_or_below = (bin_idx <= j)
                all_enclosed_mass[i, j] = np.sum(m[in_or_below])
            
            if (i + 1) % 100 == 0:
                logger.info(f"      Computed {i+1}/{n_halos} profiles")
        
        # Save to HDF5
        outfile = self.output_base / 'profiles' / f'profiles_snap{snap_num:03d}.h5'
        with h5py.File(outfile, 'w') as f:
            # Metadata
            f.attrs['mode'] = self.config.mode
            f.attrs['bcm_model'] = self.config.bcm_model if self.config.mode == 'bcm' else 'N/A'
            f.attrs['n_halos'] = n_halos
            f.attrs['mass_min'] = self.config.mass_min
            f.attrs['mass_max'] = self.config.mass_max
            f.attrs['profile_rmax_R200'] = self.config.profile_rmax
            f.attrs['n_bins'] = n_bins
            
            # Radial bins
            f.create_dataset('r_bins_R200', data=r_bins_R200)
            f.create_dataset('r_centers_R200', data=r_centers_R200)
            
            # Halo info
            f.create_dataset('halo_ids', data=halo_ids)
            f.create_dataset('halo_M200', data=halo_masses)
            f.create_dataset('halo_R200', data=halo_radii)
            
            # Profiles (n_halos × n_bins)
            f.create_dataset('density_profiles', data=all_profiles, 
                           compression='gzip', compression_opts=4)
            f.create_dataset('enclosed_mass', data=all_enclosed_mass,
                           compression='gzip', compression_opts=4)
        
        logger.info(f"    Saved {n_halos} halo profiles to {outfile}")
    
    def write_lux_input(self, particles: Dict, snap_num: int):
        """
        Write modified snapshot in TNG-compatible format for lux.
        
        Lux expects IllustrisTNG format with:
        - Header group with simulation metadata
        - PartType1 (DM) with Coordinates and Masses
        
        We create a minimal snapshot that lux can read by setting
        simulation_format=IllustrisTNG in the lux config.
        
        File structure mimics TNG: snapdir_XXX/snap_XXX.0.hdf5
        """
        logger.info(f"    Writing lux input (TNG-compatible format)...")
        
        # Create TNG-like directory structure: snapdir_XXX/snap_XXX.0.hdf5
        snap_dir = self.output_base / 'lux_input' / f'snapdir_{snap_num:03d}'
        snap_dir.mkdir(parents=True, exist_ok=True)
        outfile = snap_dir / f'snap_{snap_num:03d}.0.hdf5'
        
        n_particles = len(particles['positions'])
        
        # Get header info from original snapshot
        try:
            orig_header = load_snapshot_header(self.config.dmo_basepath, snap_num)
            redshift = orig_header['Redshift']
            time_val = orig_header['Time']
            omega_m = orig_header.get('Omega0', 0.3089)
            omega_L = orig_header.get('OmegaLambda', 0.6911)
            hubble = orig_header.get('HubbleParam', 0.6774)
        except:
            # Defaults if header not readable
            redshift = 0.0
            time_val = 1.0
            omega_m = 0.3089
            omega_L = 0.6911
            hubble = 0.6774
        
        with h5py.File(outfile, 'w') as f:
            # Header (matching TNG format)
            header = f.create_group('Header')
            header.attrs['BoxSize'] = self.config.box_size * 1e3  # Mpc/h -> kpc/h
            header.attrs['NumPart_Total'] = np.array([0, n_particles, 0, 0, 0, 0], dtype=np.uint32)
            header.attrs['NumPart_ThisFile'] = np.array([0, n_particles, 0, 0, 0, 0], dtype=np.uint32)
            header.attrs['NumPart_Total_HighWord'] = np.array([0, 0, 0, 0, 0, 0], dtype=np.uint32)
            header.attrs['NumFilesPerSnapshot'] = 1
            header.attrs['Redshift'] = redshift
            header.attrs['Time'] = time_val
            header.attrs['Omega0'] = omega_m
            header.attrs['OmegaLambda'] = omega_L
            header.attrs['HubbleParam'] = hubble
            header.attrs['Flag_DoublePrecision'] = 0
            
            # Mass table (index 1 = DM particles)
            mass_table = np.zeros(6, dtype=np.float64)
            if isinstance(particles['masses'], np.ndarray) and len(particles['masses']) > 1:
                # Variable masses - leave MassTable[1] = 0, masses stored in dataset
                mass_table[1] = 0.0
            else:
                # Uniform mass - store in MassTable
                mass_val = particles['masses'] if np.isscalar(particles['masses']) else particles['masses'][0]
                mass_table[1] = mass_val / 1e10  # Convert to 10^10 Msun/h
            header.attrs['MassTable'] = mass_table
            
            # PartType1 (DM particles)
            pt1 = f.create_group('PartType1')
            
            # Coordinates in kpc/h (TNG convention)
            coords = particles['positions'] * 1e3  # Mpc/h -> kpc/h
            pt1.create_dataset('Coordinates', data=coords.astype(np.float32),
                             compression='gzip', compression_opts=4)
            
            # Masses - only if variable (otherwise use MassTable)
            if isinstance(particles['masses'], np.ndarray) and len(particles['masses']) > 1:
                masses = particles['masses'] / 1e10  # Msun/h -> 10^10 Msun/h
                pt1.create_dataset('Masses', data=masses.astype(np.float32),
                                 compression='gzip', compression_opts=4)
            
            # Particle IDs (required by lux)
            pt1.create_dataset('ParticleIDs', 
                             data=np.arange(1, n_particles + 1, dtype=np.uint64),
                             compression='gzip', compression_opts=4)
            
            # Velocities (zeros - not used by lux for lensing)
            pt1.create_dataset('Velocities', 
                             data=np.zeros((n_particles, 3), dtype=np.float32),
                             compression='gzip', compression_opts=4)
        
        logger.info(f"    Saved lux input: {outfile}")
        logger.info(f"    ({n_particles:,} particles, z={redshift:.3f})")
    
    def generate_lux_config(self) -> Path:
        """
        Generate a lux configuration file for the modified snapshots.
        
        This creates an .ini file that points to the modified snapshot directory
        and sets appropriate ray-tracing parameters.
        
        Returns:
            Path to the generated config file
        """
        logger.info("Generating lux config file...")
        
        # Determine simulation format based on mode
        if self.config.mode in ['dmo', 'bcm']:
            sim_format = 'IllustrisTNG-Dark'  # DM-only format
        else:
            sim_format = 'IllustrisTNG'  # Hydro format (with baryons)
        
        # Build snapshot list and stack flags
        snaps = self.config.snapshots
        snap_list = ', '.join(str(s) for s in snaps)
        # Stack adjacent snapshots for low-z (typical threshold ~z<0.5, snap<52)
        stack_flags = ', '.join('true' if s < 52 else 'false' for s in snaps)
        
        # Config content
        config_content = f"""# Lux configuration for {self.config.mode} mode
# Generated by hydro_replace_pipeline.py
# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

input_dir={self.output_base / 'lux_input'}
snapshot_list={snap_list}
snapshot_stack={stack_flags}
LP_output_dir={self.output_base / 'lux_LP_output'}
RT_output_dir={self.output_base / 'lux_RT_output'}
LP_grid={self.config.grid_resolution}
RT_grid=1024
planes_per_snapshot=2
projection_direction=-1
translation_rotation=True
LP_random_seed=2020
RT_random_seed=1992
RT_randomization=True
simulation_format={sim_format}
angle=5.0
verbose=True
"""
        
        config_file = self.output_base / f'lux_{self.config.mode}.ini'
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        logger.info(f"  Saved lux config: {config_file}")
        return config_file
    
    # =========================================================================
    # Ray-Tracing
    # =========================================================================
    
    def run_lux(self):
        """Run lux ray-tracing on modified snapshots."""
        if rank != 0:
            return
        
        logger.info("Running lux ray-tracing...")
        
        # Generate config file
        config_file = self.generate_lux_config()
        
        # Create output directories
        (self.output_base / 'lux_LP_output').mkdir(parents=True, exist_ok=True)
        (self.output_base / 'lux_RT_output').mkdir(parents=True, exist_ok=True)
        
        # Lux executable
        lux_exe = Path('/mnt/home/mlee1/lux/lux')
        
        if not lux_exe.exists():
            logger.error(f"Lux executable not found: {lux_exe}")
            return
        
        logger.info(f"  Running: {lux_exe} {config_file}")
        
        # Note: lux should be run via MPI in a batch job
        # This is just for reference - actual execution should use:
        # mpirun -np N /mnt/home/mlee1/lux/lux config_file.ini
        logger.warning("Lux should be run via MPI in a separate batch job.")
        logger.warning(f"  Command: mpirun -np <N> {lux_exe} {config_file}")
    
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
                        choices=['Arico20', 'Schneider19', 'Schneider25'],
                        help='BCM model (if mode=bcm). Available: Arico20, Schneider19, Schneider25')
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
