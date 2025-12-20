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

import numpy as np
import h5py
import yaml
import argparse
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import time

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

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the hydro replace pipeline."""
    
    # Paths
    dmo_basepath: str = "/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output"
    hydro_basepath: str = "/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output"
    output_dir: str = "/mnt/home/mlee1/ceph/hydro_replace"
    lux_dir: str = "/mnt/home/mlee1/lux"
    
    # Simulation parameters
    box_size: float = 205.0  # Mpc/h
    
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
        if rank == 0:
            print(f"  Loading particles and catalogs...")
        
        dmo_particles = self.load_particles(self.config.dmo_basepath, snap_num, 'dmo')
        hydro_particles = self.load_particles(self.config.hydro_basepath, snap_num, 'hydro')
        matched_halos = self.load_halo_matches(snap_num)
        
        # 2. Apply modification
        if rank == 0:
            print(f"  Applying modification: {self.config.mode}")
        
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
        
        if self.config.save_profiles:
            self.save_profiles(dmo_particles, hydro_particles, matched_halos, snap_num)
        
        # 4. Write lux input
        self.write_lux_input(modified, snap_num)
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    def load_particles(self, basepath: str, snap_num: int, 
                       particle_type: str) -> Dict[str, np.ndarray]:
        """Load particle data from TNG snapshot."""
        # TODO: Implement actual loading with illustris_python
        # This is a placeholder
        raise NotImplementedError("Implement particle loading")
    
    def load_halo_matches(self, snap_num: int) -> Dict:
        """Load pre-computed halo matches for this snapshot."""
        # TODO: Load from matches file or compute on the fly
        raise NotImplementedError("Implement halo match loading")
    
    # =========================================================================
    # Modifications
    # =========================================================================
    
    def apply_replacement(self, dmo_particles: Dict, hydro_particles: Dict,
                          matched_halos: Dict) -> Dict:
        """
        Replace DMO halo particles with hydro particles.
        
        For each matched halo in the mass range:
        1. Remove DMO particles within radius_factor * R_200
        2. Insert hydro particles (DM + gas + stars) from matched hydro halo
        """
        # TODO: Implement replacement logic
        raise NotImplementedError("Implement replacement")
    
    def apply_bcm(self, dmo_particles: Dict, matched_halos: Dict) -> Dict:
        """
        Apply BaryonForge BCM to DMO halos.
        
        For each halo in the mass range:
        1. Compute BCM displacement field
        2. Apply radial displacement to DMO particles
        """
        # TODO: Implement BCM application
        # import BaryonForge as bfg
        # ...
        raise NotImplementedError("Implement BCM")
    
    # =========================================================================
    # Output Products
    # =========================================================================
    
    def save_projected_maps(self, particles: Dict, snap_num: int):
        """Save 2D projected surface density maps."""
        if rank == 0:
            print(f"  Saving projected maps...")
        # TODO: Implement map projection and saving
    
    def save_power_spectrum(self, particles: Dict, snap_num: int):
        """Compute and save 3D matter power spectrum."""
        if rank == 0:
            print(f"  Computing power spectrum...")
        # TODO: Implement P(k) computation with Pylians
    
    def save_profiles(self, dmo_particles: Dict, hydro_particles: Dict,
                      matched_halos: Dict, snap_num: int):
        """Save radial density profiles for halos."""
        if rank == 0:
            print(f"  Computing profiles...")
        # TODO: Implement profile computation
    
    def write_lux_input(self, particles: Dict, snap_num: int):
        """Write modified snapshot in format lux can read."""
        if rank == 0:
            print(f"  Writing lux input...")
        # TODO: Write HDF5 in TNG-like format
    
    # =========================================================================
    # Ray-Tracing
    # =========================================================================
    
    def run_lux(self):
        """Run lux ray-tracing on modified snapshots."""
        if rank != 0:
            return
        
        # TODO: Implement lux execution
        # subprocess.run([...])
        pass
    
    def analyze_peaks(self):
        """Analyze peak counts from convergence maps."""
        if rank != 0:
            return
        
        # TODO: Implement peak analysis
        pass


# =============================================================================
# Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Hydro Replace Pipeline")
    
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
        config = PipelineConfig()
    
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
