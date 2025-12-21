#!/usr/bin/env python3
"""
Fast spatial halo matching using KDTree.

This is MUCH faster than particle-based bijective matching:
- Particle matching: O(N_particles) per halo → hours
- Spatial matching: O(N_halos * log(N_halos)) → seconds

Usage:
    python spatial_match.py --resolution 625 --snapshot 99
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import h5py

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_halo_catalog(basepath: str, snap_num: int, mass_min: float = 1e13):
    """Load halo catalog with mass cut."""
    import illustris_python.groupcat as gc
    
    # Load halo properties
    fields = ['GroupPos', 'Group_M_Crit200', 'Group_R_Crit200']
    halos = gc.loadHalos(basepath, snap_num, fields=fields)
    
    # Handle single field return (returns array directly)
    if isinstance(halos, np.ndarray):
        raise ValueError("Need multiple fields, got single array")
    
    positions = halos['GroupPos'] / 1e3  # kpc/h → Mpc/h
    masses = halos['Group_M_Crit200'] * 1e10  # 10^10 Msun/h → Msun/h
    radii = halos['Group_R_Crit200'] / 1e3  # kpc/h → Mpc/h
    
    # Apply mass cut
    mask = masses >= mass_min
    
    logger.info(f"Loaded {mask.sum()} halos above {mass_min:.1e} Msun/h (of {len(masses)} total)")
    
    return {
        'positions': positions[mask],
        'masses': masses[mask],
        'radii': radii[mask],
        'indices': np.where(mask)[0]  # Original indices
    }


def spatial_match(dmo_cat: dict, hydro_cat: dict, 
                  max_separation_r200: float = 1.0,
                  mass_ratio_min: float = 0.2,
                  mass_ratio_max: float = 5.0,
                  box_size: float = 205.0) -> dict:
    """
    Match halos by spatial proximity and mass ratio.
    
    Parameters
    ----------
    dmo_cat, hydro_cat : dict
        Halo catalogs with 'positions', 'masses', 'radii', 'indices'
    max_separation_r200 : float
        Maximum separation in units of DMO R_200
    mass_ratio_min, mass_ratio_max : float
        Acceptable range of hydro/dmo mass ratio
    box_size : float
        Periodic box size in Mpc/h
        
    Returns
    -------
    matches : dict
        Matched catalog with DMO and hydro properties
    """
    t_start = time.time()
    
    n_dmo = len(dmo_cat['positions'])
    n_hydro = len(hydro_cat['positions'])
    
    logger.info(f"Matching {n_dmo} DMO halos to {n_hydro} hydro halos...")
    
    # Build KDTree of hydro positions
    # Handle periodic boundaries by searching with boxsize
    tree = cKDTree(hydro_cat['positions'], boxsize=box_size)
    
    # Storage for matches
    dmo_matched = []
    hydro_matched = []
    separations = []
    
    # For each DMO halo, find best hydro match
    for i in range(n_dmo):
        pos = dmo_cat['positions'][i]
        r200 = dmo_cat['radii'][i]
        mass_dmo = dmo_cat['masses'][i]
        
        # Search radius
        search_radius = max_separation_r200 * r200
        
        # Find nearby hydro halos
        candidates = tree.query_ball_point(pos, r=search_radius)
        
        if len(candidates) == 0:
            continue
            
        # Filter by mass ratio and find closest
        best_match = None
        best_dist = np.inf
        
        for j in candidates:
            mass_hydro = hydro_cat['masses'][j]
            ratio = mass_hydro / mass_dmo
            
            if mass_ratio_min < ratio < mass_ratio_max:
                # Compute distance (periodic)
                delta = hydro_cat['positions'][j] - pos
                delta = delta - box_size * np.round(delta / box_size)
                dist = np.linalg.norm(delta)
                
                if dist < best_dist:
                    best_dist = dist
                    best_match = j
        
        if best_match is not None:
            dmo_matched.append(i)
            hydro_matched.append(best_match)
            separations.append(best_dist)
    
    dmo_matched = np.array(dmo_matched)
    hydro_matched = np.array(hydro_matched)
    separations = np.array(separations)
    
    t_elapsed = time.time() - t_start
    logger.info(f"Found {len(dmo_matched)} matches in {t_elapsed:.2f}s")
    logger.info(f"  Match rate: {100*len(dmo_matched)/n_dmo:.1f}% of DMO halos")
    logger.info(f"  Mean separation: {separations.mean()*1e3:.1f} kpc/h")
    
    # Build output catalog
    return {
        'dmo_indices': dmo_cat['indices'][dmo_matched],
        'hydro_indices': hydro_cat['indices'][hydro_matched],
        'dmo_masses': dmo_cat['masses'][dmo_matched],
        'hydro_masses': hydro_cat['masses'][hydro_matched],
        'dmo_positions': dmo_cat['positions'][dmo_matched],
        'hydro_positions': hydro_cat['positions'][hydro_matched],
        'dmo_radii': dmo_cat['radii'][dmo_matched],
        'hydro_radii': hydro_cat['radii'][hydro_matched],
        'separations': separations,
        'n_matches': len(dmo_matched),
    }


def save_matches(matches: dict, output_path: Path):
    """Save matched catalog to HDF5."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        for key, value in matches.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
            else:
                f.attrs[key] = value
        
        # Add metadata
        f.attrs['method'] = 'spatial_kdtree'
        f.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    logger.info(f"Saved matches to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Fast spatial halo matching')
    parser.add_argument('--resolution', type=int, default=625)
    parser.add_argument('--snapshot', type=int, default=99)
    parser.add_argument('--mass-min', type=float, default=1e13)
    parser.add_argument('--max-separation', type=float, default=1.0,
                        help='Max separation in units of R_200')
    parser.add_argument('--output-dir', type=str,
                        default='/mnt/home/mlee1/ceph/hydro_replace')
    args = parser.parse_args()
    
    # Paths
    res = args.resolution
    dmo_path = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L205n{res}TNG_DM/output'
    hydro_path = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L205n{res}TNG/output'
    
    output_dir = Path(args.output_dir) / f'L205n{res}TNG' / 'matches'
    output_file = output_dir / f'spatial_match_snap{args.snapshot:03d}.h5'
    
    logger.info(f"=== Spatial Halo Matching ===")
    logger.info(f"Resolution: {res}^3")
    logger.info(f"Snapshot: {args.snapshot}")
    logger.info(f"Mass min: {args.mass_min:.1e} Msun/h")
    
    # Check if already exists
    if output_file.exists():
        logger.info(f"Output already exists: {output_file}")
        return
    
    # Load catalogs
    logger.info("Loading DMO catalog...")
    dmo_cat = load_halo_catalog(dmo_path, args.snapshot, args.mass_min)
    
    logger.info("Loading hydro catalog...")
    hydro_cat = load_halo_catalog(hydro_path, args.snapshot, args.mass_min)
    
    # Match
    matches = spatial_match(
        dmo_cat, hydro_cat,
        max_separation_r200=args.max_separation,
    )
    
    # Save
    save_matches(matches, output_file)
    
    # Summary
    logger.info("=== Summary ===")
    logger.info(f"  Matched: {matches['n_matches']} pairs")
    logger.info(f"  DMO mass range: {matches['dmo_masses'].min():.2e} - {matches['dmo_masses'].max():.2e}")
    logger.info(f"  Output: {output_file}")


if __name__ == '__main__':
    main()
