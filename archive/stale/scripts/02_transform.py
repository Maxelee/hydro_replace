#!/usr/bin/env python3
"""
Simple particle transformation for Replace/BCM modes.

This version uses serial processing with optimized numpy/scipy operations.
For the 625^3 resolution, this should complete in a few minutes.

Usage:
    python 02_transform.py --mode replace --resolution 625
    python 02_transform.py --mode bcm --bcm-model Arico20 --resolution 625
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

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


def load_particles(basepath: str, snap_num: int) -> Dict:
    """Load DM particle positions and masses."""
    import illustris_python.snapshot as snapshot
    
    # Get mass from header
    snap_file = f"{basepath}/snapdir_{snap_num:03d}/snap_{snap_num:03d}.0.hdf5"
    with h5py.File(snap_file, 'r') as f:
        dm_mass = f['Header'].attrs['MassTable'][1] * 1e10  # Msun/h
    
    # Load coordinates
    coords = snapshot.loadSubset(basepath, snap_num, 'dm', fields=['Coordinates'])
    
    # Handle both dict and array returns
    if isinstance(coords, dict):
        pos = coords['Coordinates']
    else:
        pos = coords
    
    pos = pos / 1e3  # kpc/h → Mpc/h
    masses = np.full(len(pos), dm_mass, dtype=np.float32)
    
    return {
        'positions': pos.astype(np.float32),
        'masses': masses,
    }


def load_matches(match_file: Path) -> Dict:
    """Load pre-computed halo matches."""
    with h5py.File(match_file, 'r') as f:
        return {key: f[key][:] for key in f.keys()}


def transform_replace(
    dmo_particles: Dict,
    hydro_particles: Dict,
    matches: Dict,
    radius_factor: float = 5.0,
    box_size: float = 205.0,
) -> Dict:
    """
    Replace DMO particles with hydro particles around matched halos.
    """
    n_halos = len(matches['dmo_indices'])
    logger.info(f"Processing {n_halos} halos for replacement...")
    
    # Build KDTrees
    logger.info("  Building KDTrees...")
    dmo_tree = cKDTree(dmo_particles['positions'], boxsize=box_size)
    hydro_tree = cKDTree(hydro_particles['positions'], boxsize=box_size)
    
    # Track particles
    dmo_remove_mask = np.zeros(len(dmo_particles['positions']), dtype=bool)
    hydro_add_indices = []
    hydro_add_offsets = []  # Offset to apply to hydro positions
    
    t_start = time.time()
    for i in range(n_halos):
        if i % 500 == 0:
            logger.info(f"  {i}/{n_halos} halos processed...")
        
        # Halo properties
        dmo_pos = matches['dmo_positions'][i]
        hydro_pos = matches['hydro_positions'][i]
        r200_dmo = matches['dmo_radii'][i]
        r200_hydro = matches['hydro_radii'][i]
        
        search_radius_dmo = radius_factor * r200_dmo
        search_radius_hydro = radius_factor * r200_hydro
        
        # Find DMO particles to remove
        dmo_in_halo = dmo_tree.query_ball_point(dmo_pos, r=search_radius_dmo)
        dmo_remove_mask[dmo_in_halo] = True
        
        # Find hydro particles to add
        hydro_in_halo = hydro_tree.query_ball_point(hydro_pos, r=search_radius_hydro)
        
        # Store indices and offset (shift hydro to DMO center)
        offset = dmo_pos - hydro_pos
        offset = offset - box_size * np.round(offset / box_size)  # Periodic
        
        for idx in hydro_in_halo:
            hydro_add_indices.append(idx)
            hydro_add_offsets.append(offset)
    
    logger.info(f"  Halo loop took {time.time()-t_start:.1f}s")
    
    # Apply transformation
    n_removed = dmo_remove_mask.sum()
    n_added = len(hydro_add_indices)
    logger.info(f"  Removing {n_removed} DMO particles, adding {n_added} hydro particles")
    
    # Keep DMO particles not in halos
    keep_positions = dmo_particles['positions'][~dmo_remove_mask]
    keep_masses = dmo_particles['masses'][~dmo_remove_mask]
    
    # Add hydro particles with offset
    if n_added > 0:
        hydro_add_indices = np.array(hydro_add_indices)
        hydro_add_offsets = np.array(hydro_add_offsets)
        
        add_positions = hydro_particles['positions'][hydro_add_indices] + hydro_add_offsets
        add_positions = add_positions % box_size  # Wrap
        add_masses = hydro_particles['masses'][hydro_add_indices]
        
        final_positions = np.vstack([keep_positions, add_positions])
        final_masses = np.hstack([keep_masses, add_masses])
    else:
        final_positions = keep_positions
        final_masses = keep_masses
    
    return {
        'positions': final_positions,
        'masses': final_masses,
    }


def transform_bcm(
    dmo_particles: Dict,
    matches: Dict,
    bcm_model: str = 'Arico20',
    radius_factor: float = 5.0,
    box_size: float = 205.0,
    redshift: float = 0.0,
) -> Dict:
    """
    Apply BCM displacements to DMO particles.
    """
    from BaryonForge import Baryonification3D
    from BaryonForge.DensityProfiles import DarkMatterOnly, DarkMatterBaryon
    import pyccl as ccl
    
    n_halos = len(matches['dmo_indices'])
    logger.info(f"Processing {n_halos} halos for BCM ({bcm_model})...")
    
    # Setup cosmology
    h = 0.6774
    cosmo = ccl.Cosmology(
        Omega_c=0.2589, Omega_b=0.0486, h=h,
        sigma8=0.8159, n_s=0.9667,
        matter_power_spectrum='linear'
    )
    
    # Get BCM parameters
    bpar = get_bcm_params(bcm_model, h)
    
    # Setup profiles
    logger.info("  Setting up BCM profiles...")
    dmo_profile = DarkMatterOnly(Model=bcm_model, cosmo=cosmo, z=redshift, **bpar)
    baryon_profile = DarkMatterBaryon(Model=bcm_model, cosmo=cosmo, z=redshift, **bpar)
    
    baryonifier = Baryonification3D(DMO=dmo_profile, DMB=baryon_profile, epsilon=1e-8)
    baryonifier.setup_interpolator(r_range=[1e-5, 20], M_range=[1e12, 1e16], N_samples=64)
    
    # Build KDTree
    logger.info("  Building KDTree...")
    tree = cKDTree(dmo_particles['positions'], boxsize=box_size)
    
    # Initialize displacements
    n_particles = len(dmo_particles['positions'])
    displacements = np.zeros((n_particles, 3), dtype=np.float32)
    
    t_start = time.time()
    for i in range(n_halos):
        if i % 200 == 0:
            logger.info(f"  {i}/{n_halos} halos processed...")
        
        pos = matches['dmo_positions'][i]
        mass = matches['dmo_masses'][i]  # Msun/h
        r200 = matches['dmo_radii'][i]
        
        mass_msun = mass / h  # BaryonForge expects Msun
        
        # Find particles
        search_radius = radius_factor * r200
        particle_indices = tree.query_ball_point(pos, r=search_radius)
        
        if len(particle_indices) == 0:
            continue
        
        # Relative positions
        particle_pos = dmo_particles['positions'][particle_indices]
        rel_pos = particle_pos - pos
        rel_pos = rel_pos - box_size * np.round(rel_pos / box_size)
        
        radii = np.linalg.norm(rel_pos, axis=1)
        radii = np.clip(radii, 1e-6, None)
        
        # BCM displacement
        try:
            factors = baryonifier.displacement_factor(radii, mass_msun)
            unit_vectors = rel_pos / radii[:, np.newaxis]
            displacement = unit_vectors * (radii * (factors - 1))[:, np.newaxis]
            displacements[particle_indices] += displacement
        except Exception as e:
            logger.warning(f"  BCM failed for halo {i}: {e}")
            continue
    
    logger.info(f"  Halo loop took {time.time()-t_start:.1f}s")
    
    # Apply displacements
    new_positions = dmo_particles['positions'] + displacements
    new_positions = new_positions % box_size
    
    mean_disp = np.linalg.norm(displacements, axis=1).mean() * 1e3
    logger.info(f"  Mean displacement: {mean_disp:.2f} kpc/h")
    
    return {
        'positions': new_positions.astype(np.float32),
        'masses': dmo_particles['masses'].copy(),
    }


def get_bcm_params(model: str, h: float) -> dict:
    """Get BCM parameters for each model."""
    if model == 'Schneider19':
        return dict(
            theta_ej=4, theta_co=0.1, mu_beta=1, M_c=1e14/h,
            eta=0.3, eta_delta=0.3, tau=0, tau_delta=0,
            A=0.045, M1=2.5e11/h, epsilon_h=0.015, eta_cga=0.6,
            a=0.3, n=2, epsilon=4, p=0.3, q=0.707, cdelta=6.71,
            gamma=2, delta=7,
        )
    elif model == 'Schneider25':
        return dict(
            epsilon0=4, epsilon1=0.5, alpha_excl=0.4, p=0.3, q=0.707,
            M_c=1e15, mu=0.8,
            q0=0.075, q1=0.25, q2=0.7, nu_q0=0, nu_q1=1, nu_q2=0, nstep=3/2,
            theta_c=0.3, nu_theta_c=0.5, c_iga=0.1, nu_c_iga=1.5, r_min_iga=1e-3,
            alpha=1, gamma=1.5, delta=7, tau=-1.376, tau_delta=0,
            Mstar=3e11, Nstar=0.03, eta=0.1, eta_delta=0.22, epsilon_cga=0.03,
        )
    elif model == 'Arico20':
        from numpy import sqrt
        return dict(
            cdelta=4, alpha_g=2, epsilon_h=0.015, M1_0=2.2e11/h,
            alpha_fsat=1, M1_fsat=1, delta_fsat=1, gamma_fsat=1, eps_fsat=1,
            M_c=1.2e14/h, eta=0.6, mu=0.31, beta=0.6, epsilon_hydro=sqrt(5),
            M_inn=3.3e13/h, M_r=1e16, beta_r=2, theta_inn=0.1, theta_out=3,
            theta_rg=0.3, sigma_rg=0.1, a=0.3, n=2, p=0.3, q=0.707,
            A_nt=0.495, alpha_nt=0.1, mean_molecular_weight=0.59,
        )
    else:
        raise ValueError(f"Unknown BCM model: {model}")


def save_particles(particles: Dict, output_path: Path):
    """Save transformed particles to HDF5."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('positions', data=particles['positions'],
                        compression='gzip', compression_opts=4)
        f.create_dataset('masses', data=particles['masses'],
                        compression='gzip', compression_opts=4)
        f.attrs['n_particles'] = len(particles['positions'])
        f.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    logger.info(f"Saved {len(particles['positions'])} particles to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['dmo', 'hydro', 'replace', 'bcm'])
    parser.add_argument('--bcm-model', type=str, default='Arico20',
                        choices=['Arico20', 'Schneider19', 'Schneider25'])
    parser.add_argument('--resolution', type=int, default=625)
    parser.add_argument('--snapshot', type=int, default=99)
    parser.add_argument('--radius', type=float, default=5.0)
    parser.add_argument('--output-dir', type=str,
                        default='/mnt/home/mlee1/ceph/hydro_replace')
    args = parser.parse_args()
    
    t_start = time.time()
    
    # Paths
    res = args.resolution
    dmo_path = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L205n{res}TNG_DM/output'
    hydro_path = f'/mnt/sdceph/users/sgenel/IllustrisTNG/L205n{res}TNG/output'
    
    base_dir = Path(args.output_dir) / f'L205n{res}TNG'
    match_file = base_dir / 'matches' / f'spatial_match_snap{args.snapshot:03d}.h5'
    
    # Mode name for output
    mode_name = f'bcm-{args.bcm_model}' if args.mode == 'bcm' else args.mode
    output_file = base_dir / mode_name / f'particles_snap{args.snapshot:03d}.h5'
    
    logger.info("=" * 60)
    logger.info(f"PARTICLE TRANSFORM - {mode_name.upper()}")
    logger.info("=" * 60)
    logger.info(f"Resolution: {res}^3")
    logger.info(f"Snapshot: {args.snapshot}")
    logger.info(f"Output: {output_file}")
    
    if output_file.exists():
        logger.info(f"Output exists, skipping")
        return
    
    # Load DMO particles
    logger.info("Loading DMO particles...")
    dmo_particles = load_particles(dmo_path, args.snapshot)
    logger.info(f"  Loaded {len(dmo_particles['positions'])} particles")
    
    # Process based on mode
    if args.mode == 'dmo':
        result = dmo_particles
        
    elif args.mode == 'hydro':
        logger.info("Loading hydro particles...")
        result = load_particles(hydro_path, args.snapshot)
        
    elif args.mode == 'replace':
        logger.info("Loading matches and hydro particles...")
        matches = load_matches(match_file)
        hydro_particles = load_particles(hydro_path, args.snapshot)
        
        logger.info("Applying replacement...")
        result = transform_replace(dmo_particles, hydro_particles, matches,
                                   radius_factor=args.radius)
        
    elif args.mode == 'bcm':
        logger.info("Loading matches...")
        matches = load_matches(match_file)
        
        logger.info(f"Applying BCM ({args.bcm_model})...")
        result = transform_bcm(dmo_particles, matches,
                              bcm_model=args.bcm_model,
                              radius_factor=args.radius)
    
    # Save
    save_particles(result, output_file)
    
    t_elapsed = time.time() - t_start
    logger.info(f"Total time: {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
