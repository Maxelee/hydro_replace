#!/usr/bin/env python3
"""
MPI-parallel particle transformation for Replace/BCM modes.

Each rank handles a subset of halos, computes displacements,
then results are gathered and applied.

Usage:
    # Single process
    python 02_transform_mpi.py --mode replace --resolution 625
    
    # MPI parallel (recommended)
    srun -n 16 python 02_transform_mpi.py --mode bcm --bcm-model Arico20 --resolution 625
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
import h5py

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# MPI setup - must be before other imports that might use MPI
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1

# Setup logging (only rank 0)
logging.basicConfig(
    level=logging.INFO if rank == 0 else logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_particles(basepath: str, snap_num: int, ptypes: list = [1]) -> Dict:
    """Load particle positions and masses."""
    import illustris_python.snapshot as snapshot
    
    positions = []
    masses = []
    
    # Get mass table from first snapshot file
    snap_file = f"{basepath}/snapdir_{snap_num:03d}/snap_{snap_num:03d}.0.hdf5"
    with h5py.File(snap_file, 'r') as f:
        mass_table = f['Header'].attrs['MassTable'] * 1e10  # 10^10 Msun/h → Msun/h
    
    for ptype in ptypes:
        ptype_name = {0: 'gas', 1: 'dm', 4: 'stars', 5: 'bh'}[ptype]
        
        # Load coordinates - loadSubset returns array directly for single field
        coords = snapshot.loadSubset(basepath, snap_num, ptype_name, 
                                      fields=['Coordinates'])
        
        # Handle both dict (multiple fields) and array (single field) returns
        if isinstance(coords, dict):
            pos = coords['Coordinates']
            count = coords['count']
        else:
            pos = coords
            count = len(pos) if pos is not None else 0
        
        if count > 0:
            pos = pos / 1e3  # kpc/h → Mpc/h
            positions.append(pos)
            
            # DM particles have fixed mass from header
            if ptype == 1:
                particle_mass = mass_table[1]
                masses.append(np.full(len(pos), particle_mass))
            else:
                # Other types may have variable masses
                mass_data = snapshot.loadSubset(basepath, snap_num, ptype_name,
                                                 fields=['Masses'])
                if isinstance(mass_data, dict):
                    masses.append(mass_data['Masses'] * 1e10)
                else:
                    masses.append(mass_data * 1e10)
    
    return {
        'positions': np.vstack(positions) if positions else np.array([]),
        'masses': np.hstack(masses) if masses else np.array([]),
    }


def load_matches(match_file: Path) -> Dict:
    """Load pre-computed halo matches."""
    with h5py.File(match_file, 'r') as f:
        return {
            'dmo_indices': f['dmo_indices'][:],
            'hydro_indices': f['hydro_indices'][:],
            'dmo_masses': f['dmo_masses'][:],
            'hydro_masses': f['hydro_masses'][:],
            'dmo_positions': f['dmo_positions'][:],
            'hydro_positions': f['hydro_positions'][:],
            'dmo_radii': f['dmo_radii'][:],
            'hydro_radii': f['hydro_radii'][:],
        }


def transform_replace(
    dmo_particles: Dict,
    hydro_particles: Dict,
    matches: Dict,
    radius_factor: float = 5.0,
    box_size: float = 205.0,
) -> Dict:
    """
    Replace DMO particles with hydro particles around matched halos.
    
    MPI parallel: each rank handles a subset of halos.
    """
    n_halos = len(matches['dmo_indices'])
    my_halos = list(range(rank, n_halos, size))
    
    logger.info(f"Processing {len(my_halos)} halos on rank {rank} of {size}")
    
    # Build KDTree for DMO particles (to find particles to remove)
    dmo_tree = cKDTree(dmo_particles['positions'], boxsize=box_size)
    
    # Build KDTree for hydro particles (to find replacement particles)  
    hydro_tree = cKDTree(hydro_particles['positions'], boxsize=box_size)
    
    # Track which DMO particles to remove and which hydro to add
    dmo_remove_mask = np.zeros(len(dmo_particles['positions']), dtype=bool)
    hydro_particles_to_add = []
    hydro_masses_to_add = []
    
    for i, halo_idx in enumerate(my_halos):
        if i % 100 == 0 and rank == 0:
            logger.info(f"  Rank 0: {i}/{len(my_halos)} halos processed")
        
        # Halo properties
        dmo_pos = matches['dmo_positions'][halo_idx]
        hydro_pos = matches['hydro_positions'][halo_idx]
        r200_dmo = matches['dmo_radii'][halo_idx]
        r200_hydro = matches['hydro_radii'][halo_idx]
        
        # Search radii
        search_radius_dmo = radius_factor * r200_dmo
        search_radius_hydro = radius_factor * r200_hydro
        
        # Find DMO particles to remove
        dmo_in_halo = dmo_tree.query_ball_point(dmo_pos, r=search_radius_dmo)
        dmo_remove_mask[dmo_in_halo] = True
        
        # Find hydro DM particles to add
        hydro_in_halo = hydro_tree.query_ball_point(hydro_pos, r=search_radius_hydro)
        
        # Shift hydro particles to DMO halo center
        for idx in hydro_in_halo:
            # Get position relative to hydro center
            rel_pos = hydro_particles['positions'][idx] - hydro_pos
            # Apply periodic wrapping
            rel_pos = rel_pos - box_size * np.round(rel_pos / box_size)
            # New position at DMO center
            new_pos = dmo_pos + rel_pos
            # Wrap back into box
            new_pos = new_pos % box_size
            
            hydro_particles_to_add.append(new_pos)
            hydro_masses_to_add.append(hydro_particles['masses'][idx])
    
    # Gather results from all ranks
    if comm is not None and size > 1:
        # Gather remove masks
        all_remove_masks = comm.gather(dmo_remove_mask, root=0)
        all_add_pos = comm.gather(hydro_particles_to_add, root=0)
        all_add_mass = comm.gather(hydro_masses_to_add, root=0)
        
        if rank == 0:
            # Combine remove masks (OR)
            combined_remove = np.zeros_like(dmo_remove_mask)
            for mask in all_remove_masks:
                combined_remove |= mask
            
            # Combine added particles
            combined_pos = []
            combined_mass = []
            for pos_list, mass_list in zip(all_add_pos, all_add_mass):
                combined_pos.extend(pos_list)
                combined_mass.extend(mass_list)
            
            dmo_remove_mask = combined_remove
            hydro_particles_to_add = combined_pos
            hydro_masses_to_add = combined_mass
        else:
            return None  # Non-root ranks return None
    
    # Apply transformation (rank 0 only in MPI mode)
    n_removed = dmo_remove_mask.sum()
    n_added = len(hydro_particles_to_add)
    
    logger.info(f"Removing {n_removed} DMO particles, adding {n_added} hydro particles")
    
    # Keep DMO particles not in halos
    keep_positions = dmo_particles['positions'][~dmo_remove_mask]
    keep_masses = dmo_particles['masses'][~dmo_remove_mask]
    
    # Add hydro particles
    if len(hydro_particles_to_add) > 0:
        add_positions = np.array(hydro_particles_to_add)
        add_masses = np.array(hydro_masses_to_add)
        
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
    
    MPI parallel: each rank computes displacements for its halos,
    then displacements are gathered and summed.
    """
    from BaryonForge import Baryonification3D
    from BaryonForge.DensityProfiles import DarkMatterOnly, DarkMatterBaryon
    import pyccl as ccl
    
    n_halos = len(matches['dmo_indices'])
    my_halos = list(range(rank, n_halos, size))
    
    logger.info(f"Processing {len(my_halos)} BCM halos on rank {rank} of {size}")
    
    # Setup cosmology
    h = 0.6774
    cosmo = ccl.Cosmology(
        Omega_c=0.2589, Omega_b=0.0486, h=h,
        sigma8=0.8159, n_s=0.9667,
        matter_power_spectrum='linear'
    )
    
    # Get BCM parameters based on model
    bpar = get_bcm_params(bcm_model, h)
    
    # Setup density profiles
    dmo_profile = DarkMatterOnly(
        Model=bcm_model, cosmo=cosmo, z=redshift, **bpar
    )
    baryon_profile = DarkMatterBaryon(
        Model=bcm_model, cosmo=cosmo, z=redshift, **bpar
    )
    
    # Setup Baryonification
    baryonifier = Baryonification3D(
        DMO=dmo_profile,
        DMB=baryon_profile,
        epsilon=1e-8,
    )
    baryonifier.setup_interpolator(
        r_range=[1e-5, 20],
        M_range=[1e12, 1e16],
        N_samples=64,
    )
    
    # Initialize displacement array
    n_particles = len(dmo_particles['positions'])
    my_displacements = np.zeros((n_particles, 3), dtype=np.float32)
    
    # Build KDTree for particles
    tree = cKDTree(dmo_particles['positions'], boxsize=box_size)
    
    for i, halo_idx in enumerate(my_halos):
        if i % 50 == 0:
            print(f"Rank {rank}: {i}/{len(my_halos)} halos processed", flush=True)
        
        # Halo properties
        pos = matches['dmo_positions'][halo_idx]
        mass = matches['dmo_masses'][halo_idx]  # Msun/h
        r200 = matches['dmo_radii'][halo_idx]
        
        # Convert mass to Msun (BaryonForge expects Msun)
        mass_msun = mass / h
        
        # Find particles near this halo
        search_radius = radius_factor * r200
        particle_indices = tree.query_ball_point(pos, r=search_radius)
        
        if len(particle_indices) == 0:
            continue
        
        # Get particle positions relative to halo center
        particle_pos = dmo_particles['positions'][particle_indices]
        rel_pos = particle_pos - pos
        
        # Handle periodic boundaries
        rel_pos = rel_pos - box_size * np.round(rel_pos / box_size)
        
        # Compute radii
        radii = np.linalg.norm(rel_pos, axis=1)
        radii = np.clip(radii, 1e-6, None)  # Avoid zero
        
        # Compute displacement factors
        try:
            factors = baryonifier.displacement_factor(radii, mass_msun)
        except Exception as e:
            logger.warning(f"BCM failed for halo {halo_idx}: {e}")
            continue
        
        # Apply radial displacement
        unit_vectors = rel_pos / radii[:, np.newaxis]
        displacement = unit_vectors * (radii * (factors - 1))[:, np.newaxis]
        
        # Accumulate displacements
        my_displacements[particle_indices] += displacement
    
    # Gather and sum displacements from all ranks
    if comm is not None and size > 1:
        total_displacements = np.zeros_like(my_displacements)
        comm.Reduce(my_displacements, total_displacements, op=MPI.SUM, root=0)
        
        if rank != 0:
            return None
        
        my_displacements = total_displacements
    
    # Apply displacements
    new_positions = dmo_particles['positions'] + my_displacements
    
    # Wrap into box
    new_positions = new_positions % box_size
    
    logger.info(f"Applied BCM displacements, mean |dr| = {np.linalg.norm(my_displacements, axis=1).mean()*1e3:.2f} kpc/h")
    
    return {
        'positions': new_positions,
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
    
    # Determine output mode name
    if args.mode == 'bcm':
        mode_name = f'bcm-{args.bcm_model}'
    else:
        mode_name = args.mode
    
    output_file = base_dir / mode_name / f'particles_snap{args.snapshot:03d}.h5'
    
    if rank == 0:
        logger.info("=" * 60)
        logger.info(f"PARTICLE TRANSFORM - {mode_name.upper()}")
        logger.info("=" * 60)
        logger.info(f"Resolution: {res}^3")
        logger.info(f"Snapshot: {args.snapshot}")
        logger.info(f"MPI ranks: {size}")
        logger.info(f"Output: {output_file}")
    
    # Check if output exists
    if output_file.exists():
        logger.info(f"Output exists, skipping: {output_file}")
        return
    
    # Load data
    if rank == 0:
        logger.info("Loading DMO particles...")
    dmo_particles = load_particles(dmo_path, args.snapshot, ptypes=[1])
    if rank == 0:
        logger.info(f"  Loaded {len(dmo_particles['positions'])} DMO particles")
    
    # Process based on mode
    if args.mode == 'dmo':
        # Just use DMO particles as-is
        result = dmo_particles
        
    elif args.mode == 'hydro':
        # Load and use hydro DM particles
        if rank == 0:
            logger.info("Loading hydro particles...")
        result = load_particles(hydro_path, args.snapshot, ptypes=[1])
        
    elif args.mode == 'replace':
        # Load matches and hydro particles
        if rank == 0:
            logger.info("Loading matches...")
        matches = load_matches(match_file)
        
        if rank == 0:
            logger.info("Loading hydro particles...")
        hydro_particles = load_particles(hydro_path, args.snapshot, ptypes=[1])
        
        if rank == 0:
            logger.info("Applying replacement...")
        result = transform_replace(
            dmo_particles, hydro_particles, matches,
            radius_factor=args.radius
        )
        
    elif args.mode == 'bcm':
        # Load matches and apply BCM
        if rank == 0:
            logger.info("Loading matches...")
        matches = load_matches(match_file)
        
        if rank == 0:
            logger.info(f"Applying BCM ({args.bcm_model})...")
        result = transform_bcm(
            dmo_particles, matches,
            bcm_model=args.bcm_model,
            radius_factor=args.radius
        )
    
    # Save (rank 0 only)
    if result is not None and rank == 0:
        save_particles(result, output_file)
        
        t_elapsed = time.time() - t_start
        logger.info(f"Total time: {t_elapsed:.1f}s")


if __name__ == '__main__':
    main()
