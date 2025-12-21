#!/usr/bin/env python
"""
Generate 3D pixelized density fields for DMO, Hydro, and Replace modes.

Based on working code from:
  /mnt/home/mlee1/Hydro_replacement/extract_and_pixelize_full3D.py
  /mnt/home/mlee1/Hydro_replacement/dmo_hydro_extract_full.py

Usage:
    mpirun -np 64 python generate_fields.py --snap 99 --sim-res 2500 --mode all
    mpirun -np 64 python generate_fields.py --snap 99 --sim-res 625 --mode dmo
    mpirun -np 64 python generate_fields.py --snap 99 --sim-res 2500 --mode replace --mass-min 13.0
"""

import numpy as np
import h5py
import glob
import argparse
import os
from mpi4py import MPI
from scipy.spatial import cKDTree
from illustris_python import groupcat
import MAS_library as MASL

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================================
# Simulation paths for different resolutions
# ============================================================================

SIM_PATHS = {
    2500: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG/output',
        'dmo_mass': 0.0047271638660809,
        'hydro_dm_mass': 0.00398342749867548,
    },
    1250: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG/output',
        'dmo_mass': 0.0378173109,
        'hydro_dm_mass': 0.0318674199,
    },
    625: {
        'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG_DM/output',
        'hydro': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG/output',
        'dmo_mass': 0.3025384873,
        'hydro_dm_mass': 0.2549393594,
    },
}

CONFIG = {
    'box_size': 205.0,  # Mpc/h
    'mass_unit': 1e10,  # Convert to Msun/h
    'output_dir': '/mnt/home/mlee1/ceph/hydro_replace_fields',
    'radius_multiplier': 3,  # Replace within 3 * R_200
}


def pixelize_3d(pos, mass, box_size, grid_res, mas='CIC'):
    """Assign particles to 3D grid using Mass Assignment Scheme."""
    if len(pos) == 0:
        return np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
    
    field = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
    MASL.MA(pos.astype(np.float32),
            field,
            np.float32(box_size),
            MAS=mas,
            W=mass.astype(np.float32),
            verbose=False)
    return field


def load_dmo_files(files):
    """Load DMO particle coordinates from a list of files."""
    coords_list = []
    for filepath in files:
        with h5py.File(filepath, 'r') as f:
            if 'PartType1' not in f:
                continue
            coords = f['PartType1']['Coordinates'][:].astype(np.float32) / 1e3
            coords_list.append(coords)
    
    if len(coords_list) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(coords_list)


def load_hydro_files(files, hydro_dm_mass, mass_unit):
    """Load hydro particle coordinates and masses from a list of files."""
    coords_list = []
    masses_list = []
    
    for filepath in files:
        with h5py.File(filepath, 'r') as f:
            # Gas
            if 'PartType0' in f:
                coords_list.append(f['PartType0']['Coordinates'][:].astype(np.float32) / 1e3)
                masses_list.append(f['PartType0']['Masses'][:].astype(np.float32) * mass_unit)
            
            # DM (in hydro sim)
            if 'PartType1' in f:
                c = f['PartType1']['Coordinates'][:].astype(np.float32) / 1e3
                m = np.ones(len(c), dtype=np.float32) * hydro_dm_mass * mass_unit
                coords_list.append(c)
                masses_list.append(m)
            
            # Stars
            if 'PartType4' in f:
                coords_list.append(f['PartType4']['Coordinates'][:].astype(np.float32) / 1e3)
                masses_list.append(f['PartType4']['Masses'][:].astype(np.float32) * mass_unit)
    
    if len(coords_list) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
    
    return np.concatenate(coords_list), np.concatenate(masses_list)


def generate_dmo_field(snapNum, sim_config, grid_res):
    """Generate full DMO density field."""
    if rank == 0:
        print(f"\n=== Generating DMO field for snapshot {snapNum} ===")
    
    dmo_basePath = sim_config['dmo']
    dmo_mass = sim_config['dmo_mass']
    
    # Get file list
    snap_dir = f"{dmo_basePath}/snapdir_{snapNum:03d}/"
    all_files = sorted(glob.glob(f"{snap_dir}/snap_{snapNum:03d}.*.hdf5"))
    my_files = [f for i, f in enumerate(all_files) if i % size == rank]
    
    if rank == 0:
        print(f"  Total files: {len(all_files)}, this rank: {len(my_files)}")
    
    # Load particles
    coords = load_dmo_files(my_files)
    masses = np.ones(len(coords), dtype=np.float32) * dmo_mass * CONFIG['mass_unit']
    
    if rank == 0:
        print(f"  Rank 0 loaded {len(coords):,} particles")
    
    # Pixelize local data
    local_field = pixelize_3d(coords, masses, CONFIG['box_size'], grid_res)
    
    # Reduce to rank 0
    if rank == 0:
        global_field = np.zeros_like(local_field)
    else:
        global_field = None
    
    comm.Reduce(local_field, global_field, op=MPI.SUM, root=0)
    
    return global_field


def generate_hydro_field(snapNum, sim_config, grid_res):
    """Generate full Hydro density field."""
    if rank == 0:
        print(f"\n=== Generating Hydro field for snapshot {snapNum} ===")
    
    hydro_basePath = sim_config['hydro']
    hydro_dm_mass = sim_config['hydro_dm_mass']
    
    # Get file list
    snap_dir = f"{hydro_basePath}/snapdir_{snapNum:03d}/"
    all_files = sorted(glob.glob(f"{snap_dir}/snap_{snapNum:03d}.*.hdf5"))
    my_files = [f for i, f in enumerate(all_files) if i % size == rank]
    
    if rank == 0:
        print(f"  Total files: {len(all_files)}, this rank: {len(my_files)}")
    
    # Load particles
    coords, masses = load_hydro_files(my_files, hydro_dm_mass, CONFIG['mass_unit'])
    
    if rank == 0:
        print(f"  Rank 0 loaded {len(coords):,} particles")
    
    # Pixelize local data
    local_field = pixelize_3d(coords, masses, CONFIG['box_size'], grid_res)
    
    # Reduce to rank 0
    if rank == 0:
        global_field = np.zeros_like(local_field)
    else:
        global_field = None
    
    comm.Reduce(local_field, global_field, op=MPI.SUM, root=0)
    
    return global_field


def generate_replace_field(snapNum, sim_config, grid_res, log_mass_min=13.0):
    """
    Generate replacement field: DMO with halo regions replaced by hydro.
    """
    if rank == 0:
        print(f"\n=== Generating Replace field for snapshot {snapNum} ===")
        print(f"  Mass cut: log10(M) > {log_mass_min}")
    
    dmo_basePath = sim_config['dmo']
    hydro_basePath = sim_config['hydro']
    dmo_mass = sim_config['dmo_mass']
    hydro_dm_mass = sim_config['hydro_dm_mass']
    
    # Load halo catalog (all ranks)
    halo_info = groupcat.loadHalos(
        dmo_basePath, snapNum,
        fields=['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos']
    )
    
    # Select halos above mass cut
    masses = halo_info['Group_M_Crit200'] * CONFIG['mass_unit']
    mask = np.log10(masses) >= log_mass_min
    
    halo_positions = halo_info['GroupPos'][mask] / 1e3  # kpc -> Mpc
    halo_radii = halo_info['Group_R_Crit200'][mask] / 1e3  # kpc -> Mpc
    
    if rank == 0:
        print(f"  Selected {mask.sum()} halos above mass cut")
    
    # Get file lists
    dmo_dir = f"{dmo_basePath}/snapdir_{snapNum:03d}/"
    hydro_dir = f"{hydro_basePath}/snapdir_{snapNum:03d}/"
    
    dmo_files = sorted(glob.glob(f"{dmo_dir}/snap_{snapNum:03d}.*.hdf5"))
    hydro_files = sorted(glob.glob(f"{hydro_dir}/snap_{snapNum:03d}.*.hdf5"))
    
    my_dmo_files = [f for i, f in enumerate(dmo_files) if i % size == rank]
    my_hydro_files = [f for i, f in enumerate(hydro_files) if i % size == rank]
    
    # Load particles
    dmo_coords = load_dmo_files(my_dmo_files)
    dmo_masses = np.ones(len(dmo_coords), dtype=np.float32) * dmo_mass * CONFIG['mass_unit']
    
    hydro_coords, hydro_masses = load_hydro_files(my_hydro_files, hydro_dm_mass, CONFIG['mass_unit'])
    
    if rank == 0:
        print(f"  Rank 0: {len(dmo_coords):,} DMO, {len(hydro_coords):,} hydro particles")
    
    # Build KD-trees
    dmo_tree = cKDTree(dmo_coords) if len(dmo_coords) > 0 else None
    hydro_tree = cKDTree(hydro_coords) if len(hydro_coords) > 0 else None
    
    # Create masks
    radius_mult = CONFIG['radius_multiplier']
    
    # DMO: keep everything EXCEPT within halo regions
    dmo_keep_mask = np.ones(len(dmo_coords), dtype=bool)
    if dmo_tree is not None:
        for pos, r200 in zip(halo_positions, halo_radii):
            idx = dmo_tree.query_ball_point(pos, radius_mult * r200)
            dmo_keep_mask[idx] = False
    
    # Hydro: keep ONLY within halo regions
    hydro_keep_mask = np.zeros(len(hydro_coords), dtype=bool)
    if hydro_tree is not None:
        for pos, r200 in zip(halo_positions, halo_radii):
            idx = hydro_tree.query_ball_point(pos, radius_mult * r200)
            hydro_keep_mask[idx] = True
    
    if rank == 0:
        print(f"  Rank 0: DMO kept {dmo_keep_mask.sum():,}, Hydro extracted {hydro_keep_mask.sum():,}")
    
    # Pixelize both components
    local_dmo_field = pixelize_3d(
        dmo_coords[dmo_keep_mask], dmo_masses[dmo_keep_mask],
        CONFIG['box_size'], grid_res
    )
    local_hydro_field = pixelize_3d(
        hydro_coords[hydro_keep_mask], hydro_masses[hydro_keep_mask],
        CONFIG['box_size'], grid_res
    )
    
    # Reduce both to rank 0
    if rank == 0:
        global_dmo = np.zeros_like(local_dmo_field)
        global_hydro = np.zeros_like(local_hydro_field)
    else:
        global_dmo = None
        global_hydro = None
    
    comm.Reduce(local_dmo_field, global_dmo, op=MPI.SUM, root=0)
    comm.Reduce(local_hydro_field, global_hydro, op=MPI.SUM, root=0)
    
    # Combine on rank 0
    if rank == 0:
        replace_field = global_dmo + global_hydro
        return replace_field, global_dmo, global_hydro
    return None, None, None


def save_field(field, filename, metadata=None):
    """Save 3D field to compressed npz file."""
    if field is None:
        return
    
    save_dict = {'field': field}
    if metadata:
        save_dict.update(metadata)
    
    np.savez_compressed(filename, **save_dict)
    print(f"  Saved: {filename} ({field.nbytes / 1e9:.2f} GB uncompressed)")


def main():
    parser = argparse.ArgumentParser(description='Generate 3D density fields')
    parser.add_argument('--snap', type=int, required=True, help='Snapshot number')
    parser.add_argument('--sim-res', type=int, required=True,
                        choices=[625, 1250, 2500], help='Simulation resolution (particles per side)')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'dmo', 'hydro', 'replace'],
                        help='Which fields to generate')
    parser.add_argument('--mass-min', type=float, default=13.0,
                        help='log10(M_min) for replacement halos')
    parser.add_argument('--grid-res', type=int, default=1024,
                        help='Grid resolution for pixelization')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Get simulation config
    if args.sim_res not in SIM_PATHS:
        raise ValueError(f"Unknown resolution: {args.sim_res}")
    sim_config = SIM_PATHS[args.sim_res]
    
    # Output directory
    output_dir = args.output_dir or CONFIG['output_dir']
    snap_dir = f"{output_dir}/L205n{args.sim_res}TNG/snap{args.snap:03d}"
    
    if rank == 0:
        os.makedirs(snap_dir, exist_ok=True)
        print("=" * 70)
        print(f"Generating fields for snapshot {args.snap}")
        print(f"Simulation: L205n{args.sim_res}TNG")
        print(f"Grid resolution: {args.grid_res}")
        print(f"Output: {snap_dir}")
        print("=" * 70)
    
    comm.Barrier()
    
    metadata = {
        'box_size': CONFIG['box_size'],
        'grid_resolution': args.grid_res,
        'sim_resolution': args.sim_res,
        'snapshot': args.snap,
    }
    
    # Generate requested fields
    if args.mode in ['all', 'dmo']:
        field = generate_dmo_field(args.snap, sim_config, args.grid_res)
        if rank == 0:
            save_field(field, f"{snap_dir}/dmo.npz", metadata)
    
    if args.mode in ['all', 'hydro']:
        field = generate_hydro_field(args.snap, sim_config, args.grid_res)
        if rank == 0:
            save_field(field, f"{snap_dir}/hydro.npz", metadata)
    
    if args.mode in ['all', 'replace']:
        replace_field, dmo_excl, hydro_incl = generate_replace_field(
            args.snap, sim_config, args.grid_res, log_mass_min=args.mass_min
        )
        if rank == 0:
            meta = metadata.copy()
            meta['log_mass_min'] = args.mass_min
            meta['radius_multiplier'] = CONFIG['radius_multiplier']
            save_field(replace_field, f"{snap_dir}/replace_gt{args.mass_min:.1f}.npz", meta)
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Done!")
        print("=" * 70)


if __name__ == "__main__":
    main()
