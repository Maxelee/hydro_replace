#!/usr/bin/env python
"""
Generate lens planes with halo replacement for ray-tracing.

This script generates lens planes where DMO halos are replaced with their
Hydro counterparts within a configurable mass range and radius.

Usage:
    # Replace halos with M > 10^12.5
    mpirun -np 16 python generate_lensplanes_replace.py --sim-res 625 --snap 99 \\
        --mass-min 12.5

    # Replace only halos in mass range 10^12.5 - 10^14
    mpirun -np 16 python generate_lensplanes_replace.py --sim-res 625 --snap 99 \\
        --mass-min 12.5 --mass-max 14.0

    # Custom radius factor (default 5×R200)
    mpirun -np 16 python generate_lensplanes_replace.py --sim-res 625 --snap 99 \\
        --mass-min 12.5 --radius-factor 3.0

    # All 20 ray-tracing snapshots
    mpirun -np 16 python generate_lensplanes_replace.py --sim-res 625 --snap rt \\
        --mass-min 12.5

Output: Binary density planes for lux ray-tracing in:
    {output_dir}/L205n{res}TNG/seed{seed}/replace_Mgt{mass_min:.1f}/density{plane:02d}.dat
"""

import numpy as np
import h5py
import glob
import argparse
import os
import struct
import time
from mpi4py import MPI
import MAS_library as MASL
import pyccl as ccl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================================
# Configuration
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

# Snapshot configuration for ray-tracing
SNAPSHOT_CONFIG = [
    (96, 0.04, False),  (90, 0.15, False), (85, 0.27, False), (80, 0.40, False),
    (76, 0.50, False),  (71, 0.64, False), (67, 0.78, False), (63, 0.93, False),
    (59, 1.07, False),  (56, 1.18, False), (52, 1.36, True),  (49, 1.50, True),
    (46, 1.65, True),   (43, 1.82, True),  (41, 1.93, True),  (38, 2.12, True),
    (35, 2.32, True),   (33, 2.49, True),  (31, 2.68, True),  (29, 2.87, True),
]
SNAP_TO_IDX = {cfg[0]: i for i, cfg in enumerate(SNAPSHOT_CONFIG)}

CONFIG = {
    'box_size': 205.0,  # Mpc/h
    'mass_unit': 1e10,  # Convert to Msun/h
    'grid_res': 4096,   # Lens plane resolution
    'planes_per_snapshot': 2,
    'output_base': '/mnt/home/mlee1/ceph/hydro_replace_lensplanes',
    'cache_base': '/mnt/home/mlee1/ceph/hydro_replace_fields',
}

# Cosmology
COSMO = ccl.Cosmology(
    Omega_c=0.3089 - 0.0486, Omega_b=0.0486, h=0.6774,
    sigma8=0.8158, n_s=0.9649, matter_power_spectrum='linear'
)
Omega_m = COSMO.cosmo.params.Omega_m
rho_c0 = 27.7536627  # Critical density in 10^10 Msun/h / (Mpc/h)^3


# ============================================================================
# Randomization (matches lux)
# ============================================================================

class RandomizationState:
    """Generate consistent randomization matching lux."""
    
    def __init__(self, seed=2020):
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.n_snapshots = len(SNAPSHOT_CONFIG)
        
        self.proj_dirs = self.rng.integers(0, 3, size=self.n_snapshots)
        self.displacements = self.rng.uniform(0, CONFIG['box_size'], 
                                               size=(self.n_snapshots, 3))
        self.flips = self.rng.integers(0, 2, size=(self.n_snapshots, 3)).astype(bool)
    
    def get_params(self, snap_idx):
        return {
            'proj_dir': self.proj_dirs[snap_idx],
            'displacement': self.displacements[snap_idx],
            'flip': self.flips[snap_idx],
        }


def apply_randomization(coords, params, box_size):
    """Apply translation and flip transformations."""
    coords_out = coords.copy()
    coords_out = coords_out + params['displacement']
    for axis in range(3):
        if params['flip'][axis]:
            coords_out[:, axis] = -coords_out[:, axis]
    coords_out = np.mod(coords_out, box_size)
    return coords_out


def get_projection_axes(proj_dir):
    """Get 2D axes for projection direction."""
    if proj_dir == 0:
        return 1, 2
    elif proj_dir == 1:
        return 2, 0
    else:
        return 0, 1


# ============================================================================
# Particle Loading
# ============================================================================

def load_dmo_particles(basePath, snapNum, my_files, dmo_mass, mass_unit):
    """Load DMO particle coordinates, masses, and IDs."""
    coords_list = []
    ids_list = []
    
    for filepath in my_files:
        with h5py.File(filepath, 'r') as f:
            if 'PartType1' not in f:
                continue
            coords_list.append(f['PartType1']['Coordinates'][:].astype(np.float32) / 1e3)
            ids_list.append(f['PartType1']['ParticleIDs'][:])
    
    if not coords_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
    
    coords = np.concatenate(coords_list)
    ids = np.concatenate(ids_list)
    masses = np.ones(len(coords), dtype=np.float32) * dmo_mass * mass_unit
    return coords, masses, ids


def load_hydro_particles(basePath, snapNum, my_files, hydro_dm_mass, mass_unit):
    """Load hydro particle coordinates and masses (gas + DM + stars)."""
    coords_list = []
    masses_list = []
    ids_list = []
    
    for filepath in my_files:
        with h5py.File(filepath, 'r') as f:
            # Gas
            if 'PartType0' in f:
                coords_list.append(f['PartType0']['Coordinates'][:].astype(np.float32) / 1e3)
                masses_list.append(f['PartType0']['Masses'][:].astype(np.float32) * mass_unit)
                ids_list.append(f['PartType0']['ParticleIDs'][:])
            
            # DM
            if 'PartType1' in f:
                c = f['PartType1']['Coordinates'][:].astype(np.float32) / 1e3
                coords_list.append(c)
                masses_list.append(np.ones(len(c), dtype=np.float32) * hydro_dm_mass * mass_unit)
                ids_list.append(f['PartType1']['ParticleIDs'][:])
            
            # Stars
            if 'PartType4' in f:
                coords_list.append(f['PartType4']['Coordinates'][:].astype(np.float32) / 1e3)
                masses_list.append(f['PartType4']['Masses'][:].astype(np.float32) * mass_unit)
                ids_list.append(f['PartType4']['ParticleIDs'][:])
    
    if not coords_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
    
    return np.concatenate(coords_list), np.concatenate(masses_list), np.concatenate(ids_list)


# ============================================================================
# Cache Loading
# ============================================================================

def load_particle_cache(snapNum, mass_min, mass_max, radius_factor, sim_res):
    """
    Load particle IDs from cache for halos within mass range.
    
    Returns dict with:
    - halo_indices: array of halo indices
    - dmo_ids: list of DMO particle ID arrays
    - hydro_ids: list of Hydro particle ID arrays  
    - halo_positions: halo center positions (Mpc/h)
    - halo_radii: R200 values (Mpc/h)
    - halo_masses: M200 values (Msun/h)
    """
    cache_file = f'{CONFIG["cache_base"]}/L205n{sim_res}TNG/particle_cache/cache_snap{snapNum:03d}.h5'
    
    if not os.path.exists(cache_file):
        if rank == 0:
            print(f"  Warning: Cache file not found: {cache_file}")
        return None
    
    try:
        with h5py.File(cache_file, 'r') as f:
            # Get halo info
            halo_masses = f['halo_info/masses'][:]  # Msun/h
            halo_positions = f['halo_info/positions_dmo'][:]  # Mpc/h
            halo_radii = f['halo_info/radii_dmo'][:]  # Mpc/h
            
            log_masses = np.log10(halo_masses)
            
            # Apply mass cut
            if mass_max is not None:
                mask = (log_masses >= mass_min) & (log_masses < mass_max)
            else:
                mask = log_masses >= mass_min
            
            halo_indices = np.where(mask)[0]
            
            if len(halo_indices) == 0:
                return None
            
            # Load particle IDs
            dmo_ids = []
            hydro_ids = []
            
            for idx in halo_indices:
                # DMO particles
                dmo_key = f'dmo/halo_{idx}'
                if dmo_key in f:
                    dmo_ids.append(f[dmo_key][:])
                else:
                    dmo_ids.append(np.array([], dtype=np.int64))
                
                # Hydro particles - use hydro_at_dmo (same center as DMO)
                hydro_key = f'hydro_at_dmo/halo_{idx}'
                if hydro_key in f:
                    hydro_ids.append(f[hydro_key][:])
                else:
                    hydro_ids.append(np.array([], dtype=np.int64))
            
            return {
                'halo_indices': halo_indices,
                'dmo_ids': dmo_ids,
                'hydro_ids': hydro_ids,
                'halo_positions': halo_positions[mask],
                'halo_radii': halo_radii[mask],
                'halo_masses': halo_masses[mask],
            }
    
    except Exception as e:
        if rank == 0:
            print(f"  Error loading cache: {e}")
        return None


# ============================================================================
# 2D Projection
# ============================================================================

def project_to_2d(coords, masses, box_size, grid_res, proj_dir, plane_idx, pps):
    """Project particles to 2D surface density for a lens plane slice."""
    if len(coords) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32)
    
    ax1, ax2 = get_projection_axes(proj_dir)
    
    # Slice boundaries
    slice_thickness = box_size / pps
    z_min = plane_idx * slice_thickness
    z_max = (plane_idx + 1) * slice_thickness
    
    # Select particles in slice
    z_coords = coords[:, proj_dir]
    mask = (z_coords >= z_min) & (z_coords < z_max)
    
    if mask.sum() == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32)
    
    # Get 2D positions
    pos_2d = np.column_stack([coords[mask, ax1], coords[mask, ax2]]).astype(np.float32)
    pos_2d = np.ascontiguousarray(np.mod(pos_2d, box_size))
    
    # Project using TSC
    field = np.zeros((grid_res, grid_res), dtype=np.float32)
    MASL.MA(pos_2d, field, np.float32(box_size), MAS='TSC',
            W=masses[mask].astype(np.float32), verbose=False)
    
    return field


# ============================================================================
# Binary Output
# ============================================================================

def write_density_plane(filename, delta_dz, grid_size):
    """Write density plane in lux binary format."""
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', grid_size))
        f.write(delta_dz.astype(np.float64).tobytes())
        f.write(struct.pack('i', grid_size))


def write_config(output_dir, snap_config, random_state, grid_res, pps):
    """Write lux config.dat file."""
    Ns = len(snap_config)
    Np = Ns * pps
    L = CONFIG['box_size']
    
    a = np.zeros(Np + 1)
    chi = np.zeros(Np + 1)
    chi_out = np.zeros(Np)
    Ll = np.zeros(Ns)
    Lt = np.zeros(Ns)
    
    a[0] = 1.0
    chi[0] = 0.0
    
    for s, (snap, z, stack) in enumerate(snap_config):
        Ll[s] = L
        Lt[s] = 2 * L if stack else L
    
    cumulative_dist = 0.0
    for p in range(1, Np + 1):
        s = (p - 1) // pps
        plane_in_snap = (p - 1) % pps
        chi[p] = cumulative_dist + Ll[s] / pps * (plane_in_snap + 0.5)
        if plane_in_snap == pps - 1:
            cumulative_dist += Ll[s]
    
    cumulative_dist = 0.0
    for p in range(Np):
        s = p // pps
        plane_in_snap = p % pps
        chi_out[p] = cumulative_dist + Ll[s] / pps * (plane_in_snap + 1)
        if plane_in_snap == pps - 1:
            cumulative_dist += Ll[s]
    
    for p in range(1, Np + 1):
        a[p] = ccl.scale_factor_of_chi(COSMO, chi[p])
    
    proj_dirs = random_state.proj_dirs.astype(np.int32)
    disp = random_state.displacements.flatten()
    flip = random_state.flips.flatten().astype(np.uint8)
    
    config_file = os.path.join(output_dir, 'config.dat')
    with open(config_file, 'wb') as f:
        f.write(struct.pack('i', Np))
        f.write(struct.pack('i', Ns))
        f.write(a.astype(np.float64).tobytes())
        f.write(chi.astype(np.float64).tobytes())
        f.write(chi_out.astype(np.float64).tobytes())
        f.write(Ll.astype(np.float64).tobytes())
        f.write(Lt.astype(np.float64).tobytes())
        f.write(proj_dirs.tobytes())
        f.write(disp.astype(np.float64).tobytes())
        f.write(flip.tobytes())
    
    if rank == 0:
        print(f"  Wrote config: {config_file}")


# ============================================================================
# Replace Coordinates
# ============================================================================

def build_replace_particles(dmo_coords, dmo_masses, dmo_ids,
                            hydro_coords, hydro_masses, hydro_ids,
                            cache, id_to_idx_dmo, id_to_idx_hydro):
    """
    Build replace coordinates: DMO base + swap halo particles for Hydro.
    
    For each halo in cache:
      1. Remove DMO particles (set mass to 0 or exclude)
      2. Add Hydro particles instead
    
    Returns:
        replace_coords, replace_masses
    """
    # Start with DMO as base
    replace_coords = dmo_coords.copy()
    replace_masses = dmo_masses.copy()
    
    # Track which DMO particles to exclude
    dmo_exclude = np.zeros(len(dmo_coords), dtype=bool)
    
    # Collect hydro particles to add
    hydro_coords_add = []
    hydro_masses_add = []
    
    n_halos = len(cache['halo_indices'])
    
    for i in range(n_halos):
        dmo_pids = cache['dmo_ids'][i]
        hydro_pids = cache['hydro_ids'][i]
        
        # Mark DMO particles for exclusion
        for pid in dmo_pids:
            if pid in id_to_idx_dmo:
                dmo_exclude[id_to_idx_dmo[pid]] = True
        
        # Collect hydro particles
        for pid in hydro_pids:
            if pid in id_to_idx_hydro:
                idx = id_to_idx_hydro[pid]
                hydro_coords_add.append(hydro_coords[idx])
                hydro_masses_add.append(hydro_masses[idx])
    
    # Build final arrays
    keep_mask = ~dmo_exclude
    replace_coords = replace_coords[keep_mask]
    replace_masses = replace_masses[keep_mask]
    
    if hydro_coords_add:
        hydro_coords_add = np.array(hydro_coords_add, dtype=np.float32)
        hydro_masses_add = np.array(hydro_masses_add, dtype=np.float32)
        replace_coords = np.concatenate([replace_coords, hydro_coords_add])
        replace_masses = np.concatenate([replace_masses, hydro_masses_add])
    
    return replace_coords, replace_masses


# ============================================================================
# Main Processing
# ============================================================================

def process_snapshot(args, snap_idx, snap_info, seed):
    """Process a single snapshot for a given random seed."""
    snapNum, redshift, stack = snap_info
    sim_config = SIM_PATHS[args.sim_res]
    
    box_size = CONFIG['box_size']
    grid_res = args.grid_res
    pps = CONFIG['planes_per_snapshot']
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Snapshot {snapNum} (z={redshift:.2f}), Seed {seed}")
        print(f"{'='*70}")
    
    # Load particle cache
    if rank == 0:
        print("  Loading particle cache...")
    cache = load_particle_cache(snapNum, args.mass_min, args.mass_max, 
                                 args.radius_factor, args.sim_res)
    
    if cache is None:
        if rank == 0:
            print("  ERROR: No cache available, cannot generate replace lens planes")
        return False
    
    if rank == 0:
        print(f"  Found {len(cache['halo_indices'])} halos in mass range")
    
    # Load particles
    if rank == 0:
        print("  Loading particles...")
        t0 = time.time()
    
    dmo_dir = f"{sim_config['dmo']}/snapdir_{snapNum:03d}/"
    hydro_dir = f"{sim_config['hydro']}/snapdir_{snapNum:03d}/"
    
    dmo_files = sorted(glob.glob(f"{dmo_dir}/snap_{snapNum:03d}.*.hdf5"))
    hydro_files = sorted(glob.glob(f"{hydro_dir}/snap_{snapNum:03d}.*.hdf5"))
    
    my_dmo_files = [f for i, f in enumerate(dmo_files) if i % size == rank]
    my_hydro_files = [f for i, f in enumerate(hydro_files) if i % size == rank]
    
    dmo_coords, dmo_masses, dmo_ids = load_dmo_particles(
        None, snapNum, my_dmo_files, sim_config['dmo_mass'], CONFIG['mass_unit']
    )
    hydro_coords, hydro_masses, hydro_ids = load_hydro_particles(
        None, snapNum, my_hydro_files, sim_config['hydro_dm_mass'], CONFIG['mass_unit']
    )
    
    if rank == 0:
        print(f"  Loaded particles in {time.time()-t0:.1f}s")
    
    # Build ID mappings
    if rank == 0:
        print("  Building ID→index mappings...")
    id_to_idx_dmo = {pid: idx for idx, pid in enumerate(dmo_ids)}
    id_to_idx_hydro = {pid: idx for idx, pid in enumerate(hydro_ids)}
    
    # Setup randomization
    random_state = RandomizationState(seed=seed)
    rand_params = random_state.get_params(snap_idx)
    
    # Apply randomization
    dmo_coords_rand = apply_randomization(dmo_coords, rand_params, box_size)
    hydro_coords_rand = apply_randomization(hydro_coords, rand_params, box_size)
    
    # Build replace coordinates
    if rank == 0:
        print("  Building replace particles...")
        t0 = time.time()
    
    replace_coords, replace_masses = build_replace_particles(
        dmo_coords_rand, dmo_masses, dmo_ids,
        hydro_coords_rand, hydro_masses, hydro_ids,
        cache, id_to_idx_dmo, id_to_idx_hydro
    )
    
    if rank == 0:
        print(f"  Built replace particles in {time.time()-t0:.1f}s")
        print(f"  Replace: {len(replace_coords):,} particles (vs DMO: {len(dmo_coords):,})")
    
    # Output directory
    if args.mass_max is not None:
        dir_name = f"replace_M{args.mass_min:.1f}-{args.mass_max:.1f}"
    else:
        dir_name = f"replace_Mgt{args.mass_min:.1f}"
    
    output_dir = os.path.join(
        args.output_dir, f'L205n{args.sim_res}TNG',
        f'seed{seed}', dir_name
    )
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    comm.Barrier()
    
    # Compute mean surface density
    slice_thickness = box_size / pps
    sigma_mean = Omega_m * rho_c0 * CONFIG['mass_unit'] * slice_thickness
    
    # Generate lens planes
    proj_dir = rand_params['proj_dir']
    if rank == 0:
        print(f"  Projecting along axis {proj_dir} ({'xyz'[proj_dir]})")
    
    for plane_idx in range(pps):
        plane_num = snap_idx * pps + plane_idx + 1
        output_file = os.path.join(output_dir, f'density{plane_num:02d}.dat')
        
        if args.skip_existing and os.path.exists(output_file):
            continue
        
        # Project to 2D
        field_local = project_to_2d(
            replace_coords, replace_masses, box_size, grid_res,
            proj_dir, plane_idx, pps
        )
        
        # Reduce across ranks
        field_global = np.zeros_like(field_local)
        comm.Reduce(field_local.copy(), field_global, op=MPI.SUM, root=0)
        
        # Write on rank 0
        if rank == 0:
            # Convert to delta * dz
            delta_dz = (field_global / (box_size / grid_res)**2 / sigma_mean - 1.0) * slice_thickness
            delta_dz = np.nan_to_num(delta_dz, nan=0.0)
            write_density_plane(output_file, delta_dz, grid_res)
            print(f"    Wrote plane {plane_num}: {output_file}")
    
    # Write config if this is the last snapshot
    if rank == 0 and snap_idx == len(SNAPSHOT_CONFIG) - 1:
        write_config(output_dir, SNAPSHOT_CONFIG, random_state, grid_res, pps)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate lens planes with halo replacement',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--sim-res', type=int, required=True, choices=[625, 1250, 2500])
    parser.add_argument('--snap', type=str, default='99',
                       help='Snapshot: number, "rt" (20 ray-tracing), or comma-separated')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed')
    
    # Mass selection
    parser.add_argument('--mass-min', type=float, default=12.5,
                       help='Minimum log10(M200) for replacement')
    parser.add_argument('--mass-max', type=float, default=None,
                       help='Maximum log10(M200) for replacement (optional)')
    parser.add_argument('--radius-factor', type=float, default=5.0,
                       help='Replacement radius as multiple of R200')
    
    # Output control
    parser.add_argument('--grid-res', type=int, default=4096, help='Grid resolution')
    parser.add_argument('--output-dir', type=str, default=CONFIG['output_base'])
    parser.add_argument('--skip-existing', action='store_true')
    
    args = parser.parse_args()
    
    # Parse snapshots
    if args.snap == 'rt':
        snapshots = [cfg[0] for cfg in SNAPSHOT_CONFIG]
    else:
        snapshots = [int(s) for s in args.snap.split(',')]
    
    if rank == 0:
        print("="*70)
        print("LENS PLANE GENERATION WITH HALO REPLACEMENT")
        print("="*70)
        print(f"Simulation: L205n{args.sim_res}TNG")
        print(f"Snapshots:  {snapshots}")
        print(f"Seed:       {args.seed}")
        print(f"Mass range: 10^{args.mass_min} - 10^{args.mass_max or '∞'} Msun/h")
        print(f"Radius:     {args.radius_factor}×R200")
        print(f"Grid:       {args.grid_res}")
        print(f"MPI ranks:  {size}")
        print("="*70)
    
    t_start = time.time()
    
    for snap in snapshots:
        if snap in SNAP_TO_IDX:
            snap_idx = SNAP_TO_IDX[snap]
            snap_info = SNAPSHOT_CONFIG[snap_idx]
        else:
            # Single snapshot not in ray-tracing config
            snap_idx = 0
            snap_info = (snap, 0.0, False)
        
        process_snapshot(args, snap_idx, snap_info, args.seed)
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"COMPLETE - Total time: {(time.time()-t_start)/60:.1f} minutes")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
