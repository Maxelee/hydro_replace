#!/usr/bin/env python
"""
Generate BCM lens planes for ray-tracing.

This applies BCM displacement to DMO particles within halos and generates
lens planes compatible with the lux ray-tracing code.

The approach:
1. Load DMO particles
2. Apply BCM displacement to particles within radius_factor × R200 of halos
3. Project to 2D with randomization matching lux
4. Write binary lens plane files

Usage:
    mpirun -np 4 python generate_lensplanes_bcm.py --sim-res 625 --snap 99 --bcm-model Arico20
    mpirun -np 4 python generate_lensplanes_bcm.py --sim-res 625 --snap rt --bcm-model Arico20
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
import BaryonForge as bfg

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================================
# Configuration
# ============================================================================

h = 0.6774

SIM_PATHS = {
    2500: {'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG_DM/output', 'dmo_mass': 0.0047271638660809},
    1250: {'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n1250TNG_DM/output', 'dmo_mass': 0.0378173109},
    625: {'dmo': '/mnt/sdceph/users/sgenel/IllustrisTNG/L205n625TNG_DM/output', 'dmo_mass': 0.3025384873},
}

SNAPSHOT_CONFIG = [
    (96, 0.02, False), (90, 0.10, False), (85, 0.18, False), (80, 0.27, False),
    (76, 0.35, False), (71, 0.46, False), (67, 0.55, False), (63, 0.65, False),
    (59, 0.76, False), (56, 0.85, False), (52, 0.97, True),  (49, 1.08, True),
    (46, 1.21, True),  (43, 1.36, True),  (41, 1.47, True),  (38, 1.63, True),
    (35, 1.82, True),  (33, 1.97, True),  (31, 2.14, True),  (29, 2.32, True),
]
SNAP_TO_IDX = {cfg[0]: i for i, cfg in enumerate(SNAPSHOT_CONFIG)}
RT_SNAPSHOTS = [cfg[0] for cfg in SNAPSHOT_CONFIG]

CONFIG = {
    'box_size': 205.0,
    'mass_unit': 1e10,
    'grid_res': 4096,
    'planes_per_snapshot': 2,
    'output_base': '/mnt/home/mlee1/ceph/hydro_replace_lensplanes',
    'cache_base': '/mnt/home/mlee1/ceph/hydro_replace_fields',
}

COSMO = ccl.Cosmology(
    Omega_c=0.2589, Omega_b=0.0486, h=h,
    sigma8=0.8159, n_s=0.9667,
    transfer_function='boltzmann_camb',
    matter_power_spectrum='halofit'
)
Omega_m = COSMO.cosmo.params.Omega_m
rho_c0 = 27.7536627

BCM_PARAMS = {
    'Arico20': dict(
        M_c=3.3e13, M1_0=8.63e11, eta=0.54, beta=0.12, mu=0.31, M_inn=3.3e13,
        theta_inn=0.1, theta_out=3, epsilon_h=0.015, alpha_g=2,
        epsilon_hydro=np.sqrt(5), theta_rg=0.3, sigma_rg=0.1,
        a=0.3, n=2, p=0.3, q=0.707,
        alpha_fsat=1, M1_fsat=1, delta_fsat=1, gamma_fsat=1, eps_fsat=1,
        M_r=1e16, beta_r=2, A_nt=0.495, alpha_nt=0.1,
    ),
    'Schneider19': dict(
        theta_ej=4, theta_co=0.1, M_c=1e14/h, mu_beta=0.4,
        gamma=2, delta=7, eta=0.3, eta_delta=0.3, tau=-1.5, tau_delta=0,
        A=0.09/2, M1=2.5e11/h, epsilon_h=0.015,
        a=0.3, n=2, epsilon=4, p=0.3, q=0.707,
    ),
    'Schneider25': dict(
        M_c=1e15, mu=0.8,
        q0=0.075, q1=0.25, q2=0.7, nu_q0=0, nu_q1=1, nu_q2=0, nstep=3/2,
        theta_c=0.3, nu_theta_c=1/2, c_iga=0.1, nu_c_iga=3/2, r_min_iga=1e-3,
        alpha=1, gamma=3/2, delta=7,
        tau=-1.376, tau_delta=0, Mstar=3e11, Nstar=0.03,
        eta=0.1, eta_delta=0.22, epsilon_cga=0.03,
        alpha_nt=0.1, nu_nt=0.5, gamma_nt=0.8, mean_molecular_weight=0.6125,
    ),
}


def build_cosmodict(cosmo):
    return {
        'Omega_m': cosmo.cosmo.params.Omega_m,
        'Omega_b': cosmo.cosmo.params.Omega_b,
        'sigma8': cosmo.cosmo.params.sigma8,
        'h': cosmo.cosmo.params.h,
        'n_s': cosmo.cosmo.params.n_s,
        'w0': cosmo.cosmo.params.w0,
        'wa': cosmo.cosmo.params.wa,
    }


def setup_bcm_model(model_name):
    params = BCM_PARAMS[model_name]
    if model_name == 'Arico20':
        DMB = bfg.Profiles.Arico20.DarkMatterBaryon(**params)
        DMO = bfg.Profiles.Arico20.DarkMatterOnly(**params)
    elif model_name == 'Schneider19':
        DMB = bfg.Profiles.Schneider19.DarkMatterBaryon(**params)
        DMO = bfg.Profiles.Schneider19.DarkMatterOnly(**params)
    elif model_name == 'Schneider25':
        DMB = bfg.Profiles.Schneider25.DarkMatterBaryon(**params)
        DMO = bfg.Profiles.Schneider25.DarkMatterOnly(**params)
    else:
        raise ValueError(f"Unknown BCM model: {model_name}")
    
    Displacement = bfg.Baryonification3D(DMO, DMB, COSMO, N_int=50_000)
    Displacement.setup_interpolator(z_min=0, z_max=3, z_linear_sampling=True, N_samples_R=10000, Rdelta_sampling=True)
    return Displacement


class RandomizationState:
    def __init__(self, seed=2020):
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.n_snapshots = len(SNAPSHOT_CONFIG)
        self.proj_dirs = self.rng.integers(0, 3, size=self.n_snapshots)
        self.displacements = self.rng.uniform(0, CONFIG['box_size'], size=(self.n_snapshots, 3))
        self.flips = self.rng.integers(0, 2, size=(self.n_snapshots, 3)).astype(bool)
    
    def get_params(self, snap_idx):
        return {
            'proj_dir': self.proj_dirs[snap_idx],
            'displacement': self.displacements[snap_idx],
            'flip': self.flips[snap_idx],
        }


def apply_randomization(coords, params, box_size):
    coords_out = coords.copy()
    coords_out = coords_out + params['displacement']
    for axis in range(3):
        if params['flip'][axis]:
            coords_out[:, axis] = -coords_out[:, axis]
    return np.mod(coords_out, box_size)


def get_projection_axes(proj_dir):
    if proj_dir == 0: return 1, 2
    elif proj_dir == 1: return 2, 0
    else: return 0, 1


def load_dmo_particles(basePath, snapNum, my_files, dmo_mass, mass_unit):
    coords_list, ids_list = [], []
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


def load_halo_info(snapNum, mass_min, mass_max, sim_res):
    cache_file = f'{CONFIG["cache_base"]}/L205n{sim_res}TNG/particle_cache/cache_snap{snapNum:03d}.h5'
    
    if not os.path.exists(cache_file):
        return None
    
    with h5py.File(cache_file, 'r') as f:
        masses = f['halo_info/masses'][:]
        positions = f['halo_info/positions_dmo'][:]
        radii = f['halo_info/radii_dmo'][:]
        
        log_masses = np.log10(masses)
        mask = log_masses >= mass_min
        if mass_max:
            mask &= log_masses < mass_max
        
        return {
            'positions': positions[mask],
            'radii': radii[mask],
            'masses': masses[mask],
        }


def apply_bcm_to_particles(coords, masses, halo_info, radius_factor, redshift, displacement_model, cosmo_dict):
    """Apply BCM displacement to particles near halos."""
    displaced = coords.copy()
    box_size = CONFIG['box_size']
    
    for i in range(len(halo_info['positions'])):
        halo_pos = halo_info['positions'][i]
        halo_r200 = halo_info['radii'][i]
        halo_mass = halo_info['masses'][i]
        
        # Find particles in region
        dx = coords - halo_pos
        dx = np.where(dx > box_size/2, dx - box_size, dx)
        dx = np.where(dx < -box_size/2, dx + box_size, dx)
        r = np.linalg.norm(dx, axis=1)
        in_region = r < radius_factor * halo_r200
        
        if not in_region.any():
            continue
        
        try:
            region_coords = coords[in_region]
            region_masses = masses[in_region]
            
            Snap = bfg.ParticleSnapshot(
                x=region_coords[:, 0], y=region_coords[:, 1], z=region_coords[:, 2],
                L=box_size / h, redshift=redshift, cosmo=cosmo_dict,
                M=region_masses[0] if len(region_masses) > 0 else 1e10
            )
            HCat = bfg.HaloNDCatalog(
                x=np.array([halo_pos[0]]), y=np.array([halo_pos[1]]), z=np.array([halo_pos[2]]),
                M_200c=np.array([halo_mass]), redshift=redshift, cosmo=cosmo_dict, is_central=np.array([True])
            )
            
            disp_x, disp_y, disp_z = displacement_model.displace(Snap, HCat, verbose=False)
            
            displaced[in_region, 0] += disp_x
            displaced[in_region, 1] += disp_y
            displaced[in_region, 2] += disp_z
        except:
            pass
    
    return np.mod(displaced, box_size)


def project_to_plane(coords, masses, z_min, z_max, params, grid_res, box_size):
    """Project particles in z-slice to 2D plane."""
    proj_dir = params['proj_dir']
    ax1, ax2 = get_projection_axes(proj_dir)
    
    mask = (coords[:, proj_dir] >= z_min) & (coords[:, proj_dir] < z_max)
    sel_coords = coords[mask]
    sel_masses = masses[mask]
    
    if len(sel_coords) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float64)
    
    pos_2d = np.zeros((len(sel_coords), 3), dtype=np.float32)
    pos_2d[:, 0] = sel_coords[:, ax1]
    pos_2d[:, 1] = sel_coords[:, ax2]
    pos_2d[:, 2] = 0
    
    field = np.zeros((grid_res, grid_res, 1), dtype=np.float32)
    MASL.MA(pos_2d, field, box_size, MAS='TSC', W=sel_masses.astype(np.float32), verbose=False)
    
    return field[:, :, 0].astype(np.float64)


def write_density_plane(filename, delta_dz, grid_size):
    """Write lux binary format."""
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', grid_size))
        f.write(delta_dz.astype(np.float64).tobytes())
        f.write(struct.pack('i', grid_size))


def process_snapshot(snapNum, sim_res, mass_min, mass_max, radius_factor, bcm_model, 
                     displacement_model, cosmo_dict, randomization, output_dir):
    """Process single snapshot to generate lens planes."""
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Processing snapshot {snapNum}")
        print(f"{'='*60}")
    
    snap_idx = SNAP_TO_IDX.get(snapNum)
    if snap_idx is None:
        if rank == 0:
            print(f"  Snapshot {snapNum} not in ray-tracing config, skipping")
        return
    
    redshift = SNAPSHOT_CONFIG[snap_idx][1]
    params = randomization.get_params(snap_idx)
    
    # Load halo info
    if rank == 0:
        halo_info = load_halo_info(snapNum, mass_min, mass_max, sim_res)
        if halo_info is None:
            print(f"  No cache found for snapshot {snapNum}")
            return
        n_halos = len(halo_info['positions'])
        print(f"  {n_halos} halos in mass range")
    else:
        halo_info = None
    
    halo_info = comm.bcast(halo_info, root=0)
    if halo_info is None:
        return
    
    # Load DMO particles
    basePath = SIM_PATHS[sim_res]['dmo']
    dmo_mass = SIM_PATHS[sim_res]['dmo_mass']
    snap_dir = f"{basePath}/snapdir_{snapNum:03d}/"
    all_files = sorted(glob.glob(f"{snap_dir}/snap_{snapNum:03d}.*.hdf5"))
    my_files = [f for i, f in enumerate(all_files) if i % size == rank]
    
    local_coords, local_masses, local_ids = load_dmo_particles(
        basePath, snapNum, my_files, dmo_mass, CONFIG['mass_unit']
    )
    
    local_n = len(local_coords)
    total_n = comm.reduce(local_n, op=MPI.SUM, root=0)
    if rank == 0:
        print(f"  Loaded {total_n:,} DMO particles")
    
    # Apply BCM displacement
    if rank == 0:
        print(f"  Applying BCM ({bcm_model}) displacement...")
    
    local_coords = apply_bcm_to_particles(
        local_coords, local_masses, halo_info, radius_factor,
        redshift, displacement_model, cosmo_dict
    )
    
    # Apply randomization
    local_coords = apply_randomization(local_coords, params, CONFIG['box_size'])
    
    # Generate lens planes
    box_size = CONFIG['box_size']
    grid_res = CONFIG['grid_res']
    n_planes = CONFIG['planes_per_snapshot']
    dz = box_size / n_planes
    
    cell_area = (box_size / grid_res) ** 2
    mean_sigma = Omega_m * rho_c0 * box_size * CONFIG['mass_unit']
    
    for plane_idx in range(n_planes):
        z_min = plane_idx * dz
        z_max = z_min + dz
        
        local_field = project_to_plane(
            local_coords, local_masses, z_min, z_max, params, grid_res, box_size
        )
        
        if rank == 0:
            global_field = np.zeros((grid_res, grid_res), dtype=np.float64)
        else:
            global_field = None
        
        comm.Reduce(local_field, global_field, op=MPI.SUM, root=0)
        
        if rank == 0:
            sigma = global_field / cell_area
            delta = sigma / (mean_sigma / n_planes) - 1
            delta_dz = delta * dz
            delta_dz = np.nan_to_num(delta_dz, nan=0.0)
            
            plane_num = snap_idx * n_planes + plane_idx
            filename = os.path.join(output_dir, f'density{plane_num:02d}.dat')
            write_density_plane(filename, delta_dz, grid_res)
            
            print(f"    Plane {plane_num:02d}: δ range [{delta.min():.3f}, {delta.max():.3f}]")


def write_config(output_dir, sim_res):
    """Write config.dat for lux."""
    config_file = os.path.join(output_dir, 'config.dat')
    with open(config_file, 'w') as f:
        f.write(f"box_size = {CONFIG['box_size']}\n")
        f.write(f"grid_res = {CONFIG['grid_res']}\n")
        f.write(f"Omega_m = {Omega_m}\n")
        f.write(f"h = {h}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate BCM lens planes')
    parser.add_argument('--sim-res', type=int, default=625, choices=[625, 1250, 2500])
    parser.add_argument('--snap', type=str, default='99')
    parser.add_argument('--bcm-model', type=str, default='Arico20', choices=['Arico20', 'Schneider19', 'Schneider25'])
    parser.add_argument('--mass-min', type=float, default=12.5)
    parser.add_argument('--mass-max', type=float, default=None)
    parser.add_argument('--radius-factor', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=2020)
    args = parser.parse_args()
    
    # Parse snapshots
    if args.snap == 'rt':
        snapshots = RT_SNAPSHOTS
    elif args.snap == 'all':
        snapshots = RT_SNAPSHOTS + [99]
    else:
        snapshots = [int(s) for s in args.snap.split(',')]
    
    if rank == 0:
        print("=" * 70)
        print(f"BCM LENS PLANE GENERATION - {args.bcm_model}")
        print("=" * 70)
        print(f"Resolution: L205n{args.sim_res}TNG")
        print(f"BCM Model: {args.bcm_model}")
        print(f"Mass range: 10^{args.mass_min} - 10^{args.mass_max or '∞'}")
        print(f"Radius: {args.radius_factor}×R200")
        print(f"Snapshots: {snapshots}")
        print("=" * 70)
    
    t_start = time.time()
    
    # Setup BCM
    if rank == 0:
        print("\nSetting up BCM model...")
    displacement_model = setup_bcm_model(args.bcm_model)
    cosmo_dict = build_cosmodict(COSMO)
    
    # Setup output directory
    model_lower = args.bcm_model.lower()
    if args.mass_max:
        model_dir = f"bcm_{model_lower}_M{args.mass_min:.1f}-{args.mass_max:.1f}"
    else:
        model_dir = f"bcm_{model_lower}_Mgt{args.mass_min}"
    
    output_dir = os.path.join(
        CONFIG['output_base'], f'L205n{args.sim_res}TNG', f'seed{args.seed}', model_dir
    )
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        write_config(output_dir, args.sim_res)
        print(f"\nOutput: {output_dir}")
    
    comm.Barrier()
    
    # Setup randomization
    randomization = RandomizationState(args.seed)
    
    # Process snapshots
    for snap in snapshots:
        process_snapshot(
            snap, args.sim_res, args.mass_min, args.mass_max, args.radius_factor,
            args.bcm_model, displacement_model, cosmo_dict, randomization, output_dir
        )
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"COMPLETE - Total time: {time.time()-t_start:.1f}s")
        n_planes = len(glob.glob(f'{output_dir}/density*.dat'))
        print(f"Generated {n_planes} lens planes")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
