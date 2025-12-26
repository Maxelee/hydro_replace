#!/usr/bin/env python
"""
Convert mass planes from generate_all_unified.py to lux lensing potential format.

This script takes the raw mass density planes (unnormalized pixel masses) and:
1. Converts to density contrast δ = (Σ - Σ̄)/Σ̄ × Δχ
2. FFT solves Poisson equation for lensing potential
3. Computes 5 potential derivatives (∂ψ/∂x, ∂ψ/∂y, ∂²ψ/∂x², ∂²ψ/∂x∂y, ∂²ψ/∂y²)
4. Writes output in lux format for ray-tracing

The FFT calculation matches lux/fourier.cpp exactly.

Parallelization:
    - Uses MPI to distribute planes across ranks
    - Each rank processes its assigned planes independently
    - Scales well since planes are independent

Usage:
    # Serial mode
    python convert_to_lenspot.py --input-dir /path/to/LP_output --model dmo \\
        --realization 0 --output-dir /path/to/lux_input
    
    # MPI parallel mode (recommended for large jobs)
    mpirun -np 40 python convert_to_lenspot.py --input-dir /path/to/LP_output \\
        --model dmo --all-realizations --output-dir /path/to/lux_input

Output structure:
    {output_dir}/{model}/LP_{realization:02d}/
        lenspot01.dat - lenspot40.dat  (1-indexed for lux)
        config.dat                      (geometry configuration)
"""

import numpy as np
import os
import argparse
import time
from numpy.fft import fft2, ifft2, fftfreq

# Try to import MPI - fall back to serial if not available
try:
    from mpi4py import MPI
    HAS_MPI = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('using MPI')
except ImportError:
    HAS_MPI = False
    rank = 0
    size = 1
    comm = None


# ============================================================================
# Configuration - Must match generate_all_unified.py and κTNG
# ============================================================================

BOX_SIZE = 205.0  # Mpc/h
GRID_RES = 4096   # Default grid resolution
N_SNAPSHOTS = 20
PPS = 2           # Planes per snapshot
N_PLANES = N_SNAPSHOTS * PPS  # 40 total

# TNG Cosmology
OMEGA_M = 0.3089
OMEGA_DE = 1.0 - OMEGA_M
H0 = 67.74  # km/s/Mpc
c = 299792.458  # km/s

# Critical density in units of 10^10 Msun/h / (Mpc/h)^3
RHO_CRIT_0 = 27.7536627  # 10^10 Msun/h / (Mpc/h)^3

# Snapshot order (from z≈0 to z≈2.5)
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]

# κTNG Table 2: Lens plane positions (exact values from arXiv:2010.09731)
# Format: plane_num (1-40) -> (chi_L [Mpc/h], z_L)
LENS_PLANES = {
    1:  (51.25,    0.017),   2:  (153.75,   0.052),   3:  (256.25,   0.087),   4:  (358.75,   0.123),
    5:  (461.25,   0.160),   6:  (563.75,   0.197),   7:  (666.25,   0.236),   8:  (768.75,   0.275),
    9:  (871.25,   0.314),   10: (973.75,   0.355),   11: (1076.25,  0.397),   12: (1178.75,  0.440),
    13: (1281.25,  0.484),   14: (1383.75,  0.529),   15: (1486.25,  0.576),   16: (1588.75,  0.623),
    17: (1691.25,  0.673),   18: (1793.75,  0.723),   19: (1896.25,  0.776),   20: (1998.75,  0.830),
    21: (2101.25,  0.886),   22: (2203.75,  0.944),   23: (2306.25,  1.003),   24: (2408.75,  1.065),
    25: (2511.25,  1.130),   26: (2613.75,  1.197),   27: (2716.25,  1.266),   28: (2818.75,  1.338),
    29: (2921.25,  1.413),   30: (3023.75,  1.492),   31: (3126.25,  1.573),   32: (3228.75,  1.659),
    33: (3331.25,  1.748),   34: (3433.75,  1.841),   35: (3536.25,  1.938),   36: (3638.75,  2.041),
    37: (3741.25,  2.148),   38: (3843.75,  2.260),   39: (3946.25,  2.379),   40: (4048.75,  2.503),
}

# Source plane positions (far edge of each slice)
SOURCE_PLANES = {
    1:  (102.5,    0.034),   2:  (205.0,    0.070),   3:  (307.5,    0.105),   4:  (410.0,    0.142),
    5:  (512.5,    0.179),   6:  (615.0,    0.216),   7:  (717.5,    0.255),   8:  (820.0,    0.294),
    9:  (922.5,    0.335),   10: (1025.0,   0.376),   11: (1127.5,   0.418),   12: (1230.0,   0.462),
    13: (1332.5,   0.506),   14: (1435.0,   0.552),   15: (1537.5,   0.599),   16: (1640.0,   0.648),
    17: (1742.5,   0.698),   18: (1845.0,   0.749),   19: (1947.5,   0.803),   20: (2050.0,   0.858),
    21: (2152.5,   0.914),   22: (2255.0,   0.973),   23: (2357.5,   1.034),   24: (2460.0,   1.097),
    25: (2562.5,   1.163),   26: (2665.0,   1.231),   27: (2767.5,   1.302),   28: (2870.0,   1.375),
    29: (2972.5,   1.452),   30: (3075.0,   1.532),   31: (3177.5,   1.615),   32: (3280.0,   1.703),
    33: (3382.5,   1.794),   34: (3485.0,   1.889),   35: (3587.5,   1.989),   36: (3690.0,   2.094),
    37: (3792.5,   2.203),   38: (3895.0,   2.319),   39: (3997.5,   2.440),   40: (4100.0,   2.568),
}


# ============================================================================
# Geometry Functions
# ============================================================================

def compute_scale_factor(chi, Omega_m=OMEGA_M, w_de=-1.0, tol=1e-10, max_iter=100):
    """
    Compute scale factor a at comoving distance chi using Newton's method.
    
    Matches lux/lenspot.cpp:find_a() - solves chi(a) = integral for a.
    
    Args:
        chi: comoving distance in Mpc/h
        Omega_m: matter density parameter
        w_de: dark energy equation of state
        tol: convergence tolerance
        max_iter: maximum iterations
    
    Returns:
        scale factor a
    """
    if chi <= 0:
        return 1.0
    
    Omega_de = 1.0 - Omega_m
    
    # Newton's method to find a such that chi(a) = chi_target
    a = 0.5  # Initial guess
    
    for _ in range(max_iter):
        # Compute chi(a) by numerical integration
        # chi = (c/H0) * integral_a^1 da' / (a'^2 * E(a'))
        # where E(a) = sqrt(Omega_m/a^3 + Omega_de * a^(-3*(1+w)))
        
        n_int = 1000
        a_vals = np.linspace(a, 1.0, n_int)
        
        # E(a) = sqrt(Omega_m * a + Omega_de * a^(1-3*w_de))
        # For w=-1: E(a) = sqrt(Omega_m * a + Omega_de * a^4)
        # Actually lux uses: E = sqrt(Omega_m*a + Omega_de*a^(1-3*w))
        E_vals = np.sqrt(Omega_m * a_vals + Omega_de * np.power(a_vals, 1.0 - 3.0 * w_de))
        
        # Integrand: c/100 / E  (H0 = 100h km/s/Mpc)
        integrand = (c / 100.0) / E_vals
        
        chi_computed = np.trapz(integrand, a_vals)
        
        # Derivative: d(chi)/da = -c/100 / E(a)
        E_a = np.sqrt(Omega_m * a + Omega_de * np.power(a, 1.0 - 3.0 * w_de))
        dchi_da = -(c / 100.0) / E_a
        
        # Newton update
        a_new = a - (chi_computed - chi) / dchi_da
        
        if abs(a_new - a) < tol:
            return a_new
        
        a = a_new
        
        # Keep a in valid range
        a = max(0.01, min(0.9999, a))
    
    return a


def build_geometry(box_size=BOX_SIZE, n_snapshots=N_SNAPSHOTS, pps=PPS):
    """
    Build lightcone geometry arrays matching lux conventions.
    
    Returns:
        dict with:
            - Np: number of planes
            - Ns: number of snapshots
            - chi: comoving distances to plane centers (Np+1 array, chi[0]=0)
            - chi_out: comoving distances for output (Np array, source planes)
            - a: scale factors at plane centers (Np+1 array)
            - Ll: longitudinal box sizes (Ns array)
            - Lt: transverse box sizes (Ns array)
    """
    Np = n_snapshots * pps
    Ns = n_snapshots
    
    # Box sizes - same for all snapshots in our case (no stacking)
    Ll = np.full(Ns, box_size)  # Longitudinal (along LOS)
    Lt = np.full(Ns, box_size)  # Transverse
    
    # Comoving distances to plane centers
    # chi[p] = sum of Ll for snapshots before p + offset within current snapshot
    chi = np.zeros(Np + 1)
    chi[0] = 0.0  # Observer
    
    for p in range(1, Np + 1):
        # Use κTNG Table 2 values
        chi[p] = LENS_PLANES[p][0]
    
    # Output distances (source planes)
    chi_out = np.zeros(Np)
    for p in range(Np):
        chi_out[p] = SOURCE_PLANES[p + 1][0]
    
    # Scale factors
    a = np.zeros(Np + 1)
    a[0] = 1.0  # Observer
    for p in range(1, Np + 1):
        # Use redshift from κTNG Table 2
        z = LENS_PLANES[p][1]
        a[p] = 1.0 / (1.0 + z)
    
    return {
        'Np': Np,
        'Ns': Ns,
        'chi': chi,
        'chi_out': chi_out,
        'a': a,
        'Ll': Ll,
        'Lt': Lt,
    }


# ============================================================================
# File I/O Functions
# ============================================================================

def load_mass_plane(filepath, expected_grid=None):
    """
    Load a mass plane from generate_all_unified.py output format.
    
    Format: [int32: grid_size] [float64[grid²]: mass_data] [int32: grid_size]
    
    Returns:
        tuple: (data array, grid_size)
    """
    with open(filepath, 'rb') as f:
        header = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.frombuffer(f.read(header * header * 8), dtype=np.float64)
        footer = np.frombuffer(f.read(4), dtype=np.int32)[0]
    
    if header != footer:
        raise ValueError(f"Header ({header}) != Footer ({footer}) in {filepath}")
    
    if expected_grid is not None and header != expected_grid:
        raise ValueError(f"Grid size {header} != expected {expected_grid}")
    
    return data.reshape((header, header)), header


def write_lenspot_lux(filepath, phi, grid_res):
    """
    Write lensing potential in lux format.
    
    Format: [int32: grid_size] [float64[5×grid²]: phi_derivatives] [int32: grid_size]
    
    The 5 fields are stored interleaved: phi[f + 5*(j + grid*i)] for field f at pixel (i,j)
        f=0: ∂ψ/∂x
        f=1: ∂ψ/∂y  
        f=2: ∂²ψ/∂x²
        f=3: ∂²ψ/∂x∂y
        f=4: ∂²ψ/∂y²
    
    Args:
        filepath: output file path
        phi: (5, grid_res, grid_res) array of potential derivatives
        grid_res: grid size
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Interleave the 5 fields: phi[f, i, j] -> output[f + 5*(j + grid*i)]
    output = np.zeros(5 * grid_res * grid_res, dtype=np.float64)
    for i in range(grid_res):
        for j in range(grid_res):
            for f in range(5):
                output[f + 5 * (j + grid_res * i)] = phi[f, i, j]
    
    with open(filepath, 'wb') as f:
        f.write(np.array([grid_res], dtype=np.int32).tobytes())
        f.write(output.tobytes())
        f.write(np.array([grid_res], dtype=np.int32).tobytes())


def write_config_dat(filepath, geometry, proj_dirs=None, disp=None, flip=None):
    """
    Write config.dat in lux binary format.
    
    This file contains the lightcone geometry needed by lux raytracing.
    
    Format (binary):
        int Np
        int Ns
        double[Np+1] a
        double[Np+1] chi
        double[Np] chi_out
        double[Ns] Ll
        double[Ns] Lt
        int[Ns] proj_dirs
        double[3*Ns] disp
        bool[3*Ns] flip
    """
    Np = geometry['Np']
    Ns = geometry['Ns']
    
    # Default projection directions and displacements (not used for pre-computed planes)
    if proj_dirs is None:
        proj_dirs = np.zeros(Ns, dtype=np.int32)
    if disp is None:
        disp = np.zeros(3 * Ns, dtype=np.float64)
    if flip is None:
        flip = np.zeros(3 * Ns, dtype=np.bool_)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        f.write(np.array([Np], dtype=np.int32).tobytes())
        f.write(np.array([Ns], dtype=np.int32).tobytes())
        f.write(geometry['a'].astype(np.float64).tobytes())
        f.write(geometry['chi'].astype(np.float64).tobytes())
        f.write(geometry['chi_out'].astype(np.float64).tobytes())
        f.write(geometry['Ll'].astype(np.float64).tobytes())
        f.write(geometry['Lt'].astype(np.float64).tobytes())
        f.write(proj_dirs.astype(np.int32).tobytes())
        f.write(disp.astype(np.float64).tobytes())
        f.write(flip.astype(np.bool_).tobytes())


# ============================================================================
# FFT and Potential Calculation
# ============================================================================

def mass_to_density_contrast(mass_plane, grid_res, Lt, thickness, Omega_m=OMEGA_M):
    """
    Convert mass plane to density contrast δ × Δχ.
    
    Following lux/lenspot.cpp:
        bar = rho_c0 * Omega_m * Lt² * Ll / pps / grid²
        delta = (mass / bar - 1) * (Ll / pps)
    
    Args:
        mass_plane: (grid_res, grid_res) array of pixel masses in Msun/h
                   (from generate_all_unified.py which multiplies by 1e10)
        grid_res: grid resolution
        Lt: transverse box size (Mpc/h)
        thickness: slice thickness Ll/pps (Mpc/h)
        Omega_m: matter density parameter
    
    Returns:
        delta: (grid_res, grid_res) density contrast × thickness
    """
    # Convert mass from Msun/h to 10^10 Msun/h to match RHO_CRIT_0 units
    # generate_all_unified.py multiplies TNG masses by 1e10, so we divide back
    mass_plane_unit = mass_plane / 1e10  # Now in 10^10 Msun/h
    
    # Mean mass per pixel: ρ_crit,0 × Ωm × (Lt/grid)² × thickness
    # RHO_CRIT_0 is in 10^10 Msun/h / (Mpc/h)^3
    bar = RHO_CRIT_0 * Omega_m * (Lt ** 2) * thickness / (grid_res ** 2)
    
    # Density contrast: δ = (Σ/Σ̄ - 1) × Δχ
    delta = (mass_plane_unit / bar - 1.0) * thickness
    
    return delta


def compute_lensing_potential_fft(delta, Lt, grid_res, Omega_m=OMEGA_M):
    """
    Compute lensing potential derivatives via FFT.
    
    This exactly matches lux/fourier.cpp:Fourier_trs().
    
    Solves in Fourier space:
        ψ̃ = 3 Ωm (H0/c)² δ̃ / k²
    
    Returns 5 derivatives:
        f=0: ∂ψ/∂x  = i*kx * ψ̃
        f=1: ∂ψ/∂y  = i*ky * ψ̃
        f=2: ∂²ψ/∂x² = kx² * ψ̃  (lux convention)
        f=3: ∂²ψ/∂x∂y = kx*ky * ψ̃
        f=4: ∂²ψ/∂y² = ky² * ψ̃
    
    Args:
        delta: (grid_res, grid_res) density contrast × thickness
        Lt: transverse box size (Mpc/h)
        grid_res: grid resolution
        Omega_m: matter density parameter
    
    Returns:
        phi: (5, grid_res, grid_res) potential derivatives
    """
    # Fundamental mode
    ku = 2.0 * np.pi / Lt
    
    # FFT of delta
    delta_fft = fft2(delta)
    
    # Wavenumber arrays
    # lux convention: k = ku * I where I = i if i <= N/2 else i-N
    kx_1d = fftfreq(grid_res, d=1.0/grid_res) * ku  # This gives ku * [0,1,2,...,N/2-1,-N/2,...,-1]
    ky_1d = fftfreq(grid_res, d=1.0/grid_res) * ku
    
    kx, ky = np.meshgrid(kx_1d, kx_1d, indexing='ij')
    k2 = kx**2 + ky**2
    
    # Avoid division by zero at k=0
    k2[0, 0] = 1.0  # Will set this mode to zero anyway
    
    # Prefactor: 3 Ωm (H0/c)² / k²
    # H0 = 100 h km/s/Mpc, so (H0/c)² = (100/c)² in (Mpc/h)^-2
    # 
    # IMPORTANT: lux uses FFTW which is unnormalized for both forward and backward FFT.
    # lux includes a /grid² factor to compensate after backward FFT.
    # NumPy's ifft2 automatically divides by grid², so we should NOT include /grid² here.
    prefactor = 3.0 * Omega_m * (100.0 / c) ** 2 / k2
    prefactor[0, 0] = 0.0  # Zero mode
    
    # Compute 5 potential derivatives
    phi = np.zeros((5, grid_res, grid_res), dtype=np.float64)
    
    # f=0: ∂ψ/∂x -> multiply by i*kx in Fourier space
    # In lux: psi[0] = delta_fa[1]*kx, psi[1] = -delta_fa[0]*kx (real/imag swap for i*kx)
    psi_fft = 1j * kx * delta_fft * prefactor
    phi[0] = np.real(ifft2(psi_fft))
    
    # f=1: ∂ψ/∂y -> multiply by i*ky
    psi_fft = 1j * ky * delta_fft * prefactor
    phi[1] = np.real(ifft2(psi_fft))
    
    # f=2: ∂²ψ/∂x² -> multiply by kx² (lux uses positive sign convention)
    psi_fft = kx**2 * delta_fft * prefactor
    phi[2] = np.real(ifft2(psi_fft))
    
    # f=3: ∂²ψ/∂x∂y -> multiply by kx*ky
    psi_fft = kx * ky * delta_fft * prefactor
    phi[3] = np.real(ifft2(psi_fft))
    
    # f=4: ∂²ψ/∂y² -> multiply by ky²
    psi_fft = ky**2 * delta_fft * prefactor
    phi[4] = np.real(ifft2(psi_fft))
    
    return phi


# ============================================================================
# Main Conversion Functions (MPI Parallelized)
# ============================================================================

def convert_plane(input_path, output_path, plane_idx, Lt, thickness, grid_res, verbose=True):
    """
    Convert a single mass plane to lensing potential.
    
    Args:
        input_path: path to input mass plane (generate_all_unified format)
        output_path: path to output lenspot file (lux format)
        plane_idx: 0-based plane index
        Lt: transverse box size
        thickness: slice thickness (Ll / pps)
        grid_res: expected grid resolution
        verbose: print progress
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(input_path):
        if verbose:
            print(f"  [Rank {rank}] Warning: Missing input file {input_path}")
        return False
    
    try:
        # Load mass plane
        mass_plane, grid = load_mass_plane(input_path, expected_grid=grid_res)
        
        # Convert to density contrast × thickness
        delta = mass_to_density_contrast(mass_plane, grid, Lt, thickness)
        
        # FFT to get lensing potential derivatives
        phi = compute_lensing_potential_fft(delta, Lt, grid)
        
        # Write in lux format
        write_lenspot_lux(output_path, phi, grid)
        
        if verbose:
            print(f"  [Rank {rank}] Converted plane {plane_idx:02d} -> {os.path.basename(output_path)}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"  [Rank {rank}] Error converting plane {plane_idx}: {e}")
        return False


def convert_realization_parallel(input_dir, output_dir, model, realization, grid_res=GRID_RES, verbose=True):
    """
    Convert all planes for one realization using MPI parallelization.
    
    Planes are distributed across MPI ranks for parallel processing.
    
    Args:
        input_dir: base input directory (contains {model}/LP_{real:02d}/)
        output_dir: base output directory 
        model: 'dmo', 'hydro', or replace config label
        realization: realization index (0-9)
        grid_res: grid resolution
        verbose: print progress
    
    Returns:
        Number of successfully converted planes (local to this rank)
    """
    # Input directory structure: {input_dir}/{model}/LP_{real:02d}/lenspot{idx:02d}.dat
    in_real_dir = os.path.join(input_dir, model, f'LP_{realization:02d}')
    
    # Output directory structure: {output_dir}/{model}/LP_{real:02d}/lenspot{idx+1:02d}.dat
    out_real_dir = os.path.join(output_dir, model, f'LP_{realization:02d}')
    
    if not os.path.exists(in_real_dir):
        if rank == 0 and verbose:
            print(f"Input directory not found: {in_real_dir}")
        return 0
    
    # Create output directory (rank 0 only to avoid race condition)
    if rank == 0:
        os.makedirs(out_real_dir, exist_ok=True)
        if verbose:
            print(f"\nConverting {model} realization {realization}:")
            print(f"  Input:  {in_real_dir}")
            print(f"  Output: {out_real_dir}")
    
    # Synchronize before processing
    if HAS_MPI:
        comm.Barrier()
    
    # Build geometry
    geometry = build_geometry()
    
    # Thickness per pps slice
    thickness = BOX_SIZE / PPS  # 102.5 Mpc/h
    
    # Distribute planes across ranks
    my_planes = [p for p in range(N_PLANES) if p % size == rank]
    
    if verbose and len(my_planes) > 0:
        print(f"  [Rank {rank}] Processing {len(my_planes)} planes: {my_planes[0]}-{my_planes[-1]}")
    
    # Convert each assigned plane
    n_converted_local = 0
    for plane_idx in my_planes:
        # Input file: 0-indexed
        input_path = os.path.join(in_real_dir, f'lenspot{plane_idx:02d}.dat')
        
        # Output file: 1-indexed (lux convention)
        output_path = os.path.join(out_real_dir, f'lenspot{plane_idx + 1:02d}.dat')
        
        # Transverse box size (same for all in our case)
        snapshot_idx = plane_idx // PPS
        Lt = geometry['Lt'][snapshot_idx]
        
        if convert_plane(input_path, output_path, plane_idx, Lt, thickness, grid_res, verbose=False):
            n_converted_local += 1
    
    # Synchronize and gather counts
    if HAS_MPI:
        comm.Barrier()
        n_converted_total = comm.reduce(n_converted_local, op=MPI.SUM, root=0)
    else:
        n_converted_total = n_converted_local
    
    # Write config.dat (rank 0 only)
    if rank == 0:
        config_path = os.path.join(out_real_dir, 'config.dat')
        write_config_dat(config_path, geometry)
        if verbose:
            print(f"  Wrote config.dat")
            print(f"  Converted {n_converted_total}/{N_PLANES} planes")
    
    return n_converted_local


def convert_all_parallel(input_dir, output_dir, models, n_realizations=10, grid_res=GRID_RES, verbose=True):
    """
    Convert all realizations for multiple models using MPI parallelization.
    
    Work is distributed at the (model, realization, plane) level for maximum parallelism.
    
    Args:
        input_dir: base input directory
        output_dir: base output directory
        models: list of models to convert
        n_realizations: number of realizations to convert
        grid_res: grid resolution
        verbose: print progress
    
    Returns:
        Total number of successfully converted planes (local to this rank)
    """
    t_start = time.time()
    
    if rank == 0 and verbose:
        print(f"\n{'='*60}")
        print(f"Converting {len(models)} model(s), {n_realizations} realizations each")
        print(f"Using {size} MPI rank(s)")
        print(f"{'='*60}")
    
    # Build list of all (model, realization, plane) work units
    work_units = []
    for model in models:
        for real_idx in range(n_realizations):
            for plane_idx in range(N_PLANES):
                work_units.append((model, real_idx, plane_idx))
    
    total_work = len(work_units)
    
    if rank == 0 and verbose:
        print(f"Total work units: {total_work} ({len(models)} models × {n_realizations} realizations × {N_PLANES} planes)")
    
    # Distribute work across ranks
    my_work = [w for i, w in enumerate(work_units) if i % size == rank]
    
    if verbose:
        print(f"[Rank {rank}] Assigned {len(my_work)} work units")
    
    # Build geometry once
    geometry = build_geometry()
    thickness = BOX_SIZE / PPS
    
    # Create all output directories (rank 0)
    if rank == 0:
        for model in models:
            for real_idx in range(n_realizations):
                out_dir = os.path.join(output_dir, model, f'LP_{real_idx:02d}')
                os.makedirs(out_dir, exist_ok=True)
    
    if HAS_MPI:
        comm.Barrier()
    
    # Process assigned work
    n_converted_local = 0
    n_processed = 0
    
    for model, real_idx, plane_idx in my_work:
        in_real_dir = os.path.join(input_dir, model, f'LP_{real_idx:02d}')
        out_real_dir = os.path.join(output_dir, model, f'LP_{real_idx:02d}')
        
        input_path = os.path.join(in_real_dir, f'lenspot{plane_idx:02d}.dat')
        output_path = os.path.join(out_real_dir, f'lenspot{plane_idx + 1:02d}.dat')
        
        snapshot_idx = plane_idx // PPS
        Lt = geometry['Lt'][snapshot_idx]
        
        if convert_plane(input_path, output_path, plane_idx, Lt, thickness, grid_res, verbose=False):
            n_converted_local += 1
        
        n_processed += 1
        
        # Progress report every 10 planes
        if verbose and n_processed % 10 == 0:
            print(f"[Rank {rank}] Progress: {n_processed}/{len(my_work)} ({100*n_processed/len(my_work):.1f}%)")
    
    # Synchronize
    if HAS_MPI:
        comm.Barrier()
        n_converted_total = comm.reduce(n_converted_local, op=MPI.SUM, root=0)
    else:
        n_converted_total = n_converted_local
    
    # Write config.dat files (rank 0 only)
    if rank == 0:
        for model in models:
            for real_idx in range(n_realizations):
                out_real_dir = os.path.join(output_dir, model, f'LP_{real_idx:02d}')
                config_path = os.path.join(out_real_dir, 'config.dat')
                write_config_dat(config_path, geometry)
    
    t_end = time.time()
    
    if rank == 0 and verbose:
        print(f"\n{'='*60}")
        print(f"Conversion complete!")
        print(f"  Total planes converted: {n_converted_total}")
        print(f"  Time elapsed: {t_end - t_start:.1f} s")
        print(f"  Throughput: {n_converted_total / (t_end - t_start):.1f} planes/s")
        print(f"{'='*60}")
    
    return n_converted_local


# Legacy serial functions for backward compatibility
def convert_realization(input_dir, output_dir, model, realization, grid_res=GRID_RES, verbose=True):
    """Serial version - wraps parallel version for single rank."""
    return convert_realization_parallel(input_dir, output_dir, model, realization, grid_res, verbose)


def convert_all(input_dir, output_dir, model, n_realizations=10, grid_res=GRID_RES, verbose=True):
    """Serial version for single model."""
    return convert_all_parallel(input_dir, output_dir, [model], n_realizations, grid_res, verbose)


# ============================================================================
# Utility Functions
# ============================================================================

def verify_conversion(input_path, output_path, verbose=True):
    """
    Verify a converted file by checking basic statistics.
    
    Args:
        input_path: original mass plane
        output_path: converted lenspot file
        verbose: print results
    
    Returns:
        dict with verification results
    """
    results = {}
    
    # Load original
    mass_plane, grid = load_mass_plane(input_path)
    results['input_grid'] = grid
    results['input_mean'] = mass_plane.mean()
    results['input_std'] = mass_plane.std()
    results['input_min'] = mass_plane.min()
    results['input_max'] = mass_plane.max()
    
    # Load converted
    with open(output_path, 'rb') as f:
        out_grid = np.frombuffer(f.read(4), dtype=np.int32)[0]
        phi_flat = np.frombuffer(f.read(5 * out_grid * out_grid * 8), dtype=np.float64)
        footer = np.frombuffer(f.read(4), dtype=np.int32)[0]
    
    results['output_grid'] = out_grid
    results['output_size'] = len(phi_flat)
    
    # Reshape to (5, grid, grid)
    phi = np.zeros((5, out_grid, out_grid))
    for i in range(out_grid):
        for j in range(out_grid):
            for f in range(5):
                phi[f, i, j] = phi_flat[f + 5 * (j + out_grid * i)]
    
    for f in range(5):
        results[f'phi{f}_mean'] = phi[f].mean()
        results[f'phi{f}_std'] = phi[f].std()
    
    if verbose:
        print(f"\nVerification:")
        print(f"  Input:  grid={results['input_grid']}, mean={results['input_mean']:.2e}, std={results['input_std']:.2e}")
        print(f"  Output: grid={results['output_grid']}, 5 fields")
        for f in range(5):
            print(f"    phi[{f}]: mean={results[f'phi{f}_mean']:.2e}, std={results[f'phi{f}_std']:.2e}")
    
    return results


def list_available_models(input_dir):
    """List available models in the input directory."""
    models = []
    if os.path.exists(input_dir):
        for name in os.listdir(input_dir):
            path = os.path.join(input_dir, name)
            if os.path.isdir(path):
                # Check if it has LP_* subdirectories
                lp_dirs = [d for d in os.listdir(path) if d.startswith('LP_')]
                if lp_dirs:
                    models.append(name)
    return sorted(models)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert mass planes to lux lensing potential format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python convert_to_lenspot.py --input-dir /path/to/LP_output --list-models
  
  # Convert a single realization (serial)
  python convert_to_lenspot.py --input-dir /path/to/LP_output --model dmo \\
      --realization 0 --output-dir /path/to/lux_input
  
  # Convert all realizations for a model (serial)
  python convert_to_lenspot.py --input-dir /path/to/LP_output --model dmo \\
      --all-realizations --output-dir /path/to/lux_input
  
  # Convert with MPI parallelization
  mpirun -np 40 python convert_to_lenspot.py --input-dir /path/to/LP_output \\
      --models dmo hydro --all-realizations --output-dir /path/to/lux_input
  
  # SLURM job with MPI
  srun -n 64 python convert_to_lenspot.py --input-dir /path/to/LP_output \\
      --model dmo --all-realizations --output-dir /path/to/lux_input
  
  # Verify a conversion
  python convert_to_lenspot.py --verify --input-file /path/to/input/lenspot00.dat \\
      --output-file /path/to/output/lenspot01.dat
"""
    )
    
    parser.add_argument('--input-dir', type=str,
                        help='Input directory containing model subdirectories')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for lux-format files')
    parser.add_argument('--model', type=str,
                        help='Single model to convert (dmo, hydro, or replace config)')
    parser.add_argument('--models', type=str, nargs='+',
                        help='Multiple models to convert')
    parser.add_argument('--realization', type=int,
                        help='Single realization index to convert (0-9)')
    parser.add_argument('--all-realizations', action='store_true',
                        help='Convert all realizations (0-9)')
    parser.add_argument('--n-realizations', type=int, default=10,
                        help='Number of realizations (default: 10)')
    parser.add_argument('--grid', type=int, default=GRID_RES,
                        help=f'Grid resolution (default: {GRID_RES})')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit')
    parser.add_argument('--verify', action='store_true',
                        help='Verify a conversion')
    parser.add_argument('--input-file', type=str,
                        help='Input file for verification')
    parser.add_argument('--output-file', type=str,
                        help='Output file for verification')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Only rank 0 should print header and handle non-conversion tasks
    if rank == 0 and verbose and HAS_MPI:
        print(f"\n{'='*60}")
        print(f"MPI-enabled convert_to_lenspot.py")
        print(f"Running with {size} MPI rank(s)")
        print(f"{'='*60}")
    
    # Verification mode (single rank only)
    if args.verify:
        if rank != 0:
            return
        if not args.input_file or not args.output_file:
            print("Error: --verify requires --input-file and --output-file")
            return
        verify_conversion(args.input_file, args.output_file, verbose=True)
        return
    
    # List models mode (single rank only)
    if args.list_models:
        if rank != 0:
            return
        if not args.input_dir:
            print("Error: --list-models requires --input-dir")
            return
        models = list_available_models(args.input_dir)
        print(f"\nAvailable models in {args.input_dir}:")
        for m in models:
            print(f"  {m}")
        return
    
    # Conversion mode
    if not args.input_dir or not args.output_dir:
        if rank == 0:
            parser.print_help()
            print("\nError: --input-dir and --output-dir are required for conversion")
        return
    
    # Determine models to convert
    if args.models:
        models = args.models
    elif args.model:
        models = [args.model]
    else:
        # Auto-detect (rank 0 does this, broadcast to others)
        if rank == 0:
            models = list_available_models(args.input_dir)
            if not models:
                print(f"Error: No models found in {args.input_dir}")
        else:
            models = None
        
        if HAS_MPI:
            models = comm.bcast(models, root=0)
        
        if not models:
            return
        
        if rank == 0 and verbose:
            print(f"Auto-detected models: {models}")
    
    # Determine realizations
    n_real = args.n_realizations
    if args.realization is not None:
        # Single realization mode
        if rank == 0 and verbose:
            print(f"\nConverting single realization {args.realization} for {models}")
        
        for model in models:
            convert_realization_parallel(args.input_dir, args.output_dir, model,
                                        args.realization, args.grid, verbose)
    else:
        # All realizations mode - use fully parallel function
        convert_all_parallel(args.input_dir, args.output_dir, models, 
                            n_real, args.grid, verbose)


if __name__ == '__main__':
    main()

