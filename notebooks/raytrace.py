#!/usr/bin/env python
"""
Generate weak lensing convergence maps from lensplanes using LensTools.

This script reads the unnormalized mass density planes from generate_all_unified.py
and performs ray-tracing to produce convergence/shear maps.

Based on κTNG methodology (arXiv:2010.09731):
- 20 snapshots × 2 pps = 40 lens planes per realization
- Each snapshot spans Δχ = 205 Mpc/h (box size)
- Each pps slice has thickness = 102.5 Mpc/h
- Lens planes are at the CENTER of each slice
- Source planes are at the FAR EDGE of each slice

File format: [int32: grid_size] [float64[grid²]: mass] [int32: grid_size]
Units: Msun/h (total mass in each pixel, NOT density contrast)

To convert to the dimensionless density contrast δ = (ρ - ρ̄)/ρ̄ needed by LensTools:
1. Divide by pixel area to get surface density Σ
2. Compute mean surface density Σ̄ = ρ̄ × thickness
3. δ = (Σ - Σ̄) / Σ̄
"""

import numpy as np
import h5py
import argparse
import os
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.constants import c, G

# Try importing LensTools
try:
    from lenstools import ConvergenceMap, ShearMap
    from lenstools.simulations import DensityPlane, RayTracer
    HAS_LENSTOOLS = True
except ImportError:
    print("Warning: LensTools not installed. Install with: pip install lenstools")
    HAS_LENSTOOLS = False


# ============================================================================
# Configuration - Match κTNG (arXiv:2010.09731) Table 2
# ============================================================================

BOX_SIZE = 205.0  # Mpc/h
THICKNESS = 102.5  # Mpc/h (half the box, one pps slice)
MASS_UNIT = 1e10  # Msun/h (masses were multiplied by this when saving)

# Snapshot order (from z≈0 to z≈2.5, as used in generate_all_unified.py)
# These correspond to snapshots at χ_center = 102.5, 307.5, 512.5, ... Mpc/h
SNAPSHOT_ORDER = [96, 90, 85, 80, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 35, 33, 31, 29]
SNAPSHOT_TO_INDEX = {snap: idx for idx, snap in enumerate(SNAPSHOT_ORDER)}
N_SNAPSHOTS = len(SNAPSHOT_ORDER)
PPS = 2  # Planes per snapshot
N_PLANES = N_SNAPSHOTS * PPS  # 40 total

# TNG cosmology
COSMO = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486)


# ============================================================================
# κTNG Table 2: Lens Plane (L) and Source Plane (S) Configuration
# From arXiv:2010.09731 - exact values
# ============================================================================

# Lens planes are at CENTER of each slice
# Format: L[i] -> (chi_L [Mpc/h], z_L)
LENS_PLANES = {
    1:  (51.25,    0.017),
    2:  (153.75,   0.052),
    3:  (256.25,   0.087),
    4:  (358.75,   0.123),
    5:  (461.25,   0.160),
    6:  (563.75,   0.197),
    7:  (666.25,   0.236),
    8:  (768.75,   0.275),
    9:  (871.25,   0.314),
    10: (973.75,   0.355),
    11: (1076.25,  0.397),
    12: (1178.75,  0.440),
    13: (1281.25,  0.484),
    14: (1383.75,  0.529),
    15: (1486.25,  0.576),
    16: (1588.75,  0.623),
    17: (1691.25,  0.673),
    18: (1793.75,  0.723),
    19: (1896.25,  0.776),
    20: (1998.75,  0.830),
    21: (2101.25,  0.886),
    22: (2203.75,  0.944),
    23: (2306.25,  1.003),
    24: (2408.75,  1.065),
    25: (2511.25,  1.130),
    26: (2613.75,  1.197),
    27: (2716.25,  1.266),
    28: (2818.75,  1.338),
    29: (2921.25,  1.413),
    30: (3023.75,  1.492),
    31: (3126.25,  1.573),
    32: (3228.75,  1.659),
    33: (3331.25,  1.748),
    34: (3433.75,  1.841),
    35: (3536.25,  1.938),
    36: (3638.75,  2.041),
    37: (3741.25,  2.148),
    38: (3843.75,  2.260),
    39: (3946.25,  2.379),
    40: (4048.75,  2.503),
}

# Source planes are at FAR EDGE of each slice (where convergence is evaluated)
# Format: S[i] -> (chi_S [Mpc/h], z_S)
SOURCE_PLANES = {
    1:  (102.5,    0.034),
    2:  (205.0,    0.070),
    3:  (307.5,    0.105),
    4:  (410.0,    0.142),
    5:  (512.5,    0.179),
    6:  (615.0,    0.216),
    7:  (717.5,    0.255),
    8:  (820.0,    0.294),
    9:  (922.5,    0.335),
    10: (1025.0,   0.376),
    11: (1127.5,   0.418),
    12: (1230.0,   0.462),
    13: (1332.5,   0.506),
    14: (1435.0,   0.552),
    15: (1537.5,   0.599),
    16: (1640.0,   0.648),
    17: (1742.5,   0.698),
    18: (1845.0,   0.749),
    19: (1947.5,   0.803),
    20: (2050.0,   0.858),
    21: (2152.5,   0.914),
    22: (2255.0,   0.973),
    23: (2357.5,   1.034),
    24: (2460.0,   1.097),
    25: (2562.5,   1.163),
    26: (2665.0,   1.231),
    27: (2767.5,   1.302),
    28: (2870.0,   1.375),
    29: (2972.5,   1.452),
    30: (3075.0,   1.532),
    31: (3177.5,   1.615),
    32: (3280.0,   1.703),
    33: (3382.5,   1.794),
    34: (3485.0,   1.889),
    35: (3587.5,   1.989),
    36: (3690.0,   2.094),
    37: (3792.5,   2.203),
    38: (3895.0,   2.319),
    39: (3997.5,   2.440),
    40: (4100.0,   2.568),
}

# Map plane index (0-39) to L/S plane number (1-40)
def plane_idx_to_number(plane_idx):
    """Convert 0-based plane index to 1-based plane number."""
    return plane_idx + 1

# TNG snapshot redshifts (for reference only - not used for plane positions)
TNG_SNAPSHOT_REDSHIFTS = {
    99: 0.0,    98: 0.01,   97: 0.02,   96: 0.03,   95: 0.04,
    94: 0.05,   93: 0.06,   92: 0.07,   91: 0.08,   90: 0.10,
    89: 0.11,   88: 0.12,   87: 0.14,   86: 0.15,   85: 0.17,
    84: 0.18,   83: 0.20,   82: 0.21,   81: 0.23,   80: 0.25,
    79: 0.27,   78: 0.29,   77: 0.31,   76: 0.33,   75: 0.35,
    74: 0.38,   73: 0.40,   72: 0.42,   71: 0.45,   70: 0.47,
    69: 0.50,   68: 0.52,   67: 0.55,   66: 0.58,   65: 0.60,
    64: 0.64,   63: 0.67,   62: 0.70,   61: 0.73,   60: 0.76,
    59: 0.79,   58: 0.82,   57: 0.85,   56: 0.89,   55: 0.92,
    54: 0.95,   53: 0.99,   52: 1.04,   51: 1.07,   50: 1.11,
    49: 1.15,   48: 1.20,   47: 1.25,   46: 1.30,   45: 1.36,
    44: 1.41,   43: 1.47,   42: 1.53,   41: 1.60,   40: 1.67,
    39: 1.74,   38: 1.82,   37: 1.90,   36: 1.98,   35: 2.07,
    34: 2.15,   33: 2.24,   32: 2.32,   31: 2.44,   30: 2.52,
    29: 2.58,
}


def get_snapshot_redshift(snap_num):
    """Get redshift for a TNG snapshot."""
    return TNG_SNAPSHOT_REDSHIFTS.get(snap_num, None)


def get_plane_info(plane_idx, pps=PPS):
    """
    Get lens plane info from κTNG Table 2 configuration.
    
    Args:
        plane_idx: Plane index (0-39 for 40 total planes)
        pps: Planes per snapshot (default 2)
    
    Returns:
        dict with:
          - 'plane_number': 1-based plane number (L1-L40)
          - 'snapshot': TNG snapshot number
          - 'snapshot_idx': Index in SNAPSHOT_ORDER
          - 'pps_slice': 0 or 1 (which half of snapshot)
          - 'plane_idx': Same as input (0-based)
          - 'lens_redshift': Redshift at lens plane center (z_L)
          - 'lens_chi': Comoving distance to lens plane (Mpc/h)
          - 'source_redshift': Redshift at source plane (z_S)  
          - 'source_chi': Comoving distance to source plane (Mpc/h)
          - 'thickness': Slice thickness (Mpc/h)
    """
    if plane_idx < 0 or plane_idx >= N_PLANES:
        raise ValueError(f"Plane index {plane_idx} out of range (0-{N_PLANES-1})")
    
    plane_num = plane_idx_to_number(plane_idx)  # 1-based
    
    # Get lens and source plane positions from Table 2
    lens_chi, lens_z = LENS_PLANES[plane_num]
    source_chi, source_z = SOURCE_PLANES[plane_num]
    
    # Determine which snapshot and pps slice
    snapshot_idx = plane_idx // pps
    pps_slice = plane_idx % pps
    snap_num = SNAPSHOT_ORDER[snapshot_idx]
    
    return {
        'plane_number': plane_num,
        'snapshot': snap_num,
        'snapshot_idx': snapshot_idx,
        'pps_slice': pps_slice,
        'plane_idx': plane_idx,
        'lens_redshift': lens_z,
        'lens_chi': lens_chi,  # Mpc/h
        'source_redshift': source_z,
        'source_chi': source_chi,  # Mpc/h
        'thickness': THICKNESS,  # Mpc/h
        # For backward compatibility
        'redshift': lens_z,
        'comoving_distance': lens_chi,
    }


def get_source_plane_for_redshift(z_source):
    """
    Find the source plane index closest to a given source redshift.
    
    Args:
        z_source: Target source redshift
        
    Returns:
        tuple: (plane_idx, source_info) where source_info contains chi and z
    """
    best_idx = None
    best_diff = float('inf')
    
    for plane_num, (chi_s, z_s) in SOURCE_PLANES.items():
        diff = abs(z_s - z_source)
        if diff < best_diff:
            best_diff = diff
            best_idx = plane_num - 1  # Convert to 0-based
    
    if best_idx is not None:
        return best_idx, SOURCE_PLANES[best_idx + 1]
    return None, None


def load_lensplane(filepath, expected_grid=None):
    """
    Load a lensplane from the lux binary format.
    
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


def mass_to_convergence_born(mass_planes, plane_infos, source_z, box_size, cosmo):
    """
    Convert mass planes to convergence using Born approximation.
    
    κ(θ) = (3/2) * (H0/c)² * Ωm * Σ_i [χ_L,i * (χ_S - χ_L,i) / χ_S] * (1+z_L,i) * δ_i * Δχ
    
    Using the κTNG configuration (arXiv:2010.09731):
    - Lens planes are at χ_L (center of each slice)
    - Source planes are at χ_S (far edge)
    - Each slice has thickness Δχ = 102.5 Mpc/h
    
    Args:
        mass_planes: list of 2D mass arrays (Msun/h)
        plane_infos: list of plane info dicts from get_plane_info()
        source_z: source redshift (will be mapped to nearest source plane)
        box_size: box size in Mpc/h
        cosmo: astropy cosmology
    
    Returns:
        convergence map (2D array)
    """
    if len(mass_planes) == 0:
        raise ValueError("No planes provided")
    
    grid_res = mass_planes[0].shape[0]
    h = cosmo.h
    
    # Find the nearest source plane for the given source_z
    source_idx, source_info = get_source_plane_for_redshift(source_z)
    if source_info is not None:
        chi_s = source_info[0]  # Mpc/h
        z_s = source_info[1]
        print(f"Using source plane S{source_idx+1}: χ_S={chi_s:.2f} Mpc/h, z_S={z_s:.3f}")
    else:
        # Fall back to computing from cosmology
        chi_s = cosmo.comoving_distance(source_z).to(u.Mpc).value / h  # Mpc/h
        z_s = source_z
        print(f"Using computed χ_S={chi_s:.2f} Mpc/h for z_S={z_s:.3f}")
    
    # Pixel area in (Mpc/h)²
    pixel_size = box_size / grid_res
    pixel_area = pixel_size ** 2
    
    # Mean matter density in Msun/h / (Mpc/h)³
    rho_crit_0 = cosmo.critical_density0.to(u.Msun / u.Mpc**3).value * h**2
    rho_m_0 = cosmo.Om0 * rho_crit_0  # Msun/h / (Mpc/h)³
    
    # Prefactor for lensing: (3/2) * (H0/c)² * Ωm
    # (H0/c)² = (100 h km/s/Mpc / 299792 km/s)² = (h / 2997.92 Mpc)²
    prefactor = 1.5 * cosmo.Om0 * (cosmo.H0.value / 299792.458)**2  # (Mpc/h)^-2
    
    # Initialize convergence
    kappa = np.zeros((grid_res, grid_res), dtype=np.float64)
    
    for mass_plane, info in zip(mass_planes, plane_infos):
        # Use exact χ values from κTNG Table 2
        z_lens = info['lens_redshift']
        chi_lens = info['lens_chi']  # Mpc/h (center of slice)
        thickness = info['thickness']  # Mpc/h
        
        # Skip planes at or beyond source
        if chi_lens >= chi_s:
            continue
        
        # Lensing kernel: W = χ_L * (χ_S - χ_L) / χ_S
        W = chi_lens * (chi_s - chi_lens) / chi_s
        
        # Convert mass to surface density Σ (Msun/h / (Mpc/h)²)
        sigma = mass_plane / pixel_area
        
        # Mean surface density: Σ̄ = ρ̄_m,0 * Δχ (comoving)
        sigma_mean = rho_m_0 * thickness
        
        # Density contrast: δ = (Σ - Σ̄) / Σ̄
        delta_sigma = (sigma - sigma_mean) / sigma_mean
        
        # Contribution to convergence: 
        # κ += prefactor * W * (1+z) * Δχ * δ
        kappa += prefactor * W * (1 + z_lens) * thickness * delta_sigma
    
    return kappa


def generate_convergence_map_born(lp_dir, model, realization, source_z, grid_res=4096):
    """
    Generate a convergence map using Born approximation from lensplanes.
    
    Uses κTNG methodology with exact plane positions from Table 2.
    
    Args:
        lp_dir: Base directory containing model directories
        model: 'dmo', 'hydro', or a replace config label
        realization: LP realization index (0-9)
        source_z: Source redshift (will be matched to nearest source plane)
        grid_res: Expected grid resolution
    
    Returns:
        ConvergenceMap instance (or numpy array if LensTools not available)
    """
    # Path to this realization
    real_dir = os.path.join(lp_dir, model, f'LP_{realization:02d}')
    
    if not os.path.exists(real_dir):
        raise FileNotFoundError(f"Realization directory not found: {real_dir}")
    
    # Find which source plane matches
    source_idx, source_info = get_source_plane_for_redshift(source_z)
    if source_info is not None:
        source_chi, source_z_exact = source_info
        print(f"Target source z={source_z:.3f} -> Source plane S{source_idx+1}: z={source_z_exact:.3f}, χ={source_chi:.2f} Mpc/h")
    
    # Load all planes up to the source plane
    mass_planes = []
    plane_infos = []
    
    for plane_idx in range(N_PLANES):
        filepath = os.path.join(real_dir, f'lenspot{plane_idx:02d}.dat')
        
        if not os.path.exists(filepath):
            print(f"Warning: Missing plane file {filepath}")
            continue
        
        try:
            info = get_plane_info(plane_idx)
            
            # Only include lens planes in front of source
            if info['lens_chi'] < source_chi:
                mass_data, grid = load_lensplane(filepath, expected_grid=grid_res)
                mass_planes.append(mass_data)
                plane_infos.append(info)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    print(f"Loaded {len(mass_planes)} lens planes for source at z_S={source_z_exact:.3f}")
    
    # Compute convergence
    kappa = mass_to_convergence_born(mass_planes, plane_infos, source_z, BOX_SIZE, COSMO)
    
    # Compute field of view in degrees
    # FOV = box_size / chi_s in radians
    fov_rad = BOX_SIZE / source_chi  # radians
    fov_deg = np.degrees(fov_rad)
    
    print(f"Field of view: {fov_deg:.4f} deg ({fov_deg * 60:.2f} arcmin)")
    
    if HAS_LENSTOOLS:
        conv_map = ConvergenceMap(data=kappa, angle=fov_deg * u.deg)
        return conv_map
    else:
        return kappa, fov_deg


def list_source_planes():
    """Print available source planes from κTNG configuration."""
    print("\nAvailable κTNG source planes (arXiv:2010.09731 Table 2):")
    print("-" * 60)
    print(f"{'Plane':<8} {'z_S':<10} {'χ_S [Mpc/h]':<15} {'FOV [deg]':<12}")
    print("-" * 60)
    for plane_num in sorted(SOURCE_PLANES.keys()):
        chi_s, z_s = SOURCE_PLANES[plane_num]
        fov_deg = np.degrees(BOX_SIZE / chi_s)
        print(f"S{plane_num:<7} {z_s:<10.3f} {chi_s:<15.2f} {fov_deg:<12.4f}")
    print("-" * 60)


def generate_convergence_with_raytracer(lp_dir, model, realization, source_z, 
                                         fov_deg=3.5, map_resolution=512, grid_res=4096):
    """
    Generate convergence map using LensTools RayTracer.
    
    This uses the full multi-plane ray-tracing algorithm with κTNG plane positions.
    
    Args:
        lp_dir: Base directory containing model directories
        model: 'dmo', 'hydro', or a replace config label
        realization: LP realization index (0-9)
        source_z: Source redshift (will be matched to nearest source plane)
        fov_deg: Field of view in degrees
        map_resolution: Output map resolution
        grid_res: Input plane grid resolution
    
    Returns:
        ConvergenceMap instance
    """
    if not HAS_LENSTOOLS:
        raise ImportError("LensTools required for ray-tracing")
    
    real_dir = os.path.join(lp_dir, model, f'LP_{realization:02d}')
    
    # Find the matching source plane
    source_idx, source_info = get_source_plane_for_redshift(source_z)
    if source_info is not None:
        source_chi, source_z_exact = source_info
        print(f"Target source z={source_z:.3f} -> Source plane S{source_idx+1}: z={source_z_exact:.3f}")
    else:
        source_chi = COSMO.comoving_distance(source_z).to(u.Mpc).value / COSMO.h
        source_z_exact = source_z
    
    # Initialize ray tracer with DensityPlane type
    tracer = RayTracer(lens_type=DensityPlane)
    
    h = COSMO.h
    
    # Collect all planes with their info, then sort by redshift before adding
    planes_to_add = []
    
    # Mean 3D matter density (comoving) in Msun/h / (Mpc/h)³
    rho_m_0 = COSMO.Om0 * COSMO.critical_density0.to(u.Msun / u.Mpc**3).value * h**2
    
    for plane_idx in range(N_PLANES):
        filepath = os.path.join(real_dir, f'lenspot{plane_idx:02d}.dat')
        
        if not os.path.exists(filepath):
            continue
        
        try:
            info = get_plane_info(plane_idx)
            
            # Use κTNG Table 2 values
            chi_lens = info['lens_chi']  # Mpc/h
            z_lens = info['lens_redshift']
            thickness = info['thickness']
            
            # Skip planes at or beyond source
            if chi_lens >= source_chi:
                continue
            
            mass_data, grid = load_lensplane(filepath, expected_grid=grid_res)
            
            # Convert mass to dimensionless surface density contrast
            # δ_Σ = (Σ - Σ̄) / Σ̄
            pixel_size = BOX_SIZE / grid
            pixel_area = pixel_size ** 2
            
            # Surface density and mean
            sigma = mass_data / pixel_area
            sigma_mean = rho_m_0 * thickness
            
            # Density contrast
            delta = (sigma - sigma_mean) / sigma_mean
            
            # Create DensityPlane with exact κTNG positions
            plane = DensityPlane(
                data=delta.astype(np.float64),
                angle=BOX_SIZE * u.Mpc / h,  # Physical length (proper)
                redshift=z_lens,
                cosmology=COSMO,
                comoving_distance=chi_lens * u.Mpc / h  # Use exact Table 2 value
            )
            
            planes_to_add.append((z_lens, chi_lens, plane, info['plane_number']))
            
        except Exception as e:
            print(f"Error processing plane {plane_idx}: {e}")
            continue
    
    # Sort planes by redshift and add to tracer
    planes_to_add.sort(key=lambda x: x[0])
    
    for z_lens, chi_lens, plane, plane_num in planes_to_add:
        tracer.addLens(plane)
        print(f"  Added L{plane_num}: z={z_lens:.3f}, χ={chi_lens:.2f} Mpc/h")
    
    print(f"Added {len(tracer.lens)} lens planes to ray tracer")
    
    # Create initial ray positions
    b = np.linspace(0, fov_deg, map_resolution)
    xx, yy = np.meshgrid(b, b)
    pos = np.array([xx, yy]) * u.deg
    
    # Compute convergence with Born approximation
    print(f"Ray-tracing to z_S={source_z_exact:.3f}...")
    kappa = tracer.convergenceBorn(pos, z=source_z_exact)
    
    # Create convergence map
    conv_map = ConvergenceMap(data=kappa, angle=fov_deg * u.deg, 
                               redshift=source_z_exact, cosmology=COSMO)
    
    return conv_map


def write_info_file(output_dir, n_planes=N_PLANES):
    """
    Write an info file with lens plane positions from κTNG Table 2.
    
    This is useful for documentation and validation.
    """
    info_file = os.path.join(output_dir, 'plane_info.txt')
    
    with open(info_file, 'w') as f:
        f.write("# κTNG Lens Plane Configuration (arXiv:2010.09731 Table 2)\n")
        f.write("# plane_idx: 0-based index (file naming)\n")
        f.write("# L#: Lens plane number (1-based)\n")
        f.write("# chi_L: Comoving distance to lens plane center [Mpc/h]\n")
        f.write("# z_L: Redshift at lens plane\n")
        f.write("# S#: Corresponding source plane number\n")
        f.write("# chi_S: Comoving distance to source plane [Mpc/h]\n")
        f.write("# z_S: Redshift at source plane\n")
        f.write("# snap: TNG snapshot number\n")
        f.write("# pps: Slice within snapshot (0 or 1)\n")
        f.write("#\n")
        f.write(f"# {'idx':<4} {'L#':<4} {'chi_L':<10} {'z_L':<8} {'S#':<4} {'chi_S':<10} {'z_S':<8} {'snap':<5} {'pps':<3}\n")
        
        for plane_idx in range(n_planes):
            info = get_plane_info(plane_idx)
            plane_num = info['plane_number']
            f.write(f"  {plane_idx:<4} L{plane_num:<3} {info['lens_chi']:<10.2f} {info['lens_redshift']:<8.4f} ")
            f.write(f"S{plane_num:<3} {info['source_chi']:<10.2f} {info['source_redshift']:<8.4f} ")
            f.write(f"{info['snapshot']:<5} {info['pps_slice']:<3}\n")
    
    print(f"Wrote plane info file: {info_file}")
    return info_file


def generate_convergence(lp_dir, model='dmo', realization=0, source_z=1.0, 
                          grid_res=4096, method='born', fov=None, map_res=512,
                          output=None, verbose=True):
    """
    Generate a weak lensing convergence map from lensplanes.
    
    This is the main entry point for generating convergence maps using κTNG methodology.
    Can be called from notebooks or scripts.
    
    Args:
        lp_dir: Base lensplane directory (e.g., '/path/to/L205n2500TNG')
        model: Model type - 'dmo', 'hydro', or replace config label (default: 'dmo')
        realization: LP realization index 0-9 (default: 0)
        source_z: Source redshift, will be matched to nearest κTNG source plane (default: 1.0)
        grid_res: Input plane grid resolution (default: 4096)
        method: 'born' (direct Born approximation) or 'raytracer' (LensTools) (default: 'born')
        fov: Field of view in degrees. If None, auto-computed from source χ (default: None)
        map_res: Output map resolution for raytracer method (default: 512)
        output: Output file path. If None, map is returned but not saved (default: None)
        verbose: Print progress messages (default: True)
    
    Returns:
        If LensTools available: ConvergenceMap instance
        If LensTools not available: tuple (kappa_array, fov_deg)
    
    Example:
        >>> from generate_wl_maps import generate_convergence, list_source_planes
        >>> list_source_planes()  # See available source redshifts
        >>> conv_map = generate_convergence('/path/to/lensplanes', model='hydro', source_z=1.0)
        >>> print(f"κ mean: {conv_map.data.mean():.4f}, std: {conv_map.data.std():.4f}")
    """
    if verbose:
        print(f"Generating convergence map (κTNG methodology):")
        print(f"  Model: {model}")
        print(f"  Realization: {realization}")
        print(f"  Source z: {source_z}")
        print(f"  Method: {method}")
    
    if method == 'born':
        result = generate_convergence_map_born(
            lp_dir, model, realization, source_z, grid_res
        )
        
        if HAS_LENSTOOLS:
            conv_map = result
        else:
            kappa, fov_deg = result
            if verbose:
                print(f"\nConvergence statistics:")
                print(f"  Mean: {kappa.mean():.6f}")
                print(f"  Std: {kappa.std():.6f}")
                print(f"  Min: {kappa.min():.6f}")
                print(f"  Max: {kappa.max():.6f}")
            
            if output:
                np.savez(output.replace('.fits', '.npz'), 
                         kappa=kappa, fov_deg=fov_deg, source_z=source_z)
                if verbose:
                    print(f"\nSaved to {output.replace('.fits', '.npz')}")
            
            return kappa, fov_deg
    else:
        # For raytracer, compute default FOV from source distance
        if fov is None:
            source_idx, source_info = get_source_plane_for_redshift(source_z)
            if source_info:
                source_chi = source_info[0]
                fov = np.degrees(BOX_SIZE / source_chi)
                if verbose:
                    print(f"  Auto FOV: {fov:.4f} deg")
            else:
                fov = 3.5  # Default fallback
                
        conv_map = generate_convergence_with_raytracer(
            lp_dir, model, realization, source_z, fov, map_res, grid_res
        )
    
    if verbose:
        print(f"\nConvergence statistics:")
        print(f"  Mean: {conv_map.data.mean():.6f}")
        print(f"  Std: {conv_map.data.std():.6f}")
        print(f"  Min: {conv_map.data.min():.6f}")
        print(f"  Max: {conv_map.data.max():.6f}")
    
    if output:
        conv_map.save(output)
        if verbose:
            print(f"\nSaved to {output}")
    
    return conv_map


def main():
    """Command-line interface for generate_wl_maps."""
    parser = argparse.ArgumentParser(
        description='Generate weak lensing maps from lensplanes using κTNG methodology',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available source redshifts
  python generate_wl_maps.py --list-sources
  
  # Generate convergence map at z=1.0 using Born approximation  
  python generate_wl_maps.py --lp-dir /path/to/L205n2500TNG --model dmo --source-z 1.0
  
  # Use ray-tracer method with custom FOV
  python generate_wl_maps.py --lp-dir /path/to/L205n2500TNG --model hydro \\
      --source-z 0.5 --method raytracer --fov 5.0
  
  # Write plane configuration info file
  python generate_wl_maps.py --write-info --lp-dir /path/to/output

For notebook usage:
  >>> from generate_wl_maps import generate_convergence, list_source_planes
  >>> conv_map = generate_convergence('/path/to/lensplanes', source_z=1.0)
"""
    )
    parser.add_argument('--lp-dir', type=str,
                        help='Base lensplane directory (e.g., /path/to/L205n2500TNG)')
    parser.add_argument('--model', type=str, default='dmo',
                        help='Model: dmo, hydro, or replace config label')
    parser.add_argument('--realization', type=int, default=0,
                        help='LP realization index (0-9)')
    parser.add_argument('--source-z', type=float, default=1.0,
                        help='Source redshift (will be matched to nearest κTNG source plane)')
    parser.add_argument('--grid', type=int, default=4096,
                        help='Input plane grid resolution')
    parser.add_argument('--output', type=str, default='convergence.fits',
                        help='Output file path')
    parser.add_argument('--method', type=str, default='born',
                        choices=['born', 'raytracer'],
                        help='Method: born (direct) or raytracer (LensTools)')
    parser.add_argument('--fov', type=float, default=None,
                        help='Field of view in degrees (for raytracer; default: auto from source χ)')
    parser.add_argument('--map-res', type=int, default=512,
                        help='Output map resolution (for raytracer method)')
    parser.add_argument('--write-info', action='store_true',
                        help='Write plane info file with κTNG configuration')
    parser.add_argument('--list-sources', action='store_true',
                        help='List available source planes and exit')
    
    args = parser.parse_args()
    
    if args.list_sources:
        list_source_planes()
        return
    
    if args.write_info:
        if not args.lp_dir:
            print("Error: --lp-dir required for --write-info")
            return
        write_info_file(args.lp_dir)
        return
    
    if not args.lp_dir:
        parser.print_help()
        print("\nError: --lp-dir is required")
        return
    
    # Use the main generate function
    conv_map = generate_convergence(
        lp_dir=args.lp_dir,
        model=args.model,
        realization=args.realization,
        source_z=args.source_z,
        grid_res=args.grid,
        method=args.method,
        fov=args.fov,
        map_res=args.map_res,
        output=args.output,
        verbose=True
    )
    
    # Compute and print power spectrum if we have a ConvergenceMap
    if HAS_LENSTOOLS and hasattr(conv_map, 'powerSpectrum'):
        l_edges = np.logspace(2, 4, 30)
        l, Pl = conv_map.powerSpectrum(l_edges)
        
        print(f"\nPower spectrum (first 5 bins):")
        for i in range(min(5, len(l))):
            print(f"  l={l[i]:.0f}: P(l)={Pl[i]:.2e}")


if __name__ == '__main__':
    main()
