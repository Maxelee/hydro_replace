#!/usr/bin/env python3
"""
Quick analysis of existing power spectra from hydro replacement.
Uses pre-computed P(k) data at /mnt/home/mlee1/ceph/power_spectra/

Run interactively or via: python scripts/analyze_existing_pk.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Configuration
PK_DIR = Path("/mnt/home/mlee1/ceph/power_spectra")
OUTPUT_DIR = Path("/mnt/home/mlee1/hydro_replace2/figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_pk_3d(filepath):
    """Load 3D power spectrum file (k, Pk0, Pk2, Pk4, Pkphase, Nmodes)."""
    data = np.loadtxt(filepath)
    return {
        'k': data[:, 0],
        'Pk': data[:, 1],  # monopole
        'Pk2': data[:, 2],
        'Pk4': data[:, 3],
        'Nmodes': data[:, 5]
    }

def get_pk_files(mode='normal', radius=5):
    """Get all P(k) files for a given mode and radius."""
    pattern = f"pixelized_maps_res1024_{mode}_rad{radius}_*_Pk3D.txt"
    files = sorted(glob.glob(str(PK_DIR / pattern)))
    return files

def parse_mass_label(filename):
    """Extract mass label from filename."""
    parts = Path(filename).stem.split('_')
    mass_idx = [i for i, p in enumerate(parts) if p.startswith('mass')][0]
    mass_label = '_'.join(parts[mass_idx:]).replace('_Pk3D', '')
    return mass_label

def plot_suppression_by_mass(mode='normal', radius=5, save=True):
    """
    Plot P(k) suppression S(k) = P_replaced / P_DMO for different mass bins.
    """
    files = get_pk_files(mode, radius)
    
    if not files:
        print(f"No files found for mode={mode}, radius={radius}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Separate cumulative and regular mass bins
    cumul_files = [f for f in files if 'cumul' in f]
    regular_files = [f for f in files if 'cumul' not in f]
    
    # Colors for mass bins
    colors = plt.cm.viridis(np.linspace(0, 0.9, 6))
    
    # Plot cumulative bins
    ax = axes[0]
    ax.set_title(f'Cumulative Mass Bins ({mode}, {radius}×R_200)')
    for i, f in enumerate(sorted(cumul_files)):
        pk = load_pk_3d(f)
        label = parse_mass_label(f).replace('massgt', 'M > 10^').replace('_cumul', '')
        ax.semilogx(pk['k'], pk['Pk'], label=label, color=colors[i % len(colors)], lw=2)
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('P(k) [(Mpc/h)³]')
    ax.legend(fontsize=8)
    ax.set_xlim(0.03, 10)
    ax.grid(True, alpha=0.3)
    
    # Plot regular bins
    ax = axes[1]
    ax.set_title(f'Regular Mass Bins ({mode}, {radius}×R_200)')
    for i, f in enumerate(sorted(regular_files)):
        pk = load_pk_3d(f)
        label = parse_mass_label(f).replace('mass', 'M: ')
        ax.semilogx(pk['k'], pk['Pk'], label=label, color=colors[i % len(colors)], lw=2)
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('P(k) [(Mpc/h)³]')
    ax.legend(fontsize=8)
    ax.set_xlim(0.03, 10)
    ax.grid(True, alpha=0.3)
    
    # Plot ratio of normal to inverse (suppression vs enhancement)
    ax = axes[2]
    ax.set_title('Suppression: P_normal / P_inverse')
    inv_files = get_pk_files('inverse', radius)
    cumul_inv = [f for f in inv_files if 'cumul' in f]
    
    for i, (nf, invf) in enumerate(zip(sorted(cumul_files), sorted(cumul_inv))):
        pk_n = load_pk_3d(nf)
        pk_inv = load_pk_3d(invf)
        # Interpolate to same k
        ratio = pk_n['Pk'] / pk_inv['Pk']
        label = parse_mass_label(nf).replace('massgt', 'M > 10^').replace('_cumul', '')
        ax.semilogx(pk_n['k'], ratio, label=label, color=colors[i % len(colors)], lw=2)
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('S(k) = P_normal / P_inverse')
    ax.legend(fontsize=8)
    ax.set_xlim(0.03, 10)
    ax.set_ylim(0.8, 1.2)
    ax.grid(True, alpha=0.3)
    
    # Radius comparison for cumulative M>12
    ax = axes[3]
    ax.set_title('Radius Dependence (M > 10¹² cumulative)')
    for rad, color in zip([1, 3, 5], ['blue', 'orange', 'green']):
        files_rad = get_pk_files('normal', rad)
        f = [f for f in files_rad if 'gt12.0_cumul' in f]
        if f:
            pk = load_pk_3d(f[0])
            ax.semilogx(pk['k'], pk['Pk'], label=f'{rad}×R_200', color=color, lw=2)
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('P(k) [(Mpc/h)³]')
    ax.legend()
    ax.set_xlim(0.03, 10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        outfile = OUTPUT_DIR / f'pk_summary_{mode}_rad{radius}.png'
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"Saved: {outfile}")
    
    plt.show()

def plot_all_radii_comparison():
    """Compare P(k) across all three radii for key mass bins."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    mass_bins = ['gt12.0_cumul', 'gt13.0_cumul', 'gt14.0_cumul']
    radii = [1, 3, 5]
    
    for j, mass_bin in enumerate(mass_bins):
        # Normal mode
        ax = axes[0, j]
        ax.set_title(f'Normal Mode: M > 10^{mass_bin.split(".")[0][-2:]}')
        for rad, color in zip(radii, ['blue', 'orange', 'green']):
            files = get_pk_files('normal', rad)
            f = [f for f in files if mass_bin in f]
            if f:
                pk = load_pk_3d(f[0])
                ax.semilogx(pk['k'], pk['Pk'], label=f'{rad}×R_200', color=color, lw=2)
        ax.set_xlabel('k [h/Mpc]')
        ax.set_ylabel('P(k)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.03, 10)
        
        # Inverse mode
        ax = axes[1, j]
        ax.set_title(f'Inverse Mode: M > 10^{mass_bin.split(".")[0][-2:]}')
        for rad, color in zip(radii, ['blue', 'orange', 'green']):
            files = get_pk_files('inverse', rad)
            f = [f for f in files if mass_bin in f]
            if f:
                pk = load_pk_3d(f[0])
                ax.semilogx(pk['k'], pk['Pk'], label=f'{rad}×R_200', color=color, lw=2)
        ax.set_xlabel('k [h/Mpc]')
        ax.set_ylabel('P(k)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.03, 10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pk_all_radii_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'pk_all_radii_comparison.png'}")
    plt.show()

def analyze_bcm_profiles():
    """Analyze existing BCM profile outputs."""
    import h5py
    
    halo_dir = Path("/mnt/home/mlee1/ceph/baryonification_output/halos")
    halo_files = sorted(halo_dir.glob("*.h5"))[:10]  # Sample first 10
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Density profiles
    ax = axes[0, 0]
    ax.set_title('Sample Density Profiles')
    
    for i, hf in enumerate(halo_files[:5]):
        with h5py.File(hf, 'r') as f:
            bins = f['profile_bins'][:]
            r = 0.5 * (bins[:-1] + bins[1:])  # bin centers
            rho_hydro = f['hydro_density'][:]
            rho_dmo = f['dmo_density'][:]
            rho_bcm = f['baryon_density'][:]
        
        if i == 0:
            ax.loglog(r, rho_hydro, 'b-', lw=2, alpha=0.5, label='Hydro')
            ax.loglog(r, rho_dmo, 'k-', lw=2, alpha=0.5, label='DMO')
            ax.loglog(r, rho_bcm, 'r-', lw=2, alpha=0.5, label='BCM')
        else:
            ax.loglog(r, rho_hydro, 'b-', lw=1, alpha=0.3)
            ax.loglog(r, rho_dmo, 'k-', lw=1, alpha=0.3)
            ax.loglog(r, rho_bcm, 'r-', lw=1, alpha=0.3)
    
    ax.set_xlabel('r / R_200')
    ax.set_ylabel('ρ(r) [M☉/h / (Mpc/h)³]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(1, color='gray', linestyle='--', alpha=0.5, label='R_200')
    
    # BCM/Hydro ratio
    ax = axes[0, 1]
    ax.set_title('BCM / Hydro Ratio')
    
    for hf in halo_files[:10]:
        with h5py.File(hf, 'r') as f:
            bins = f['profile_bins'][:]
            r = 0.5 * (bins[:-1] + bins[1:])
            rho_hydro = f['hydro_density'][:]
            rho_bcm = f['baryon_density'][:]
        
        # Avoid division by zero
        valid = (rho_hydro > 0) & (rho_bcm > 0)
        ratio = np.where(valid, rho_bcm / rho_hydro, np.nan)
        ax.semilogx(r, ratio, alpha=0.5)
    
    ax.axhline(1, color='gray', linestyle='--')
    ax.set_xlabel('r / R_200')
    ax.set_ylabel('ρ_BCM / ρ_Hydro')
    ax.set_ylim(0.5, 2.0)
    ax.grid(True, alpha=0.3)
    
    # Distribution of profile errors
    ax = axes[1, 0]
    ax.set_title('Distribution of Profile Errors at R_200')
    
    errors_r200 = []
    for hf in halo_files:
        with h5py.File(hf, 'r') as f:
            bins = f['profile_bins'][:]
            r = 0.5 * (bins[:-1] + bins[1:])
            rho_hydro = f['hydro_density'][:]
            rho_bcm = f['baryon_density'][:]
            
            # Find bin closest to R_200
            idx = np.argmin(np.abs(r - 1.0))
            if rho_hydro[idx] > 0:
                errors_r200.append((rho_bcm[idx] - rho_hydro[idx]) / rho_hydro[idx])
    
    ax.hist(errors_r200, bins=20, edgecolor='black')
    ax.set_xlabel('(ρ_BCM - ρ_Hydro) / ρ_Hydro at R_200')
    ax.set_ylabel('Count')
    ax.axvline(0, color='gray', linestyle='--')
    ax.axvline(np.median(errors_r200), color='red', linestyle='-', label=f'Median: {np.median(errors_r200):.2f}')
    ax.legend()
    
    # Halo count summary
    ax = axes[1, 1]
    ax.text(0.5, 0.5, f'Total halos processed: {len(list(halo_dir.glob("*.h5")))}',
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.text(0.5, 0.3, f'Sample analyzed: {len(halo_files)}',
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bcm_profile_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'bcm_profile_analysis.png'}")
    plt.show()

if __name__ == "__main__":
    print("Analyzing existing power spectra data...")
    print(f"P(k) directory: {PK_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # List available files
    all_files = list(PK_DIR.glob("*_Pk3D.txt"))
    print(f"Found {len(all_files)} P(k) files")
    
    # Run analyses
    print("\n1. Plotting P(k) summary for normal mode, 5×R_200...")
    plot_suppression_by_mass(mode='normal', radius=5)
    
    print("\n2. Plotting all radii comparison...")
    plot_all_radii_comparison()
    
    print("\n3. Analyzing BCM profiles...")
    analyze_bcm_profiles()
    
    print("\nDone! Check figures/ directory for outputs.")
