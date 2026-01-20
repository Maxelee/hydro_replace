"""
plot_baryonic_response_combined.py

Combined plotting script with both simple and comprehensive visualizations.
Complete version with all functions and robust error handling.

Usage:
    python plot_baryonic_response_combined.py --style simple    # Your original style
    python plot_baryonic_response_combined.py --style full      # All figures
    python plot_baryonic_response_combined.py --style both      # Everything (default)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path
import pickle
import argparse
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')

# =============================================================================
# Journal-ready plotting configuration
# =============================================================================

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'

plt.rcParams['figure.figsize'] = (7, 4.5)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.grid'] = False
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.9

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path('./response_analysis_output')
FIGURE_DIR = Path('./figures')
FIGURE_DIR.mkdir(exist_ok=True)

MASS_BIN_EDGES = np.array([1.00e12, 3.16e12, 1.00e13, 3.16e13, 1.00e15])
ALPHA_EDGES = np.array([0.0, 0.5, 1.0, 3.0, 5.0])
R_FACTORS = [0.5, 1.0, 3.0, 5.0]
MASS_THRESHOLDS = [1.00e12, 3.16e12, 1.00e13, 3.16e13]

# =============================================================================
# Load data
# =============================================================================

def load_results(stat_key='C_ell', z=23):
    """Load analysis results."""
    npz_path = OUTPUT_DIR / f'response_{stat_key}_z{z:02d}.npz'
    pkl_path = OUTPUT_DIR / f'response_{stat_key}_z{z:02d}_dicts.pkl'
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Results not found: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    with open(pkl_path, 'rb') as f:
        dicts = pickle.load(f)
    
    results = {
        'stat_key': stat_key,
        'z': int(data['z']),
        'S_D': data['S_D'],
        'S_H': data['S_H'],
        'Delta_S': data['Delta_S'],
        'S_D_err': data['S_D_err'] if 'S_D_err' in data else None,
        'S_H_err': data['S_H_err'] if 'S_H_err' in data else None,
        'ell': data['ell'] if 'ell' in data else None,
        'snr_mid': data['snr_mid'] if 'snr_mid' in data else None,
        'mass_bin_edges': data['mass_bin_edges'],
        'alpha_edges': data['alpha_edges'],
    }
    results.update(dicts)
    
    return results

# =============================================================================
# SIMPLE PLOTS (your original style)
# =============================================================================

def plot_F_vs_ell_simple(stat_key='C_ell', z=23, save=True):
    """
    Plot F_S vs ell (or nu), grouped by radius factor.
    One panel per alpha, multiple M_min curves per panel.
    
    This matches your original plotting style.
    """
    print(f"  Creating simple F_S plot for {stat_key}...")
    
    results = load_results(stat_key, z)
    cumulative = results['cumulative_responses']
    
    if stat_key == 'C_ell':
        x = results['ell']
        xlabel = r'$\ell$'
        title_prefix = r'$F_{C_\ell}$'
    elif stat_key == 'peaks':
        x = results['snr_mid']
        xlabel = r'Peak height $\nu$ (SNR)'
        title_prefix = r'$F_{\rm peaks}$'
    else:
        x = results['snr_mid']
        xlabel = r'Minima depth $\nu$ (SNR)'
        title_prefix = r'$F_{\rm minima}$'
    
    # Create figure: 4 panels (one per r_factor)
    fig, axs = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(12, 4))
    
    # Identify bad bins (where Delta_S is too small)
    Delta_S = results['Delta_S']
    S_D = results['S_D']
    bad_mask = np.abs(Delta_S / (np.abs(S_D) + 1e-30)) < 0.01
    
    for i, r_factor in enumerate(R_FACTORS):
        ax = axs[i]
        ax.set_title(r'$\alpha=$' + f'{r_factor}' + r'$R_{200}$', fontsize=11)
        
        # Find all models with this r_factor
        plotted_any = False
        for label, data in cumulative.items():
            alpha_max = data['alpha_max']
            
            if np.isclose(alpha_max, r_factor):
                M_min = data['M_min']
                F_S = data['F_S'].copy()
                
                # Mask bad bins (set to 1 for visualization)
                F_S[bad_mask] = 1.0
                
                # Plot
                mass_thresh = np.round(np.log10(M_min), 2)
                
                if stat_key == 'C_ell':
                    ax.semilogx(x, F_S, label=f'M > {mass_thresh}', linewidth=1.5)
                else:
                    ax.plot(x, F_S, 'o-', label=f'M > {mass_thresh}', 
                           linewidth=1.5, markersize=4)
                
                plotted_any = True
        
        if not plotted_any:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')
        
        # Formatting
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    # Legend on first panel
    axs[0].legend(ncols=2, fontsize=8, loc='best')
    
    # Labels
    for i in range(4):
        axs[i].set_xlabel(xlabel, fontsize=11)
    axs[0].set_ylabel(title_prefix, fontsize=11)
    
    plt.tight_layout()
    
    if save:
        fname = FIGURE_DIR / f'F_{stat_key}_simple.pdf'
        plt.savefig(fname)
        print(f'    Saved {fname}')
    
    return fig

# =============================================================================
# COMPREHENSIVE PLOTS
# =============================================================================

def plot_tile_response_heatmap(results, stat_key='C_ell', k_range=None, save=True):
    """
    Create 2D heatmap of tile responses Delta F_S(a,i) in halo mass-radius space.
    """
    print(f"  Creating tile heatmap for {stat_key}...")
    
    tile_responses = results.get('tile_responses', {})
    
    if len(tile_responses) == 0:
        print(f"    Warning: No tile responses found for {stat_key}")
        return None
    
    # Build 2D array of responses
    n_mass = len(MASS_BIN_EDGES) - 1
    n_alpha = len(ALPHA_EDGES) - 1
    
    response_map = np.zeros((n_mass, n_alpha))
    response_map[:] = np.nan  # Default to NaN for missing data
    
    for (a, i), Delta_F in tile_responses.items():
        # Average over relevant range
        if stat_key == 'C_ell' and k_range is not None:
            ell = results['ell']
            mask = (ell >= k_range[0]) & (ell <= k_range[1])
            response_map[a, i] = np.nanmean(Delta_F[mask])
        elif stat_key in ['peaks', 'minima'] and k_range is not None:
            snr = results['snr_mid']
            mask = (snr >= k_range[0]) & (snr <= k_range[1])
            response_map[a, i] = np.nanmean(Delta_F[mask])
        else:
            response_map[a, i] = np.nanmean(Delta_F)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Use diverging colormap centered at 0
    vmax = np.nanmax(np.abs(response_map))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0
    
    im = ax.imshow(
        response_map,
        origin='lower',
        aspect='auto',
        cmap='RdBu_r',
        vmin=-vmax,
        vmax=vmax,
        extent=[-0.5, n_alpha-0.5, -0.5, n_mass-0.5]
    )
    
    # Add values as text
    for a in range(n_mass):
        for i in range(n_alpha):
            val = response_map[a, i]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.5*vmax else 'black'
                ax.text(i, a, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=9, weight='bold')
    
    # Set ticks and labels
    ax.set_xticks(range(n_alpha))
    ax.set_xticklabels([r'$[{:.1f}, {:.1f})$'.format(ALPHA_EDGES[i], ALPHA_EDGES[i+1]) 
                        for i in range(n_alpha)], rotation=0, fontsize=9)
    
    ax.set_yticks(range(n_mass))
    ax.set_yticklabels([r'$[10^{{{:.1f}}}, 10^{{{:.1f}}})$'.format(
        np.log10(MASS_BIN_EDGES[a]), np.log10(MASS_BIN_EDGES[a+1])) 
        for a in range(n_mass)], fontsize=9)
    
    ax.set_xlabel(r'Radius shell ($R_{200}$ units)', fontsize=11)
    ax.set_ylabel(r'Halo mass bin ($M_\odot/h$)', fontsize=11)
    
    # Title
    if stat_key == 'C_ell' and k_range:
        title = (r'$\Delta F_{C_\ell}(M,\alpha)$: ' +
                f'$\ell\in[{k_range[0]}, {k_range[1]}]$')
    elif stat_key == 'peaks' and k_range:
        title = (r'$\Delta F_{\rm peaks}(M,\alpha)$: ' +
                f'$\\nu\in[{k_range[0]}, {k_range[1]}]$')
    elif stat_key == 'minima' and k_range:
        title = (r'$\Delta F_{\rm minima}(M,\alpha)$: ' +
                f'$\\nu\in[{k_range[0]}, {k_range[1]}]$')
    else:
        title = f'Tile response: {stat_key}'
    
    ax.set_title(title, fontsize=11, pad=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\Delta F_S(a,i)$', fontsize=10)
    
    plt.tight_layout()
    
    if save:
        fname = FIGURE_DIR / f'tile_heatmap_{stat_key}.pdf'
        plt.savefig(fname)
        print(f'    Saved {fname}')
    
    return fig

def plot_additivity_test(results, stat_key='C_ell', k_range=None, save=True):
    """
    Plot additivity: F_true vs F_lin and epsilon_S.
    With robust NaN/Inf handling.
    """
    print(f"  Creating additivity test for {stat_key}...")
    
    additivity_results = results.get('additivity_results', {})
    
    if len(additivity_results) == 0:
        print(f"    Warning: No additivity results for {stat_key}")
        return None
    
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    
    # Panel (a): F_true vs F_lin scatter
    ax1 = fig.add_subplot(gs[0])
    
    F_true_all = []
    F_lin_all = []
    
    for label, data in additivity_results.items():
        F_true = data['F_true']
        F_lin = data['F_lin']
        
        if stat_key == 'C_ell' and k_range:
            ell = results['ell']
            mask = (ell >= k_range[0]) & (ell <= k_range[1])
            F_true_mean = np.nanmean(F_true[mask])
            F_lin_mean = np.nanmean(F_lin[mask])
        else:
            F_true_mean = np.nanmean(F_true)
            F_lin_mean = np.nanmean(F_lin)
        
        # Only add if finite
        if np.isfinite(F_true_mean) and np.isfinite(F_lin_mean):
            F_true_all.append(F_true_mean)
            F_lin_all.append(F_lin_mean)
    
    if len(F_true_all) == 0:
        print(f"    Warning: No valid additivity data for {stat_key}, skipping plot")
        plt.close(fig)
        return None
    
    F_true_all = np.array(F_true_all)
    F_lin_all = np.array(F_lin_all)
    
    ax1.scatter(F_true_all, F_lin_all, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    # 1:1 line with finite limits
    lims = [
        min(F_true_all.min(), F_lin_all.min()) - 0.05,
        max(F_true_all.max(), F_lin_all.max()) + 0.05
    ]
    
    # Safety check
    if not (np.isfinite(lims[0]) and np.isfinite(lims[1])):
        lims = [-1, 2]  # Default fallback
    
    ax1.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5, label='1:1')
    
    ax1.set_xlabel(r'$F_S^{\rm (true)}$', fontsize=11)
    ax1.set_ylabel(r'$F_S^{\rm (lin)}$ (sum of tiles)', fontsize=11)
    ax1.set_title('(a) Additivity check', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    try:
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
    except:
        pass  # Skip if limits still problematic
    
    # Panel (b): Epsilon vs M_min
    ax2 = fig.add_subplot(gs[1])
    
    plotted_any = False
    for alpha_max in R_FACTORS:
        M_vals = []
        eps_vals = []
        
        for label, data in additivity_results.items():
            try:
                parsed = label.split('_')
                M_min = float(parsed[0][1:])  # Remove 'M'
                alpha_val = float(parsed[1][1:])  # Remove 'a'
                
                if np.isclose(alpha_val, alpha_max):
                    eps = data['mean_eps']
                    if np.isfinite(eps):
                        M_vals.append(M_min)
                        eps_vals.append(eps)
            except:
                continue
        
        if len(M_vals) > 0:
            idx = np.argsort(M_vals)
            M_vals = np.array(M_vals)[idx]
            eps_vals = np.array(eps_vals)[idx]
            
            label = r'$\alpha=' + f'{alpha_max:.1f}' + r'R_{200}$'
            ax2.plot(M_vals, eps_vals, 'o-', label=label, linewidth=2, markersize=6)
            plotted_any = True
    
    if not plotted_any:
        ax2.text(0.5, 0.5, 'No finite data', transform=ax2.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
    
    ax2.set_xlabel(r'$M_{\rm min}$ [$M_\odot/h$]', fontsize=11)
    ax2.set_ylabel(r'$\langle|\epsilon_S|\rangle$', fontsize=11)
    ax2.set_xscale('log')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    if plotted_any:
        ax2.legend(loc='best', fontsize=8)
    ax2.set_title(r'(b) Non-additivity vs mass', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Panel (c): Epsilon vs alpha
    ax3 = fig.add_subplot(gs[2])
    
    plotted_any = False
    for M_min in MASS_THRESHOLDS:
        alpha_vals = []
        eps_vals = []
        
        for label, data in additivity_results.items():
            try:
                parsed = label.split('_')
                M_val = float(parsed[0][1:])
                alpha_val = float(parsed[1][1:])
                
                if np.isclose(M_val, M_min):
                    eps = data['mean_eps']
                    if np.isfinite(eps):
                        alpha_vals.append(alpha_val)
                        eps_vals.append(eps)
            except:
                continue
        
        if len(alpha_vals) > 0:
            idx = np.argsort(alpha_vals)
            alpha_vals = np.array(alpha_vals)[idx]
            eps_vals = np.array(eps_vals)[idx]
            
            label = r'$M_{\rm min}=10^{' + f'{np.log10(M_min):.1f}' + r'}M_\odot/h$'
            ax3.plot(alpha_vals, eps_vals, 'o-', label=label, linewidth=2, markersize=6)
            plotted_any = True
    
    if not plotted_any:
        ax3.text(0.5, 0.5, 'No finite data', transform=ax3.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
    
    ax3.set_xlabel(r'$\alpha$ (radius factor)', fontsize=11)
    ax3.set_ylabel(r'$\langle|\epsilon_S|\rangle$', fontsize=11)
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    if plotted_any:
        ax3.legend(loc='best', fontsize=8)
    ax3.set_title(r'(c) Non-additivity vs radius', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    if save:
        fname = FIGURE_DIR / f'additivity_{stat_key}.pdf'
        plt.savefig(fname, bbox_inches='tight')
        print(f'    Saved {fname}')
    
    return fig

def plot_cumulative_response_curves(results, stat_key='C_ell', k_range=None, save=True):
    """
    Plot cumulative response F_S as a function of M_min (at fixed alpha)
    and as a function of alpha (at fixed M_min).
    """
    print(f"  Creating cumulative response curves for {stat_key}...")
    
    cumulative_responses = results['cumulative_responses']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Panel (a): F_S vs alpha at fixed M_min
    ax = axes[0]
    for M_min in MASS_THRESHOLDS:
        alpha_vals = []
        F_vals = []
        F_errs = []
        
        for label, data in cumulative_responses.items():
            if np.isclose(data['M_min'], M_min):
                alpha_vals.append(data['alpha_max'])
                
                # Average over range
                if stat_key == 'C_ell' and k_range:
                    ell = results['ell']
                    mask = (ell >= k_range[0]) & (ell <= k_range[1])
                    F_vals.append(np.nanmean(data['F_S'][mask]))
                    F_errs.append(np.nanmean(data['F_S_err'][mask]))
                else:
                    F_vals.append(np.nanmean(data['F_S']))
                    F_errs.append(np.nanmean(data['F_S_err']))
        
        if len(alpha_vals) > 0:
            # Sort by alpha
            idx = np.argsort(alpha_vals)
            alpha_vals = np.array(alpha_vals)[idx]
            F_vals = np.array(F_vals)[idx]
            F_errs = np.array(F_errs)[idx]
            
            label = r'$M_{\rm min}=' + f'10^{{{np.log10(M_min):.1f}}}' + r'M_\odot/h$'
            ax.plot(alpha_vals, F_vals, 'o-', label=label, linewidth=2, markersize=6)
            ax.fill_between(alpha_vals, F_vals - F_errs, F_vals + F_errs, alpha=0.2)
    
    ax.set_xlabel(r'$\alpha$ (radius factor)', fontsize=12)
    ax.set_ylabel(r'$F_S(M_{\rm min}, \alpha)$', fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='best', fontsize=9)
    ax.set_title('(a) Response vs radius (fixed mass)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel (b): F_S vs M_min at fixed alpha
    ax = axes[1]
    for alpha_max in R_FACTORS:
        M_vals = []
        F_vals = []
        F_errs = []
        
        for label, data in cumulative_responses.items():
            if np.isclose(data['alpha_max'], alpha_max):
                M_vals.append(data['M_min'])
                
                if stat_key == 'C_ell' and k_range:
                    ell = results['ell']
                    mask = (ell >= k_range[0]) & (ell <= k_range[1])
                    F_vals.append(np.nanmean(data['F_S'][mask]))
                    F_errs.append(np.nanmean(data['F_S_err'][mask]))
                else:
                    F_vals.append(np.nanmean(data['F_S']))
                    F_errs.append(np.nanmean(data['F_S_err']))
        
        if len(M_vals) > 0:
            idx = np.argsort(M_vals)
            M_vals = np.array(M_vals)[idx]
            F_vals = np.array(F_vals)[idx]
            F_errs = np.array(F_errs)[idx]
            
            label = r'$\alpha=' + f'{alpha_max:.1f}' + r'R_{200}$'
            ax.plot(M_vals, F_vals, 'o-', label=label, linewidth=2, markersize=6)
            ax.fill_between(M_vals, F_vals - F_errs, F_vals + F_errs, alpha=0.2)
    
    ax.set_xlabel(r'$M_{\rm min}$ [$M_\odot/h$]', fontsize=12)
    ax.set_ylabel(r'$F_S(M_{\rm min}, \alpha)$', fontsize=12)
    ax.set_xscale('log')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='best', fontsize=9)
    ax.set_title('(b) Response vs mass (fixed radius)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Overall title
    if stat_key == 'C_ell' and k_range:
        fig.suptitle(f'Cumulative response: $C_\\ell$ for $\\ell\\in[{k_range[0]}, {k_range[1]}]$',
                    fontsize=13, y=1.02)
    else:
        fig.suptitle(f'Cumulative response: {stat_key}', fontsize=13, y=1.02)
    
    plt.tight_layout()
    
    if save:
        fname = FIGURE_DIR / f'cumulative_response_{stat_key}.pdf'
        plt.savefig(fname)
        print(f'    Saved {fname}')
    
    return fig

def plot_response_spectrum(results, stat_key='C_ell', save=True):
    """
    Plot how tile responses vary with ell (for C_ell) or nu (for peaks).
    Show a few representative tiles.
    """
    print(f"  Creating response spectrum for {stat_key}...")
    
    tile_responses = results.get('tile_responses', {})
    
    if len(tile_responses) == 0:
        print(f"    Warning: No tile responses for {stat_key}")
        return None
    
    # Select representative tiles
    representative_tiles = [
        (0, 1),  # Low mass, inner
        (1, 2),  # Mid mass, virial
        (2, 2),  # High mass, virial
        (3, 3),  # Very high mass, outskirts
    ]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for tile_key in representative_tiles:
        if tile_key not in tile_responses:
            continue
        
        a, i = tile_key
        Delta_F = tile_responses[tile_key]
        
        M_label = f'$10^{{{np.log10(MASS_BIN_EDGES[a]):.1f}}}$-$10^{{{np.log10(MASS_BIN_EDGES[a+1]):.1f}}}$'
        alpha_label = f'{ALPHA_EDGES[i]:.1f}-{ALPHA_EDGES[i+1]:.1f}'
        label = f'{M_label} $M_\\odot/h$, {alpha_label}$R_{{200}}$'
        
        if stat_key == 'C_ell':
            x = results['ell']
            ax.plot(x, Delta_F, label=label, linewidth=2)
            ax.set_xlabel(r'Multipole $\ell$', fontsize=12)
            ax.set_xscale('log')
        else:
            x = results['snr_mid']
            ax.plot(x, Delta_F, 'o-', label=label, linewidth=2, markersize=4)
            ax.set_xlabel(r'Peak height $\nu$ (SNR)', fontsize=12)
    
    ax.set_ylabel(r'$\Delta F_S(M,\alpha)$', fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='best', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    
    if stat_key == 'C_ell':
        title = 'Tile responses vs multipole (representative tiles)'
    else:
        title = f'Tile responses vs SNR ({stat_key}, representative tiles)'
    ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    
    if save:
        fname = FIGURE_DIR / f'response_spectrum_{stat_key}.pdf'
        plt.savefig(fname)
        print(f'    Saved {fname}')
    
    return fig

def plot_cross_statistic_comparison(save=True):
    """
    Compare tile response patterns for C_ell, peaks, and minima.
    Three heatmaps side-by-side.
    """
    print("  Creating cross-statistic comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    stat_keys = ['C_ell', 'peaks', 'minima']
    titles = [
        r'(a) $C_\ell$ ($\ell\in[500,3000]$)',
        r'(b) Peaks ($\nu\in[2,5]$)',
        r'(c) Minima ($\nu\in[2,5]$)'
    ]
    k_ranges = [(500, 3000), (2, 5), (2, 5)]
    
    n_mass = len(MASS_BIN_EDGES) - 1
    n_alpha = len(ALPHA_EDGES) - 1
    
    # Find global vmax for consistent color scale
    all_maps = []
    for stat_key, k_range in zip(stat_keys, k_ranges):
        try:
            results = load_results(stat_key)
            tile_responses = results.get('tile_responses', {})
            
            response_map = np.zeros((n_mass, n_alpha))
            response_map[:] = np.nan
            
            for (a, i), Delta_F in tile_responses.items():
                if stat_key == 'C_ell':
                    ell = results['ell']
                    mask = (ell >= k_range[0]) & (ell <= k_range[1])
                    response_map[a, i] = np.nanmean(Delta_F[mask])
                else:
                    snr = results['snr_mid']
                    mask = (snr >= k_range[0]) & (snr <= k_range[1])
                    response_map[a, i] = np.nanmean(Delta_F[mask])
            
            all_maps.append(response_map)
        except Exception as e:
            print(f"    Warning: Could not load {stat_key}: {e}")
            all_maps.append(np.zeros((n_mass, n_alpha)))
    
    vmax = max([np.nanmax(np.abs(m)) for m in all_maps if not np.all(np.isnan(m))])
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0
    
    # Plot each heatmap
    for ax, response_map, title in zip(axes, all_maps, titles):
        im = ax.imshow(
            response_map,
            origin='lower',
            aspect='auto',
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax,
            extent=[-0.5, n_alpha-0.5, -0.5, n_mass-0.5]
        )
        
        # Add values
        for a in range(n_mass):
            for i in range(n_alpha):
                val = response_map[a, i]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 0.5*vmax else 'black'
                    ax.text(i, a, f'{val:.2f}', ha='center', va='center',
                           color=color, fontsize=8, weight='bold')
        
        ax.set_xticks(range(n_alpha))
        ax.set_xticklabels([f'{ALPHA_EDGES[i]:.1f}' for i in range(n_alpha)], fontsize=9)
        
        ax.set_yticks(range(n_mass))
        if ax == axes[0]:
            ax.set_yticklabels([f'{np.log10(MASS_BIN_EDGES[a]):.1f}' for a in range(n_mass)])
            ax.set_ylabel(r'$\log_{10}(M/[M_\odot/h])$ bin', fontsize=10)
        else:
            ax.set_yticklabels([])
        
        ax.set_xlabel(r'$\alpha/R_{200}$ shell', fontsize=10)
        ax.set_title(title, fontsize=11)
    
    # Single colorbar for all
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$\Delta F_S(M,\alpha)$', fontsize=10)
    
    plt.suptitle('Halo-space response patterns across statistics', fontsize=12, y=0.98)
    
    if save:
        fname = FIGURE_DIR / 'cross_statistic_heatmaps.pdf'
        plt.savefig(fname, bbox_inches='tight')
        print(f'    Saved {fname}')
    
    return fig

# =============================================================================
# Main execution
# =============================================================================

def main(style='both'):
    """
    Generate figures based on style selection.
    
    Args:
        style: 'simple', 'full', or 'both'
    """
    print("="*70)
    print(f"Generating {style} figures")
    print("="*70)
    
    # Check if output directory exists
    if not OUTPUT_DIR.exists():
        print(f"ERROR: {OUTPUT_DIR} not found. Run analysis first.")
        return
    
    statistics = ['C_ell', 'peaks', 'minima']
    
    if style in ['simple', 'both']:
        print("\n=== SIMPLE PLOTS (your original style) ===\n")
        for stat_key in statistics:
            try:
                plot_F_vs_ell_simple(stat_key)
            except Exception as e:
                print(f"ERROR plotting {stat_key} (simple): {e}")
                import traceback
                traceback.print_exc()
    
    if style in ['full', 'both']:
        print("\n=== COMPREHENSIVE PLOTS ===\n")
        
        # Individual statistic plots
        for stat_key in statistics:
            try:
                print(f"Processing {stat_key}...")
                results = load_results(stat_key)
                
                # Set k_range
                if stat_key == 'C_ell':
                    k_range = (500, 3000)
                else:
                    k_range = (2, 5)
                
                # Tile heatmap
                plot_tile_response_heatmap(results, stat_key=stat_key, k_range=k_range)
                
                # Cumulative curves
                plot_cumulative_response_curves(results, stat_key=stat_key, k_range=k_range)
                
                # Additivity
                plot_additivity_test(results, stat_key=stat_key, k_range=k_range)
                
                # Response spectrum
                plot_response_spectrum(results, stat_key=stat_key)
                
            except Exception as e:
                print(f"ERROR in comprehensive plots for {stat_key}: {e}")
                import traceback
                traceback.print_exc()
        
        # Cross-statistic comparison
        try:
            plot_cross_statistic_comparison()
        except Exception as e:
            print(f"ERROR in cross-statistic comparison: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"Figures saved to: {FIGURE_DIR}")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate baryonic response plots')
    parser.add_argument('--style', type=str, default='both',
                       choices=['simple', 'full', 'both'],
                       help='Plot style: simple (original), full (comprehensive), or both')
    
    args = parser.parse_args()
    
    main(style=args.style)
