#!/usr/bin/env python
"""
Test script for the unified statistics pipeline.

Tests individual components with actual data from the cluster.
"""

import os
import sys
import numpy as np

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from constants import RT_BASE, LP_BASE, N_LP, KAPPA_TARGETS, SMOOTHING_ARCMIN, PIXEL_SCALE_ARCMIN, SN_BINS
from stats_utils import load_kappa, smooth_kappa, compute_Cl, compute_peaks, compute_minima, compute_pdf

def test_load_kappa():
    """Test loading convergence maps from RT directory."""
    print("=" * 80)
    print("TEST 1: Loading convergence maps")
    print("=" * 80)
    
    # Test with DMO data
    kappa_path = os.path.join(RT_BASE, 'dmo', 'LP_00', 'run001', 'kappa20.dat')
    
    if not os.path.exists(kappa_path):
        print(f"SKIP: File not found: {kappa_path}")
        return False
    
    print(f"Loading: {kappa_path}")
    
    try:
        kappa = load_kappa(kappa_path, ng=1024)
        print(f"✓ Loaded successfully")
        print(f"  Shape: {kappa.shape}")
        print(f"  Range: [{np.min(kappa):.6f}, {np.max(kappa):.6f}]")
        print(f"  Mean: {np.mean(kappa):.6f}")
        print(f"  RMS: {np.std(kappa):.6f}")
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_smooth_kappa():
    """Test smoothing convergence maps."""
    print("\n" + "=" * 80)
    print("TEST 2: Smoothing convergence maps")
    print("=" * 80)
    
    kappa_path = os.path.join(RT_BASE, 'dmo', 'LP_00', 'run001', 'kappa20.dat')
    
    if not os.path.exists(kappa_path):
        print(f"SKIP: File not found: {kappa_path}")
        return False
    
    try:
        kappa = load_kappa(kappa_path)
        print(f"Original RMS: {np.std(kappa):.6f}")
        
        kappa_smooth = smooth_kappa(kappa, SMOOTHING_ARCMIN, PIXEL_SCALE_ARCMIN)
        print(f"✓ Smoothed successfully")
        print(f"  Smoothed RMS: {np.std(kappa_smooth):.6f}")
        print(f"  Smoothing scale: {SMOOTHING_ARCMIN} arcmin")
        print(f"  Pixel scale: {PIXEL_SCALE_ARCMIN:.3f} arcmin/pixel")
        
        sigma_pix = SMOOTHING_ARCMIN / PIXEL_SCALE_ARCMIN
        print(f"  Sigma (pixels): {sigma_pix:.1f}")
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_compute_Cl():
    """Test computing angular power spectrum."""
    print("\n" + "=" * 80)
    print("TEST 3: Computing angular power spectrum C_ℓ")
    print("=" * 80)
    
    kappa_path = os.path.join(RT_BASE, 'dmo', 'LP_00', 'run001', 'kappa20.dat')
    
    if not os.path.exists(kappa_path):
        print(f"SKIP: File not found: {kappa_path}")
        return False
    
    try:
        kappa = load_kappa(kappa_path)
        ell, Cl = compute_Cl(kappa, fov_deg=5.0)
        
        print(f"✓ Computed C_ℓ successfully")
        print(f"  Number of ℓ bins: {len(ell)}")
        print(f"  ℓ range: [{ell[0]:.1f}, {ell[-1]:.1f}]")
        print(f"  C_ℓ range: [{np.min(Cl):.3e}, {np.max(Cl):.3e}]")
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compute_peaks():
    """Test computing peak counts."""
    print("\n" + "=" * 80)
    print("TEST 4: Computing peak counts")
    print("=" * 80)
    
    kappa_path = os.path.join(RT_BASE, 'dmo', 'LP_00', 'run001', 'kappa20.dat')
    
    if not os.path.exists(kappa_path):
        print(f"SKIP: File not found: {kappa_path}")
        return False
    
    try:
        kappa = load_kappa(kappa_path)
        kappa_smooth = smooth_kappa(kappa, SMOOTHING_ARCMIN, PIXEL_SCALE_ARCMIN)
        rms = np.std(kappa_smooth)
        
        peaks = compute_peaks(kappa_smooth, rms, SN_BINS)
        minima = compute_minima(kappa_smooth, rms, SN_BINS)
        pdf = compute_pdf(kappa_smooth, rms, SN_BINS)
        
        print(f"✓ Computed statistics successfully")
        print(f"  S/N bins: {len(SN_BINS) - 1}")
        print(f"  S/N range: [{SN_BINS[0]}, {SN_BINS[-1]}]")
        print(f"  Total peaks: {np.sum(peaks)}")
        print(f"  Total minima: {np.sum(minima)}")
        print(f"  Total pixels: {np.sum(pdf)}")
        print(f"  Expected pixels: {1024 * 1024}")
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_models():
    """Test loading data from multiple models."""
    print("\n" + "=" * 80)
    print("TEST 5: Loading data from multiple models")
    print("=" * 80)
    
    models = ['dmo', 'hydro']
    
    # Add some Replace models that exist
    replace_models = [
        'hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.0_Ro_0.5',
        'hydro_replace_Ml_1.00e12_Mu_3.16e12_Ri_0.5_Ro_1.0',
    ]
    
    success_count = 0
    for model in models + replace_models:
        kappa_path = os.path.join(RT_BASE, model, 'LP_00', 'run001', 'kappa20.dat')
        
        if os.path.exists(kappa_path):
            try:
                kappa = load_kappa(kappa_path)
                rms = np.std(kappa)
                print(f"✓ {model:60s} RMS = {rms:.6f}")
                success_count += 1
            except Exception as e:
                print(f"✗ {model:60s} ERROR: {e}")
        else:
            print(f"- {model:60s} (file not found)")
    
    print(f"\nLoaded {success_count}/{len(models) + len(replace_models)} models")
    return success_count > 0


def test_lensplane_paths():
    """Test that lensplane paths are accessible."""
    print("\n" + "=" * 80)
    print("TEST 6: Checking lensplane directory structure")
    print("=" * 80)
    
    # Check DMO lensplanes
    lp_dir = os.path.join(LP_BASE, 'dmo')
    
    if not os.path.exists(lp_dir):
        print(f"✗ LP base directory not found: {lp_dir}")
        return False
    
    print(f"✓ LP base directory exists: {lp_dir}")
    
    # Check LP directories
    lp_dirs = [os.path.join(lp_dir, f'LP_{i:02d}') for i in range(5)]
    existing = [d for d in lp_dirs if os.path.exists(d)]
    
    print(f"  Found {len(existing)}/{len(lp_dirs)} LP directories")
    
    if existing:
        # Check lenspot files in first LP
        lenspot_path = os.path.join(existing[0], 'lenspot00.dat')
        if os.path.exists(lenspot_path):
            print(f"✓ Example lenspot file: {lenspot_path}")
            file_size = os.path.getsize(lenspot_path)
            print(f"  File size: {file_size / 1e6:.1f} MB")
            return True
        else:
            print(f"✗ Lenspot file not found: {lenspot_path}")
            return False
    
    return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "UNIFIED STATISTICS PIPELINE TESTS" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    tests = [
        test_load_kappa,
        test_smooth_kappa,
        test_compute_Cl,
        test_compute_peaks,
        test_multiple_models,
        test_lensplane_paths,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ TEST FAILED WITH EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
