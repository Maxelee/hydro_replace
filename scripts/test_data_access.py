#!/usr/bin/env python
"""
Quick test script to verify data access and module imports.
Run interactively: source /mnt/home/mlee1/venvs/hydro_replace/bin/activate && python scripts/test_data_access.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Testing Hydro Replace Data Access")
print("=" * 60)

# Test 1: Import modules
print("\n[1] Testing module imports...")
try:
    import numpy as np
    import h5py
    import yaml
    print("    ✓ numpy, h5py, yaml")
except ImportError as e:
    print(f"    ✗ Error: {e}")

try:
    import illustris_python as il
    print("    ✓ illustris_python")
except ImportError as e:
    print(f"    ✗ illustris_python: {e}")

try:
    import MAS_library as MASL
    print("    ✓ MAS_library (Pylians)")
except ImportError as e:
    print(f"    ✗ MAS_library: {e}")

# Test 2: Load config
print("\n[2] Testing config loading...")
config_path = Path(__file__).parent.parent / 'config' / 'simulation_paths.yaml'
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    hydro_path = config['simulations']['tng300']['hydro']
    dmo_path = config['simulations']['tng300']['dmo']
    print(f"    ✓ Config loaded")
    print(f"    Hydro: {hydro_path}")
    print(f"    DMO:   {dmo_path}")
except Exception as e:
    print(f"    ✗ Error: {e}")
    sys.exit(1)

# Test 3: Load halo catalogs
print("\n[3] Testing halo catalog access...")
try:
    # Load hydro group catalog header
    halos_hydro = il.groupcat.loadHalos(hydro_path, 99, fields=['GroupMass', 'Group_M_Crit200', 'Group_R_Crit200', 'GroupPos'])
    n_halos_hydro = len(halos_hydro['GroupMass'])
    print(f"    ✓ Hydro: {n_halos_hydro:,} total halos")
    
    # Load DMO group catalog
    halos_dmo = il.groupcat.loadHalos(dmo_path, 99, fields=['GroupMass', 'Group_M_Crit200', 'Group_R_Crit200', 'GroupPos'])
    n_halos_dmo = len(halos_dmo['GroupMass'])
    print(f"    ✓ DMO:   {n_halos_dmo:,} total halos")
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Count halos by mass bin
print("\n[4] Counting halos by mass bin...")
try:
    # M200c is in 10^10 Msun/h, convert to Msun/h
    m200c_hydro = halos_hydro['Group_M_Crit200'] * 1e10
    m200c_dmo = halos_dmo['Group_M_Crit200'] * 1e10
    
    mass_bins = [
        (1e10, 1e12, '10-12'),
        (1e12, 10**12.5, '12-12.5'),
        (10**12.5, 1e13, '12.5-13'),
        (1e13, 10**13.5, '13-13.5'),
        (10**13.5, 1e14, '13.5-14'),
        (1e14, 1e15, '>14'),
    ]
    
    print(f"    {'Mass Bin':<12} {'Hydro':>10} {'DMO':>10}")
    print(f"    {'-'*12} {'-'*10} {'-'*10}")
    
    for m_min, m_max, label in mass_bins:
        n_hydro = np.sum((m200c_hydro >= m_min) & (m200c_hydro < m_max))
        n_dmo = np.sum((m200c_dmo >= m_min) & (m200c_dmo < m_max))
        print(f"    {label:<12} {n_hydro:>10,} {n_dmo:>10,}")
    
    # Total above 10^12
    n_hydro_12 = np.sum(m200c_hydro >= 1e12)
    n_dmo_12 = np.sum(m200c_dmo >= 1e12)
    print(f"    {'-'*12} {'-'*10} {'-'*10}")
    print(f"    {'M > 10^12':<12} {n_hydro_12:>10,} {n_dmo_12:>10,}")
    
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check existing pixelized data
print("\n[5] Checking existing pixelized data...")
pix_dir = Path('/mnt/home/mlee1/ceph/pixelized')
if pix_dir.exists():
    npz_files = list(pix_dir.glob('*.npz'))
    print(f"    ✓ Found {len(npz_files)} pixelized map files")
    
    # Load one to check structure
    if npz_files:
        sample = np.load(npz_files[0])
        print(f"    Sample file: {npz_files[0].name}")
        print(f"    Arrays: {list(sample.keys())}")
        for key in sample.keys():
            print(f"      {key}: shape {sample[key].shape}, dtype {sample[key].dtype}")
else:
    print(f"    ✗ Directory not found: {pix_dir}")

print("\n" + "=" * 60)
print("Data access tests complete!")
print("=" * 60)
