"""
Tests for utility functions.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path


class TestPeriodicBoundary:
    """Tests for periodic boundary utilities."""
    
    def test_apply_periodic_boundary(self):
        """Test periodic boundary wrapping."""
        from hydro_replace.utils import apply_periodic_boundary
        
        box_size = 100.0
        coords = np.array([[110.0, -10.0, 50.0]])
        
        wrapped = apply_periodic_boundary(coords, box_size)
        
        assert wrapped[0, 0] == pytest.approx(10.0)
        assert wrapped[0, 1] == pytest.approx(90.0)
        assert wrapped[0, 2] == pytest.approx(50.0)
    
    def test_periodic_distance(self):
        """Test periodic distance calculation."""
        from hydro_replace.utils import periodic_distance
        
        box_size = 100.0
        
        # Regular distance
        d = periodic_distance([10, 10, 10], [20, 10, 10], box_size)
        assert d == pytest.approx(10.0)
        
        # Distance that wraps around
        d = periodic_distance([5, 10, 10], [95, 10, 10], box_size)
        assert d == pytest.approx(10.0)
    
    def test_periodic_distance_array(self):
        """Test periodic distance with arrays."""
        from hydro_replace.utils import periodic_distance
        
        box_size = 100.0
        
        pos1 = np.array([[10, 10, 10], [5, 10, 10]])
        pos2 = np.array([[20, 10, 10], [95, 10, 10]])
        
        d = periodic_distance(pos1, pos2, box_size)
        
        assert len(d) == 2
        assert d[0] == pytest.approx(10.0)
        assert d[1] == pytest.approx(10.0)


class TestParallel:
    """Tests for parallel utilities."""
    
    def test_distribute_items_single_rank(self):
        """Test distribution with single rank (no MPI)."""
        from hydro_replace.utils.parallel import distribute_items
        
        items = list(range(10))
        local = distribute_items(items, comm=None)
        
        # Without MPI, should return all items
        assert local == items
    
    def test_is_root_no_mpi(self):
        """Test is_root without MPI."""
        from hydro_replace.utils.parallel import is_root
        
        # Should return True when MPI not available
        assert is_root() == True


class TestIOHelpers:
    """Tests for I/O helper functions."""
    
    def test_save_load_hdf5(self):
        """Test HDF5 save and load."""
        from hydro_replace.utils import save_hdf5, load_hdf5
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.h5'
            
            data = {
                'array': np.array([1, 2, 3]),
                'matrix': np.random.randn(10, 3),
            }
            attrs = {
                'name': 'test',
                'value': 42,
            }
            
            save_hdf5(filepath, data, attrs=attrs)
            loaded = load_hdf5(filepath)
            
            assert np.array_equal(loaded['array'], data['array'])
            assert np.allclose(loaded['matrix'], data['matrix'])
            assert loaded['name'] == 'test'
            assert loaded['value'] == 42
    
    def test_create_output_path(self):
        """Test output path creation."""
        from hydro_replace.utils import create_output_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_output_path(
                tmpdir,
                'power_spectrum',
                mass_bin='M3',
                radius='5R200',
            )
            
            assert 'power_spectrum' in str(path)
            assert 'mass_bin_M3' in str(path)
            assert 'radius_5R200' in str(path)
            assert str(path).endswith('.h5')


class TestStatistics:
    """Tests for statistics utilities."""
    
    def test_stack_profiles_mean(self):
        """Test profile stacking with mean."""
        from hydro_replace.analysis.statistics import stack_profiles
        
        profiles = [
            np.array([1, 2, 3]),
            np.array([2, 3, 4]),
            np.array([3, 4, 5]),
        ]
        
        stacked, scatter = stack_profiles(profiles, method='mean')
        
        assert stacked[0] == pytest.approx(2.0)
        assert stacked[1] == pytest.approx(3.0)
        assert stacked[2] == pytest.approx(4.0)
    
    def test_stack_profiles_median(self):
        """Test profile stacking with median."""
        from hydro_replace.analysis.statistics import stack_profiles
        
        profiles = [
            np.array([1, 2, 3]),
            np.array([2, 3, 4]),
            np.array([10, 10, 10]),  # Outlier
        ]
        
        stacked, scatter = stack_profiles(profiles, method='median')
        
        assert stacked[0] == pytest.approx(2.0)
        assert stacked[1] == pytest.approx(3.0)
        assert stacked[2] == pytest.approx(4.0)
    
    def test_bootstrap_error(self):
        """Test bootstrap error estimation."""
        from hydro_replace.analysis.statistics import bootstrap_error
        
        data = np.random.randn(100) + 5.0
        
        mean, lower, upper = bootstrap_error(data, n_bootstrap=100, random_state=42)
        
        assert abs(mean - 5.0) < 0.5
        assert lower < mean < upper
    
    def test_bin_data(self):
        """Test data binning."""
        from hydro_replace.analysis.statistics import bin_data
        
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.randn(100) * 0.1
        
        result = bin_data(x, y, bins=5)
        
        assert len(result['x_centers']) == 5
        assert len(result['y_stat']) == 5
        # Check approximately linear relationship preserved
        assert result['y_stat'][0] < result['y_stat'][-1]


class TestProfiles:
    """Tests for profile analysis."""
    
    def test_compute_density_profile(self):
        """Test density profile computation."""
        from hydro_replace.analysis.profiles import compute_density_profile
        
        # Create particles in a uniform sphere
        n_particles = 10000
        r_max = 1.0
        
        # Random points in sphere
        phi = np.random.uniform(0, 2*np.pi, n_particles)
        costheta = np.random.uniform(-1, 1, n_particles)
        u = np.random.uniform(0, 1, n_particles)
        
        r = r_max * u**(1/3)
        theta = np.arccos(costheta)
        
        coords = np.column_stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ])
        
        # Center at (5, 5, 5)
        center = np.array([5.0, 5.0, 5.0])
        coords += center
        
        masses = np.ones(n_particles)
        
        r_bins = np.linspace(0.1, 1.0, 10)
        
        profile = compute_density_profile(
            coords, masses, center, r_bins, box_size=10.0
        )
        
        # Check that density is computed
        assert len(profile.density) == len(r_bins) - 1
        assert np.all(profile.density > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
