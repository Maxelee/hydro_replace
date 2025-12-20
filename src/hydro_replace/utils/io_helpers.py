"""
I/O Helpers Module
==================

Utilities for file input/output operations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def save_hdf5(
    filepath: Union[str, Path],
    data: Dict[str, Any],
    attrs: Optional[Dict[str, Any]] = None,
    compression: str = 'gzip',
    compression_opts: int = 4,
) -> Path:
    """
    Save data to HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    data : dict
        Dictionary of arrays to save.
    attrs : dict, optional
        File-level attributes.
    compression : str
        Compression algorithm.
    compression_opts : int
        Compression level.

    Returns
    -------
    filepath : Path
        Path to saved file.

    Examples
    --------
    >>> save_hdf5('output.h5', {'coords': coords, 'masses': masses})
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Add metadata
        f.attrs['created'] = datetime.now().isoformat()
        f.attrs['format_version'] = '1.0'
        
        if attrs:
            for key, value in attrs.items():
                if isinstance(value, (str, int, float, bool)):
                    f.attrs[key] = value
                elif isinstance(value, np.ndarray):
                    f.attrs[key] = value
                elif isinstance(value, (list, tuple)):
                    f.attrs[key] = np.array(value)
                elif isinstance(value, dict):
                    # Store dicts as JSON string
                    f.attrs[key] = json.dumps(value)
        
        # Save datasets
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(
                    key,
                    data=value,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            elif isinstance(value, dict):
                # Create group for nested dict
                grp = f.create_group(key)
                _save_dict_to_group(grp, value, compression, compression_opts)
            else:
                # Try to convert to array
                try:
                    f.create_dataset(key, data=np.asarray(value))
                except Exception as e:
                    logger.warning(f"Could not save {key}: {e}")
    
    logger.debug(f"Saved HDF5: {filepath}")
    return filepath


def _save_dict_to_group(
    group: h5py.Group,
    data: Dict[str, Any],
    compression: str,
    compression_opts: int,
) -> None:
    """Recursively save dictionary to HDF5 group."""
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            group.create_dataset(
                key,
                data=value,
                compression=compression,
                compression_opts=compression_opts,
            )
        elif isinstance(value, dict):
            subgroup = group.create_group(key)
            _save_dict_to_group(subgroup, value, compression, compression_opts)
        elif isinstance(value, (str, int, float, bool)):
            group.attrs[key] = value
        else:
            try:
                group.create_dataset(key, data=np.asarray(value))
            except Exception:
                pass


def load_hdf5(
    filepath: Union[str, Path],
    keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load data from HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.
    keys : list, optional
        Specific keys to load. If None, load all.

    Returns
    -------
    data : dict
        Dictionary with loaded arrays and attributes.

    Examples
    --------
    >>> data = load_hdf5('output.h5')
    >>> coords = data['coords']
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = {}
    
    with h5py.File(filepath, 'r') as f:
        # Load attributes
        for key in f.attrs:
            value = f.attrs[key]
            # Try to parse JSON strings
            if isinstance(value, str) and value.startswith('{'):
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    data[key] = value
            else:
                data[key] = value
        
        # Load datasets
        dataset_keys = keys if keys else list(f.keys())
        
        for key in dataset_keys:
            if key in f:
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    data[key] = item[:]
                elif isinstance(item, h5py.Group):
                    data[key] = _load_group_to_dict(item)
    
    return data


def _load_group_to_dict(group: h5py.Group) -> Dict[str, Any]:
    """Recursively load HDF5 group to dictionary."""
    data = {}
    
    # Load attributes
    for key in group.attrs:
        data[key] = group.attrs[key]
    
    # Load datasets and subgroups
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            data[key] = item[:]
        elif isinstance(item, h5py.Group):
            data[key] = _load_group_to_dict(item)
    
    return data


def create_output_path(
    base_dir: Union[str, Path],
    prefix: str,
    suffix: str = '.h5',
    timestamp: bool = False,
    **kwargs,
) -> Path:
    """
    Create output file path with optional timestamp.

    Parameters
    ----------
    base_dir : str or Path
        Base output directory.
    prefix : str
        File prefix.
    suffix : str
        File extension.
    timestamp : bool
        If True, add timestamp to filename.
    **kwargs
        Additional components to include in filename.

    Returns
    -------
    filepath : Path
        Output file path.

    Examples
    --------
    >>> path = create_output_path('/output', 'power_spectrum', mass_bin='1e14')
    >>> print(path)  # /output/power_spectrum_mass_bin_1e14.h5
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    components = [prefix]
    
    for key, value in kwargs.items():
        components.append(f"{key}_{value}")
    
    if timestamp:
        components.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    filename = '_'.join(components) + suffix
    return base_dir / filename


def get_checkpoint_path(
    output_dir: Union[str, Path],
    stage: str,
    step: int,
) -> Path:
    """
    Get path for a checkpoint file.

    Parameters
    ----------
    output_dir : str or Path
        Output directory.
    stage : str
        Pipeline stage name.
    step : int
        Step number.

    Returns
    -------
    path : Path
        Checkpoint file path.
    """
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    return checkpoint_dir / f'{stage}_step{step:04d}.h5'


def save_checkpoint(
    filepath: Union[str, Path],
    data: Dict[str, Any],
    step: int,
    stage: str,
    completed_items: Optional[List] = None,
) -> Path:
    """
    Save pipeline checkpoint.

    Parameters
    ----------
    filepath : str or Path
        Checkpoint file path.
    data : dict
        Data to checkpoint.
    step : int
        Current step number.
    stage : str
        Pipeline stage.
    completed_items : list, optional
        List of completed item IDs.

    Returns
    -------
    filepath : Path
        Path to checkpoint file.
    """
    attrs = {
        'step': step,
        'stage': stage,
        'timestamp': datetime.now().isoformat(),
    }
    
    if completed_items is not None:
        data['_completed_items'] = np.array(completed_items)
    
    return save_hdf5(filepath, data, attrs=attrs)


def load_checkpoint(
    filepath: Union[str, Path],
) -> Dict[str, Any]:
    """
    Load pipeline checkpoint.

    Parameters
    ----------
    filepath : str or Path
        Checkpoint file path.

    Returns
    -------
    data : dict
        Checkpoint data with metadata.
    """
    return load_hdf5(filepath)


def find_latest_checkpoint(
    output_dir: Union[str, Path],
    stage: str,
) -> Optional[Path]:
    """
    Find the latest checkpoint for a stage.

    Parameters
    ----------
    output_dir : str or Path
        Output directory.
    stage : str
        Pipeline stage.

    Returns
    -------
    path : Path or None
        Path to latest checkpoint, or None if not found.
    """
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = sorted(checkpoint_dir.glob(f'{stage}_step*.h5'))
    
    if checkpoints:
        return checkpoints[-1]
    return None


def save_yaml(
    filepath: Union[str, Path],
    data: Dict[str, Any],
) -> Path:
    """
    Save data to YAML file.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    data : dict
        Data to save.

    Returns
    -------
    filepath : Path
        Path to saved file.
    """
    import yaml
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    return filepath


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from YAML file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    data : dict
        Loaded data.
    """
    import yaml
    
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)
