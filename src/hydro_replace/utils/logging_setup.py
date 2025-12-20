"""
Logging Setup Module
====================

Utilities for configuring logging across the pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    include_rank: bool = False,
) -> logging.Logger:
    """
    Set up logging for the hydro_replace pipeline.

    Parameters
    ----------
    level : int or str
        Logging level (e.g., logging.INFO, 'DEBUG').
    log_file : str or Path, optional
        Path to log file. If None, logs to stdout only.
    format_string : str, optional
        Custom format string.
    include_rank : bool
        If True, include MPI rank in log messages.

    Returns
    -------
    logger : Logger
        Configured root logger.

    Examples
    --------
    >>> setup_logging(level='DEBUG', log_file='run.log')
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("Starting analysis...")
    """
    # Get root logger for the package
    logger = logging.getLogger('hydro_replace')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format string
    if format_string is None:
        if include_rank:
            try:
                from mpi4py import MPI
                rank = MPI.COMM_WORLD.Get_rank()
                format_string = f'%(asctime)s [Rank {rank}] %(levelname)s - %(name)s - %(message)s'
            except ImportError:
                format_string = '%(asctime)s %(levelname)s - %(name)s - %(message)s'
        else:
            format_string = '%(asctime)s %(levelname)s - %(name)s - %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Parameters
    ----------
    name : str
        Module name (typically __name__).

    Returns
    -------
    logger : Logger
        Logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing halo...")
    """
    return logging.getLogger(name)


class MPIFileHandler(logging.FileHandler):
    """
    File handler that only writes from MPI rank 0.

    Parameters
    ----------
    filename : str
        Log file path.
    mode : str
        File mode.
    """
    
    def __init__(self, filename: str, mode: str = 'a'):
        try:
            from mpi4py import MPI
            self.rank = MPI.COMM_WORLD.Get_rank()
        except ImportError:
            self.rank = 0
        
        super().__init__(filename, mode)
    
    def emit(self, record):
        """Only emit from rank 0."""
        if self.rank == 0:
            super().emit(record)


def setup_mpi_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """
    Set up MPI-aware logging.

    Only rank 0 writes to file; all ranks write to stdout.

    Parameters
    ----------
    level : int or str
        Logging level.
    log_file : str or Path, optional
        Log file path.

    Returns
    -------
    logger : Logger
        Configured logger.
    """
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
    except ImportError:
        rank = 0
        size = 1
    
    logger = logging.getLogger('hydro_replace')
    logger.setLevel(level)
    logger.handlers.clear()
    
    format_string = f'%(asctime)s [Rank {rank}/{size}] %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_string, datefmt='%H:%M:%S')
    
    # Console handler for all ranks
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler only for rank 0
    if log_file is not None and rank == 0:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
