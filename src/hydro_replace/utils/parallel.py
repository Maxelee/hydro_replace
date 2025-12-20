"""
Parallel Utilities Module
=========================

MPI parallelization utilities.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, TypeVar, Union

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


def get_mpi_comm():
    """
    Get MPI communicator.

    Returns
    -------
    comm : MPI.Comm or None
        MPI communicator, or None if MPI not available.
    """
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        return None


def get_rank() -> int:
    """Get MPI rank (0 if MPI not available)."""
    comm = get_mpi_comm()
    if comm is None:
        return 0
    return comm.Get_rank()


def get_size() -> int:
    """Get MPI size (1 if MPI not available)."""
    comm = get_mpi_comm()
    if comm is None:
        return 1
    return comm.Get_size()


def is_root() -> bool:
    """
    Check if current process is root (rank 0).

    Returns
    -------
    is_root : bool
        True if rank 0 or MPI not available.
    """
    return get_rank() == 0


def distribute_items(
    items: List[T],
    comm=None,
) -> List[T]:
    """
    Distribute items across MPI ranks.

    Parameters
    ----------
    items : list
        Items to distribute.
    comm : MPI.Comm, optional
        MPI communicator.

    Returns
    -------
    local_items : list
        Items assigned to this rank.

    Examples
    --------
    >>> halo_ids = list(range(100))
    >>> local_ids = distribute_items(halo_ids)
    >>> for hid in local_ids:
    ...     process_halo(hid)
    """
    if comm is None:
        comm = get_mpi_comm()
    
    if comm is None:
        return items
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n_items = len(items)
    items_per_rank = n_items // size
    remainder = n_items % size
    
    # Distribute remainder evenly
    if rank < remainder:
        start = rank * (items_per_rank + 1)
        end = start + items_per_rank + 1
    else:
        start = rank * items_per_rank + remainder
        end = start + items_per_rank
    
    return items[start:end]


def distribute_range(
    n_total: int,
    comm=None,
) -> Tuple[int, int]:
    """
    Get range of indices for this rank.

    Parameters
    ----------
    n_total : int
        Total number of items.
    comm : MPI.Comm, optional
        MPI communicator.

    Returns
    -------
    start : int
        Starting index (inclusive).
    end : int
        Ending index (exclusive).
    """
    if comm is None:
        comm = get_mpi_comm()
    
    if comm is None:
        return 0, n_total
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    items_per_rank = n_total // size
    remainder = n_total % size
    
    if rank < remainder:
        start = rank * (items_per_rank + 1)
        end = start + items_per_rank + 1
    else:
        start = rank * items_per_rank + remainder
        end = start + items_per_rank
    
    return start, end


def gather_arrays(
    local_array: np.ndarray,
    comm=None,
    root: int = 0,
) -> Optional[np.ndarray]:
    """
    Gather arrays from all ranks to root.

    Parameters
    ----------
    local_array : ndarray
        Array from this rank.
    comm : MPI.Comm, optional
        MPI communicator.
    root : int
        Rank to gather to.

    Returns
    -------
    gathered : ndarray or None
        Concatenated arrays on root, None on other ranks.

    Examples
    --------
    >>> local_result = compute_local()
    >>> all_results = gather_arrays(local_result)
    >>> if is_root():
    ...     save_results(all_results)
    """
    if comm is None:
        comm = get_mpi_comm()
    
    if comm is None:
        return local_array
    
    # Get counts from all ranks
    local_count = len(local_array)
    counts = comm.gather(local_count, root=root)
    
    if comm.Get_rank() == root:
        # Prepare receive buffer
        total_count = sum(counts)
        if local_array.ndim == 1:
            gathered = np.empty(total_count, dtype=local_array.dtype)
        else:
            gathered = np.empty((total_count,) + local_array.shape[1:], dtype=local_array.dtype)
        
        # Gather
        offset = 0
        for rank in range(comm.Get_size()):
            if rank == root:
                gathered[offset:offset+counts[rank]] = local_array
            else:
                comm.Recv(gathered[offset:offset+counts[rank]], source=rank, tag=rank)
            offset += counts[rank]
        
        return gathered
    else:
        # Send to root
        comm.Send(local_array, dest=root, tag=comm.Get_rank())
        return None


def gather_list(
    local_list: List[T],
    comm=None,
    root: int = 0,
) -> Optional[List[T]]:
    """
    Gather lists from all ranks to root.

    Parameters
    ----------
    local_list : list
        List from this rank.
    comm : MPI.Comm, optional
        MPI communicator.
    root : int
        Rank to gather to.

    Returns
    -------
    gathered : list or None
        Concatenated lists on root, None on other ranks.
    """
    if comm is None:
        comm = get_mpi_comm()
    
    if comm is None:
        return local_list
    
    all_lists = comm.gather(local_list, root=root)
    
    if comm.Get_rank() == root:
        # Flatten
        gathered = []
        for lst in all_lists:
            gathered.extend(lst)
        return gathered
    return None


def allgather_arrays(
    local_array: np.ndarray,
    comm=None,
) -> np.ndarray:
    """
    Gather arrays from all ranks to all ranks.

    Parameters
    ----------
    local_array : ndarray
        Array from this rank.
    comm : MPI.Comm, optional
        MPI communicator.

    Returns
    -------
    gathered : ndarray
        Concatenated arrays on all ranks.
    """
    if comm is None:
        comm = get_mpi_comm()
    
    if comm is None:
        return local_array
    
    # Use Allgather
    all_arrays = comm.allgather(local_array)
    return np.concatenate(all_arrays)


def broadcast_array(
    array: Optional[np.ndarray],
    comm=None,
    root: int = 0,
) -> np.ndarray:
    """
    Broadcast array from root to all ranks.

    Parameters
    ----------
    array : ndarray or None
        Array on root (can be None on other ranks).
    comm : MPI.Comm, optional
        MPI communicator.
    root : int
        Rank to broadcast from.

    Returns
    -------
    array : ndarray
        Broadcast array on all ranks.
    """
    if comm is None:
        comm = get_mpi_comm()
    
    if comm is None:
        return array
    
    # Broadcast shape and dtype first
    if comm.Get_rank() == root:
        shape = array.shape
        dtype = array.dtype
    else:
        shape = None
        dtype = None
    
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    
    if comm.Get_rank() != root:
        array = np.empty(shape, dtype=dtype)
    
    comm.Bcast(array, root=root)
    
    return array


def reduce_sum(
    local_value: Union[float, np.ndarray],
    comm=None,
    root: int = 0,
) -> Optional[Union[float, np.ndarray]]:
    """
    Sum values across all ranks.

    Parameters
    ----------
    local_value : float or ndarray
        Value from this rank.
    comm : MPI.Comm, optional
        MPI communicator.
    root : int
        Rank to reduce to.

    Returns
    -------
    total : float or ndarray or None
        Sum on root, None on other ranks.
    """
    if comm is None:
        comm = get_mpi_comm()
    
    if comm is None:
        return local_value
    
    from mpi4py import MPI
    return comm.reduce(local_value, op=MPI.SUM, root=root)


def barrier(comm=None) -> None:
    """
    MPI barrier synchronization.

    Parameters
    ----------
    comm : MPI.Comm, optional
        MPI communicator.
    """
    if comm is None:
        comm = get_mpi_comm()
    
    if comm is not None:
        comm.Barrier()


def print_rank0(message: str, comm=None) -> None:
    """
    Print message only from rank 0.

    Parameters
    ----------
    message : str
        Message to print.
    comm : MPI.Comm, optional
        MPI communicator.
    """
    if is_root():
        print(message)
