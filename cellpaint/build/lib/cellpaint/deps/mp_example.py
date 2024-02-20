"""Write a NumPy array in parallel from multiple CPUs/processes, using shared memory."""

from contextlib import closing
import multiprocessing as mp
import os

import numpy as np


def _init(shared_arr_1, shared_arr_2):
    # The shared array pointer is a global variable so that it can be accessed by the
    # child processes. It is a tuple (pointer, dtype, shape).
    global shared_arr___1, shared_arr___2
    shared_arr___1 = shared_arr_1
    shared_arr___2 = shared_arr_2


def shared_to_numpy(shared_arr, dtype, shape):
    """Get a NumPy array from a shared memory buffer, with a given dtype and shape.
    No copy is involved, the array reflects the underlying shared buffer."""
    return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)


def create_shared_array(dtype, shape):
    """Create a new shared array. Return the shared array pointer, and a NumPy array view to it.
    Note that the buffer values are not initialized.
    """
    dtype = np.dtype(dtype)
    # Get a ctype type from the NumPy dtype.
    cdtype = np.ctypeslib.as_ctypes_type(dtype)
    # Create the RawArray instance.
    # print(shape)
    shared_arr = mp.RawArray(cdtype, int(np.prod(shape)))
    # Get a NumPy array view.
    arr = shared_to_numpy(shared_arr, dtype, shape)
    return shared_arr, arr


def parallel_function(index_range):
    """This function is called in parallel in the different processes. It takes an index range in
    the shared array, and fills in some values there."""
    i0, i1 = index_range
    arr1 = shared_to_numpy(*shared_arr___1)
    arr2 = shared_to_numpy(*shared_arr___2)
    # WARNING: you need to make sure that two processes do not write to the same part of the array.
    # Here, this is the case because the ranges are not overlapping, and all processes receive
    # a different range.
    arr1[i0:i1] = np.arange(i0, i1)
    arr2[i0:i1] = np.arange(i0, i1)+3


def main():

    # For simplicity, make sure the total size is a multiple of the number of processes.
    n_processes = os.cpu_count()
    N = 80
    N = N - (N % n_processes)
    assert N % n_processes == 0

    # Initialize a shared array.
    dtype = np.int32
    shape = (N,)
    shared_arr, arr1 = create_shared_array(dtype, shape)
    shared_arr2, arr2 = create_shared_array(dtype, shape)
    arr1.flat[:] = np.zeros(N)
    arr2.flat[:] = np.ones(N)
    # Show [0, 0, 0, ...].
    # print(arr)

    # Create a Pool of processes and expose the shared array to the processes, in a global variable
    # (_init() function).
    with closing(mp.Pool(
            n_processes,
            initializer=_init,
            initargs=((shared_arr, dtype, shape), (shared_arr2, dtype, shape))
    )) as p:
        n = N // n_processes
        # Call parallel_function in parallel.
        p.map(parallel_function, [(k * n, (k + 1) * n) for k in range(n_processes)])
    # Close the processes.
    p.join()
    # Show [0, 1, 2, 3...]
    print(arr1)
    print(arr2)


if __name__ == '__main__':
    mp.freeze_support()
    main()