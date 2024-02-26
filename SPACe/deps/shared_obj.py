import numpy as np
from multiprocessing import Array


def get_shared_array(name: str, shape=None):
    """ Get a shared memory array (multiprocessing.Array) with name <name> and view it as a numpy array.

    The shared memory array is a flattened array but if created by share_array.share_array.make_shared_array(), it will
    be reshaped to the original shape of the numpy array. Optionally, a shape <shape> can be explicitly specified.

    Parameters
    ----------
    name : str
        Name of the shared memory array as string.
    shape : tuple or None
        The shared memory array is a flattened array but if created by share_array.share_array.make_shared_array(),
        it will be reshaped to the original shape of the numpy array.
        Optionally, a shape <shape> can be explicitly specified.

    Returns
    ----------
    np_array : np.ndarray
        Shared memory array viewed as numpy array.
    """
    mp_array = globals()[name]
    np_array = np.frombuffer(mp_array.get_obj(), dtype=np.dtype(mp_array.get_obj()._type_))
    if (shape is None) and (name + '_shape' in globals().keys()):
        shape = globals()[name + '_shape']
        shape = np.frombuffer(shape.get_obj(), dtype=np.int)
    if shape is not None:
        np_array = np_array.reshape(shape)
    return np_array


def make_shared_array(np_array: np.ndarray, name: str):
    """ Crate a shared memory array (multiprocessing.Array) with name <name> from a numpy array.
    Shared memory array will be automatically initialized with the data from the numpy array.
    Shared memory array can accessed as numpy array using ShareArray.get_shared_array().

    Flattened shared memory array will be created as global variable with name <name> and it's shape information will
    be stored in another shared memory array <name>_shape, which is used by get_shared_array() to get the numpy shape.

    Parameters
    ----------
    np_array : np.ndarray
        Numpy array to create shared memory array from.
        Numpy datatype will be converted to ctypes datatype automatically.
        Data from numpy array will be used to initialize the shared array.
    name : str
        Name of the shared memory array as string.
        Shared memory array can be accessed as numpy array from multiple processes using get_shared_array(<name>).
    """

    """
    Example
    -------
    >>> from share_array.share_array import make_shared_array, get_shared_array
    >>> import numpy as np
    >>> np_array = np.arange(5*3).reshape(5,3)  # create a numpy array
    >>> if __name__ == '__main__':
    >>>     make_shared_array(np_array, name='my_shared_array')  # create shared memory array from numpy array
    >>> shared_np_array = get_shared_array('my_shared_array')  # get process safe shared memory array as numpy array
    """

    mp_dtype = np.ctypeslib.as_ctypes(np_array.dtype.type())._type_
    mp_array = Array(typecode_or_type=mp_dtype, size_or_initializer=int(np.prod(np_array.shape)))
    globals()[name] = mp_array
    shared_np_array = get_shared_array(name, shape=np_array.shape)
    shared_np_array[:] = np_array

    mp_array_shape = Array(typecode_or_type='l', size_or_initializer=len(np_array.shape))
    globals()[name + '_shape'] = mp_array_shape
    shared_np_array = get_shared_array(name + '_shape')
    shared_np_array[:] = np_array.shape


