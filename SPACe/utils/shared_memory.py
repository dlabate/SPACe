import types
import ctypes
import inspect

import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing.managers import BaseManager, NamespaceProxy


class MyBaseManager(BaseManager):
    """https://stackoverflow.com/questions/3671666/sharing-a-complex-object-between-processes"""
    """https://stackoverflow.com/questions/26499548/accessing-an-attribute-of-a-multiprocessing-proxy-of-a-class"""
    pass


class TestProxy(NamespaceProxy):
    """Needed to expose attributes when using MyBaseManger to register a python simple class as shared!
     Without it the MyBaseManger can expose the methods of the class to all processes, but not the attributes."""
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__',)


class TestProxy2(NamespaceProxy):
    """Needed to expose attributes when using MyBaseManger to register a python simple class as shared!
     Without it the MyBaseManger can expose the methods of the class to all processes, but not the attributes."""
    def __init__(self, test_class):
        self._exposed_ = tuple(dir(test_class))
        # print(f"TestProxy self._exposed_:\n{self._exposed_}\n")

    def __getattr__(self, name):
        result = super().__getattr__(name)
        if isinstance(result, types.MethodType):
            def wrapper(*args, **kwargs):
                return self._callmethod(name, args, kwargs)  # Note the return here
            return wrapper
        return result


def create_shared_proxy_obj(target_class):
    dic = {'types': types}
    exec('''def __getattr__(self, key):
        result = self._callmethod('__getattribute__', (key,))
        if isinstance(result, types.MethodType):
            def wrapper(*args, **kwargs):
                self._callmethod(key, args)
            return wrapper
        return result''', dic)
    proxyName = target_class.__name__ + "Proxy"
    ProxyType = type(proxyName, (NamespaceProxy,), dic)
    ProxyType._exposed_ = tuple(dir(target_class))
    print(ProxyType._exposed_)
    return ProxyType


def create_shared_pd_df(df):
    """https://stackoverflow.com/questions/22487296/
    multiprocessing-in-python-sharing-large-object-e-g-pandas-dataframe-between"""
    # the origingal dataframe is df, store the columns/dtypes pairs
    df_dtypes_dict = dict(list(zip(df.columns, df.dtypes)))
    # declare a shared Array with data from df
    mparr = mp.Array(ctypes.c_double, df.values.reshape(-1))
    # create a new df based on the shared array
    df_shared = pd.DataFrame(np.frombuffer(mparr.get_obj()).reshape(df.shape),
                             columns=df.columns).astype(df_dtypes_dict)
    return df_shared


def create_shared_np_num_arr(shape, c_dtype):
    """
    1) https://stackoverflow.com/questions/5549190/
    is-shared-readonly-data-copied-to-different-processes-for-multiprocessing/5550156#5550156

    2) https://stackoverflow.com/questions/69283001/access-and-modify-a-2d-array-using-python-multiprocessing"""
    flattened_shape = int(np.prod([it for it in shape]))
    shared_array_base = mp.Array(getattr(ctypes, c_dtype), flattened_shape)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(*shape)
    return shared_array


def create_shared_np_str_arr(shape, max_chars):
    """https://stackoverflow.com/questions/50844290/
    numpy-matrix-of-strings-converted-to-shared-array-of-strings-creates-type-mismat"""
    input_array = np.zeros(shape, dtype=f"|S{max_chars}")
    shared_memory = mp.Array(ctypes.c_char, input_array.size * input_array.itemsize, lock=False)
    np_wrapper = np.frombuffer(shared_memory, dtype=input_array.dtype).reshape(input_array.shape)
    np.copyto(np_wrapper, input_array)
    return np_wrapper


def create_shared_namespace_obj(manager, args):
    """https://github.com/prickly-pythons/prickly-pythons/issues/14
    [HowTo] Shared data in multiprocessing with Manager.Namespace()

    https://stackoverflow.com/questions/2597278/python-load-variables-in-a-dict-into-namespace
    https://www.programiz.com/python-programming/methods/built-in/setattr
    """
    # Create manager object in module-level namespace
    # # manager = mp.Manager()
    # Then create a container of things that you want to share to processes as Manager.Namespace() object.
    config = manager.Namespace()
    # fill in the config shared-namespace object with normal name space object keys and values using a for loop
    args = vars(args)
    for key, value in args.items():
        setattr(config, key, value)
    return config