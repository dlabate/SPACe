import numpy as np
import pandas as pd
from functools import lru_cache, wraps
from timeit import default_timer
from copy import copy
from time import sleep


# Caching implementation for functions whose first argument is a pandas dataframes:
"""https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75"""
# Joblib Memory
"""https://joblib.readthedocs.io/en/latest/auto_examples/memory_basic_usage.html"""
# Caching not more than available memory
"""https://stackoverflow.com/questions/23477284/memory-aware-lru-caching-in-python"""
#  lru_cache.py source code python
"""https://gist.github.com/wmayner/0245b7d9c329e498d42b"""


def df_cache(*args, **kwargs):
    """LRU cache implementation for functions whose FIRST parameter is a pandas dataframe"""

    def decorator(function):
        @wraps(function)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            df_tuple = dataframe_to_tuple(df)
            return cached_wrapper(df_tuple, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(df_tuple, *args, **kwargs):
            df = pd.DataFrame(index=df_tuple[0], columns=df_tuple[1], data=df_tuple[2])
            return function(df, *args, **kwargs)

        def array_to_tuple(np_array):
            """Iterates recursivelly."""
            try:
                return tuple(array_to_tuple(_) for _ in np_array)
            except TypeError:
                return np_array

        def dataframe_to_tuple(df: pd.DataFrame):
            try:
                return (
                    tuple(df.index), tuple(df.columns), array_to_tuple(df.values)
                )
            except TypeError:
                return df

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator


def vocalTimeit(*args, **kwargs):
    ''' provides the decorator @vocalTime which will print the name of the function as well as the
        execution time in seconds '''

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            start = default_timer()
            results = function(*args, **kwargs)
            end = default_timer()
            print('{} execution time: {} s'.format(function.__name__, end-start))
            return results
        return wrapper
    return decorator


def npCacheMap(*args, **kwargs):
    ''' LRU cache implementation for functions whose FIRST parameter is a numpy array
        forked from: https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75 '''

    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            hashable_array = tuple(map(tuple, np_array))
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper
    return decorator


def npCacheFun(*args, **kwargs):
    ''' LRU cache implementation for functions whose FIRST parameter is a numpy array
        Source: https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75 '''

    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            hashable_array = array_to_tuple(np_array)
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        def array_to_tuple(np_array):
            '''Iterates recursively.'''
            try:
                return tuple(array_to_tuple(_) for _ in np_array)
            except TypeError:
                return np_array

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper
    return decorator


def npCacheMethod(*args, **kwargs):
    ''' LRU cache implementation for methods whose FIRST parameter is a numpy array
        modified from: https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75 '''

    def decorator(function):
        @wraps(function)
        def wrapper(s, np_array, *args, **kwargs):
            hashable_array = tuple(map(tuple, np_array))
            return cached_wrapper(s, hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(s, hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(s, array, *args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper
    return decorator


if __name__ == "__main__":

    array = np.random.rand(900, 600)
    param = int(np.random.rand()*1000)

    print('\nwith @npCacheFun:')
    @vocalTimeit()
    @npCacheFun()
    def doSomethingFun(array, param):
        print("Calculating...")
        sleep(1)
        return True

    for i in range(5):
        res = copy(doSomethingFun(array, param))

    print('\nwith @npCacheMap:')
    @vocalTimeit()
    @npCacheMap()
    def doSomethingMap(array, param):
        print("Calculating...")
        sleep(1)
        return True

    for i in range(5):
        res = copy(doSomethingMap(array, param))

    print('\nwith @npCacheMethod')

    class DummyClass():
        def __init__(self):
            for i in range(5):
                res = copy(self.doSomethingMethod(array, param))

        @vocalTimeit()
        @npCacheMethod()
        def doSomethingMethod(self, array, param):
            print("Calculating...")
            sleep(1)
            return True
    dummyClass = DummyClass()