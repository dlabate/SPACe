import warnings
import math
from math import gcd
from collections import namedtuple, Counter

import numpy as np
from numpy import array, asarray, ma
from numpy.lib import NumpyVersion
from numpy.testing import suppress_warnings

# from scipy.spatial.distance import cdist
# from scipy.ndimage import _measurements
# from scipy._lib._util import (check_random_state, MapWrapper,
#                               rng_integers, _rename_parameter)
#
# import scipy.special as special
# from scipy import linalg
# from . import distributions
# from . import _mstats_basic as mstats_basic
# from ._stats_mstats_common import (_find_repeats, linregress, theilslopes,
#                                    siegelslopes)
# from ._stats import (_kendall_dis, _toint64, _weightedrankedtau,
#                      _local_correlations)
# from dataclasses import make_dataclass
# from ._hypotests import _all_partitions
# from ._hypotests_pythran import _compute_outer_prob_inside_method
# from ._resampling import _batch_generator
# from ._axis_nan_policy import (_axis_nan_policy_factory,
#                                _broadcast_concatenate)
# from ._binomtest import _binary_search_for_binom_tst as _binary_search
# from scipy._lib._bunch import _make_tuple_bunch
# from scipy import stats


def _validate_distribution(values, weights):
    """
    Validate the values and weights from a distribution input of `cdf_distance`
    and return them as ndarray objects.
    Parameters
    ----------
    values : array_like
        Values observed in the (empirical) distribution.
    weights : array_like
        Weight for each value.
    Returns
    -------
    values : ndarray
        Values as ndarray.
    weights : ndarray
        Weights as ndarray.
    """
    # Validate the value array.
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Distribution can't be empty.")

    # Validate the weight array, if specified.
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(values):
            raise ValueError('Value and weight array-likes for the same '
                             'empirical distribution must be of the same size.')
        if np.any(weights < 0):
            raise ValueError('All weights must be non-negative.')
        if not 0 < np.sum(weights) < np.inf:
            raise ValueError('Weight array-like sum must be positive and '
                             'finite. Set as None for an equal distribution of '
                             'weight.')

        return values, weights

    return values, None


def _cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:
    .. math::
        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}
    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.
    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.
    Returns
    -------
    distance : float
        The computed distance between the distributions.
    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.
    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.
    """
    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)
    print("u_values, v_values", u_values, v_values, '\n')

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)
    print("u_sorter, v_sorter", u_sorter, v_sorter, '\n')

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')
    print("all_values", all_values, '\n')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)
    print("deltas", deltas, '\n')

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')
    print("u_cdf_indices, v_cdf_indices", u_cdf_indices, v_cdf_indices, '\n')

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    print(u_cdf, v_cdf, '\n')
    print(np.abs(u_cdf - v_cdf), '\n')
    print(np.multiply(np.abs(u_cdf - v_cdf), deltas), '\n')

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
    return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p),
                                       deltas)), 1/p)

p=1
u_values = [-1, 1, 1, 3, 4, 2, 5, 9, 7, 10, 11]
v_values = [-3, 5, 7, 15, 25, 10, 17, 12, 7, 9, 13, 10, 11]

u_values = np.arange(10)
v_values = np.arange(12,50, 3)
print(_cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None))