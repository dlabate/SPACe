import sys
import time
from typing import Optional

import numpy as np

import torch
import torch.nn as nn


def cdf_distance_torch_2d(u_values, v_values):
    """
    pytorch implementation of the wassertein distance between two distributions.

    u_values: MxN1 tensor
    where M is the number of columns/features and N1 is the number of rows/cells

    v_values: MxN2 anchor tensor where
    M is the number of columns/features and N2 is the number of rows/cells in the
    anchor condition
    """
    all_values = torch.cat((u_values, v_values), dim=1)
    all_values, _ = torch.sort(all_values, dim=1)

    u_sorted, _ = torch.sort(u_values, dim=1)
    v_sorted, _ = torch.sort(v_values, dim=1)

    # Compute the differences between pairs of successive values of u and v.
    deltas = torch.diff(all_values, dim=1)
    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    # all_values = all_values[:, :-1].contiguous()
    u_cdf_indices = torch.searchsorted(u_sorted, all_values.contiguous()[:, :-1], right=True)
    v_cdf_indices = torch.searchsorted(v_sorted, all_values.contiguous()[:, :-1], right=True)
    print(f"u-vals:     \n{u_values}\n"
          f"v-vals:     \n{v_values}\n"
          f"u-sorted:   \n{u_sorted}\n"
          f"v-sorted:   \n{v_sorted}\n"
          f"all-vals:   \n{all_values}\n"
          f"u-indices: {u_cdf_indices.size()} \n{u_cdf_indices}\n"
          f"v-indices: {v_cdf_indices.size()} \n{v_cdf_indices}")
    # Calculate the CDFs of u and v using their weights, if specified.
    u_cdf = u_cdf_indices / u_values.size(1)
    v_cdf = v_cdf_indices / v_values.size(1)

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    return torch.sum(torch.mul(torch.abs(u_cdf - v_cdf), deltas), dim=1)


def apply_along_axis(function, x, axis):
    return torch.stack([function(x_i) for x_i in torch.unbind(x, dim=axis)], dim=axis)


def get_tensor_size(mytensor, units="gb"):
    if units.lower() == "gb":
        divisor = 1024 * 1024 * 1024
    elif units.lower() == "mb":
        divisor = 1024 * 1024
    elif units.lower() == "kb":
        divisor = 1024
    else:
        raise ValueError("Not acceptable")
    return sys.getsizeof(mytensor.storage()) / divisor


def skewness(x, dim=1):
    """
    Taken from https://github.com/ExamDay/InfoTorch/blob/main/infotorch.py

    x is a 2d tensor of shape (num_features, num_cells),
    and we would like to average over the cell population
    to get a separate stats over each single feature column

    Calculates skewness of data "x" along dimension "dim"."""
    std, mean = torch.std_mean(x, dim)
    n = torch.Tensor([x.shape[dim]]).to(x.device)
    eps = 1e-6  # for stability

    sample_bias_adjustment = torch.sqrt(n * (n - 1)) / (n - 2)
    return sample_bias_adjustment * (
        (torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(3), dim) / n)
        / std.pow(3).clamp(min=eps))


def kurtosis(x, dim=1):
    """
    Taken from https://github.com/ExamDay/InfoTorch/blob/main/infotorch.py

    x is a 2d tensor of shape (num_features, num_cells),
    and we would like to average over the cell population
    to get a separate stats over each single feature column

    Calculates kurtosis of data "x" along dimension "dim".
    """
    std, mean = torch.std_mean(x, dim)
    n = torch.Tensor([x.shape[dim]]).to(x.device)
    eps = 1e-6  # for stability

    sample_bias_adjustment = (n - 1) / ((n - 2) * (n - 3))
    return sample_bias_adjustment * (
        (n + 1)
        * (
            (torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(4), dim) / n)
            / std.pow(4).clamp(min=eps)
        )
        - 3 * (n - 1))


def skewness_measures(x, dim=1):
    """
    Taken from https://github.com/ExamDay/InfoTorch/blob/main/infotorch.py

    x is a 2d tensor of shape (num_features, num_cells),
    and we would like to average over the cell population
    to get a separate stats over each single feature column


    Used to detect bimodality (or multimodality) of dataset(s) given a tensor "x" containing the
    data and a dimension "dim" along which to calculate.  The logic behind this index is that a
    bimodal (or multimodal) distribution with light tails will have very low kurtosis, an asymmetric
    character, or both – all of which increase this index.  The smaller this value is the more
    likely the data are to follow a unimodal distribution.  As a rule: if return value ≤ 0.555
    (bimodal index for uniform distribution), the data are considered to follow a unimodal
    distribution. Otherwise, they follow a bimodal or multimodal distribution.
    """
    eps = 1e-6  # for stability

    # # convert x to a probability vector
    # sum_ = torch.unsqueeze(torch.sum(x, dim), dim=1)
    # # print(x.shape, sum_.shape)
    # x = x/(sum_+eps*torch.sign(sum_))

    # calculate standard deviation and mean of dataset(s)
    std, mean = torch.std_mean(x, dim)
    # get number of samples in dataset(s)
    n = torch.Tensor([x.shape[dim]]).to(x.device)

    # calculate skewness:
    # repeating most of the skewness function here to avoid recomputation of standard devation and mean
    sample_bias_adjustment = torch.sqrt(n * (n - 1)) / (n - 2)
    skew = sample_bias_adjustment * ((torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(3), dim) / n)/
                                     std.pow(3).clamp(min=eps))
    # calculate kurtosis:
    sample_bias_adjustment = (n - 1) / ((n - 2) * (n - 3))
    kurt = sample_bias_adjustment * (
        (n + 1) * ((torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(4), dim) / n) / std.pow(4).clamp(min=eps))
        - 3 * (n - 1))
    # calculate bimodality index:
    bi_index = (skew.pow(2) + 1) / (kurt + 3 * ((n - 2).pow(2) / ((n - 2) * (n - 3))))

    return bi_index








