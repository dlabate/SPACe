from __future__ import annotations

import math
from math import sqrt
from typing import Any

import torch
import torch.nn.functional as F

from kornia.core import Device, Dtype, Tensor, concatenate, stack, tensor, where, zeros, zeros_like
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.utils import deprecated

from kornia.filters.kernels import _unpack_2d_ks
torch.set_printoptions(linewidth=100)


def get_binary_kernel2d(
    window_size: tuple[int, int] | int, *, device: Device | None = None, dtype: Dtype = torch.float32
) -> Tensor:
    """Create a binary kernel to extract the patches.

    If the window size is HxW will create a (H*W)x1xHxW kernel.
    """
    # TODO: add default dtype as None when kornia relies on torch > 1.12

    ky, kx = _unpack_2d_ks(window_size)

    window_range = kx * ky
    print(window_range)
    kernel = zeros((window_range, window_range), device=device, dtype=dtype)
    print(kernel)
    idx = torch.arange(window_range, device=device)
    print(idx)
    kernel[idx, idx] += 1.0
    print(kernel)
    print('\n')
    return kernel.view(window_range, 1, ky, kx)


# out = get_binary_kernel2d(2)
# print(out.size())
# print(out)


# With square kernels and equal stride
filters = torch.randn(8, 4, 3, 3)
inputs = torch.randn(1, 4, 5, 5)
out = F.conv2d(inputs, filters, padding=1)