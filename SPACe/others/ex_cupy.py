import time
import cupy as cp
import numpy as np
from cupyimg.skimage.exposure import rescale_intensity
def cupy_isin(arr, test_arrs):
    arr = cp.asarray(arr)
    shape = arr.shape
    # arr = arr.reshape(-1)
    isin_ = cp.zeros_like(arr, dtype=cp.bool_)
    for item in test_arrs:
        isin_ += arr == item
    return isin_


def cupy_setdiff1d(ar1, ar2):
    return ar1[~cp.in1d(ar1, ar2)]


# Test the function
ar1 = cp.array([1, 2, 3, 2, 4, 1])
ar2 = cp.array([3, 4, 5, 6])
print(cupy_setdiff1d(ar1, ar2))  # Output: array([1, 2, 2, 1])


# Test the function
arr = cp.random.random_integers(-1, 4, (6, 6))
test_elements = cp.array([0, 2])
print(cupy_isin(arr, test_elements))
print(cp.isin(arr, test_elements))
print(cp.setdiff1d(arr, test_elements))
print(cp.where(arr>1, arr, 0))


s_time = time.time()
prcs = np.percentile(cp.asnumpy(arr), np.array((10, 20, 30)))
print("prcs", prcs)
print(time.time()-s_time)

s_time = time.time()
prcs = cp.percentile(arr, cp.array((10, 20, 30)))
print("prcs", prcs)
print(time.time()-s_time)


s_time = time.time()
prcs = cp.percentile(arr+3, cp.array((10, 20, 30)))
print("prcs", prcs)
print(time.time()-s_time)

print(cp.percentile(arr, (3, 20)))
w3_img_tmp = rescale_intensity(
    arr,
    in_range=tuple(cp.asnumpy(cp.percentile(arr, (3, 20)))))
