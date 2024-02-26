import numpy as np
import numba as nb
from scipy import ndimage as ndi
import time

# N = 10000
# a = np.float32(np.random.random((N, N), ))
# b = np.int64(np.random.randint(0, 100, size=(N, N)))
# slices = ndi.find_objects(b)
# print(type(slices[0]))
# print(slices[0])
#
#
# def get_slice(img, mask, slice):
#     return img[slice] * mask.astype(bool)
#
# nb.njit(ngil=True)
# def get_slice_nb(img, mask, slice):
#     return img[slice] * mask.astype(bool)
#
#
# s_time = time.time()
# res = get_slice(a, b, slices[10])
# e_time = time.time()
# print(e_time-s_time)
#
# res1 = get_slice_nb(a, b, slices[10])
#
# s_time = time.time()
# res2 = get_slice_nb(a, b, slices[10])
# e_time = time.time()
# print(e_time-s_time)

#####################################
# N = 10000
# cyto = np.int64(np.random.randint(0, 100, size=(N, N)))
# mito = np.int64(np.random.randint(0, 2, size=(N, N)))
#
# s_time = time.time()
# a = mito*cyto
# print(time.time()-s_time)
#
# s_time = time.time()
# mito[mito > 0] = cyto[mito > 0]
# print(time.time()-s_time)
############################################

####################################
N = 10000
mito = np.int64(np.random.randint(0, 2, size=(N, N)))

s_time = time.time()
a = mito*(mito != 0)
print(time.time()-s_time)

s_time = time.time()
b = mito[mito > 0]
print(time.time()-s_time)
###########################################
