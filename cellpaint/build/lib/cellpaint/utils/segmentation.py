import numpy as np
import SimpleITK as sitk

def get_sitk_img_thresholding_mask(img, myfilter):
    tmp = img.copy()
    # tmp = (tmp - np.amin(tmp)) / (np.amax(tmp) - np.amin(tmp) + 1e-6)
    tmp1 = tmp[tmp > 0]
    tmp1 = tmp1.reshape((-1, 1))
    size = tmp1.shape[0]
    tmp1 = sitk.GetImageFromArray(tmp1)
    try:
        tmp1 = myfilter.Execute(tmp1)
        tmp1 = sitk.GetArrayFromImage(tmp1)
    except:
        tmp1 = np.zeros(size, dtype=np.uint16)
    tmp1 = tmp1.reshape(-1)
    tmp[tmp > 0] = tmp1
    return tmp


def remove_small_objects_mod(ar, min_size):
    """ar: 2D labelled numpy array, assuming labels are positive integers and background is zero."""
    component_sizes = np.bincount(ar.ravel())
    ar[(component_sizes < min_size)[ar]] = 0
    return ar


def remove_large_objects_mod(ar, max_size):
    """ar: 2D labelled numpy array, assuming labels are positive integers and background is zero."""
    component_sizes = np.bincount(ar.ravel())
    ar[(component_sizes > max_size)[ar]] = 0
    return ar


def remove_small_and_large_objects_mod(ar, min_size, max_size):
    """ar: 2D labelled numpy array, assuming labels are positive integers and background is zero."""
    component_sizes = np.bincount(ar.ravel())
    ar[(component_sizes < min_size)[ar]] = 0
    ar[(component_sizes > max_size)[ar]] = 0
    return ar