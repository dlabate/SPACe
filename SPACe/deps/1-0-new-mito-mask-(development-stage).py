from pprint import pprint
from tqdm import tqdm

import multiprocessing as mp
from functools import partial

import numpy as np
from pathlib import Path, WindowsPath
import re
import itertools
import tifffile

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import SimpleITK as sitk
from scipy.ndimage.morphology import binary_fill_holes

from skimage import exposure
from skimage.io import imread, imsave
from skimage.measure import label, regionprops_table
from skimage.segmentation import find_boundaries, clear_border, mark_boundaries
from skimage.morphology import disk, erosion, dilation, \
    binary_dilation, binary_erosion, binary_closing, remove_small_objects
from skimage.segmentation import watershed, expand_labels
from skimage.filters import threshold_triangle, threshold_otsu, gaussian
from skimage import feature
# from skimage.color import label2rgb
# from skimage.restoration import rolling_ball, ellipsoid_kernel
from skimage.exposure import rescale_intensity

from PIL import Image

from utils.args import args, ignore_imgaeio_warning, create_shared_multiprocessing_name_space_object
from utils.helpers import get_matching_img_group_nuc_mask_cyto_mask, \
    load_img, get_sitk_img_thresholding_mask_v1, \
    get_overlay, move_figure, creat_segmentation_example_fig

img_path_groups, mask0_paths, mask1_paths, args.num_channels = \
    get_matching_img_group_nuc_mask_cyto_mask(args.main_path, args.experiment, args.lab, mask_folder="Masks")
mask2_paths = [it.parents[0] / f"w2_{'_'.join(it.stem.split('_')[1:])}.png" for it in mask0_paths]

f1 = sitk.OtsuThresholdImageFilter()
f1.SetInsideValue(0)
f1.SetOutsideValue(1)

f2 = sitk.MomentsThresholdImageFilter()
f2.SetInsideValue(0)
f2.SetOutsideValue(1)

f3 = sitk.MaximumEntropyThresholdImageFilter()
f3.SetInsideValue(0)
f3.SetOutsideValue(1)


def get_mito_mask():
    f1 = sitk.OtsuThresholdImageFilter()
    f1.SetInsideValue(0)
    f1.SetOutsideValue(1)

    for it0, it1, it2, it3 in zip(img_path_groups, mask0_paths, mask1_paths, mask2_paths):
        img = load_img(it0, args.num_channels, args.height, args.width)
        mask0 = np.array(Image.open(it1)).astype(np.uint16)  # nucleus mask
        mask1 = np.array(Image.open(it2)).astype(np.uint16)  # cyto/mito mask
        mask2 = np.array(Image.open(it3)).astype(np.uint16)  # nucleoli mask
        # relabel mask2 with the nuclei/cytoplasm masks (already matching) labels
        mask2[mask2 > 0] = mask0[mask2 > 0]
        tmp_img = img[-1].copy()
        tmp_img[(dilation(mask1, disk(6)) == 0) | (erosion(mask0, disk(4)) > 0)] = 0
        mask_global = get_sitk_img_thresholding_mask_v1(tmp_img, f1)

        mask_local = np.zeros_like(mask1)
        mask1_mod = mask1.copy()
        mask1_mod[mask0 > 0] = 0
        props = regionprops_table(mask1_mod, tmp_img, properties=["label", "bbox", "intensity_image"])
        for lb, limg, y0, x0, y1, x1 in zip(
                props["label"],
                props["intensity_image"],
                props["bbox-0"],
                props["bbox-1"],
                props["bbox-2"],
                props["bbox-3"]):
            # scaled_img = rescale_intensity(limg, in_range=tuple(np.percentile(limg, [5, 99])))
            tmp_mask = get_sitk_img_thresholding_mask_v1(limg, f1).astype(bool)
            mask_local[y0:y1, x0:x1][tmp_mask > 0] = (tmp_mask[tmp_mask > 0])

        img[-1] = rescale_intensity(img[-1], in_range=tuple(np.percentile(img[-1], [10, 99.9])))
        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
        axes[0, 0].imshow(img[-1], cmap="gray")
        axes[0, 0].set_title("raw image")

        axes[0, 1].imshow(get_overlay(img[-1], mask0, colors=args.colors), cmap="gray")
        axes[0, 1].set_title("raw image overlay mask0")

        axes[0, 2].imshow(get_overlay(img[-1], mask1, colors=args.colors), cmap="gray")
        axes[0, 2].set_title("raw image overlay mask1")

        axes[1, 0].imshow(mask_global, cmap="gray")
        axes[1, 0].set_title("global thresholding mask")

        axes[1, 1].imshow(mask_local, cmap="gray")
        axes[1, 1].set_title("local thresholding mask")

        axes[1, 2].imshow(np.logical_or(mask_global, mask_local), cmap="gray")
        axes[1, 2].set_title("local thresholding and global thresholding mask")
        plt.suptitle(it1.stem)
        plt.show()
        return np.logical_or(mask_global, mask_local)


get_mito_mask()