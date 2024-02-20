import tifffile
from tqdm import tqdm

import multiprocessing as mp
from functools import partial

import numpy as np

import SimpleITK as sitk
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import find_objects


from skimage.measure._regionprops import RegionProperties
from skimage import exposure
from skimage.io import imread, imsave
from skimage.measure import label, regionprops_table
from skimage.segmentation import find_boundaries, clear_border, mark_boundaries
from skimage.morphology import disk, erosion, dilation, \
    binary_dilation, binary_erosion, binary_closing, remove_small_objects
from skimage.segmentation import watershed, expand_labels
from skimage.filters import threshold_triangle, threshold_otsu, gaussian

import matplotlib.pyplot as plt

from utils.args import args, ignore_imgaeio_warning, create_shared_multiprocessing_name_space_object
from utils.helpers import get_matching_img_group_nuc_mask_cyto_mask, \
    load_img, get_sitk_img_thresholding_mask_v1

img_path_groups, w0_mask_paths, w1_mask_paths, args.num_channels = \
    get_matching_img_group_nuc_mask_cyto_mask(args.main_path, args.experiment, args.lab)


idx = 5000

for channel in [0, 1, 2, 3, 4]:
    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(21.3, 13.3)
    axes.imshow(tifffile.imread(img_path_groups[idx][channel]), cmap="gray")
    axes.set_axis_off()
    plt.savefig(args.main_path/args.experiment/f"{channel}.png", bbox_inches='tight', dpi=300)
    # plt.show()
