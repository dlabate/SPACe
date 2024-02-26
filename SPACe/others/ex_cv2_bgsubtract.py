import time
import tifffile
import torch
from tqdm import tqdm
from pathlib import WindowsPath
from functools import lru_cache
import multiprocessing as mp

import cv2
import numpy as np
from skimage.filters import median
from skimage.measure._regionprops import _cached
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.restoration import rolling_ball, ball_kernel
from skimage.measure import label
from skimage.morphology import remove_small_holes, remove_small_objects, \
    binary_dilation, binary_erosion, binary_closing, isotropic_closing, dilation, erosion, flood_fill,\
    disk, square, white_tophat
from scipy import ndimage as ndi


from cellpaint.steps_single_plate.step0_args import Args
from cellpose import models

from kornia.filters import median_blur
import matplotlib.pyplot as plt


def set_mancini_datasets_hyperparameters(args):
    # image channels order
    args.nucleus_idx = 0
    args.cyto_idx = 1
    args.nucleoli_idx = 2
    args.actin_idx = 3
    args.mito_idx = 4

    # hyperparameters/constants used in Cellpaint Step 1
    #######################################################################
    args.cellpose_nucleus_diam = 100
    args.cellpose_cyto_diam = 100
    args.cellpose_batch_size = 64
    # hyperparameters/constants used in Cellpaint Step 2
    #######################################################################
    args.min_fov_cell_count = 20
    args.w0_bd_size = 300
    # min_cyto_size=1500,
    args.min_nucleus_size = 1000
    args.nucleus_area_to_cyto_area_thresh = .75

    # nucleoli segmentation hyperparameters
    args.nucleoli_channel_rescale_intensity_percentile_ub = 99
    args.nucleoli_bd_area_to_nucleoli_area_threshold = .2
    args.min_nucleoli_size_multiplier = .005
    args.max_nucleoli_size_multiplier = .3

    # hyperparameters/constants used in Cellpaint Step 4 & Cellpaint Step 5
    #######################################################################
    args.min_well_cell_count = 100
    args.qc_min_feat_rows = 450
    args.distmap_min_feat_rows = 1000
    return args


camii_server_flav = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid"
camii_server_seema = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Seema"
camii_server_jump_cpg0012 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0012"
camii_server_jump_cpg0001 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0001"

main_path = WindowsPath(camii_server_flav)
exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"

args = Args(experiment=exp_fold, main_path=main_path, mode="full").args
args = set_mancini_datasets_hyperparameters(args)

# Create a background subtractor
# bs1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
bs2 = cv2.createBackgroundSubtractorKNN(detectShadows=True)
my_kernel = ball_kernel(radius=100, ndim=2)
filterSize = (100, 100)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize)
cellpose_model = models.Cellpose(gpu=True, model_type="cyto2", net_avg=False)

for ii in range(0, 50):
    # Read the TIFF image
    # img = tifffile.imread(args.img_filepaths[ii])
    # prcs = tuple(np.percentile(img, [10, 99.9]))
    # img = rescale_intensity(img, in_range=prcs)
    # img_tensor = torch.as_tensor(np.float32(img[np.newaxis, np.newaxis])).to("cuda:0")
    # output = median_blur(img_tensor, (21, 21))[0, 0].cpu().numpy()
    # diff = img-output
    # diff[diff < 0] = 0
    # print(img_tensor.shape, output.shape)
    # fig, axes = plt.subplots(1, 3, sharey=True, sharex=True)
    # fig.suptitle(args.img_filepaths[ii].stem)
    # axes[0].imshow(img, cmap="gray")
    # axes[1].imshow(output, cmap="gray")
    # axes[2].imshow(img-output, cmap="gray")
    # plt.show()

    # # Apply background subtraction
    img = cv2.imread(str(args.img_filepaths[ii]), cv2.IMREAD_UNCHANGED)
    prcs = tuple(np.percentile(img, [5, 99]))
    img = rescale_intensity(img, in_range=prcs)
    ###################################################################################
    # tmp1 = bs2.apply(img)
    # binary_erosion(tmp1, square(15), out=tmp1)
    # binary_dilation(tmp1, square(30), out=tmp1)
    # tmp1 = label(tmp1 > 0, connectivity=2)
    # print(len(np.unique(tmp1)))
    # remove_small_objects(tmp1, min_size=3000, out=tmp1)
    # img[tmp1 == 0] = 0
    # img = np.float32(equalize_adapthist(img, clip_limit=0.06))
    ##############################################################################
    s_time = time.time()
    img1 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    print(f"OPENCV TOPHAT took {time.time()-s_time} seconds ....")
    # img2 = img.copy()
    # s_time = time.time()
    # white_tophat(img2,  footprint=my_kernel, out=img2)
    print(f"SKIMAGE TOPHAT took {time.time()-s_time} seconds ....")

    ################################################################################
    # w_mask, _, _, _ = cellpose_model.eval(
    #     img,
    #     diameter=args.cellpose_nucleus_diam,
    #     channels=[0, 0],
    #     batch_size=args.cellpose_batch_size,
    #     z_axis=None,
    #     channel_axis=None,
    #     resample=False, )
    ##############################################################################
    print(img.shape, img.dtype, np.amin(img), np.amax(img))
    print(img1.shape, img1.dtype, np.amin(img1), np.amax(img1))
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
    fig.suptitle(args.img_filepaths[ii].stem)
    axes[0].imshow(img, cmap="gray")
    axes[1].imshow(img1, cmap="gray")
    # axes[2].imshow(img2, cmap="gray")
    plt.show()