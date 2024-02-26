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
# from skimage import feature
# from skimage.color import label2rgb
# from skimage.restoration import rolling_ball, ellipsoid_kernel
# from skimage.exposure import rescale_intensity


from utils.args import args, ignore_imgaeio_warning, create_shared_multiprocessing_name_space_object
from utils.helpers import get_matching_img_group_nuc_mask_cyto_mask, \
    load_img, get_sitk_img_thresholding_mask_v1, \
    get_overlay, move_figure, creat_segmentation_example_fig


def get_w0_mask_and_w1_masks_step1(w0_mask_path, w1_mask_path, file_path_group, args):
    img_filter = sitk.OtsuThresholdImageFilter()
    img_filter.SetInsideValue(0)
    img_filter.SetOutsideValue(1)
    img = load_img(file_path_group, args.num_channels, args.height, args.width)
    # make sure w0_mask and w1_mask are both cast to uint16
    w1_mask = imread(str(w1_mask_path)).astype(np.uint16)
    w1_mask = clear_border(w1_mask)

    w0_mask = imread(str(w0_mask_path)).astype(np.uint16)
    w0_mask = clear_border(w0_mask)
    w0_mask[remove_small_objects(
        w0_mask.astype(bool),
        min_size=args.min_nucleus_size,
        connectivity=8).astype(int) == 0] = 0
    # remove nuclei with no cytoplasm
    w0_mask[w1_mask == 0] = 0

    props00 = regionprops_table(
        w1_mask,
        intensity_image=w0_mask,
        properties=["intensity_image"])
    props10 = regionprops_table(
        w1_mask,
        intensity_image=img[args.nucleus_idx],
        properties=["label", "image", "intensity_image", "bbox"])
    # total, cnt0, cnt1, cnt2 = len(props10["label"]), 0, 0, 0
    unix0 = np.unique(w0_mask)
    unix1 = np.unique(w1_mask)
    counter0 = unix0[-1]
    counter1 = unix1[-1]

    for iiiiii, (ll,
                 ii,
                 tmp0,
                 mm,
                 y0,
                 x0,
                 y1,
                 x1) in \
            enumerate(zip(
                props10["label"],
                props10["intensity_image"],
                props00["intensity_image"],
                props10["image"],
                props10["bbox-0"],
                props10["bbox-1"],
                props10["bbox-2"],
                props10["bbox-3"],
            )):

        if np.sum(mm) < args.min_nucleus_size:
            w1_mask[w1_mask == ll] = 0
            w0_mask[w1_mask == ll] = 0
            continue
        tmp0 = label(tmp0, background=0)
        # # removing edge artifact, so that the two masks do not touch on the bd
        tmp0_props = regionprops_table(tmp0, properties=["label", "area"])
        idxs = tmp0_props["label"][tmp0_props["area"] < 400]
        tmp0[np.isin(tmp0, idxs)] = 0

        uu = np.unique(tmp0)
        # print(ll, len(uu), uu)
        if len(uu) == 2:
            # cnt1 += 1
            continue

        # there is cytoplasm but no nuclei inside
        # segment inside of it in nuclei channel
        elif len(uu) == 1:

            tmp0 = get_sitk_img_thresholding_mask_v1(ii, img_filter)
            tmp0 = binary_closing(tmp0, disk(8))
            tmp0 = binary_fill_holes(tmp0, structure=np.ones((2, 2)))
            if np.sum(tmp0 > 0) < args.min_nucleus_size:
                w1_mask[w1_mask == ll] = 0
                continue
            # # remove small noisy dots
            # tmp0 = label(tmp0, connectivity=2)
            # tmp_props = regionprops_table(tmp0, properties=("label", "area"))
            # idxs = tmp_props["label"][tmp_props["area"] < 400]
            # tmp0[np.isin(tmp0, idxs)] = 0

            # fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
            # axes[0].imshow(tmp0, cmap="gray")
            # axes[1].imshow(ii, cmap="gray")
            # plt.show()
            w0_mask[y0:y1, x0:x1][tmp0 > 0] = tmp0[tmp0 > 0] + counter0
            # cnt0 += 1
            counter0 += 1

        elif len(uu) > 2:  # there is cytoplasm but more than one nuclei

            # segment and break down this cyto based on the nucleous
            tmp1 = watershed(
                mm,
                markers=tmp0,
                connectivity=tmp0.ndim,
                mask=mm)
            shape = tmp1.shape
            tmp_unix, tmp1 = np.unique(tmp1, return_inverse=True)
            tmp1 = tmp1.reshape(shape)

            w1_mask[y0:y1, x0:x1][tmp1 > 0] = tmp1[tmp1 > 0] + counter1
            counter1 += len(np.setdiff1d(tmp_unix, [0]))
            # cnt2 += 1
            # fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
            # axes[1, 0].imshow(label2rgb(tmp0, bg_label=0), cmap="gray")
            # axes[2, 0].imshow(label2rgb(mm, bg_label=0), cmap="gray")
            # axes[2, 1].imshow(label2rgb(tmp1, bg_label=0), cmap="gray")
            # plt.show()
    # print(total, 100*cnt0/total, 100*cnt1/total, 100*cnt2/total)
    # # remove noisy bd effects
    mask_bd = find_boundaries(
        w1_mask,
        connectivity=2,
        mode="inner").astype(np.uint16)
    mask_bd = binary_dilation(mask_bd, disk(2))
    w0_mask[(w0_mask > 0) & mask_bd] = 0
    w0_mask[remove_small_objects(
        w0_mask.astype(bool),
        min_size=400,
        connectivity=8).astype(int) == 0] = 0

    # match the labels for w1_mask and w0_mask
    _, w1_mask = np.unique(w1_mask, return_inverse=True)
    w1_mask = w1_mask.reshape((args.height, args.width))
    w1_mask = np.uint16(w1_mask)
    w0_mask[w0_mask > 0] = w1_mask[w0_mask > 0]
    # remove the leftovers from matching step
    diff = np.setdiff1d(np.unique(w1_mask), np.unique(w0_mask))
    w1_mask[np.isin(w1_mask, diff)] = 0
    # # trim w1_mask a bit
    # w1_mask = erosion(w1_mask, disk(1))
    # if args.testing and args.show_intermediate_steps:
    #     if args.rescale_image:
    #         img[args.nucleus_idx] = rescale_intensity(img[
    #         args.nucleus_idx],
    #         in_range=tuple(np.percentile(img[args.nucleus_idx], (70, 99.99))))
    #         img[args.cyto_channel_index] = rescale_intensity(
    #         img[args.cyto_channel_index],
    #         in_range=tuple(np.percentile(img[args.cyto_channel_index], (50, 99.9))))
    #     fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    #     axes[0, 0].imshow(img[args.nucleus_idx], cmap="gray")
    #     axes[1, 0].imshow(img[args.cyto_channel_index], cmap="gray")
    #     axes[0, 1].imshow(get_overlay(img[args.nucleus_idx], w0_mask, args.colors), cmap="gray")
    #     axes[1, 1].imshow(get_overlay(img[args.cyto_channel_index], w0_mask, args.colors), cmap="gray")
    #     axes[0, 2].imshow(get_overlay(img[args.nucleus_idx], w1_mask, args.colors), cmap="gray")
    #     axes[1, 2].imshow(get_overlay(img[args.cyto_channel_index], w1_mask, args.colors), cmap="gray")
    #     plt.show()

    assert np.array_equal(np.unique(w0_mask), np.unique(w1_mask))
    return w0_mask, w1_mask, img


def get_w0_mask_step2(w0_mask, w1_mask, img, args):
    img_filter = sitk.OtsuThresholdImageFilter()
    img_filter.SetInsideValue(0)
    img_filter.SetOutsideValue(1)

    assert np.array_equal(np.unique(w0_mask), np.unique(w1_mask))
    # Condition: if ratio of nuclei mask area to its cyto mask area is more than overlap than 70 percent,
    # then we will redo that nuclei mask
    props0 = regionprops_table(w0_mask,
                               properties=("area", "label"))
    props1 = regionprops_table(w1_mask,
                               img[args.nucleus_idx],
                               properties=("area", "bbox", "image", "intensity_image"))
    cond = (props0["area"] / props1["area"]) > .6
    if np.sum(cond) >= 1:
        for (y0, x0, y1, x1, llabel, area, lw1_mask, limg01) in zip(
                props1["bbox-0"][cond],
                props1["bbox-1"][cond],
                props1["bbox-2"][cond],
                props1["bbox-3"][cond],
                props0["label"][cond],
                props0["area"][cond],
                props1["image"][cond],
                props1["intensity_image"][cond]):

            limg01 = gaussian(limg01, sigma=3)
            limg01 = exposure.rescale_intensity(
                limg01,
                in_range=tuple(np.percentile(limg01, (50, 90))))
            # rewrite the nuclei mask with a new mask
            lmask2 = get_sitk_img_thresholding_mask_v1(limg01, myfilter=img_filter)
            # remove small nuclei
            if np.sum(lmask2) < args.min_nucleus_size:
                continue
            # zero out the previous mask
            w0_mask[y0:y1, x0:x1][lw1_mask > 0] = 0
            # add in the new mask at the same location
            w0_mask[y0:y1, x0:x1][lmask2 > 0] = llabel
    assert np.array_equal(np.unique(w0_mask), np.unique(w1_mask))

    # if args.testing and args.show_intermediate_steps:
    #     fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    #     axes[0, 0].imshow(img[0], cmap="gray")
    #     axes[1, 0].imshow(img[1], cmap="gray")
    #
    #     axes[0, 1].imshow(get_overlay(img[0], w0_mask, args.colors), cmap="gray")
    #     axes[1, 1].imshow(get_overlay(img[1], w0_mask, args.colors), cmap="gray")
    #
    #     axes[0, 2].imshow(get_overlay(img[0], w1_mask, args.colors), cmap="gray")
    #     axes[1, 2].imshow(get_overlay(img[1], w1_mask, args.colors), cmap="gray")
    #
    #     plt.show()

    return w0_mask, img


def get_w0_mask_step2_depped(w0_mask, w1_mask, img, args):
    img_filter = sitk.OtsuThresholdImageFilter()
    img_filter.SetInsideValue(0)
    img_filter.SetOutsideValue(1)

    # Condition: if ratio of nuclei mask area to its cyto mask area is more than overlap than 65 percent,
    # then we will redo that nuclei mask
    w0_max = int(np.amax(w0_mask))
    w1_max = int(np.amax(w1_mask))
    assert w0_max == w1_max

    w0_objects = find_objects(w0_mask, max_label=w1_max)
    w1_objects = find_objects(w1_mask, max_label=w1_max)
    for ii in range(w1_max):
        if w1_objects[ii] is None:
            continue
        obj_label = ii + 1
        w10_props = RegionProperties(
            slice=w1_objects[ii], label=obj_label, label_image=w1_mask,
            intensity_image=img[args.nucleus_idx],
            cache_active=True,)
        w00_props = RegionProperties(
            slice=w0_objects[ii], label=obj_label, label_image=w0_mask,
            intensity_image=img[args.nucleus_idx],
            cache_active=True,)
        if w00_props.area/w10_props.area <= .6:
            continue
        l_img = w10_props.intensity_image
        y0, x0, y1, x1 = w10_props.bbox

        l_img = gaussian(l_img, sigma=3)
        prc = tuple(np.percentile(l_img, (50, 90)))
        l_img = exposure.rescale_intensity(l_img, in_range=prc)
        # rewrite the nuclei mask with a new mask
        l_mask = get_sitk_img_thresholding_mask_v1(l_img, myfilter=img_filter)
        # remove small nuclei
        l_mask[remove_small_objects(
            l_mask.astype(bool),
            min_size=args.min_nucleus_size,
            connectivity=8).astype(int) == 0] = 0

        if np.sum(l_mask) < args.min_nucleus_size:
            w1_mask[w1_mask == obj_label] = 0
            w0_mask[w1_mask == obj_label] = 0
            continue
        # if np.sum(l_mask) / w10_props.area > .6:
        #     l_mask = binary_erosion(l_mask, disk(2))
        # zero out the previous mask

        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
        axes[0].imshow(get_overlay(img[0, y0:y1, x0:x1], w0_mask[y0:y1, x0:x1], args.colors), cmap="gray")
        axes[1].imshow(get_overlay(img[0, y0:y1, x0:x1], l_mask, args.colors), cmap="gray")
        axes[2].imshow(get_overlay(img[1, y0:y1, x0:x1], w1_mask[y0:y1, x0:x1], args.colors), cmap="gray")
        plt.show()

        w0_mask[y0:y1, x0:x1][w10_props.image == obj_label] = 0
        # replace the new mask at the same location
        w0_mask[y0:y1, x0:x1][(w10_props.image == obj_label) & (l_mask > 0)] = obj_label
    # w0_mask[w1_mask == 0] = 0
    # w0_mask[w0_mask > 0] = w1_mask[w0_mask > 0]
    unix0 = np.unique(w0_mask)
    unix1 = np.unique(w1_mask)
    print(len(unix0), len(unix0))
    print(np.setdiff1d(unix0, unix1))
    print(np.setdiff1d(unix1, unix0))
    assert np.array_equal(np.unique(w0_mask), np.unique(w1_mask))

    if args.testing and args.show_intermediate_steps:
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        axes[0].imshow(get_overlay(img[0], w0_mask, args.colors), cmap="gray")
        axes[1].imshow(get_overlay(img[1], w1_mask, args.colors), cmap="gray")
        plt.show()
    return w0_mask, w1_mask


def get_nucleoli_mask(w0_mask, img, args):
    # nucleoli
    # w0_maskd = dilation(w0_mask, disk(2))
    img_filter = sitk.YenThresholdImageFilter()
    img_filter.SetInsideValue(0)
    img_filter.SetOutsideValue(1)

    mask2 = np.zeros_like(w0_mask)
    props01 = regionprops_table(
        w0_mask,
        intensity_image=img[args.nucleoli_idx],
        properties=["label", "image", "intensity_image", "bbox", "area"])
    for iiiiii, (ll, ii, mm, y0, x0, y1, x1, nuc_area) in \
            enumerate(zip(
                props01["label"],
                props01["intensity_image"],
                props01["image"],
                props01["bbox-0"],
                props01["bbox-1"],
                props01["bbox-2"],
                props01["bbox-3"],
                props01["area"],
            )):
        val = threshold_otsu(ii)
        lb = np.sum(ii < val) / np.size(ii)
        ii = gaussian(ii, sigma=1)
        ii = exposure.rescale_intensity(
            ii,
            in_range=tuple(np.percentile(ii, (lb, args.nucleoli_channel_rescale_intensity_percentile_ub))))
        # (40, 88), (30, 95)
        min_nucleoli_size = args.min_nucleoli_size_multiplier * (np.sum(mm))
        max_nucleoli_size = args.max_nucleoli_size_multiplier * (np.sum(mm))

        bd = find_boundaries(
            np.pad(mm, ((5, 5), (5, 5)), mode='constant', constant_values=(0, 0)),
            connectivity=2)
        bd = bd[5:-5, 5:-5]
        bd = binary_dilation(bd, disk(3))

        tmp = get_sitk_img_thresholding_mask_v1(ii, img_filter).astype(np.uint16)
        # tmp1 = tmp.copy()
        tmp = binary_erosion(tmp, disk(1))
        # tmp2 = tmp.copy()
        tmp = label(tmp, connectivity=2)
        tmp_props = regionprops_table(tmp, properties=["label", "area"])
        tmp_bd_props = regionprops_table(tmp * bd, properties=["label", "area"])
        inds = []
        for kkk in tmp_bd_props["label"]:
            if tmp_bd_props['area'][tmp_bd_props["label"] == kkk] / \
                    tmp_props['area'][tmp_props["label"] == kkk] > args.nucleoli_bd_area_to_nucleoli_area_threshold:
                inds.append(kkk)
        tmp[np.isin(tmp, inds)] = 0
        cond = (tmp_props["area"] < min_nucleoli_size) | \
               (tmp_props["area"] > max_nucleoli_size)
        inds = tmp_props["label"][cond]
        tmp[np.isin(tmp, inds)] = 0

        # fig, axes = plt.subplots(1, 7, sharex=True, sharey=True)
        # axes[0].imshow(img[args.nucleus_idx][y0:y1, x0:x1], cmap="gray")
        # axes[1].imshow(ii, cmap="gray")
        # axes[2].imshow(mm, cmap="gray")
        # axes[3].imshow(tmp1, cmap="gray")
        # axes[4].imshow(tmp2, cmap="gray")
        # axes[5].imshow(bd, cmap="gray")
        # axes[6].imshow(tmp, cmap="gray")
        # plt.show()
        mask2[y0:y1, x0:x1][tmp > 0] = ll
    return mask2