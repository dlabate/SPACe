import os
import time
from tqdm import tqdm
from pathlib import WindowsPath

from functools import lru_cache
import matplotlib.pyplot as plt

import tifffile
import skimage.io as sio

import numpy as np
import SimpleITK as sitk
from skimage.segmentation import watershed, expand_labels, find_boundaries, clear_border
from skimage.exposure import rescale_intensity
from skimage.measure import label
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import disk, erosion, dilation, closing, \
    binary_dilation, binary_erosion, binary_closing, \
    remove_small_objects, convex_hull_image
from skimage.color import label2rgb
from scipy.ndimage import find_objects
from scipy.spatial import distance
from PIL import Image, ImageFilter

from cellpaint.utils.img_files_dep import sort_key_for_imgs, get_img_paths
from cellpaint.steps_single_plate.step0_args import Args, Base, sort_key_for_imgs
from cellpaint.utils.segmentation import get_sitk_img_thresholding_mask

from numba import jit
import torch

import cupy as cp
from cupyimg.skimage.measure import label as cp_label
from cupyimg.skimage.segmentation import find_boundaries as cp_find_boundaries
from cupyimg.skimage.filters import threshold_otsu as cp_otsu, threshold_yen as cp_yen
from cupyimg.skimage.filters import gaussian as cp_gaussian
from cupyimg.skimage.exposure.exposure import rescale_intensity as cp_rescale_intensity
from cupyimg.skimage.morphology import erosion as cp_erosion, \
    binary_dilation as cp_binary_dilation, dilation as cp_dilation, closing as cp_closing, disk as cp_disk
from cupyx.scipy.ndimage import grey_dilation, grey_erosion, grey_closing
from torchvision.ops import masks_to_boxes

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


class SegmentationPartII(Base):
    buffer_size = 3
    w1_labels_shift = 2
    w2_labels_shift = 701
    w3_local_rescale_intensity_lb = 10
    w3_local_rescale_intensity_ub = 99.2
    nucleoli_bd_pad = 5

    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)

    yen_filter = sitk.YenThresholdImageFilter()
    yen_filter.SetInsideValue(0)
    yen_filter.SetOutsideValue(1)

    maxentropy_filter = sitk.MaximumEntropyThresholdImageFilter()
    maxentropy_filter.SetInsideValue(0)
    maxentropy_filter.SetOutsideValue(1)

    canny_filter = sitk.CannyEdgeDetectionImageFilter()
    canny_filter.SetLowerThreshold(0.0)
    canny_filter.SetUpperThreshold(1.0)
    canny_filter.SetVariance([5.0, 5.0])

    label_stats_img_filter = sitk.LabelStatisticsImageFilter()
    label_shape_stats_img_filter = sitk.LabelShapeStatisticsImageFilter()
    label_intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()

    w1_dil_struct = disk(4)

    # TODO: To smoothen the boundary of each object (if possible and can be done quickly)
    #  option 1) Can use the mode filter in PIL.
    #  option 2) Can use the dilation and erosion ops in skimage. ✓
    #  option 3) Can use the convex-hull op in skimage in skimage.
    #  option 4) Apply median blur and use watershed

    def __init__(self, args):
        Base.__init__(self, args)
        _, self.img_groups, self.n_groups = self.get_img_channel_groups()
        self.w1_mask_filepaths = list(args.step2_save_path.rglob("*_W1.png"))
        self.w2_mask_filepaths = list(args.step2_save_path.rglob("*_W2.png"))

        # create a border object:
        self.border_mask = cp.zeros((self.args.height, self.args.width), dtype=bool)
        self.border_mask[0:self.buffer_size, :] = True
        self.border_mask[-self.buffer_size:, :] = True
        self.border_mask[:, 0:self.buffer_size] = True
        self.border_mask[:, -self.buffer_size:] = True

        # Set minimum object size, adjust according to your needs
        self.label_filter = sitk.ConnectedComponentImageFilter()
        self.w1_relabel_filter = sitk.RelabelComponentImageFilter()
        self.w1_relabel_filter.SetMinimumObjectSize(self.args.min_sizes["w1"])

    def get_bbox_from_mask(self, mask_arr):
        # convert to torch tensor
        mask_arr = torch.as_tensor(mask_arr, dtype=torch.int64)
        unique_labels = torch.unique(mask_arr)
        if len(unique_labels) <= 1:
            return None, 0
        nonzero_unique_labels = unique_labels[1:]
        slices = mask_arr == nonzero_unique_labels[:, None, None]
        # each element is a (xmin, ymin, xmax, ymax)
        bboxes = masks_to_boxes(slices)
        count = nonzero_unique_labels.size(0)

        # convert back to cupy
        bboxes = cp.asarray(bboxes, dtype=cp.uint16)
        nonzero_unique_labels = cp.asarray(nonzero_unique_labels, dtype=cp.uint16)
        return bboxes, nonzero_unique_labels, count

    def run(self, img_group):

        well_id, fov = sort_key_for_imgs(img_group[0], "to_get_well_id_and_fov", self.args.plate_protocol)
        w1_mask_path = list(filter(
            lambda x: (x.stem.split("_")[0] == well_id) & (x.stem.split("_")[1] == fov),
            self.w1_mask_filepaths))[0]
        w2_mask_path = list(filter(
            lambda x: (x.stem.split("_")[0] == well_id) & (x.stem.split("_")[1] == fov),
            self.w2_mask_filepaths))[0]

        stime = time.time()
        img = self.load_img(img_group)
        w1_mask = sio.imread(w1_mask_path)
        w2_mask = sio.imread(w2_mask_path)
        print(f"reading input image and masks finished in {time.time()-stime} seconds")

        stime = time.time()
        img, w1_mask, w2_mask = self.step1_preprocessing_and_w1w2_label_matching(img, w1_mask, w2_mask)
        print(f"step1 finished in {time.time()-stime} seconds")
        # stime = time.time()
        # w3_mask, w5_mask = self.step2_get_nucleoli_and_mito_masks_v1(img, w1_mask, w2_mask)
        # print(f"step2-v1 finished in {time.time()-stime} seconds")
        stime = time.time()
        w3_mask, w5_mask = self.step2_get_nucleoli_and_mito_masks_v2(img, w1_mask, w2_mask)
        print(f"step2-v2 finished in {time.time()-stime} seconds")
        print("\n")
        del w1_mask, w2_mask, w3_mask, w5_mask, img
        # free memory
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        # # stime = time.time()
        # # img = cp.asnumpy(img)
        # # w1_mask = cp.asnumpy(w1_mask)
        # # w2_mask = cp.asnumpy(w2_mask)
        # # w3_mask = cp.asnumpy(w3_mask)
        # # w5_mask = cp.asnumpy(w5_mask)
        # # print(f"conversion back to numpy finished in {time.time() - stime} seconds")
        # # fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
        # # axes[0, 0].imshow(img[0], cmap="gray")
        # # axes[0, 1].imshow(img[1], cmap="gray")
        # # axes[0, 2].imshow(img[2], cmap="gray")
        # # axes[0, 3].imshow(img[4], cmap="gray")
        # # axes[1, 0].imshow(label2rgb(w1_mask), cmap="gray")
        # # axes[1, 1].imshow(label2rgb(w2_mask), cmap="gray")
        # # axes[1, 2].imshow(label2rgb(w3_mask), cmap="gray")
        # # axes[1, 3].imshow(label2rgb(w5_mask), cmap="gray")
        # # plt.show()

    def step1_preprocessing_and_w1w2_label_matching(self, img, w1_mask, w2_mask):
        """
        This modeling is based upon observing that the intersection between nucleus channel and cyto channel happens as:
        The histogram of intersection ratios has big tails and is tiny in the middle,
        that is mostly nucleus/cyto intersection is:
        1) either the intersection is really small, the cyto barely touches a nucleus
        2) or the intersection is really large and the cyto almost covers the entire nucleus

        Here we assume w1_mask.ndim == 2 and w2_mask.ndim == 2"""
        # w1_img = rescale_intensity(w1_img, in_range=tuple(cp.percentile(w1_img, (10, 99.9))))
        # w2_img = rescale_intensity(w2_img, in_range=tuple(cp.percentile(w2_img, (15, 99.8))))
        # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        # axes[0, 0].imshow(w1_img, cmap="gray")
        # axes[0, 1].imshow(w2_img, cmap="gray")
        # axes[1, 0].imshow(label2rgb(w1_mask), cmap="gray")
        # axes[1, 1].imshow(label2rgb(w2_mask), cmap="gray")
        # plt.show()
        stime = time.time()
        # img = torch.as_tensor(img.astype(np.int64)).to("cuda:0")
        # w1_mask = torch.as_tensor(w1_mask.astype(np.int64)).to("cuda:0")
        # w2_mask = torch.as_tensor(w2_mask.astype(np.int64)).to("cuda:0")

        img = cp.asarray(img)
        w1_mask = cp.asarray(w1_mask)
        w2_mask = cp.asarray(w2_mask)
        print(f"conversion finished in {time.time()-stime} seconds")

        shape_ = w2_mask.shape
        stime_part0 = time.time()
        cp_closing(w1_mask, cp_disk(2), out=w1_mask)
        cp_closing(w2_mask, cp_disk(2), out=w2_mask)
        print(f"part 0 completion time in seconds: {time.time()-stime_part0}")

        # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        # axes[0, 0].imshow(img[0], cmap="gray")
        # axes[0, 1].imshow(img[1], cmap="gray")
        # axes[1, 0].imshow(label2rgb(w1_mask), cmap="gray")
        # axes[1, 1].imshow(label2rgb(w2_mask), cmap="gray")
        # plt.show()

        #####################################################################
        stime_partI = time.time()
        if w1_mask.ndim != 2 or w2_mask.ndim != 2:
            raise ValueError("Arrs have to be two dimensional")

        # remove small objects/cells in botth w1_mask and w2_mask
        w1_mask[(cp.bincount(w1_mask.ravel()) < self.args.min_sizes["w1"])[w1_mask]] = 0
        w2_mask[(cp.bincount(w2_mask.ravel()) < self.args.min_sizes["w2"])[w2_mask]] = 0

        # remove cells touching the border in w1_mask, and the ones
        # touching the border in w2 mask which don't contain any nucleus
        w1_border_ids = cp.unique(w1_mask[self.border_mask])
        w1_mask[cp.isin(w1_mask, w1_border_ids)] = 0
        w2_border_ids = cp.unique(w2_mask[self.border_mask & (w1_mask == 0)])
        w2_mask[cp.isin(w2_mask, w2_border_ids)] = 0
        # print(w1_border_ids, w2_border_ids)

        if cp.sum(w1_mask) == 0 or cp.sum(w2_mask) == 0:
            print("no pixels detected ...")
            return w1_mask, w2_mask

        # translate both masks by 2 and 3, to create the intersect mask
        w1_mask += self.w1_labels_shift
        w2_mask += self.w2_labels_shift
        w1_mask[w1_mask == self.w1_labels_shift] = 0
        w2_mask[w2_mask == self.w2_labels_shift] = 0
        intersect_mask = w2_mask * w1_mask
        # find ratio of area of intersecting regions
        intersect_area = cp.bincount(intersect_mask.ravel())[intersect_mask]
        intersect_area[(w1_mask == 0) | (w2_mask == 0)] = 0
        w1_area = cp.bincount(w1_mask.ravel())[w1_mask]
        intersect_ratio = (intersect_area / w1_area)
        print(f"part 1 completion time in seconds: {time.time()-stime_partI}")
        ########################################################
        stime_partII = time.time()
        # 2) First deal with nucleus ids
        # since multiple cytoplasms may intersect a nucleus,
        # intersect_ratio between a nucleus and cyto channel will happen at multiple cyto instances/labels,
        # we have to choose the one with the largest intersection portion,
        # otherwise, if the intersection is not significant we just label the cyto same as the expanded nucleus
        w1_bboxes, w1_unix, n1 = self.get_bbox_from_mask(w1_mask)
        low_intersect_id = cp.zeros((n1, ), dtype=cp.int64)
        for ii in range(n1):
            (x0, y0, x1, y1), w1_label = w1_bboxes[ii], w1_unix[ii]
            w1_mask_bbx = w1_mask[y0:y1, x0:x1] == w1_label
            ratio_bbx = cp.where(w1_mask_bbx, intersect_ratio[y0:y1, x0:x1], cp.zeros_like(w1_mask_bbx))
            rmax = cp.amax(ratio_bbx)
            # w2_bbx_before = cp.where(w1_mask_bbx, w2_mask[slc1], 0)

            if rmax < .5:  # if the intersection is small, remove the nucleus
                w2_mask[y0:y1, x0:x1] = cp.where(w1_mask_bbx, cp.zeros_like(w1_mask_bbx), w2_mask[y0:y1, x0:x1])
                low_intersect_id[ii] = w1_label
            else:  # otherwise, extend the w2_mask to contain the entire nucleus
                w2_label = cp.amax(w2_mask[y0:y1, x0:x1][ratio_bbx == rmax])
                w2_mask[y0:y1, x0:x1][w1_mask_bbx] = w2_label

            # fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
            # fig.suptitle(f"cyto-intersect-area/nucleus-area ratios: "
            #              f"{cp.unique(ratio_bbx)}")
            # axes[0, 0].imshow(img[0][slc1], cmap="gray")
            # axes[0, 1].imshow(w1_mask_bbx, cmap="gray")
            # axes[0, 2].axis("off")
            # axes[1, 0].imshow(img[1][slc1], cmap="gray")
            # axes[1, 1].imshow(w2_bbx_before, cmap="gray")
            # axes[1, 2].imshow(cp.where(w1_mask_bbx, w2_mask[slc1], 0), cmap="gray")
            # plt.show()

        low_intersect_id = low_intersect_id[low_intersect_id != 0]
        m1, m2 = cp.amax(w1_mask), cp.amax(w2_mask)
        max_ = cp.maximum(m1, m2)
        w1_mask_dil = cp_dilation(w1_mask, self.w1_dil_struct)
        w1_mask_dil = cp.where(cp.isin(w1_mask_dil, low_intersect_id), w1_mask_dil, cp.zeros_like(w1_mask_dil))
        w2_mask = cp.where(w1_mask_dil, w1_mask_dil + max_, w2_mask)
        w2_mask[(cp.bincount(w2_mask.ravel()) < self.args.min_sizes["w2"])[w2_mask]] = 0

        # stime = time.time()
        _, w2_mask = cp.unique(w2_mask, return_inverse=True)
        w2_mask = w2_mask.reshape(shape_)
        # print(f"unique and reshape {time.time()-stime}")

        # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        # axes[0, 0].imshow(img[0], cmap="gray")
        # axes[0, 1].imshow(img[1], cmap="gray")
        # axes[1, 0].imshow(label2rgb(w1_mask), cmap="gray")
        # axes[1, 1].imshow(label2rgb(w2_mask), cmap="gray")
        # plt.show()

        # Now, all nucleus have exactly one cytoplasm that completely covers them,
        # and that cytoplasm is the cytoplasm, is the cellpose cytoplasm that covered it the most.
        print(f"part 2 completion time in seconds: {time.time()-stime_partII}")
        ######################################################################
        stime_partIII = time.time()
        # 3) no deal with cyto ids
        m1, m2 = int(cp.amax(w1_mask)), int(cp.amax(w2_mask))
        max_ = int(cp.maximum(m1, m2))
        w1_count, w2_count = max_, max_
        # w2_slices = find_objects(cp.asnumpy(w2_mask), max_label=m2)
        w2_bboxes, w2_unix, n2 = self.get_bbox_from_mask(w2_mask)

        for jj in range(n2):
            (x0, y0, x1, y1), w2_label = w2_bboxes[jj], w2_unix[jj]
            w2_mask_bbx = w2_mask[y0:y1, x0:x1] == w2_label
            w1_mask_bbx = cp.where(w2_mask_bbx, w1_mask[y0:y1, x0:x1], cp.zeros_like(w2_mask_bbx))

            # kill small labeled nucleus tiny pieces in w1_mask_bbx covered by w2_mask_bbx
            # update the w2_mask under slc2 as well
            area_cond = (cp.bincount(w1_mask_bbx.ravel()) < self.args.min_sizes["w1"])[w1_mask_bbx]
            w1_mask_bbx[area_cond] = 0
            w2_mask_bbx[area_cond] = 0
            w2_mask[y0:y1, x0:x1][area_cond] = 0

            w1_unix = cp.setdiff1d(cp.unique(w1_mask_bbx), cp.zeros_like(w1_mask_bbx))
            w1_img_bbx = cp.where(w2_mask_bbx, img[0][y0:y1, x0:x1], cp.zeros_like(w2_mask_bbx))

            # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            # axes[0, 0].imshow(w1_img_bbx, cmap="gray")
            # axes[0, 1].imshow(w2_img_bbx, cmap="gray")
            # axes[1, 0].imshow(w1_mask_bbx, cmap="gray")
            # axes[1, 1].imshow(w2_mask_bbx, cmap="gray")
            # plt.show()

            n_w1 = len(w1_unix)

            if n_w1 == 0:  # no nucleus under w2_mask

                # segment the w1/nucleus channel under w2_mask

                w1_mask_bbx = w1_mask_bbx < cp_otsu(w1_img_bbx)
                w1_mask_bbx = cp_label(w1_mask_bbx, connectivity=2)
                w1_mask_bbx[(cp.bincount(w1_mask_bbx.ravel()) < self.args.min_sizes["w1"])[w1_mask_bbx]] = 0

                if cp.sum(w1_mask_bbx) > 0:
                    w1_mask[y0:y1, x0:x1] = \
                        cp.where(w1_mask_bbx, w1_count*cp.ones_like(w1_mask_bbx), w1_mask[y0:y1, x0:x1])
                    w1_count += 1
                else:
                    w2_mask[y0:y1, x0:x1][w2_mask_bbx] = 0

                # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
                # fig.suptitle(f"nw1 == 0 part \n small-intersecting-nuc-size={cp.sum(area_cond)}\n"
                #              f"w2-mask-area = {cp.sum(w2_mask_bbx)} "
                #              f"w1-mask-area = {cp.sum(cp.sum(w1_mask_bbx))}")
                # axes[0, 0].imshow(w1_img_bbx, cmap="gray")
                # axes[0, 1].imshow(w2_img_bbx, cmap="gray")
                # axes[1, 0].imshow(label2rgb(w1_mask_bbx), cmap="gray")
                # axes[1, 1].imshow(label2rgb(w2_mask_bbx), cmap="gray")
                # plt.show()

            elif n_w1 == 1:
                continue

            else:
                continue
                # w1_img_bbx_sitk = sitk.GetImageFromArray(cp.asnumpy(w1_img_bbx))
                # w1_mask_bbx_sitk = sitk.GetImageFromArray(cp.asnumpy(w1_mask_bbx))
                # # w2_img_bbx_sitk = sitk.GetImageFromArray(w2_img_bbx)
                # w2_mask_bbx_sitk = sitk.GetImageFromArray(cp.asnumpy(w2_mask_bbx).astype(np.uint8))
                #
                # # get apdc: average pairwise distances of centroid
                # self.label_intensity_stats_filter.Execute(w1_mask_bbx_sitk, w1_img_bbx_sitk)
                # w1_labels = self.label_intensity_stats_filter.GetLabels()
                # centroids = np.zeros((len(w1_labels), 2), dtype=np.float32)
                # for kk, w1_label in enumerate(w1_labels):
                #     centroids[kk] = w1_img_bbx_sitk.TransformPhysicalPointToIndex(
                #         self.label_intensity_stats_filter.GetCenterOfGravity(w1_label))
                # avg_pdist = np.mean(distance.pdist(centroids))
                #
                # if avg_pdist < self.args.multi_nucleus_dist_thresh:
                #     continue
                # else:  # segment the cytoplasm mask using watershed
                #     w2_mask_bbx_wsd = sitk.MorphologicalWatershedFromMarkers(
                #         sitk.SignedMaurerDistanceMap(w2_mask_bbx_sitk != 0),
                #         w1_mask_bbx_sitk,
                #         markWatershedLine=False)
                #     w2_mask_bbx_wsd = sitk.GetArrayFromImage(sitk.Mask(
                #         w2_mask_bbx_wsd,
                #         sitk.Cast(w2_mask_bbx_sitk, w2_mask_bbx_wsd.GetPixelID())))
                #     # fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
                #     # axes[0, 0].imshow(w1_img_bbx, cmap="gray")
                #     # axes[1, 0].imshow(w2_img_bbx, cmap="gray")
                #     # axes[0, 1].imshow(label2rgb(w1_mask_bbx), cmap="gray")
                #     # axes[1, 1].imshow(label2rgb(w2_mask_bbx), cmap="gray")
                #     # axes[0, 2].imshow(label2rgb(w2_mask_bbx_wsd), cmap="gray")
                #     # axes[1, 2].axis("off")
                #     # plt.show()
                #     w2_mask_bbx_wsd[w2_mask_bbx_wsd != 0] += w2_count
                #     w2_mask[slc2] = cp.where(w2_mask_bbx, cp.asarray(w2_mask_bbx_wsd), w2_mask[slc2])
                #     w2_count += int(cp.amax(w2_mask_bbx_wsd))
        print(f"part 3 completion time in seconds: {time.time()-stime_partIII}")

        stime_partIV = time.time()
        _, w2_mask = cp.unique(w2_mask, return_inverse=True)
        w2_mask = w2_mask.reshape(shape_)

        w1_mask[w1_mask > 0] = w2_mask[w1_mask > 0]
        w1_unix = cp.unique(w1_mask)
        w2_unix = cp.unique(w2_mask)
        diff2 = cp.setdiff1d(w2_unix, w1_unix)
        if len(diff2) > 0:
            w2_mask[cp.isin(w2_mask, diff2)] = 0
            w2_unix = cp.setdiff1d(w2_unix, diff2)
            # print(diff_)
        # print(len(w1_unix), len(w2_unix))
        assert cp.array_equal(w1_unix, w2_unix)
        cp_dilation(w2_mask, cp_disk(4), out=w2_mask)

        print(f"part 4 completion time in seconds: {time.time()-stime_partIV}")

        # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        # axes[0, 0].imshow(cp.asnumpy(img[0]), cmap="gray")
        # axes[0, 1].imshow(cp.asnumpy(img[1]), cmap="gray")
        # axes[1, 0].imshow(label2rgb(cp.asnumpy(w1_mask)), cmap="gray")
        # axes[1, 1].imshow(label2rgb(cp.asnumpy(w2_mask)), cmap="gray")
        # plt.show()
        # print('\n')S
        return img, w1_mask, w2_mask

    def step2_get_nucleoli_and_mito_masks_v2(self, img, w1_mask, w2_mask):
        # w5: mito channel
        w5_img = cp.where(cp_erosion(w1_mask, cp_disk(2)) | (w2_mask == 0), 0, img[4])
        w5_mask_global = w5_img < cp_otsu(w5_img)
        w5_mask_local = cp.zeros_like(img[4])
        # create cytoplasmic mask excluding the nucleus
        cyto_mask = cp.where(w1_mask > 0, cp.zeros_like(w1_mask), w2_mask)

        # w3: nucleoli channel
        w3_mask = cp.zeros_like(w2_mask)
        w2_bboxes, w2_unix, n2 = self.get_bbox_from_mask(w2_mask)

        for jj in range(n2):
            (x0, y0, x1, y1), w2_label = w2_bboxes[jj], w2_unix[jj]
            # w2_mask_bbx = w2_mask[y0:y1, x0:x1] == w2_label

            w3_bbx = w1_mask[y0:y1, x0:x1] == w2_label
            w5_bbx = cyto_mask[y0:y1, x0:x1] == w2_label
            ####################################################################
            # local mito mask calculation
            w5_img_tmp = cp.where(w5_bbx, w5_img[y0:y1, x0:x1], cp.zeros_like(w5_bbx))
            lb = float(cp.sum(w5_img_tmp < cp_otsu(w5_img_tmp)) / cp.size(w5_img_tmp))
            w5_in_range = tuple(cp.asnumpy(cp.percentile(w5_img_tmp, (lb, 99.9))))
            w5_img_tmp = cp_rescale_intensity(w5_img_tmp, in_range=w5_in_range)
            w5_mask_tmp = w5_img_tmp < cp_otsu(w5_img_tmp)
            # w5_mask_tmp = cp.where(w5_mask_tmp, obj_label, 0)
            w5_mask_local[y0:y1, x0:x1] = cp.where(w5_bbx, w5_mask_tmp, cp.zeros_like(w5_bbx))
            # fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
            # axes[0, 0].imshow(cp.where(w3_bbx, img[0][y0:y1, x0:x1], 0), cmap="gray")
            # axes[0, 1].imshow(cp.where(w5_bbx, img[1][y0:y1, x0:x1], 0), cmap="gray")
            # axes[0, 2].imshow(cp.where(w5_bbx, w5_img_tmp, 0), cmap="gray")
            # axes[1, 0].imshow(label2rgb(cp.where(w3_bbx, w1_mask[y0:y1, x0:x1], 0)), cmap="gray")
            # axes[1, 1].imshow(label2rgb(cp.where(w5_bbx, w2_mask[y0:y1, x0:x1], 0)), cmap="gray")
            # axes[1, 2].imshow(label2rgb(w5_mask_tmp), cmap="gray")
            # plt.show()
            ####################################################################################
            # local nucleoli calculation
            w3_img_tmp = cp.where(w3_bbx, img[2][y0:y1, x0:x1], cp.zeros_like(w3_bbx))
            if self.args.plate_protocol in ["greiner", "perkinelmer"]:
                lb = float((cp.sum(w3_img_tmp < cp_otsu(w3_img_tmp)) / cp.size(w3_img_tmp)))
            else:
                lb = .1

            w3_img_tmp = cp_gaussian(w3_img_tmp, sigma=2)
            # (40, 88), (30, 95)
            w3_in_range = tuple(cp.asnumpy(cp.percentile(w3_img_tmp, (lb, self.w3_local_rescale_intensity_ub))))
            w3_img_tmp = rescale_intensity(w3_img_tmp, in_range=w3_in_range)
            w3_mask_tmp = (w3_img_tmp < cp_yen(w3_img_tmp))
            # w3_mask_tmp = binary_erosion(w3_mask_tmp, disk(1))
            w3_mask_tmp = cp_label(w3_mask_tmp, connectivity=2)

            # remove small and large segmented nucleoli
            w3_bbx_area = cp.sum(w3_bbx)
            min_nucleoli_size = self.args.min_nucleoli_size_multiplier * w3_bbx_area
            max_nucleoli_size = self.args.max_nucleoli_size_multiplier * w3_bbx_area
            areas = cp.bincount(w3_mask_tmp.ravel())[w3_mask_tmp]
            cond = (areas < min_nucleoli_size) | (areas > max_nucleoli_size)
            w3_mask_tmp[cond] = 0
            areas[cond] = 0

            if self.args.plate_protocol in ["greiner", "perkinelmer"]:
                # remove nucleoli that intersect too much with the boundary
                w3_bbx_padded = cp.pad(
                    w3_bbx,
                    ((self.nucleoli_bd_pad, self.nucleoli_bd_pad),
                     (self.nucleoli_bd_pad, self.nucleoli_bd_pad)),
                    constant_values=(0, 0))
                bd = cp_find_boundaries(w3_bbx_padded, connectivity=2)
                bd = bd[
                     self.nucleoli_bd_pad:-self.nucleoli_bd_pad,
                     self.nucleoli_bd_pad:-self.nucleoli_bd_pad]
                if self.args.plate_protocol in ["greiner", "perkinelmer"]:
                    bd = cp_binary_dilation(bd, cp_disk(2))

                w3_tmp_bd_mask = w3_mask_tmp * bd
                bd_areas = cp.bincount(w3_tmp_bd_mask.ravel())[w3_tmp_bd_mask]
                area_ratio = areas/bd_areas
                w3_tmp_bd_mask[area_ratio < self.args.nucleoli_bd_area_to_nucleoli_area_threshold] = 0

                # fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
                # axes[0, 0].imshow(cp.where(w3_bbx, img[0][y0:y1, x0:x1], 0), cmap="gray")
                # axes[0, 1].imshow(cp.where(w3_bbx, img[2][y0:y1, x0:x1], 0), cmap="gray")
                # axes[0, 2].axis("off")
                # axes[1, 0].imshow(label2rgb(cp.where(w3_bbx, w1_mask[y0:y1, x0:x1], 0)), cmap="gray")
                # axes[1, 1].imshow(label2rgb(w3_mask_tmp), cmap="gray")
                # axes[1, 2].imshow(bd, cmap="gray")
                # plt.show()

            w3_mask[y0:y1, x0:x1][w3_mask_tmp] = w2_label

        # mito mask
        w5_mask = cp.logical_or(w5_mask_global, w5_mask_local).astype(cp.uint16)
        w5_mask *= cyto_mask.astype(cp.uint16)

        return w3_mask, w5_mask


if __name__ == "__main__":

    camii_server_flav = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid"
    camii_server_seema = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Seema"
    camii_server_jump_cpg0012 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0012"
    camii_server_jump_cpg0001 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0001"

    main_path = WindowsPath(camii_server_flav)
    exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"

    args = Args(experiment=exp_fold, main_path=main_path).args
    seg_model = SegmentationPartII(args)
    for ii in range(12):
        seg_model.run(seg_model.img_groups[ii])

    # mask_filename = "A01_F001_W1.png"
    # img_filename = "AssayPlate_PerkinElmer_CellCarrier-384_A01_T0001F001L01A01Z01C01.tif"
    # img_path = args.main_path/args.experiment/args.imgs_fold/img_filename
    # mask_path = args.step2_save_path/mask_filename
