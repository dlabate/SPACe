import time
from tqdm import tqdm
from pathlib import WindowsPath


import numpy as np
import SimpleITK as sitk

from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import find_objects

import skimage.io as sio
from skimage import exposure
from skimage.measure import label, regionprops_table
from skimage.measure._regionprops import RegionProperties
from skimage.segmentation import watershed, expand_labels, find_boundaries, clear_border
from skimage.morphology import disk, erosion, dilation, \
    binary_dilation, binary_erosion, binary_closing, remove_small_objects
from skimage.filters import threshold_otsu, gaussian
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity

from functools import partial
import multiprocessing as mp
import matplotlib.pyplot as plt

from cellpaint.utils.img_files_dep import get_all_file_paths, load_img
from cellpaint.utils.shared_memory import MyBaseManager, TestProxy
from cellpaint.utils.segmentation import remove_small_objects_mod, \
    remove_small_and_large_objects_mod, get_sitk_img_thresholding_mask
from cellpaint.utils.figure_objects import get_overlay, move_figure
from cellpaint.steps_single_plate.step0_args import Args, get_img_channel_groups


class ThresholdingSegmentation:
    """Sycn W0 and W1 masks, and also segment W2 and W4 channels using thresholding filters from SimpleITK."""
    analysis_step = 2

    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)

    yen_filter = sitk.YenThresholdImageFilter()
    yen_filter.SetInsideValue(0)
    yen_filter.SetOutsideValue(1)

    nucleoli_bd_pad = 5
    additive_constant = 1000
    # w0_bd_size = 300
    # # min_cyto_size = 1500
    # min_nucleus_size = 1000
    # nucleus_area_to_cyto_area_thresh = .75

    # # nucleoli segmentation hyperparameters
    # nucleoli_channel_rescale_intensity_percentile_ub = 99
    # nucleoli_bd_area_to_nucleoli_area_threshold = .2
    # min_nucleoli_size_multiplier = .005
    # max_nucleoli_size_multiplier = .3

    def __init__(self, args):
        self.args = args
        self.args.num_channels = 5

        # args.masks_path_p1.stem = "MasksP1"
        _, self.img_path_groups, self.n_groups = get_img_channel_groups(self.args)
        self.img_path_groups = list(self.img_path_groups)
        self.w0_mask_paths = list(self.args.step2_save_path.rglob("*_W1.png"))
        self.w1_mask_paths = list(self.args.step2_save_path.rglob("*_W2.png"))
        self.N = len(self.img_path_groups)

        # there has to be at least 8 cpu cores available on the computer.
        self.num_workers = min(8, mp.cpu_count(), self.N)

        print(f"number of images in step II: {self.N}")

    def get_masks(self, idx):
        img = load_img(self.img_path_groups[idx], self.args.num_channels, self.args.height, self.args.width)

        file_id = '_'.join(self.w0_mask_paths[idx].stem.split('_')[1:])
        mp0 = self.args.step2_save_path / f"w0_{file_id}.png"
        mp1 = self.args.step2_save_path / f"w1_{file_id}.png"
        mp2 = self.args.step2_save_path / f"w2_{file_id}.png"
        mp4 = self.args.step2_save_path / f"w4_{file_id}.png"

        if np.sum(img) == 0:  # if the image does not have the exact number of channels
            w0_mask = np.zeros_like(img[0], dtype=np.uint16)
            w1_mask = np.zeros_like(img[0], dtype=np.uint16)
            w2_mask = np.zeros_like(img[0], dtype=np.uint16)
            w4_mask = np.zeros_like(img[0], dtype=np.uint16)
        else:
            # making sure w0_mask and w1_mask are both cast to uint16
            w0_mask = sio.imread(str(self.w0_mask_paths[idx])).astype(np.uint16)
            w1_mask = sio.imread(str(self.w1_mask_paths[idx])).astype(np.uint16)

            # If the total number of segmented cells in both DAPI and CYTO channels is small then skip that image
            if (len(np.unique(w1_mask)) - 1 < self.args.min_fov_cell_count) and \
                    (len(np.unique(w0_mask)) - 1 < self.args.min_fov_cell_count):
                w0_mask = np.zeros_like(w0_mask)
                w1_mask = np.zeros_like(w0_mask)
                w2_mask = np.zeros_like(w0_mask)
                w4_mask = np.zeros_like(w0_mask)
            else:
                w0_mask, w1_mask, unix0 = self.step_1_sync_w0_and_w1_masks(w0_mask, w1_mask, img)
                # If the total number of segmented cells in both DAPI and CYTO channels is small then skip that image
                if len(unix0) - 1 < self.args.min_fov_cell_count:
                    w0_mask = np.zeros_like(w0_mask)
                    w1_mask = np.zeros_like(w0_mask)
                    w2_mask = np.zeros_like(w0_mask)
                    w4_mask = np.zeros_like(w0_mask)

                else:
                    w2_mask, w4_mask = self.step_2_get_nucleoli_and_mito_masks(w0_mask, w1_mask, img)
        if self.args.mode == "debug":
                self.show_all_masks(img, w0_mask, w1_mask, w2_mask, w4_mask)
        # assert w0_mask.dtype == w1_mask.dtype == w2_mask.dtype == w4_mask.dtype == np.uint16
        sio.imsave(mp0, w0_mask, check_contrast=False)
        sio.imsave(mp1, w1_mask, check_contrast=False)
        sio.imsave(mp2, w2_mask, check_contrast=False)
        sio.imsave(mp4, w4_mask, check_contrast=False)

    def show_all_masks(self, img, w0_mask, w1_mask, w2_mask, w4_mask):
            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            fig.set_size_inches(13, 9)
            move_figure(fig, 30, 30)
            for ii in range(0, 5):
                img[ii] = rescale_intensity(img[ii], in_range=tuple(np.percentile(img[ii], (5, 99.9))))
            axes[0, 0].imshow(get_overlay(img[self.args.nucleus_idx], w0_mask, self.args.colors), cmap="gray")
            axes[0, 1].imshow(get_overlay(img[self.args.cyto_idx], w1_mask, self.args.colors), cmap="gray")
            axes[1, 0].imshow(get_overlay(img[self.args.nucleoli_idx], w2_mask, self.args.colors), cmap="gray")
            axes[1, 1].imshow(get_overlay(img[self.args.mito_idx], w4_mask, self.args.colors), cmap="gray")
            axes[0, 0].set_title("Nucleus Channel", **self.args.csfont)
            axes[0, 1].set_title("Cytoplasm Channel", **self.args.csfont)
            axes[1, 0].set_title("Nucleoli Channel", **self.args.csfont)
            axes[1, 1].set_title("Mito Channel", **self.args.csfont)
            for ii, ax in enumerate(axes.flatten()):
                ax.set_axis_off()
            plt.show()

    def show_w0_and_w1_masks(self, w0_mask, w1_mask, img):
            from skimage.exposure import rescale_intensity
            from cellpaint.utils.figure_objects import get_overlay, move_figure
            fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
            fig.set_size_inches(13, 9)
            move_figure(fig, 30, 30)
            axes[0].imshow(get_overlay(img[self.args.nucleus_idx], w0_mask, self.args.colors), cmap="gray")
            axes[1].imshow(get_overlay(img[self.args.cyto_idx], w1_mask, self.args.colors), cmap="gray")
            axes[0].set_title("Nucleus Channel", **self.args.csfont)
            axes[1].set_title("Cytoplasm Channel", **self.args.csfont)
            for ii, ax in enumerate(axes.flatten()):
                ax.set_axis_off()
            plt.show()

    def handle_nucleus_masks_with_no_cyto_mask(self, w0_mask, w1_mask):
        """this function helps segment more cells in extreme conditions,
        for example when there are illumination issues.
        Use the dilation of w0_mask where w1_mask is zero, but w0_mask is positive"""
        # handle nuclei with no cyto
        w0_mask_nocyto = w0_mask.copy()
        # w0_mask_nocyto[(w0_mask > 0) & (w1_mask > 0)] = 0
        w0_mask_nocyto[w1_mask > 0] = 0
        # fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
        # axes[0].imshow(label2rgb(w0_mask, bg_label=0), cmap="gray")
        # axes[1].imshow(label2rgb(w1_mask, bg_label=0), cmap="gray")
        # axes[2].imshow(label2rgb(w0_mask_nocyto, bg_label=0), cmap="gray")
        # plt.show()
        w0_mask_nocyto = remove_small_objects_mod(w0_mask_nocyto, min_size=self.args.min_nucleus_size)
        if np.max(w0_mask_nocyto) > 0:
            dilation(w0_mask_nocyto, disk(4), out=w0_mask_nocyto)
            w0_mask_nocyto[w1_mask > 0] = 0
            w0_mask_nocyto[w0_mask_nocyto > 0] += self.additive_constant
            w1_mask += w0_mask_nocyto
            # print(len(np.unique(w0_mask_nocyto)), len(np.unique(w1_mask)))
            # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            # axes[0, 0].imshow(label2rgb(w0_mask, bg_label=0), cmap="gray")
            # axes[0, 1].imshow(label2rgb(w1_mask, bg_label=0), cmap="gray")
            # axes[1, 0].imshow(label2rgb(w0_mask_nocyto, bg_label=0), cmap="gray")
            # axes[1, 1].axis("off")
            # plt.show()
        return w1_mask

    def region_props_loop(self, w0_mask, w1_mask, img, w1_objects, jj, counter0, counter1):
        if w1_objects[jj] is None:
            return None
        obj_label = jj + 1
        w10_props = RegionProperties(
            slice=w1_objects[jj], label=obj_label, label_image=w1_mask,
            intensity_image=img[self.args.nucleus_idx],
            cache_active=True, )
        # within current cell, find pieces of w0_mask within the w1_mask
        w0_pieces = RegionProperties(
            slice=w1_objects[jj], label=obj_label, label_image=w1_mask,
            intensity_image=w0_mask,
            cache_active=True, ).intensity_image
        # removing edge artifact, so that the two masks do not touch on the bd
        w0_pieces = label(w0_pieces.astype(bool), background=0)
        if np.max(w0_pieces) > 1:
            w0_pieces = remove_small_objects_mod(w0_pieces, min_size=self.args.min_nucleus_size)
        y0, x0, y1, x1 = w10_props.bbox
        lw0_unix = np.unique(w0_pieces)
        # # # TODO: Rethink this step maybe, this should never happen!
        # if len(lw0_unix) == 0:  # remove bad cells
        #     print("the weird thing happened", lw0_unix)
        #     w1_mask[w1_mask == obj_label] = 0
        #     w0_mask[w1_mask == obj_label] = 0
        if len(lw0_unix) == 2:  # there is exactly one nucleus in the cyto mask
            return None
        # when there is cytoplasm but no nuclei inside segment inside of it in nuclei channel
        elif len(lw0_unix) == 1:
            w0_intensity = w10_props.intensity_image
            w0_lmask = get_sitk_img_thresholding_mask(w0_intensity, self.otsu_filter)
            binary_closing(w0_lmask, disk(8), out=w0_lmask)
            binary_fill_holes(w0_lmask, structure=np.ones((2, 2)), output=w0_lmask)
            if np.amax(w0_lmask) == 0:  # nothing was found
                w1_mask[w1_mask == obj_label] = 0
            else:
                if np.sum(w0_lmask) / w10_props.area > self.args.nucleus_area_to_cyto_area_thresh:
                    erosion(w0_lmask, disk(4), out=w0_lmask)
                # remove the segmented objects that are too small or too large
                w0_lmask = label(w0_lmask, connectivity=2, background=0)
                w0_lmask = remove_small_and_large_objects_mod(
                    w0_lmask, min_size=self.args.min_nucleus_size, max_size=.75*w10_props.area)
                unix = np.unique(w0_lmask)
                # if no object or more than 1 object was found
                if len(unix) == 0 or len(unix) == 1 or len(unix) >= 3:
                    w1_mask[w1_mask == obj_label] = 0
                else:  # success
                    w0_lmask = (w0_lmask > 0).astype(np.uint16)
                    w0_lmask[w0_lmask > 0] += counter0
                    w0_mask[y0:y1, x0:x1] += w0_lmask
                    counter0 += 1

            # # fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
            # # axes[0].imshow(w0_mask[y0:y1, x0:x1], cmap="gray")
            # # axes[1].imshow(w1_mask[y0:y1, x0:x1], cmap="gray")
            # # axes[2].imshow(w0_lmask, cmap="gray")
            # # plt.show()
            # cnt1 += 1
        elif len(lw0_unix) > 2:  # there is cytoplasm but more than one nuclei
            # segment/break down this cyto by watershed
            tmp1 = watershed(
                w10_props.image,
                markers=w0_pieces,
                connectivity=w0_pieces.ndim,
                mask=w10_props.image)
            tmp_unix, tmp1 = np.unique(tmp1, return_inverse=True)
            tmp1 = tmp1.reshape(w10_props.image.shape).astype(np.uint16)
            erosion(tmp1, disk(2), out=tmp1)
            tmp1[tmp1 > 0] += counter1
            w1_mask[y0:y1, x0:x1] = tmp1
            # w0cp = w0_mask[y0:y1, x0:x1].copy()
            # w1cp = w1_mask[y0:y1, x0:x1].copy()
            # # w0cp[np.isin(w0cp, np.setdiff1d(np.unique(w0cp), lw0_unix))] = 0
            # # w1cp[np.isin(w1cp, np.setdiff1d(np.unique(w1cp), np.unique(tmp1)))] = 0
            # fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
            # axes[0].imshow(label2rgb(w0cp, bg_label=0), cmap="gray")
            # axes[1].imshow(label2rgb(w1cp, bg_label=0), cmap="gray")
            # axes[2].imshow(label2rgb(tmp1, bg_label=0), cmap="gray")
            # plt.show()

            counter1 += len(np.setdiff1d(tmp_unix, [0]))
            # cnt2 += 1
        return w0_mask, w1_mask, counter0, counter1

    def step_1_sync_w0_and_w1_masks(self, w0_mask, w1_mask, img):
        # print("cellpose ", len(np.unique(w0_mask)), len(np.unique(w1_mask)))
        # step 1)
        clear_border(w0_mask, out=w0_mask)
        clear_border(w1_mask, out=w1_mask)
        w0_mask = remove_small_objects_mod(w0_mask, min_size=self.args.min_nucleus_size)
        w1_mask = remove_small_objects_mod(w1_mask, min_size=self.args.min_nucleus_size)
        # print("after clear border and remove small objects: ", len(np.unique(w0_mask)), len(np.unique(w1_mask)))
        #######################################################################
        # step 2)
        w1_mask = self.handle_nucleus_masks_with_no_cyto_mask(w0_mask, w1_mask)
        # print("after handling cells with no cyto", len(np.unique(w0_mask)), len(np.unique(w1_mask)))
        #################################################################
        # looking at area distribution fo each mask to check for segmentation problems is a very good idea.
        # area0 = regionprops_table(w0_mask, properties=["area"])["area"]
        # area1 = regionprops_table(w1_mask, properties=["area"])["area"]
        # fig, axes = plt.subplots(1, 1)
        # axes.hist(area0, color="red", label="w0/nucleus mask area")
        # axes.hist(area1, color="blue", label="w1/cyto mask area")
        # plt.show()
        ##############################################################################
        # step 3)
        w1_max = int(np.amax(w1_mask))
        w1_objects = find_objects(w1_mask, max_label=w1_max)
        unix0, unix1 = np.unique(w0_mask), np.unique(w1_mask)
        counter0, counter1 = unix0[-1], unix1[-1]
        # cnt1, cnt2 = 0, 0
        # total = len(unix1)
        # TODO: Remove bad nuclues and cyto around edges using circularity, maybe
        for jj in range(w1_max):
            out = self.region_props_loop(w0_mask, w1_mask, img, w1_objects, jj, counter0, counter1)
            if out is not None:
                w0_mask, w1_mask, counter0, counter1 = out
        # print("after self.region_props_loop", len(np.unique(w0_mask)), len(np.unique(w1_mask)))
        # self.show_w0_and_w1_masks(w0_mask, w1_mask, img)
        # print(total, 100*cnt1/total, 100*cnt2/total)
        # # remove noisy bd effects
        #############################################################################
        # step 4)
        mask_bd = find_boundaries(
            w1_mask,
            connectivity=2,
            mode="inner").astype(np.uint16)
        mask_bd = binary_dilation(mask_bd, disk(2))
        w0_mask[(w0_mask > 0) & mask_bd] = 0
        w0_mask = remove_small_objects_mod(w0_mask, min_size=self.args.w0_bd_size)
        # matches the labels of w1_mask and w0_mask
        w0_mask[w0_mask > 0] = w1_mask[w0_mask > 0]
        # print("after handling w0_mask boundary effects: ", len(np.unique(w0_mask)), len(np.unique(w1_mask)))
        ###########################################################################
        # step 5)
        # remove the leftovers pieces of the matching step (previous step)
        # in other words, remove segmented cytoplasm without nucleus
        diff = np.setdiff1d(np.unique(w1_mask), np.unique(w0_mask))
        w1_mask[np.isin(w1_mask, diff)] = 0
        # print(len(np.unique(w0_mask)), len(np.unique(w1_mask)))
        # # relabel w1_mask
        # _, w1_mask = np.unique(w1_mask, return_inverse=True)
        # w1_mask = np.uint16(w1_mask.reshape((self.args.height, self.args.width)))
        # print("final step removing cyto with no nucleus", len(np.unique(w0_mask)), len(np.unique(w1_mask)))
        # self.show_w0_and_w1_masks(w0_mask, w1_mask, img)

        unix0, unix1 = np.unique(w0_mask), np.unique(w1_mask)

        assert np.array_equal(unix0, unix1) and np.array_equal(w1_mask[w0_mask > 0], w0_mask[w0_mask > 0])
        return w0_mask, w1_mask, unix0

    def step_2_get_nucleoli_and_mito_masks(self, w0_mask, w1_mask, img):
        # mito mask
        tmp_img4 = img[-1].copy()
        tmp_img4[(dilation(w1_mask, disk(6)) == 0) | (erosion(w0_mask, disk(4)) > 0)] = 0
        global_mask4 = get_sitk_img_thresholding_mask(tmp_img4, self.otsu_filter)
        local_mask4 = np.zeros_like(w1_mask)

        cyto_mask = w1_mask.copy()
        cyto_mask[w0_mask > 0] = 0  # create cytoplasmic mask excluding the nucleus
        max_ = int(np.amax(cyto_mask))
        w4_objects = find_objects(cyto_mask, max_label=max_)
        ########################################################
        # nucleoli mask
        w2_mask = np.zeros_like(w0_mask)
        w0_objects = find_objects(w0_mask, max_label=max_)
        ########################################################
        for ii in range(max_):
            if w4_objects[ii] is None:
                continue
            obj_label = ii + 1
            w4_props = RegionProperties(
                slice=w4_objects[ii], label=obj_label, label_image=cyto_mask,
                intensity_image=tmp_img4,
                cache_active=True, )
            w2_props = RegionProperties(
                slice=w0_objects[ii], label=obj_label, label_image=w0_mask,
                intensity_image=img[self.args.nucleoli_idx],
                cache_active=True, )
            ####################################################################
            # local mito mask calculation
            y0, x0, y1, x1 = w4_props.bbox
            # scaled_img = rescale_intensity(limg, in_range=tuple(np.percentile(limg, [5, 99])))
            tmp4 = get_sitk_img_thresholding_mask(w4_props.intensity_image, self.otsu_filter).astype(bool)
            # local_mask4[y0:y1, x0:x1][tmp_mask > 0] = tmp_mask[tmp_mask > 0]
            local_mask4[y0:y1, x0:x1] += tmp4.astype(np.uint16)
            del tmp4
            #####################################################################
            # local nucleoli mask calculation
            l_img2 = w2_props.intensity_image
            lmask2 = w2_props.image
            y0, x0, y1, x1 = w2_props.bbox
            if self.args.plate_protocol in ["greiner", "perkinelmer"]:
                lb = np.sum(l_img2 < threshold_otsu(l_img2)) / np.size(l_img2)
            else:
                lb = .1
            # lb = np.sum(l_img2 < threshold_otsu(l_img2)) / np.size(l_img2)
            l_img2 = gaussian(l_img2, sigma=1)
            prc = tuple(np.percentile(l_img2, (lb, self.args.nucleoli_channel_rescale_intensity_percentile_ub)))
            l_img2 = exposure.rescale_intensity(l_img2, in_range=prc)  # (40, 88), (30, 95)
            min_nucleoli_size = self.args.min_nucleoli_size_multiplier * (np.sum(lmask2))
            max_nucleoli_size = self.args.max_nucleoli_size_multiplier * (np.sum(lmask2))

            lmask2_padded = np.pad(
                lmask2,
                ((self.nucleoli_bd_pad, self.nucleoli_bd_pad),
                 (self.nucleoli_bd_pad, self.nucleoli_bd_pad)),
                constant_values=(0, 0))
            bd = find_boundaries(lmask2_padded, connectivity=2)
            bd = bd[
                 self.nucleoli_bd_pad:-self.nucleoli_bd_pad,
                 self.nucleoli_bd_pad:-self.nucleoli_bd_pad]
            if self.args.plate_protocol in ["greiner", "perkinelmer"]:
                bd = binary_dilation(bd, disk(3))

            tmp2 = get_sitk_img_thresholding_mask(l_img2, self.yen_filter).astype(np.uint16)
            tmp2 = binary_erosion(tmp2, disk(1))
            tmp2 = label(tmp2, connectivity=2)
            tmax = int(np.amax(tmp2))

            tmp2_objects = find_objects(tmp2, max_label=tmax)
            tmp2_bd_objects = find_objects(tmp2 * bd, max_label=tmax)
            for jj in range(tmax):
                if tmp2_objects[jj] is None:
                    continue
                tmp2_label = jj + 1
                tmp2_props = RegionProperties(
                    slice=tmp2_objects[jj], label=tmp2_label, label_image=tmp2,
                    intensity_image=None, cache_active=True, )
                tmp2_bd_props = RegionProperties(
                    slice=tmp2_bd_objects[jj], label=tmp2_label, label_image=tmp2 * bd,
                    intensity_image=None, cache_active=True, )
                if (tmp2_bd_props.area / tmp2_props.area > self.args.nucleoli_bd_area_to_nucleoli_area_threshold or \
                        tmp2_props.area < min_nucleoli_size or tmp2_props.area > max_nucleoli_size) and \
                        self.args.plate_protocol in ["greiner", "perkinelmer"]:
                    tmp2[tmp2 == tmp2_label] = 0
            w2_mask[y0:y1, x0:x1] += (tmp2 != 0).astype(np.uint16)
            # w2_mask[y0:y1, x0:x1][tmp2 > 0] = obj_label
        # mito mask
        w4_mask = np.logical_or(global_mask4, local_mask4).astype(np.uint16)
        w4_mask *= cyto_mask
        w2_mask *= w0_mask
        # print(len(np.unique(w0_mask)),
        #       len(np.unique(w1_mask)),
        #       len(np.unique(w2_mask)),
        #       len(np.unique(w4_mask)),
        #       len(np.unique(cyto_mask)))
        # w4_mask[w4_mask > 0] = cyto_mask[w4_mask > 0]
        # assert np.array_equal(np.unique(w4_mask), np.unique(cyto_mask))
        # assert w0_mask.dtype == w1_mask.dtype == w2_mask.dtype == np.uint16
        return w2_mask, w4_mask


def step2_run_single_process_for_loop(args):
    inst = ThresholdingSegmentation(args)
    for ii in tqdm(range(inst.N)):
    # for ii in range(inst.N):
        inst.get_masks(ii)
        # print('\n')


def step2_run_multi_process_for_loop(args, myclass):
    """
    We have to Register the ThresholdingSegmentation class object as well as its attributes as shared using:
    https://stackoverflow.com/questions/26499548/accessing-an-attribute-of-a-multiprocessing-proxy-of-a-class

    """
    MyManager = MyBaseManager()
    MyManager.register(myclass.__name__, myclass, TestProxy)
    with MyManager as manager:
        inst = getattr(manager, myclass.__name__)(args)

        with mp.Pool(processes=inst.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(inst.get_masks, np.arange(inst.N)), total=inst.N):
                pass


def step2_main_run_loop(args, myclass=ThresholdingSegmentation):
    """
    Main function for cellpaint step II which:
        1) Corrects and syncs the Nucleus and Cyto masks from Cellpaint stepI.
        2) Generates Nucleoli and Mitocondria masks using Nucleus and Cytoplasm masks, respectively.

        In what follows each mask is referred to as:
        Nucleus mask: w0_mask
        Cyto mask: w1_mask
        Nucleoli mask: w2_mask
        Mito mask: w4_mask

        It saves all those masks as separate png files into:

        if args.mode.lower() == "debug":
            self.args.masks_path_p2 = args.main_path / args.experiment / "Debug" / "MasksP2"
        elif args.mode.lower() == "test":
            self.args.masks_path_p2 = args.main_path / args.experiment / "Test" / "MasksP2"
        elif args.mode.lower() == "full":
            self.args.masks_path_p2 = args.main_path / args.experiment / "MasksP2"
    """
    print("Cellpaint Step 2: \n"
          "2-1) Matching segmentation of Nucleus and Cytoplasm \n"
          "2-2) Thresholding segmentation of Nucleoli and Mitocondria ...")
    s_time = time.time()
    if args.mode == "debug":
        step2_run_single_process_for_loop(args)
    else:
        step2_run_multi_process_for_loop(args, myclass)
    print(f"Finished Cellpaint step 2 in: {(time.time()-s_time)/3600} hours\n")


if __name__ == "__main__":

    camii_server_flav = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid"
    camii_server_seema = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Seema"
    camii_server_jump_cpg0012 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0012"
    camii_server_jump_cpg0001 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0001"

    main_path = WindowsPath(camii_server_flav)
    exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"

    args = Args(experiment=exp_fold, main_path=main_path).args
    args.mode = "full"
    step2_main_run_loop(args, ThresholdingSegmentation)
