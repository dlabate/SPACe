import os
import time
from tqdm import tqdm
import multiprocessing as mp
from pathlib import WindowsPath
import matplotlib.pyplot as plt

import cv2
import sympy
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageFilter
import skimage.io as sio
from skimage.measure import label
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import watershed, expand_labels, find_boundaries, clear_border
from skimage.morphology import disk, erosion, dilation, closing, \
    binary_dilation, binary_erosion, binary_closing, remove_small_objects, convex_hull_image

from scipy.spatial import distance
from scipy.ndimage import find_objects

from cellpose import models
import pyclesperanto_prototype as cle

from cellpaint.steps_single_plate.step0_args import Args, \
    load_img, sort_key_for_imgs, sort_key_for_masks, set_mask_save_name


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = cle.select_device("RTX")


class SegmentationPartI:
    """Read all the image tif files for the experiment as a list and sort them, "img_paths".
     Then divide the files into groups each containing the 4/5 channels of a single image, "img_path_groups".
     for example,
     [[args.main_path\args.experiment\args.plate_protocol\args.plate_protocol_B02_T0001F001L01A01Z01C01.tif,
     args.main_path\args.experiment\args.plate_protocol\args.plate_protocol_B02_T0001F001L01A02Z01C02.tif,
     args.main_path\args.experiment\args.plate_protocol\args.plate_protocol_B02_T0001F001L01A03Z01C03.tif,
     args.main_path\args.experiment\args.plate_protocol\args.plate_protocol_B02_T0001F001L01A04Z01C04.tif,
     args.main_path\args.experiment\args.plate_protocol\args.plate_protocol_B02_T0001F001L01A05Z01C05.tif,],

     [args.main_path\args.experiment\args.plate_protocol\args.plate_protocol_B02_T0001F002L01A01Z01C01.tif,
     args.main_path\args.experiment\args.plate_protocol\args.plate_protocol_B02_T0001F002L01A02Z01C02.tif,
     args.main_path\args.experiment\args.plate_protocol\args.plate_protocol_B02_T0001F002L01A03Z01C03.tif,
     args.main_path\args.experiment\args.plate_protocol\args.plate_protocol_B02_T0001F002L01A04Z01C04.tif,
     args.main_path\args.experiment\args.plate_protocol\args.plate_protocol_B02_T0001F002L01A05Z01C05.tif,]

     Note that if the number of cells segmented in an image is smaller than "MIN_CELLS",
     meaning the FOV/image is mostly empty, this function will NOT save the corresponding mask into disk!!!
     """
    analysis_step = 2

    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)

    # meta_cols = ["exp-id", "well-id", "fov", "treatment", "cell-line", "density", "dosage", "other"]
    # stages = ["0-raw-image", "2-bgsub-image"]

    def __init__(self, args):
        """self.N is the total number of images (when all their channels are grouped together) in the
        args.main_path\args.experiment\args.plate_protocol folder."""
        self.args = args
        self.cellpose_model = models.Cellpose(gpu=True, model_type=self.args.cellpose_model_type, net_avg=False)

        if self.args.mode == "preview":
            self.save_path = self.args.main_path / self.args.experiment / f"Step0_MasksP1-Preview"
            self.save_path.mkdir(exist_ok=True, parents=True)
        else:
            self.save_path = self.args.main_path / self.args.experiment / f"Step{self.analysis_step}_MasksP1"
            self.save_path.mkdir(exist_ok=True, parents=True)

    def get_cellpose_masks(self, img_channels_filepaths, img_filename_key):
        """
        cellpose-segmentation of a single image:
        Segment the nucleus and cytoplasm channels using Cellpose then save the masks to disk."""

        # # stime = time.time()
        # load, rescale/contrast-enhance, and background subtraction the images using the tophat filter!!!
        img = load_img(img_channels_filepaths, self.args)

        if self.args.step2_segmentation_algorithm == "w1=cellpose_w2=cellpose":  # recommended
            # get nucleus and cyto masks using cellpose
            w1_mask, _, _, _ = self.cellpose_model.eval(
                img[0],
                diameter=self.args.cellpose_nucleus_diam,
                channels=[0, 0],
                batch_size=self.args.cellpose_batch_size,
                z_axis=None,
                channel_axis=None,
                resample=False,
                min_size=self.args.min_sizes["w1"],
            )

            w2_mask, _, _, _ = self.cellpose_model.eval(
                img[1],
                diameter=self.args.cellpose_cyto_diam,
                channels=[0, 0],
                batch_size=self.args.cellpose_batch_size,
                z_axis=None,
                channel_axis=None,
                resample=False,
                min_size=self.args.min_sizes["w2"])
        # # print(f"cellpose w1 takes {time.time()-stime} seconds")
        ###########################################################################################
        elif self.args.step2_segmentation_algorithm == "w1=pycle_w2=pycle":
            # stime = time.time()
            w1_mask = np.array(cle.voronoi_otsu_labeling(cle.push(img[0]), spot_sigma=10)).astype(np.uint16)
            w2_mask = cle.voronoi_otsu_labeling(cle.push(img[1]), spot_sigma=8, outline_sigma=1)
            # w2_mask = cle.minimum_box(cle.maximum_box(w2_mask, radius_x=10, radius_y=10), radius_x=10, radius_y=10)
            w2_mask = np.array(w2_mask).astype(np.uint16)
            # print(f"cle voronoi_otsu_labeling on w2 takes {time.time()-stime} seconds")

        elif self.args.step2_segmentation_algorithm == "w1=cellpose_w2=pycle":
            w1_mask, _, _, _ = self.cellpose_model.eval(
                img[0],
                diameter=self.args.cellpose_nucleus_diam,
                channels=[0, 0],
                batch_size=self.args.cellpose_batch_size,
                z_axis=None,
                channel_axis=None,
                resample=False, )
            w2_mask = cle.voronoi_otsu_labeling(cle.push(img[1]), spot_sigma=8, outline_sigma=1)
            # w2_mask = cle.minimum_box(cle.maximum_box(w2_mask, radius_x=10, radius_y=10), radius_x=10, radius_y=10)
            w2_mask = np.array(w2_mask).astype(np.uint16)

        elif self.args.step2_segmentation_algorithm == "w1=pycle_w2=cellpose":
            w1_mask = cle.voronoi_otsu_labeling(cle.push(img[0]), spot_sigma=8, outline_sigma=1)
            # w1_mask = cle.minimum_box(cle.maximum_box(w1_mask, radius_x=10, radius_y=10), radius_x=10, radius_y=10)
            w1_mask = np.array(w1_mask).astype(np.uint16)

            w2_mask, _, _, _ = self.cellpose_model.eval(
                img[1],
                diameter=self.args.cellpose_cyto_diam,
                channels=[0, 0],
                batch_size=self.args.cellpose_batch_size,
                z_axis=None,
                channel_axis=None,
                resample=False, )

        else:
            raise NotImplementedError()

        if self.args.mode == "test":
            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            fig.suptitle(img_filename_key)
            axes[0, 0].imshow(img[0], cmap="gray")
            axes[0, 1].imshow(label2rgb(w1_mask, bg_label=0), cmap="gray")
            axes[1, 0].imshow(img[1], cmap="gray")
            axes[1, 1].imshow(label2rgb(w2_mask, bg_label=0), cmap="gray")
            # axes[2, 0].axis("off")
            plt.show()

        return w1_mask, w2_mask

    def run_single(self, img_channels_filepaths, img_filename_key):
        w1_mask, w2_mask = self.get_cellpose_masks(img_channels_filepaths, img_filename_key)
        ########################################################################################################
        # Save the masks into disk
        # Create a savename choosing a name for the experiment name, and also using the well_id, and fov.
        # ('2022-0817-CP-benchmarking-density_20220817_120119', 'H24', 'F009')
        exp_id, well_id, fov = img_filename_key[0], img_filename_key[1], img_filename_key[2]

        w1_mask = np.uint16(w1_mask)
        w2_mask = np.uint16(w2_mask)
        sio.imsave(self.save_path / set_mask_save_name(well_id, fov, 0), w1_mask, check_contrast=False)
        sio.imsave(self.save_path / set_mask_save_name(well_id, fov, 1), w2_mask, check_contrast=False)
        # print(f"saving {time.time() - et}")


class SegmentationPartII:
    """Never put any object here that is a numpy array, because multiprocess can't pickle it!!!"""
    analysis_step = 3
    buffer_size = 3
    w1_labels_shift = 2
    w3_local_rescale_intensity_lb = 10
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

    # TODO: To smoothen the boundary of each object (if possible and can be done quickly)
    #  option 1) Can use the mode filter in PIL.
    #  option 2) Can use the dilation and erosion ops in skimage. ✓
    #  option 3) Can use the convex-hull op in skimage in skimage.
    #  option 4) Apply median blur and use watershed

    def __init__(self, args, show_masks=False):
        """Never put any object here that is a numpy array, because multiprocess can't pickle it!!!"""

        self.args = args
        self.show_masks = show_masks

        if self.args.mode == "preview":
            self.load_path = self.args.main_path / self.args.experiment / f"Step0_MasksP1-Preview"
            self.save_path = self.args.main_path / self.args.experiment / f"Step0_MasksP2-Preview"
        else:
            self.load_path = self.args.main_path / self.args.experiment / f"Step{self.analysis_step-1}_MasksP1"
            self.save_path = self.args.main_path / self.args.experiment / f"Step{self.analysis_step}_MasksP2"
        self.save_path.mkdir(exist_ok=True, parents=True)

        self.w1_mask_filepaths = list(self.load_path.rglob("*_W1.png"))
        self.w2_mask_filepaths = list(self.load_path.rglob("*_W2.png"))

        # check matching

        if self.args.mode != "preview":
            assert len(self.w1_mask_filepaths) == \
                   len(self.w2_mask_filepaths) == \
                   len(self.args.img_channels_filepaths)
            for it0, it1, it2 in zip(self.args.img_channels_filepaths, self.w1_mask_filepaths, self.w2_mask_filepaths):
                assert sort_key_for_imgs(it0[0], "to_get_well_id_and_fov", self.args.plate_protocol) == \
                sort_key_for_masks(it1) == sort_key_for_masks(it2)
                # print(it0[0].stem, '\n', it1.stem, '\n', it2.stem, '\n')

            self.num_workers = min(16, mp.cpu_count(), self.args.N)

    def run_demo(self, img_group, img, w1_mask, w2_mask):
        well_id, fov = sort_key_for_imgs(img_group[0], "to_get_well_id_and_fov", self.args.plate_protocol)
        # stime = time.time()
        w1_mask, w2_mask = self.step1_preprocessing_and_w1w2_label_matching(img, w1_mask, w2_mask)
        # print(f"step3 finished in {time.time()-stime} seconds")
        # stime = time.time()
        w3_mask, w5_mask = self.step2_get_nucleoli_and_mito_masks_v2(img, w1_mask, w2_mask)
        # fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
        # fig.set_size_inches(40, 20)
        # fig.suptitle(f"well-id={well_id}  fov={fov}")
        # axes[0, 0].imshow(img[0], cmap="gray")
        # axes[0, 1].imshow(img[1], cmap="gray")
        # axes[0, 2].imshow(img[2], cmap="gray")
        # axes[0, 3].imshow(img[4], cmap="gray")
        # axes[1, 0].imshow(label2rgb(w1_mask), cmap="gray")
        # axes[1, 1].imshow(label2rgb(w2_mask), cmap="gray")
        # axes[1, 2].imshow(label2rgb(w3_mask), cmap="gray")
        # axes[1, 3].imshow(label2rgb(w5_mask), cmap="gray")
        # plt.show()
        return w1_mask, w2_mask, w3_mask, w5_mask

    def run_single(self, index):

        # 0) get image group keys
        img_group = self.args.img_channels_filepaths[index]
        _, well_id, fov = self.args.img_filename_keys[index]

        # 1) load image and masks
        img = load_img(img_group, self.args)
        w1_mask_path = self.w1_mask_filepaths[index]
        w2_mask_path = self.w2_mask_filepaths[index]

        w1_mask = cv2.imread(str(w1_mask_path), cv2.IMREAD_UNCHANGED)
        w2_mask = cv2.imread(str(w2_mask_path), cv2.IMREAD_UNCHANGED)

        # 2) get masks
        # stime = time.time()
        w1_mask, w2_mask = self.step1_preprocessing_and_w1w2_label_matching(img, w1_mask, w2_mask)
        # print(f"step1 finished in {time.time()-stime} seconds")
        # stime = time.time()
        w3_mask, w5_mask = self.step2_get_nucleoli_and_mito_masks_v2(img, w1_mask, w2_mask)
        # print(f"step2-v2 finished in {time.time()-stime} seconds")

        # 3) show
        if self.show_masks == "test":
            fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
            axes[0, 0].imshow(img[0], cmap="gray")
            axes[0, 1].imshow(img[1], cmap="gray")
            axes[0, 2].imshow(img[2], cmap="gray")
            axes[0, 3].imshow(img[4], cmap="gray")
            axes[1, 0].imshow(label2rgb(w1_mask), cmap="gray")
            axes[1, 1].imshow(label2rgb(w2_mask), cmap="gray")
            axes[1, 2].imshow(label2rgb(w3_mask), cmap="gray")
            axes[1, 3].imshow(label2rgb(w5_mask), cmap="gray")
            plt.show()

        # 4) save
        sio.imsave(self.save_path / set_mask_save_name(well_id, fov, 0), w1_mask, check_contrast=False)
        sio.imsave(self.save_path / set_mask_save_name(well_id, fov, 1), w2_mask, check_contrast=False)
        sio.imsave(self.save_path / set_mask_save_name(well_id, fov, 2), w3_mask, check_contrast=False)
        sio.imsave(self.save_path / set_mask_save_name(well_id, fov, 4), w5_mask, check_contrast=False)

    def run_multi(self, index):
        # 0) Get keys
        # well_id, fov = sort_key_for_imgs(
        #     self.args.img_channels_filepaths[index][0],
        #     "to_get_well_id_and_fov", self.args.plate_protocol)
        _, well_id, fov = self.args.img_filename_keys[index]
        w1_mask_path = self.w1_mask_filepaths[index]
        w2_mask_path = self.w2_mask_filepaths[index]

        # 1) Read inputs
        img = load_img(self.args.img_channels_filepaths[index], self.args)
        w1_mask = cv2.imread(str(w1_mask_path), cv2.IMREAD_UNCHANGED)
        w2_mask = cv2.imread(str(w2_mask_path), cv2.IMREAD_UNCHANGED)

        # 2) Generate masks
        w1_mask, w2_mask = self.step1_preprocessing_and_w1w2_label_matching(img, w1_mask, w2_mask)
        w3_mask, w5_mask = self.step2_get_nucleoli_and_mito_masks_v2(img, w1_mask, w2_mask)

        # 3) Save
        sio.imsave(self.save_path / set_mask_save_name(well_id, fov, 0), w1_mask, check_contrast=False)
        sio.imsave(self.save_path / set_mask_save_name(well_id, fov, 1), w2_mask, check_contrast=False)
        sio.imsave(self.save_path / set_mask_save_name(well_id, fov, 2), w3_mask, check_contrast=False)
        sio.imsave(self.save_path / set_mask_save_name(well_id, fov, 4), w5_mask, check_contrast=False)

    def step1_preprocessing_and_w1w2_label_matching(self, img, w1_mask, w2_mask):
        """
        This modeling is based upon observing that the intersection between nucleus channel and cyto channel happens as:
        The histogram of intersection ratios has big tails and is tiny in the middle,
        that is mostly nucleus/cyto intersection is:
        1) either the intersection is really small, the cyto barely touches a nucleus
        2) or the intersection is really large and the cyto almost covers the entire nucleus

        Here we assume w1_mask.ndim == 2 and w2_mask.ndim == 2"""
        # create a border object:
        border_mask = np.zeros((self.args.height, self.args.width), dtype=bool)
        border_mask[0:self.buffer_size, :] = True
        border_mask[-self.buffer_size:, :] = True
        border_mask[:, 0:self.buffer_size] = True
        border_mask[:, -self.buffer_size:] = True

        label_intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()

        shape_ = w2_mask.shape
        w1_mask = w1_mask.astype(np.uint32)
        w2_mask = w2_mask.astype(np.uint32)

        # slightly smoothen the masks
        w1_mask = closing(w1_mask, disk(4))
        w2_mask = closing(w2_mask, disk(4))

        # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        # axes[0, 0].imshow(img[0], cmap="gray")
        # axes[0, 1].imshow(img[1], cmap="gray")
        # axes[1, 0].imshow(label2rgb(w1_mask), cmap="gray")
        # axes[1, 1].imshow(label2rgb(w2_mask), cmap="gray")
        # plt.show()

        # stime = time.time()
        if w1_mask.ndim != 2 or w2_mask.ndim != 2:
            raise ValueError("Arrs have to be two dimensional")

        # remove small objects/cells in botth w1_mask and w2_mask
        w1_mask[(np.bincount(w1_mask.ravel()) < self.args.min_sizes["w1"])[w1_mask]] = 0
        w2_mask[(np.bincount(w2_mask.ravel()) < self.args.min_sizes["w2"])[w2_mask]] = 0

        # remove cells touching the border in w1_mask, and the ones
        # touching the border in w2 mask which don't contain any nucleus
        w1_border_ids = np.unique(w1_mask[border_mask])
        w1_mask[np.isin(w1_mask, w1_border_ids)] = 0
        w2_border_ids = np.unique(w2_mask[border_mask & (w1_mask == 0)])
        w2_mask[np.isin(w2_mask, w2_border_ids)] = 0
        # print(w1_border_ids, w2_border_ids)

        if np.sum(w1_mask) == 0 or np.sum(w2_mask) == 0:
            print("no pixels detected ...")
            return w1_mask.astype(np.uint16), w2_mask.astype(np.uint16)

        # print(f"part 1 completion time in seconds: {time.time()-stime}")
        ########################################################
        # stime = time.time()
        # 2) First deal with nucleus ids
        # since multiple cytoplasms may intersect a nucleus,
        # intersect_ratio between a nucleus and cyto channel will happen at multiple cyto instances/labels,
        # we have to choose the one with the largest intersection portion,
        # otherwise, if the intersection is not significant we just label the cyto same as the expanded nucleus
        intersect_ratios, w1_mask, w2_mask = self.get_interect_area_over_w1_area_ratios(w1_mask, w2_mask)
        m1, m2 = np.amax(w1_mask), np.amax(w2_mask)
        w1_slices = find_objects(w1_mask, max_label=m1)
        # w1_unix = np.setdiff1d(np.unique(w1_mask), 0)
        low_intersect_ids = []
        for ii, slc1 in enumerate(w1_slices):
            if slc1 is None:
                continue
            w1_label = ii + 1
            w1_bbox = w1_mask[slc1] == w1_label
            ratio_bbox = np.where(w1_bbox, intersect_ratios[slc1], 0)
            rmax = np.amax(ratio_bbox)
            # w2_bbox_before = np.where(w1_bbox, w2_mask[slc1], 0)

            if rmax < .5:  # if the intersection is small, remove the nucleus
                w2_mask[slc1] = np.where(w1_bbox, 0, w2_mask[slc1])
                low_intersect_ids.append(w1_label)
            else:  # otherwise, extend the w2_mask to contain the entire nucleus
                w2_label = np.amax(w2_mask[slc1][ratio_bbox == rmax])
                w2_mask[slc1][w1_bbox] = w2_label

            # fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
            # fig.suptitle(f"cyto-intersect-area/nucleus-area ratios: "
            #              f"{np.unique(ratio_bbox)}")
            # axes[0, 0].imshow(img[0][slc1], cmap="gray")
            # axes[0, 1].imshow(w1_bbox, cmap="gray")
            # axes[0, 2].axis("off")
            # axes[1, 0].imshow(img[1][slc1], cmap="gray")
            # axes[1, 1].imshow(w2_bbox_before, cmap="gray")
            # axes[1, 2].imshow(np.where(w1_bbox, w2_mask[slc1], 0), cmap="gray")
            # plt.show()
        w1_mask_dil = dilation(w1_mask, disk(4))
        w1_mask_dil = np.where(np.isin(w1_mask_dil, low_intersect_ids), w1_mask_dil, 0)
        w2_mask = np.where(w1_mask_dil, w1_mask_dil, w2_mask)
        w2_mask[(np.bincount(w2_mask.ravel()) < self.args.min_sizes["w2"])[w2_mask]] = 0

        # stime = time.time()
        _, w2_mask = np.unique(w2_mask, return_inverse=True)
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
        # print(f"part 2 completion time in seconds: {time.time()-stime}")
        ######################################################################
        # stime = time.time()
        # 3) no deal with cyto ids
        m1 = np.amax(w1_mask)
        m2 = np.amax(w2_mask)
        max_ = np.maximum(m1, m2)

        w1_count = max_
        w2_count = max_
        w2_slices = find_objects(w2_mask, max_label=m2)
        for jj, slc2 in enumerate(w2_slices):
            if slc2 is None:
                continue
            w2_label = jj + 1
            w2_mask_bbox = w2_mask[slc2] == w2_label
            w1_mask_bbox = np.where(w2_mask_bbox, w1_mask[slc2], 0)

            # kill small labeled nucleus tiny pieces in w1_mask_bbox covered by w2_mask_bbox
            # update the w2_mask under slc2 as well
            area_cond = (np.bincount(w1_mask_bbox.ravel()) < self.args.min_sizes["w1"])[w1_mask_bbox]
            w1_mask_bbox[area_cond] = 0
            w2_mask_bbox[area_cond] = 0
            w2_mask[slc2][area_cond] = 0

            w1_unix = np.setdiff1d(np.unique(w1_mask_bbox), 0)
            w1_img_bbox = np.where(w2_mask_bbox, img[0][slc2], 0)
            w1_img_bbox_sitk = sitk.GetImageFromArray(w1_img_bbox)
            w1_mask_bbox_sitk = sitk.GetImageFromArray(w1_mask_bbox)

            # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            # axes[0, 0].imshow(w1_img_bbox, cmap="gray")
            # axes[0, 1].imshow(w2_img_bbox, cmap="gray")
            # axes[1, 0].imshow(w1_mask_bbox, cmap="gray")
            # axes[1, 1].imshow(w2_mask_bbox, cmap="gray")
            # plt.show()

            n_w1 = len(w1_unix)

            if n_w1 == 0:  # no nucleus under w2_mask

                # segment the w1/nucleus channel under w2_mask
                w1_mask_bbox = sitk.GetArrayFromImage(self.otsu_filter.Execute(w1_img_bbox_sitk))
                w1_mask_bbox = label(w1_mask_bbox, connectivity=2)
                w1_mask_bbox[(np.bincount(w1_mask_bbox.ravel()) < self.args.min_sizes["w1"])[w1_mask_bbox]] = 0
                if np.sum(w1_mask_bbox) > 0:
                    w1_mask[slc2] = np.where(w1_mask_bbox, w1_count, w1_mask[slc2])
                    w1_count += 1
                else:
                    w2_mask[slc2][w2_mask_bbox] = 0

                # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
                # fig.suptitle(f"nw1 == 0 part \n small-intersecting-nuc-size={np.sum(area_cond)}\n"
                #              f"w2-mask-area = {np.sum(w2_mask_bbox)} "
                #              f"w1-mask-area = {np.sum(np.sum(w1_mask_bbox))}")
                # axes[0, 0].imshow(w1_img_bbox, cmap="gray")
                # axes[0, 1].imshow(w2_img_bbox, cmap="gray")
                # axes[1, 0].imshow(label2rgb(w1_mask_bbox), cmap="gray")
                # axes[1, 1].imshow(label2rgb(w2_mask_bbox), cmap="gray")
                # plt.show()

            elif n_w1 == 1:
                continue

            else:
                # get apdc: average pairwise distances of centroid
                w2_mask_bbox_sitk = sitk.GetImageFromArray(w2_mask_bbox.astype(np.uint8))
                label_intensity_stats_filter.Execute(w1_mask_bbox_sitk, w1_img_bbox_sitk)
                w1_labels = label_intensity_stats_filter.GetLabels()

                centroids = np.zeros((len(w1_labels), 2), dtype=np.float32)
                for kk, w1_label in enumerate(w1_labels):
                    centroids[kk] = w1_img_bbox_sitk.TransformPhysicalPointToIndex(
                        label_intensity_stats_filter.GetCenterOfGravity(w1_label))
                avg_pdist = np.mean(distance.pdist(centroids))

                if avg_pdist < self.args.multi_nucleus_dist_thresh:
                    continue
                else:  # segment the cytoplasm mask using watershed
                    w2_mask_bbox_wsd = sitk.MorphologicalWatershedFromMarkers(
                        sitk.SignedMaurerDistanceMap(w2_mask_bbox_sitk != 0),
                        w1_mask_bbox_sitk,
                        markWatershedLine=False)
                    w2_mask_bbox_wsd = sitk.GetArrayFromImage(sitk.Mask(
                        w2_mask_bbox_wsd,
                        sitk.Cast(w2_mask_bbox_sitk, w2_mask_bbox_wsd.GetPixelID())))
                    # fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
                    # axes[0, 0].imshow(w1_img_bbox, cmap="gray")
                    # axes[1, 0].imshow(w2_img_bbox, cmap="gray")
                    # axes[0, 1].imshow(label2rgb(w1_mask_bbox), cmap="gray")
                    # axes[1, 1].imshow(label2rgb(w2_mask_bbox), cmap="gray")
                    # axes[0, 2].imshow(label2rgb(w2_mask_bbox_wsd), cmap="gray")
                    # axes[1, 2].axis("off")
                    # plt.show()
                    w2_mask_bbox_wsd[w2_mask_bbox_wsd != 0] += w2_count
                    w2_mask[slc2] = np.where(w2_mask_bbox, w2_mask_bbox_wsd, w2_mask[slc2])
                    w2_count += int(np.amax(w2_mask_bbox_wsd))
        # print(f"part 3 completion time in seconds: {time.time()-stime}")

        # stime = time.time()
        _, w2_mask = np.unique(w2_mask, return_inverse=True)
        w2_mask = w2_mask.reshape(shape_).astype(np.uint16)
        w1_mask = w1_mask.astype(np.uint16)

        w1_mask[w1_mask > 0] = w2_mask[w1_mask > 0]
        w1_unix, w2_unix = np.unique(w1_mask), np.unique(w2_mask)
        # diff1 = np.setdiff1d(w1_unix, w2_unix)
        diff2 = np.setdiff1d(w2_unix, w1_unix)
        if len(diff2) > 0:
            w2_mask[np.isin(w2_mask, diff2)] = 0
            w2_unix = np.setdiff1d(w2_unix, diff2)
            # print(diff_)
        # print(len(w1_unix), len(w2_unix))
        w2_mask = dilation(w2_mask, disk(4))
        assert np.array_equal(w1_unix, w2_unix)

        # print(f"part 4 completion time in seconds: {time.time()-stime}")

        # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        # axes[0, 0].imshow(img[0], cmap="gray")
        # axes[0, 1].imshow(img[1], cmap="gray")
        # axes[1, 0].imshow(label2rgb(w1_mask), cmap="gray")
        # axes[1, 1].imshow(label2rgb(w2_mask), cmap="gray")
        # plt.show()
        # print('\n')S
        return w1_mask, w2_mask

    def step2_get_nucleoli_and_mito_masks_v2(self, img, w1_mask, w2_mask):
        if len(np.unique(w1_mask)) <= 1 or len(np.unique(w2_mask)) <= 1:
            w3_mask = np.zeros_like(w1_mask)
            w5_mask = np.zeros_like(w2_mask)
            return w3_mask, w5_mask

        # w5: mito channel
        w5_img = np.where(erosion(w1_mask, disk(2)) | (w2_mask == 0), 0, img[4])
        w5_mask_global = sitk.GetArrayFromImage(self.otsu_filter.Execute(sitk.GetImageFromArray(w5_img)))
        w5_mask_global = w5_mask_global.astype(np.uint16)
        w5_mask_local = np.zeros_like(img[4], dtype=np.uint16)
        # create cytoplasmic mask excluding the nucleus
        cyto_mask = np.where(w1_mask > 0, 0, w2_mask)

        # w3: nucleoli channel
        w3_mask = np.zeros_like(cyto_mask, dtype=np.uint16)
        max_ = int(np.amax(cyto_mask))
        nucleus_slices = find_objects(w1_mask, max_label=max_)
        cyto_slices = find_objects(cyto_mask, max_label=max_)

        for ii, (slc1, slc2) in enumerate(zip(nucleus_slices, cyto_slices)):
            if slc1 is None or slc2 is None:
                continue
            obj_label = ii + 1
            w3_bbox = w1_mask[slc1] == obj_label
            w5_bbox = cyto_mask[slc2] == obj_label
            ####################################################################
            # local mito mask calculation
            w5_img_tmp = np.where(w5_bbox, w5_img[slc2], 0)
            lb = np.sum(w5_img_tmp < threshold_otsu(w5_img_tmp)) / np.size(w5_img_tmp)
            in_range = tuple(np.percentile(w5_img_tmp, (lb, 99.9)))
            w5_img_tmp = rescale_intensity(w5_img_tmp, in_range=in_range)
            w5_mask_tmp = sitk.GetArrayFromImage(self.otsu_filter.Execute(sitk.GetImageFromArray(w5_img_tmp)))
            # w5_mask_tmp = np.where(w5_mask_tmp, obj_label, 0)
            w5_mask_local[slc2] = np.where(w5_bbox, w5_mask_tmp, 0)
            # fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
            # axes[0, 0].imshow(np.where(w3_bbox, img[0][slc], 0), cmap="gray")
            # axes[0, 1].imshow(np.where(w5_bbox, img[1][slc], 0), cmap="gray")
            # axes[0, 2].imshow(np.where(w5_bbox, w5_img_tmp, 0), cmap="gray")
            # axes[1, 0].imshow(label2rgb(np.where(w3_bbox, w1_mask[slc], 0)), cmap="gray")
            # axes[1, 1].imshow(label2rgb(np.where(w5_bbox, w2_mask[slc], 0)), cmap="gray")
            # axes[1, 2].imshow(label2rgb(w5_mask_tmp), cmap="gray")
            # plt.show()
            ####################################################################################
            # local nucleoli calculation
            w3_img_tmp = np.where(w3_bbox, img[2][slc1], 0)
            if self.args.plate_protocol in ["greiner", "perkinelmer"]:
                lb = np.sum(w3_img_tmp < threshold_otsu(w3_img_tmp)) / np.size(w3_img_tmp)
            else:
                lb = .1
            w3_img_tmp = gaussian(w3_img_tmp, sigma=2)
            # prc = tuple(np.percentile(
            #     w3_img_tmp,
            #     (self.w3_local_rescale_intensity_lb, self.w3_local_rescale_intensity_ub)))
            in_range = tuple(np.percentile(w3_img_tmp, (lb, self.args.w3_local_rescale_intensity_ub)))
            w3_img_tmp = rescale_intensity(w3_img_tmp, in_range=in_range)  # (40, 88), (30, 95)
            w3_mask_tmp = sitk.GetArrayFromImage(self.yen_filter.Execute(
                sitk.GetImageFromArray(w3_img_tmp)))
            # w3_mask_tmp = binary_erosion(w3_mask_tmp, disk(1))
            w3_mask_tmp = label(w3_mask_tmp, connectivity=2)

            # remove small and large segmented nucleoli
            w3_bbox_area = np.sum(w3_bbox)
            min_nucleoli_size = self.args.min_nucleoli_size_multiplier * w3_bbox_area
            max_nucleoli_size = self.args.max_nucleoli_size_multiplier * w3_bbox_area
            areas = np.bincount(w3_mask_tmp.ravel())[w3_mask_tmp]
            cond = (areas < min_nucleoli_size) | (areas > max_nucleoli_size)
            w3_mask_tmp[cond] = 0
            areas[cond] = 0

            if self.args.plate_protocol in ["greiner", "perkinelmer"]:
                # remove nucleoli that intersect too much with the boundary
                w3_bbox_padded = np.pad(
                    w3_bbox,
                    ((self.nucleoli_bd_pad, self.nucleoli_bd_pad),
                     (self.nucleoli_bd_pad, self.nucleoli_bd_pad)),
                    constant_values=(0, 0))
                bd = find_boundaries(w3_bbox_padded, connectivity=2)
                bd = bd[
                     self.nucleoli_bd_pad:-self.nucleoli_bd_pad,
                     self.nucleoli_bd_pad:-self.nucleoli_bd_pad]
                if self.args.plate_protocol in ["greiner", "perkinelmer"]:
                    bd = binary_dilation(bd, disk(2))

                w3_tmp_bd_mask = w3_mask_tmp * bd
                bd_areas = np.bincount(w3_tmp_bd_mask.ravel())[w3_tmp_bd_mask]
                area_ratio = areas/bd_areas
                w3_tmp_bd_mask[area_ratio < self.args.nucleoli_bd_area_to_nucleoli_area_threshold] = 0

                # fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
                # axes[0, 0].imshow(np.where(w3_bbox, img[0][slc], 0), cmap="gray")
                # axes[0, 1].imshow(np.where(w3_bbox, img[2][slc], 0), cmap="gray")
                # axes[0, 2].axis("off")
                # axes[1, 0].imshow(label2rgb(np.where(w3_bbox, w1_mask[slc], 0)), cmap="gray")
                # axes[1, 1].imshow(label2rgb(w3_mask_tmp), cmap="gray")
                # axes[1, 2].imshow(bd, cmap="gray")
                # plt.show()

            w3_mask[slc1][w3_mask_tmp > 0] = obj_label

        # mito mask
        w5_mask = np.logical_or(w5_mask_global, w5_mask_local).astype(np.uint16)
        w5_mask *= cyto_mask
        w3_mask[(np.bincount(w3_mask.ravel()) < self.args.min_sizes["w3"])[w3_mask]] = 0
        w5_mask[(np.bincount(w5_mask.ravel()) < self.args.min_sizes["w5"])[w5_mask]] = 0

        unix2 = np.unique(w2_mask)
        unix5 = np.unique(w5_mask)
        # unix1 = np.unique(w1_mask)
        # unix3 = np.unique(w3_mask)

        diff_w2_w5 = np.setdiff1d(w2_mask, w5_mask)
        if len(diff_w2_w5) > 0:
            w2_mask[np.isin(w2_mask, diff_w2_w5)] = 0
            w1_mask[np.isin(w2_mask, diff_w2_w5)] = 0
            unix2 = np.setdiff1d(unix2, diff_w2_w5)

            # unix1 = np.setdiff1d(unix1, diff_w2_w5)
            # unix3 = np.setdiff1d(unix3, diff_w2_w5)

        w3_mask[w3_mask > 0] = w1_mask[w3_mask > 0]
        # print("unix1", len(unix1))
        # print("unix2", len(unix2))
        # print("unix3", len(unix3))
        # print("unix5", len(unix5))
        # print("w3 vs w1", len(np.setdiff1d(unix3, unix1)))
        # print("w1 vs w3", len(np.setdiff1d(unix1, unix3)))
        # print("w5 vs w2", len(np.setdiff1d(unix5, unix2)))
        # print("w2 vs w5", len(np.setdiff1d(unix2, unix5)))
        # print("w1 vs w2", len(np.setdiff1d(unix1, unix2)))
        # print("w2 vs w1", len(np.setdiff1d(unix2, unix1)))
        # print('\n')
        assert np.array_equal(unix2, unix5)

        return w3_mask, w5_mask

    def get_interect_area_over_w1_area_ratios(self, w1_mask, w2_mask):
        # translate both masks to make sure no two product of values are the same,
        # to create the intersect mask
        max1, max2 = np.amax(w1_mask), np.amax(w2_mask)
        max_ = np.maximum(max1, max2)
        max_p = self.PrevPrime_Reference(max_+200)

        w2_mask[w2_mask > 0] += max_p - 1
        w1_mask[w1_mask > 0] += 1

        # if max_p > 900:  # to avoid overflow
        #     w1_mask = np.uint32(w1_mask)
        #     w2_mask = np.uint32(w2_mask)

        intersect_mask = w2_mask * w1_mask
        # find ratio of area of intersecting regions
        intersect_area = np.bincount(intersect_mask.ravel())[intersect_mask]
        intersect_area[(w1_mask == 0) | (w2_mask == 0)] = 0
        w1_area = np.bincount(w1_mask.ravel())[w1_mask]
        intersect_ratio = (intersect_area / w1_area)
        return intersect_ratio, w1_mask, w2_mask

    @staticmethod
    def PrevPrime_Reference(N):
        """https://stackoverflow.com/questions/68907414/
        faster-way-to-find-the-biggest-prime-number-less-than-or-equal-to-the-given-inpu"""
        # tim = time.time()
        for i in range(1 << 14):
            p = N - i
            if (p & 1) == 0:
                continue
            if sympy.isprime(p):
                # tim = time.time() - tim
                # print(f'ReferenceTime {tim:.3f} sec', flush=True)
                return p

    @staticmethod
    def get_pairs_area_profile(w1_mask, w2_mask, w1_slices, w1_unix, w2_unix, ):
        n1, n2 = len(w1_unix), len(w2_unix)
        info_mat = np.zeros((n1, n2, 5), dtype=np.float32)
        for ii, i1 in enumerate(w1_unix):
            slc1 = w1_slices[i1 - 1]
            tmp1 = w1_mask[slc1] == i1
            for jj, i2 in enumerate(w2_unix):
                tmp2 = w2_mask[slc1] == i2
                intersect = tmp1 & tmp2
                area1 = np.sum(tmp1)
                area2 = np.sum(tmp2)
                area3 = np.sum(intersect)
                info_mat[ii, jj, 0] = i1
                info_mat[ii, jj, 1] = i2
                info_mat[ii, jj, 2] = area1
                info_mat[ii, jj, 3] = area2
                info_mat[ii, jj, 4] = area3 / area1
                # print(ii, jj, area1, area2, area3)
        return info_mat

