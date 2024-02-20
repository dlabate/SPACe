import time
from tqdm import tqdm
from pathlib import WindowsPath


import numpy as np
import pandas as pd

import cv2
import tifffile
import skimage.io as sio
from skimage.measure import label
from scipy.ndimage import find_objects
from scipy.stats import median_abs_deviation

from cellpaint.utils.shared_memory import MyBaseManager, TestProxy, create_shared_np_num_arr
from cellpaint.utils.skimage_regionprops_extension import RegionPropertiesExtension, TEXTURE_FEATURE_NAMES
from cellpaint.steps_single_plate.step0_args import Args, load_img, \
    sort_key_for_imgs, sort_key_for_masks, containsLetterAndNumber

import re
import warnings
from PIL import Image
from ctypes import c_int, c_float
import multiprocessing as mp
from functools import partial, lru_cache


class FeatureExtractor:
    """The heatmp and distmap classes slightly depend on this class because of args.
    This dependency needs to be resolved for clarity and ease of use.
    """
    analysis_step = 4
    N_ub = 900
    cache_max_size = 5
    cyto_pixel_height_lb = 30
    cyto_pixel_width_lb = 30

    def __init__(self, args):
        self.args = self.prepare_step1_add_feature_args(args)

        self.masks_load_path = self.args.main_path / self.args.experiment / f"Step{self.analysis_step - 1}_MasksP2"
        self.save_path = self.args.main_path / self.args.experiment / f"Step{self.analysis_step}_Features"
        self.save_path.mkdir(exist_ok=True, parents=True)

        self.w1_mask_filepaths = list(self.masks_load_path.rglob("*_W1.png"))
        self.w2_mask_filepaths = list(self.masks_load_path.rglob("*_W2.png"))
        self.w3_mask_filepaths = list(self.masks_load_path.rglob("*_W3.png"))
        self.w5_mask_filepaths = list(self.masks_load_path.rglob("*_W5.png"))

        # match lens
        assert len(self.w1_mask_filepaths) == \
               len(self.w2_mask_filepaths) == \
               len(self.w3_mask_filepaths) == \
               len(self.w5_mask_filepaths) == \
               len(self.args.img_channels_filepaths)

        # match (well-id, fov) pair
        for it0, it1, it2, it3, it5 in zip(
                self.args.img_channels_filepaths,
                self.w1_mask_filepaths,
                self.w2_mask_filepaths,
                self.w3_mask_filepaths,
                self.w5_mask_filepaths,
        ):
            print(it0[0])
            assert sort_key_for_imgs(it0[0], "to_get_well_id_and_fov", self.args.plate_protocol) == \
                   sort_key_for_masks(it1) == \
                   sort_key_for_masks(it2) == \
                   sort_key_for_masks(it3) == \
                   sort_key_for_masks(it5)
            # print(it0[0].stem, '\n', it1.stem, '\n', it2.stem, '\n')
        self.num_workers = min(16, mp.cpu_count(), self.args.N)
        self.prepare_step2_warn_user_about_missing_wellids()

    def prepare_step1_add_feature_args(self, args):
        """
        This step is mainly used in feature extraction/step3 of the cellpaint analysis.

        Create names, with certain rules, for all the extracted features columns, that going to be saved
        to the following csv files in "Features" folder:
        metadata_of_features.csv, misc_features.csv
        w0_features.csv, w1_features.csv, w2_features.csv, w3_features.csv, and w4_features.csv

        Also, we use median and mad statistics to summarize the haralick and moment features.
        This helps shrink the size/width of the feature-heatmap.
        """
        # args.organelles = organelles
        # args.num_channels = len(args.organelles)
        args.num_image_channels = 5
        ################################################################################################
        args.metadata_feature_cols = [
            "exp-id",
            "well-id",
            "fov",
            "treatment",
            "cell-line",
            "density",
            "dosage",
            "other",]
        args.bbox_feature_cols = [
            "Nucleus_BBox-y0",
            "Nucleus_BBox-x0",
            "Nucleus_BBox-y1",
            "Nucleus_BBox-x1",
            "Cell_BBox-y0",
            "Cell_BBox-x0",
            "Cell_BBox-y1",
            "Cell_BBox-x1",]
        args.misc_feature_cols = [
            "Misc_Count_# nucleoli",
            "Misc_Area-Ratio_nucleus/cell",
            "Misc_Area-Ratio_cyto/cell",
            "Misc_Area-Ratio_nucleoli/cell",
            "Misc_Area-Ratio_mito/cell",
            "Misc_Area-Ratio_nucleus/cyto",
            "Misc_Area-Ratio_mito/cyto",
            "Misc_Area-Ratio_nucleoli/cyto",
            "Misc_Area-Ratio_nucleoli/nucleus",]
        args.shape_feature_cols = [
            "Shape_Area_cell",
            "Shape_Area_nucleus",
            "Shape_Area_cyto",
            "Shape_Area_nucleoli",
            "Shape_Area_mito",

            "Shape_Nucleus_convex-area",
            "Shape_Nucleus_perimeter",
            "Shape_Nucleus_perimeter-crofton",
            "Shape_Nucleus_circularity",
            "Shape_Nucleus_efc-ratio",
            "Shape_Nucleus_eccentricity",
            "Shape_Nucleus_equiv-diam-area",
            "Shape_Nucleus_feret-diam-max",
            "Shape_Nucleus_solidity",
            "Shape_Nucleus_extent",

            "Shape_Cell_convex-area",
            "Shape_Cell_perimeter",
            "Shape_Cell_perimeter-crofton",
            "Shape_Cell_circularity",
            "Shape_Cell_efc-ratio",
            "Shape_Cell_eccentricity",
            "Shape_Cell_equiv-diam-area",
            "Shape_Cell_feret-diam-max",
            "Shape_Cell_solidity",
            "Shape_Cell_extent",]
        args.intensity_cols = \
            [str(it) for it in RegionPropertiesExtension.intensity_percentiles] + \
            ["median", "mad", "mean", "std"]
        args.feature_channels = [
            "W1-img-Nucleus-Mask",
            "W2-img-Cyto-Mask",
            "W3-img-Nucleoli-Mask",
            "W4-img-Cyto-Mask",
            "W5-img-Mito-Mask",
            "W1-img-Cell-Mask",
            "W2-img-Cell-Mask",
            "W3-img-Cell-Mask",
            "W4-img-Cell-Mask",
            "W5-img-Cell-Mask",]
        args.intensity_feature_cols = [
            f"Intensity_{it1}_{it2}" for it1 in args.feature_channels for it2 in args.intensity_cols]
        # args.texture_feature_cols = [
        #     f"Texture_Haralick-GLCM-{RegionPropertiesExtension.n_levels}_{it1}_{it2}_d{it3}-a{it4}"
        #     for it1 in feature_channels
        #     for it2 in RegionPropertiesExtension.texture_categories
        #     for it3 in RegionPropertiesExtension.distances
        #     for it4 in RegionPropertiesExtension.angles_str]
        args.texture_feature_cols = [f"Texture_{it0}_{it1}"
                                     for it0 in args.feature_channels
                                     for it1 in TEXTURE_FEATURE_NAMES]
        return args

    def prepare_step2_warn_user_about_missing_wellids(self, ):
        """
        if for certain wells the corresponding tiff image files are missing,
        or for certain image files the corresponding well metadata from platemap are missing,
        the user needs to be warned!!!!
        """

        wellids_from_img_files = [
            sort_key_for_imgs(it[0], "to_get_well_id", self.args.plate_protocol,)
            for it in self.args.img_channels_filepaths]
        wellids_from_platemap = self.args.wellids

        missig_wells_1 = np.setdiff1d(wellids_from_img_files, wellids_from_platemap)
        missig_wells_2 = np.setdiff1d(wellids_from_platemap, wellids_from_img_files)

        if len(missig_wells_1) > 0:
            raise ValueError(
                f"The following well-ids are in the image-folder  {self.args.experiment},\n"
                f" but are missing from the platemap file:\n"
                f"{missig_wells_1}")
        elif len(missig_wells_2) > 0:
            warnings.warn(
                f"The following well-ids are in the platemap file,\n"
                f" but are missing from the image-folder  {self.args.experiment}:\n"
                f"{missig_wells_2}\n\n"
                f"DO NOT WORRY ABOUT THIS IF args.mode==test or args.mode==debug!")
        else:
            print("no well-id is missing!!! Enjoy!!!")

    def step1_load_input_and_get_metadata(self, index):
        """
            w0_mask_path = .../w0_P000025-combchem-v3-U2OS-24h-L1-copy1_B02_1.png
            w1_mask_path = .../w1_P000025-combchem-v3-U2OS-24h-L1-copy1_B02_1.png
            w2_mask_path = .../w2_P000025-combchem-v3-U2OS-24h-L1-copy1_B02_1.png
            w5_mask_path = .../w4_P000025-combchem-v3-U2OS-24h-L1-copy1_B02_1.png
            img_channels_group:
            [
            .../P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s1_w1DCEB3369-8F24-4915-B0F6-B543ADD85297.tif,
            .../P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s1_w2C3AF00C2-E9F2-406A-953F-2ACCF649F58B.tif,
            .../P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s1_w3524F4D75-8D83-4DDC-828F-136E6A520E5D.tif,
            .../P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s1_w4568AFB8E-781D-4841-8BC8-8FD870A3147F.tif,
            .../P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s1_w5D9A405BD-1C0C-45E4-A335-CEE88A9AD244.tif,
            ]

            index is an int that refers to the index of the img_path_group in
            self.img_path_groups
            """

        # loading the image files for the 5 channels
        img_filepaths = self.args.img_channels_filepaths[index]
        img = load_img(img_filepaths, self.args)
        # loading the mask files for the 4 channels that have their own mask
        nucleus_mask = cv2.imread(str(self.w1_mask_filepaths[index]), cv2.IMREAD_UNCHANGED)
        cell_mask = cv2.imread(str(self.w2_mask_filepaths[index]), cv2.IMREAD_UNCHANGED)
        nucleoli_mask = cv2.imread(str(self.w3_mask_filepaths[index]), cv2.IMREAD_UNCHANGED)
        mito_mask = cv2.imread(str(self.w5_mask_filepaths[index]), cv2.IMREAD_UNCHANGED)
        cyto_mask = np.where(nucleus_mask > 0, 0, cell_mask)
        # remove small objects
        cyto_mask[(np.bincount(cyto_mask.ravel()) < self.args.min_sizes["w2"])[cyto_mask]] = 0
        nucleoli_mask[(np.bincount(nucleoli_mask.ravel()) < self.args.min_sizes["w3"])[nucleoli_mask]] = 0
        mito_mask[(np.bincount(mito_mask.ravel()) < self.args.min_sizes["w5"])[mito_mask]] = 0

        unix1 = np.unique(nucleus_mask)
        unix2 = np.unique(cyto_mask)
        unix0 = np.unique(cell_mask)
        unix3 = np.unique(nucleoli_mask)
        unix5 = np.unique(mito_mask)
        diff = np.setdiff1d(unix1, unix2)

        # handle the edge case where after removing the nucleus from cell, no cyto is left:
        if len(diff) > 0:

            nucleus_mask[np.isin(nucleus_mask, diff)] = 0
            nucleoli_mask[np.isin(nucleoli_mask, diff)] = 0
            mito_mask[np.isin(mito_mask, diff)] = 0
            cell_mask[np.isin(cell_mask, diff)] = 0

            unix1 = np.setdiff1d(unix1, diff)
            unix3 = np.setdiff1d(unix3, diff)
            unix5 = np.setdiff1d(unix5, diff)
            unix0 = np.setdiff1d(unix0, diff)
        # print(img_filepaths[0].stem)
        # print(f"diff {len(diff)}", len(unix0), len(unix1), len(unix2), len(unix3), len(unix5))
        # print(f"max "
        #       f"{np.amax(cell_mask)}  "
        #       f"{np.amax(nucleus_mask)}  "
        #       f"{np.amax(nucleoli_mask)}  "
        #       f"{np.amax(cyto_mask)}  "
        #       f"{np.amax(mito_mask)}  "
        #       )

        import matplotlib.pyplot as plt
        from skimage.color import label2rgb
        # fig, axes = plt.subplots(1, 5, sharex=True, sharey=True)
        # axes[0].imshow(label2rgb(nucleus_mask), cmap="gray")
        # axes[1].imshow(label2rgb(cyto_mask), cmap="gray")
        # axes[2].imshow(label2rgb(nucleoli_mask), cmap="gray")
        # axes[3].imshow(label2rgb(mito_mask), cmap="gray")
        # axes[4].imshow(label2rgb(cell_mask), cmap="gray")
        # plt.show()
        # print('\n')

        if (len(unix1) <= 1) or (len(unix2) <= 1) or (len(unix3) <= 1) or (len(unix5) <= 1) or (len(unix0) <= 1):
            return None

        mask = np.zeros((5, self.args.height, self.args.width), dtype=np.uint16)

        mask[0] = nucleus_mask
        mask[1] = cyto_mask
        mask[2] = nucleoli_mask
        mask[3] = mito_mask
        mask[4] = cell_mask

        # get metadata
        exp_id, well_id, fov = self.args.img_filename_keys[index]
        if containsLetterAndNumber(fov):
            fov = int(re.findall(r'\d+', fov)[0])
        elif fov.isdigit:
            fov = int(fov)
        else:
            raise ValueError(f"FOV value {fov} is unacceptable!")

        dosage = self.args.wellid2dosage[well_id]
        treatment = self.args.wellid2treatment[well_id]
        cell_line = self.args.wellid2cellline[well_id]
        density = self.args.wellid2density[well_id]
        other = self.args.wellid2other[well_id]

        return img, mask, (exp_id, well_id, fov, treatment, cell_line, density, dosage, other)

    def step2_get_features(self, index):
        out = self.step1_load_input_and_get_metadata(index)
        if out is None:
            return None
        img, mask, metadata = out

        max_ = np.amax(mask[-1])
        # N = len(np.unique(mask[0])) - 1
        nucleus_objects = find_objects(mask[0], max_label=max_)
        cyto_objects = find_objects(mask[1], max_label=max_)
        nucleoli_objects = find_objects(mask[2], max_label=max_)
        mito_objects = find_objects(mask[3], max_label=max_)
        cell_objects = find_objects(mask[4], max_label=max_)
        #################################################################
        cnt = 0
        range_ = tqdm(range(max_), total=max_) if self.args.mode == "test" else range(max_)
        bbox_features = np.zeros((self.N_ub, len(self.args.bbox_feature_cols)), dtype=np.float32)
        misc_features = np.zeros((self.N_ub, len(self.args.misc_feature_cols)), dtype=np.float32)
        shape_features = np.zeros((self.N_ub, len(self.args.shape_feature_cols)), dtype=np.float32)
        intensity_features = np.zeros((self.N_ub, len(self.args.intensity_feature_cols)), dtype=np.float32)
        texture_features = np.zeros((self.N_ub, len(self.args.texture_feature_cols)), dtype=np.float32)

        # Here we are assuming each object have the exact same label across all masks/channels.
        # Also since cell_object is the biggest mask, we also assume that no other labelled object
        # can be found inside a single slice/labelled_object from cell_object!
        for ii in range_:  # loops over each cell/roi/bbox
            nucleus_obj = nucleus_objects[ii]
            cyto_obj = cyto_objects[ii]
            nucleoli_obj = nucleoli_objects[ii]
            mito_obj = mito_objects[ii]
            cell_obj = cell_objects[ii]
            obj_label = ii + 1

            # if there is no cell or no nucleoli skip it the cell.
            if (nucleus_obj is None) or (cyto_obj is None) or (nucleoli_obj is None) or \
                    (mito_obj is None) or (cell_obj is None):
                # print(ii, nucleus_obj, nucleoli_obj)
                continue
            nucleus_props = RegionPropertiesExtension(cell_obj, obj_label, mask[0], img[0], "nucleus")
            cyto_props = RegionPropertiesExtension(cyto_obj, obj_label, mask[1], img[1], "cyto")
            y0, x0, y1, x1 = cyto_props.bbox
            if (y1-y0) < self.cyto_pixel_height_lb or (x1-x0) < self.cyto_pixel_height_lb:
                continue

            nucleoli_props = RegionPropertiesExtension(nucleoli_obj, obj_label, mask[2], img[2], "nucleoli")
            actin_props = RegionPropertiesExtension(cyto_obj, obj_label, mask[1], img[3], "actin")
            mito_props = RegionPropertiesExtension(mito_obj, obj_label, mask[3], img[4], "mito")

            cell_w1_props = RegionPropertiesExtension(cell_obj, obj_label, mask[4], img[0], "w1-cell")
            cell_w2_props = RegionPropertiesExtension(cell_obj, obj_label, mask[4], img[1], "w2-cell")
            cell_w3_props = RegionPropertiesExtension(cell_obj, obj_label, mask[4], img[2], "w3-cell")
            cell_w4_props = RegionPropertiesExtension(cell_obj, obj_label, mask[4], img[3], "w4-cell")
            cell_w5_props = RegionPropertiesExtension(cell_obj, obj_label, mask[4], img[4], "w5-cell")

            # get shape features AND bounding boxes for nucleus mask and cell mask
            bbox_features[cnt, :] = nucleus_props.bbox+cell_w1_props.bbox
            # get shape features AND bounding boxes for nucleus mask and cell mask
            shape_features[cnt, :] = \
                (cell_w1_props.area, nucleus_props.area, cyto_props.area, nucleoli_props.area, mito_props.area,

                 nucleus_props.area_convex, nucleus_props.perimeter, nucleus_props.perimeter_crofton,
                 nucleus_props.circularity, nucleus_props.efc_ratio,
                 nucleus_props.eccentricity, nucleus_props.equivalent_diameter_area,
                 nucleus_props.feret_diameter_max, nucleus_props.solidity, nucleus_props.extent,

                 cell_w1_props.area_convex, cell_w1_props.perimeter, cell_w1_props.perimeter_crofton,
                 cell_w1_props.circularity, cell_w1_props.efc_ratio,
                 cell_w1_props.eccentricity, cell_w1_props.equivalent_diameter_area,
                 cell_w1_props.feret_diameter_max, cell_w1_props.solidity, cell_w1_props.extent)

            # intensity and texture feature profile using each prop!
            intensity_mt1_w1 = nucleus_props.intensity_statistics
            intensity_mt1_w2 = cyto_props.intensity_statistics
            intensity_mt1_w3 = nucleoli_props.intensity_statistics
            intensity_mt1_w4 = actin_props.intensity_statistics
            intensity_mt1_w5 = mito_props.intensity_statistics

            intensity_mt2_w1 = cell_w1_props.intensity_statistics
            intensity_mt2_w2 = cell_w2_props.intensity_statistics
            intensity_mt2_w3 = cell_w3_props.intensity_statistics
            intensity_mt2_w4 = cell_w4_props.intensity_statistics
            intensity_mt2_w5 = cell_w5_props.intensity_statistics

            intensity_features[cnt, :] = \
                intensity_mt1_w1 + intensity_mt1_w2 + intensity_mt1_w3 + intensity_mt1_w4 + intensity_mt1_w5 + \
                intensity_mt2_w1 + intensity_mt2_w2 + intensity_mt2_w3 + intensity_mt2_w4 + intensity_mt2_w5

            glcm_mt1_w1 = nucleus_props.glcm_features
            glcm_mt1_w2 = cyto_props.glcm_features
            glcm_mt1_w3 = nucleoli_props.glcm_features
            glcm_mt1_w4 = actin_props.glcm_features
            glcm_mt1_w5 = mito_props.glcm_features

            glcm_mt2_w1 = cell_w1_props.glcm_features
            glcm_mt2_w2 = cell_w2_props.glcm_features
            glcm_mt2_w3 = cell_w3_props.glcm_features
            glcm_mt2_w4 = cell_w4_props.glcm_features
            glcm_mt2_w5 = cell_w5_props.glcm_features
            texture_features[cnt, :] = glcm_mt1_w1 + glcm_mt1_w2 + glcm_mt1_w3 + glcm_mt1_w4 + glcm_mt1_w5 + \
                  glcm_mt2_w1 + glcm_mt2_w2 + glcm_mt2_w3 + glcm_mt2_w4 + glcm_mt2_w5

            misc_features[cnt, 0] = np.amax(label(nucleoli_props.image, connectivity=2, background=0))
            # Now we need to extract the bounding box of the cell_object from each image channel
            cnt += 1
        if cnt == 0:
            return None
        bbox_features = bbox_features[0:cnt]
        shape_features = shape_features[0:cnt]
        intensity_features = intensity_features[0:cnt]
        texture_features = texture_features[0:cnt]
        misc_features = misc_features[0:cnt]

        misc_features[:, 1] = shape_features[:, 1] / shape_features[:, 0]  # nucleus area to cell area
        misc_features[:, 2] = shape_features[:, 2] / shape_features[:, 0]  # cyto area to cell area
        misc_features[:, 3] = shape_features[:, 3] / shape_features[:, 0]  # nucleoli area to cell area
        misc_features[:, 4] = shape_features[:, 4] / shape_features[:, 0]  # mito area to cell area

        misc_features[:, 5] = shape_features[:, 1] / shape_features[:, 2]  # nucleus area to cyto area
        misc_features[:, 6] = shape_features[:, 4] / shape_features[:, 2]  # mito area to cyto area
        misc_features[:, 7] = shape_features[:, 3] / shape_features[:, 2]  # nucleoli area to cyto area
        misc_features[:, 8] = shape_features[:, 3] / shape_features[:, 1]  # nucleoli area to nucleus area

        metadata_features = np.repeat(np.array(metadata, dtype=object)[np.newaxis], repeats=cnt, axis=0)
        return metadata_features, bbox_features, misc_features, shape_features, intensity_features, texture_features


def step4_single_run_loop(args, myclass=FeatureExtractor):

    # 2) get features
    n_rows = 0
    inst = myclass(args)
    T = inst.args.N * inst.N_ub
    metadata_features = np.zeros((T, len(inst.args.metadata_feature_cols)), dtype=object)
    bbox_features = create_shared_np_num_arr((T, len(inst.args.bbox_feature_cols)), c_dtype="c_float")
    misc_features = create_shared_np_num_arr((T, len(inst.args.misc_feature_cols)), c_dtype="c_float")
    shape_features = create_shared_np_num_arr((T, len(inst.args.shape_feature_cols)), c_dtype="c_float")
    intensity_features = create_shared_np_num_arr((T, len(inst.args.intensity_feature_cols)), c_dtype="c_float")
    texture_features = create_shared_np_num_arr((T, len(inst.args.texture_feature_cols)), c_dtype="c_float")

    # metadata_features = np.zeros((T, len(inst.args.metadata_feature_cols)), dtype=object)
    # bbox_features = np.zeros((T, len(inst.args.bbox_feature_cols)), dtype=np.float32)
    # misc_features = np.zeros((T, len(inst.args.misc_feature_cols)), dtype=np.float32)
    # shape_features = np.zeros((T, len(inst.args.shape_feature_cols)), dtype=np.float32)
    # intensity_features = np.zeros((T, len(inst.args.intensity_feature_cols)), dtype=np.float32)
    # texture_features = np.zeros((T, len(inst.args.texture_feature_cols)), dtype=np.float32)

    for idx in tqdm(range(inst.args.N), total=inst.args.N):
            out = inst.step2_get_features(idx)
            if out is not None:
                ncells = out[0].shape[0]
                start = n_rows
                end = start + ncells
                # metadata, bbox_, misc_, shape_, intensity_, texture_ = out
                metadata_features[start:end, :] = out[0]
                bbox_features[start:end, :] = out[1]
                misc_features[start:end, :] = out[2]
                shape_features[start:end, :] = out[3]
                intensity_features[start:end, :] = out[4]
                texture_features[start:end, :] = out[5]
                n_rows = end

    metadata_features = metadata_features[0:n_rows]
    bbox_features = bbox_features[0:n_rows]
    misc_features = misc_features[0:n_rows]
    shape_features = shape_features[0:n_rows]
    intensity_features = intensity_features[0:n_rows]
    texture_features = texture_features[0:n_rows]

    # save features and metadata
    print("converting features numpy arrays to a pandas dataframes and saving them as csv files ...")
    metadata_features = pd.DataFrame(metadata_features, columns=inst.args.metadata_feature_cols)
    bbox_features = pd.DataFrame(bbox_features, columns=inst.args.bbox_feature_cols)
    misc_features = pd.DataFrame(misc_features, columns=inst.args.misc_feature_cols)
    shape_features = pd.DataFrame(shape_features, columns=inst.args.shape_feature_cols)
    intensity_features = pd.DataFrame(intensity_features, columns=inst.args.intensity_feature_cols)
    texture_features = pd.DataFrame(texture_features, columns=inst.args.texture_feature_cols)

    metadata_features.to_csv(inst.save_path / f"metadata_features.csv", index=False, float_format="%.2f")
    bbox_features.to_csv(inst.save_path / f"bbox_features.csv", index=False, float_format="%.2f")
    misc_features.to_csv(inst.save_path / f"misc_features.csv", index=False, float_format="%.2f")
    shape_features.to_csv(inst.save_path / f"shape_features.csv", index=False, float_format="%.2f")
    intensity_features.to_csv(inst.save_path / f"intensity_features.csv", index=False, float_format="%.2f")
    texture_features.to_csv(inst.save_path / f"texture_features.csv", index=False, float_format="%.2f")


def step4_multi_run_loop(args, myclass=FeatureExtractor):
    """ We have to Register the FeatureExtractor class object as well as its attributes as shared using:
    https://stackoverflow.com/questions/26499548/accessing-an-attribute-of-a-multiprocessing-proxy-of-a-class"""
    start_time = time.time()
    print("Cellpaint Step 4: feature extraction ...")
    MyManager = MyBaseManager()
    # register the custom class on the custom manager
    MyManager.register(myclass.__name__, myclass, TestProxy)
    # create a new manager instance
    with MyManager as manager:
        inst = getattr(manager, myclass.__name__)(args)
        n_rows = 0
        T = inst.args.N * inst.N_ub
        metadata_features = np.zeros((T, len(inst.args.metadata_feature_cols)), dtype=object)
        bbox_features = create_shared_np_num_arr((T, len(inst.args.bbox_feature_cols)), c_dtype="c_float")
        misc_features = create_shared_np_num_arr((T, len(inst.args.misc_feature_cols)), c_dtype="c_float")
        shape_features = create_shared_np_num_arr((T, len(inst.args.shape_feature_cols)), c_dtype="c_float")
        intensity_features = create_shared_np_num_arr((T, len(inst.args.intensity_feature_cols)), c_dtype="c_float")
        texture_features = create_shared_np_num_arr((T, len(inst.args.texture_feature_cols)), c_dtype="c_float")

        # metadata_features = np.zeros((T, len(inst.args.metadata_feature_cols)), dtype=object)
        # bbox_features = np.zeros((T, len(inst.args.bbox_feature_cols)), dtype=np.float32)
        # misc_features = np.zeros((T, len(inst.args.misc_feature_cols)), dtype=np.float32)
        # shape_features = np.zeros((T, len(inst.args.shape_feature_cols)), dtype=np.float32)
        # intensity_features = np.zeros((T, len(inst.args.intensity_feature_cols)), dtype=np.float32)
        # texture_features = np.zeros((T, len(inst.args.texture_feature_cols)), dtype=np.float32)

        with mp.Pool(processes=inst.num_workers) as pool:
            """Using pool.imap whichs preserve order, so that no two processes write to the same row!!!"""
            for out in tqdm(pool.imap(inst.step2_get_features, range(inst.args.N)), total=inst.args.N):
                if out is not None:
                    ncells = out[0].shape[0]
                    start = n_rows
                    end = start + ncells
                    # metadata, bbox_, misc_, shape_, intensity_, texture_ = out
                    metadata_features[start:end, :] = out[0]
                    bbox_features[start:end, :] = out[1]
                    misc_features[start:end, :] = out[2]
                    shape_features[start:end, :] = out[3]
                    intensity_features[start:end, :] = out[4]
                    texture_features[start:end, :] = out[5]
                    n_rows = end

        metadata_features = metadata_features[0:n_rows]
        bbox_features = bbox_features[0:n_rows]
        misc_features = misc_features[0:n_rows]
        shape_features = shape_features[0:n_rows]
        intensity_features = intensity_features[0:n_rows]
        texture_features = texture_features[0:n_rows]

        # save features and metadata
        print("Converting features numpy arrays to a pandas dataframes and saving them as csv files ...")
        metadata_features = pd.DataFrame(metadata_features, columns=inst.args.metadata_feature_cols)
        bbox_features = pd.DataFrame(bbox_features, columns=inst.args.bbox_feature_cols)
        misc_features = pd.DataFrame(misc_features, columns=inst.args.misc_feature_cols)
        shape_features = pd.DataFrame(shape_features, columns=inst.args.shape_feature_cols)
        intensity_features = pd.DataFrame(intensity_features, columns=inst.args.intensity_feature_cols)
        texture_features = pd.DataFrame(texture_features, columns=inst.args.texture_feature_cols)

        metadata_features.to_csv(inst.save_path / f"metadata_features.csv", index=False, float_format="%.2f")
        bbox_features.to_csv(inst.save_path / f"bbox_features.csv", index=False, float_format="%.2f")
        misc_features.to_csv(inst.save_path / f"misc_features.csv", index=False, float_format="%.2f")
        shape_features.to_csv(inst.save_path / f"shape_features.csv", index=False, float_format="%.2f")
        intensity_features.to_csv(inst.save_path / f"intensity_features.csv", index=False, float_format="%.2f")
        texture_features.to_csv(inst.save_path / f"texture_features.csv", index=False, float_format="%.2f")

        print(f"Finished Cellpaint step 4 in: {(time.time() - start_time) / 3600} hours\n")
        return inst.args


def step4_main_run_loop(args, myclass=FeatureExtractor):
    # args.mode = "test"
    if args.mode == "test":
        step4_single_run_loop(args, myclass)
    else:
        step4_multi_run_loop(args, myclass)
