import tifffile
import time
from tqdm import tqdm
from pathlib import WindowsPath
import matplotlib.pyplot as plt

import torch
from torch import linalg as LA
from kymatio.torch import Scattering2D

import numpy as np
from scipy.ndimage import find_objects
from scipy.stats import median_abs_deviation

import skimage.io as sio
from skimage.segmentation import find_boundaries
from skimage.measure import label, find_contours
from skimage.measure._regionprops import RegionProperties, _cached

from functools import lru_cache
from pyefd import elliptic_fourier_descriptors

from cellpaint.steps_single_plate.step0_args import Args


class FeatureExtractorSingular:
    cache_max_size = 8
    testing = True
    #########################################################
    # TODO: add necessary args for feature extraction here:
    bd_val = 10
    bd_padding = ((bd_val, bd_val), (bd_val, bd_val))
    side = 500
    n_channels_calc = 10
    ################################################
    scattering = Scattering2D(J=2, shape=(side, side))

    # Pre-defining all constants, and moving to shared_memory, to avoid all unnecessary recalculation
    # ######## extremely intensive computation!!! #################
    #################################################################################
    # number of levels used to discretize an image_crop/cell/organelle locally
    n_levels = 16
    # discretization at the 0-level is done in a separate line of code
    range_levels = range(1, n_levels)
    boundaries = torch.linspace(0, 1, n_levels)
    ######################################################
    # Constants for the GLCM-Matrix
    angles = torch.as_tensor([0, 90, 180, 270])
    distances = torch.arange(1, 20)
    n_angles = angles.size(0)
    n_distances = distances.size(0)
    range_angles = range(n_angles)
    range_distances = range(n_angles)
    angles_mesh, distances_mesh = torch.meshgrid(angles, distances, indexing="ij")
    offset_row = torch.round(torch.sin(angles_mesh) * distances_mesh).long()
    offset_col = torch.round(torch.cos(angles_mesh) * distances_mesh).long()
    start_row = torch.where(offset_row > 0, 0, -offset_row)
    start_col = torch.where(offset_col > 0, 0, -offset_col)
    #######################################################
    # Constants for Haralick-Texture Features extracted from the GLCM-Matrix
    # self.n_levels -1 because we are removing the zero_level
    reduction_dims = (0, 1)
    std_tolerance = 1e-9
    # These must match the shape GLCM transpose:
    # (n_levels, n_levels, n_angles, n_dists, batch_size, n_channels)
    shape_1 = (n_levels, n_levels, 1, 1, 1, 1)
    shape_2 = (n_levels, 1, 1, 1, 1, 1)
    shape_3 = (1, n_levels, 1, 1, 1, 1)

    i_rest, j_rest = torch.meshgrid(torch.arange(0, n_levels), torch.arange(0, n_levels), indexing="ij")
    weights_contrast = ((i_rest - j_rest) ** 2).view(shape_1).type(torch.FloatTensor)
    weights_dissimilarity = torch.abs(i_rest - j_rest).view(shape_1).type(torch.FloatTensor)
    weights_homogeneity = 1 / (1 + weights_contrast)

    i_corr = torch.arange(n_levels).view(shape_2).type(torch.FloatTensor)
    j_corr = torch.arange(n_levels).view(shape_3).type(torch.FloatTensor)

    # move all the tensors into shared_memory
    angles.share_memory_()
    distances.share_memory_()
    angles_mesh.share_memory_()
    distances_mesh.share_memory_()
    offset_row.share_memory_()
    offset_col.share_memory_()
    start_row.share_memory_()
    start_col.share_memory_()

    i_rest.share_memory_()
    j_rest.share_memory_()
    weights_contrast.share_memory_()
    weights_dissimilarity.share_memory_()
    weights_homogeneity.share_memory_()
    i_corr.share_memory_()
    j_corr.share_memory_()
    boundaries.share_memory_()
    #######################################################################################
    # ######## extremely intensive computation!!! #################
    #########################################################################################
    percentiles = np.array([2, 5, 10, 15, 25, 35, 40, 60, 75, 80, 90, 95, 98, 99.5])

    def __init__(self, args):
        self.args = args
        self.rows = self.side
        self.cols = self.side
        self.end_row = torch.where(self.offset_row > 0, self.rows - self.offset_row, self.rows)
        self.end_col = torch.where(self.offset_col > 0, self.cols - self.offset_col, self.cols)

    def get_img_crop(self, img_crop, cell_mask_crop):
        """
        img_crop: object/cell_organelle within a bounding box.
        and is 3d-np.array of shape (n_channels, object_width, object_height),
        """
        out = img_crop * cell_mask_crop[np.newaxis]
        return out

    def discretize_intensity_locally(self, img_crop):
        """
        img_crop: object/cell_organelle within a bounding box.
        and is 3d-np.array of shape (n_channels, object_width, object_height),
        within a single field of view inside a particular well.

        This function assumes each channel can have a very different texture.
        As the light emission is all over the place and the confocal camera takes
        image from channel separately, this assumption physically makes sense.
        """
        # assert img.ndim == 3 # This must always the case, but commenting out for speed!!!
        img_crop = torch.as_tensor(img_crop)
        shape_ = img_crop.size()
        discrete_img_crop = torch.zeros(shape_, dtype=torch.uint8)
        for c_in in shape_:  # each channel is handled separately
            tmp = img_crop[c_in]
            min_ = torch.amin(tmp).item()
            max_ = torch.max(tmp).item()
            levels = torch.linspace(min_, max_, self.n_levels)
            for lvl in self.range_levels:
                # i0 = levels[ii - 1]
                # i2 = levels[ii]
                # chop clipping the image
                # discrete_img_crop[(i0 <= img) & (img < i1)] = ii
                # discrete_img_crop += ((i0 <= img) * (img < i1) * (ii - 1)).astype(np.uint8)
                #########################################################################
                # # round clipping the image
                # i1 = i0 + (i2 - i0) // 2
                # discrete_img_crop[(i0 <= img) & (img < i1)] = ii - 1
                # discrete_img_crop[(i1 <= img) & (img < i2)] = ii
                # ###################################
                # hard clipping (chopping) the image
                discrete_img_crop[c_in, (levels[lvl - 1] <= tmp) & (tmp < levels[lvl])] = lvl - 1
            discrete_img_crop[c_in, tmp == max_] = self.n_levels - 1
        return discrete_img_crop

    def get_padded_img_crop(self, img_crop):
        """
        img_crop: object/cell_organelle within a bounding box.
        and is 3d-np.array of shape (n_channels, object_width, object_height),
        """

        hh, ww = img_crop.shape[1], img_crop.shape[2]
        padh, padw = (self.side - hh) // 2, (self.side - ww) // 2
        padding = ((0, 0), (padh, self.side-hh-padh), (padw, self.side-ww-padw))
        return np.pad(img_crop, padding, 'constant', constant_values=(0, 0))

    def normalize_img_crop(self, img_crop):
        # img_crop = np.float32(img_crop)
        min_ = np.amin(img_crop, axis=(1, 2))
        max_ = np.amin(img_crop, axis=(1, 2))
        for ii in range(img_crop.shape[0]):
            img_crop[ii] = (img_crop[ii]-min_[ii])/(max_[ii]-min_[ii]) if max_[ii] > min_[ii] else 0
        return img_crop

    def efc_ratio(self, image):
        bd = find_boundaries(np.pad(image, self.bd_padding, 'constant', constant_values=(0, 0)))
        bd_contours = find_contours(bd, .1)[0]
        efc = elliptic_fourier_descriptors(bd_contours,
                                           normalize=True,
                                           order=15)
        # N = efc.shape[0]
        efcs = np.sqrt(efc[:, 0] ** 2 + efc[:, 1] ** 2) + np.sqrt(efc[:, 2] ** 2 + efc[:, 3] ** 2)
        ratios = efcs[0] / np.sum(efcs[1:])
        return ratios

    def circularity(self, perimeter, area):
        if perimeter > 1e-6:
            return (4 * np.pi * area) / perimeter ** 2
        else:
            return np.nan

    @lru_cache(maxsize=cache_max_size)
    def step0_preprocessing(self, img_paths, mask_paths):
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
        nucleus_img = tifffile.imread(img_paths[0])
        cyto_img = tifffile.imread(img_paths[1])
        nucleoli_img = tifffile.imread(img_paths[2])
        actin_img = tifffile.imread(img_paths[3])
        mito_img = tifffile.imread(img_paths[4])

        # loading the mask files for the 4 channels that have their own mask
        nucleus_mask = sio.imread(mask_paths[0])
        cell_mask = sio.imread(mask_paths[1])
        nucleoli_mask = sio.imread(mask_paths[2])
        mito_mask = sio.imread(mask_paths[3])

        cyto_mask = cell_mask.copy()
        cyto_mask[nucleus_mask > 0] = 0

        # fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
        # axes[0].imshow(nucleus_mask, cmap="gray")
        # axes[1].imshow(cyto_mask, cmap="gray")
        # axes[2].imshow(cell_mask, cmap="gray")
        # plt.show()
        # print(len(np.unique(cell_mask)))
        # print(len(np.unique(cyto_mask)))
        # print(len(np.unique(nucleus_mask)))

        assert np.array_equal(np.unique(cell_mask), np.unique(cyto_mask))

        img = np.zeros((self.n_channels_calc, self.args.height, self.args.width), dtype=np.uint16)
        mask = np.zeros((self.n_channels_calc, self.args.height, self.args.width), dtype=np.uint16)

        ##############################################################
        # TODO: Need to add the illumination correction step right here!!!
        ###########################################################
        img[0], mask[0] = nucleus_img, nucleus_mask
        img[1], mask[1] = cyto_img, cyto_mask
        img[2], mask[2] = nucleoli_img, nucleoli_mask
        img[3], mask[3] = actin_img, cyto_mask
        img[4], mask[4] = mito_img, mito_mask
        img[5], mask[5] = nucleus_img, cell_mask
        img[6], mask[6] = cyto_img, cell_mask
        img[7], mask[7] = nucleoli_img, nucleus_mask
        img[8], mask[8] = actin_img, cell_mask
        img[9], mask[9] = mito_img, cell_mask
        return img, mask

    def step1_get_features(self, img, mask, ):
        max_ = np.amax(mask[0])
        N = len(np.unique(mask[0])) - 1
        nucleus_objects = find_objects(mask[0], max_label=max_)
        cell_objects = find_objects(mask[5], max_label=max_)
        #################################################################
        cnt = 0
        # has_nucleoli = np.zeros(shape=(N, 1), dtype=bool)
        # misc_features = np.zeros((N, self.args.num_misc_cols), dtype=np.float32)
        # features = np.zeros(shape=(self.args.num_organelles, N, self.args.num_feat_cols), dtype=np.float32)
        range_ = tqdm(range(max_), total=max_) if self.testing else range(max_)
        bboxes = np.zeros((N, 8), dtype=np.float32)
        shape_features = np.zeros((N, 25), dtype=np.float32)
        misc_features = np.zeros((N, 4), dtype=np.float32)
        img_crops = np.zeros((N, self.n_channels_calc, self.side, self.side), dtype=np.float32)
        discrete_img_crops = torch.zeros((N, self.n_channels_calc, self.side, self.side), dtype=torch.float32)

        # Here we are assuming each object have the exact same label across all masks.
        # Also since cell_object is the biggest mask, we also assume that no other labelled object
        # can be found inside a single slice/labelled_object from cell_object!
        for ii in range_:  # loops over each cell/roi/bbox
            cell_obj = cell_objects[ii]
            nucleus_obj = nucleus_objects[ii]
            obj_label = ii + 1
            # if there is no cell or no nucleoli skip it the cell.
            if nucleus_obj is None or np.sum(img[self.args.nucleoli_idx][nucleus_obj]) == 0:
                # if nucleus_obj is None or np.sum(nucleoli_props.image) == 0:
                continue

            # TODO: Figure out the correct index of all masks
            # get shape features AND bounding boxes for nucleus mask and cyto mask
            cell_props = RegionProperties(cell_obj, obj_label, mask[5], img[1], cache_active=True)
            nucleus_props = RegionProperties(nucleus_obj, obj_label, mask[0], img[0], cache_active=True)
            cyto_props = RegionProperties(cell_obj, obj_label, mask[1], img[1], cache_active=True)
            nucleoli_props = RegionProperties(nucleus_obj, obj_label, mask[2], img[2], cache_active=True)
            mito_props = RegionProperties(cell_obj, obj_label, mask[4], img[4], cache_active=True)

            bboxes[cnt, :] = nucleus_props.bbox+cell_props.bbox
            nucleus_circ = self.circularity(nucleus_props.perimeter, nucleus_props.area)
            nucleus_efcr = self.efc_ratio(nucleus_props.image)
            cell_circ = self.circularity(cell_props.perimeter, cell_props.area)
            cell_efcr = self.efc_ratio(cell_props.image)
            shape_features[cnt, :] = \
                (cell_props.area, nucleus_props.area, cyto_props.area, nucleoli_props.area, mito_props.area,
                 nucleus_props.area_convex, nucleus_props.perimeter, nucleus_props.perimeter_crofton,
                 nucleus_circ, nucleus_efcr,
                 nucleus_props.eccentricity, nucleus_props.equivalent_diameter_area,
                 nucleus_props.feret_diameter_max, nucleus_props.solidity, nucleus_props.extent,

                 cell_props.area_convex, cell_props.perimeter, cell_props.perimeter_crofton,
                 cell_circ, cell_efcr,
                 cell_props.eccentricity, cell_props.equivalent_diameter_area,
                 cell_props.feret_diameter_max, cell_props.solidity, cell_props.extent)

            # nucleoli_mask = mask[self.args.nucleoli_cidx][nucleus_obj]
            # nucleoli_mask = nucleoli_mask*nucleus_props.image
            misc_features[cnt, 0] = np.amax(label(nucleus_props.image, connectivity=2, background=0))
            # Now we need to extract the bounding box of the cell_object from each image channel
            img_crops[cnt] = np.float32(self.get_padded_img_crop(
                self.get_img_crop(img[:, cell_obj[0], cell_obj[1]], cell_props.image)))
            discrete_img_crops[cnt] = torch.as_tensor(self.normalize_img_crop(img_crops[cnt]))
            cnt += 1
        # intensity features, haralick features, and scattering features
        s_time = time.time()
        # intensity_features = self.get_intensity_features(img_crops)
        # print(f"time it takes to calc intensity features: {time.time()-s_time}")
        # Here, discrete_img_crops.dtype is changed from float32 to int64 here!!!
        s_time = time.time()
        discrete_img_crops = torch.bucketize(discrete_img_crops, boundaries=self.boundaries, out_int32=True).type(
            torch.ByteTensor)
        glcm = self.glcm_loop_torch(discrete_img_crops)
        print(f"time it takes to calc haralick features: {time.time()-s_time}")
        energy_features, contrast_features, dissimilarity_features, \
            homogeneity_features, correlation_features = self.get_haralick_texture_features(glcm)

        # the scattering coefficients
        # by a just single call to its appropriate function
        e_time = time.time()
        scattering_features = self.scattering(img_crops)
        print(f"time it takes to calc scattering features: {time.time()-s_time}")
        return

    def get_intensity_features(self, img_crops):
        img_crops[img_crops == 0] = np.nan

        median_features = np.nanmedian(img_crops, axis=(2, 3))
        mad_features = np.nanmedian(np.abs(img_crops - median_features[:, :, np.newaxis, np.newaxis]), axis=(2, 3))
        mean_features = np.nanmean(img_crops, axis=(2, 3))
        std_features = np.nanstd(img_crops, axis=(2, 3))
        percentile_features = np.nanpercentile(img_crops, self.percentiles, axis=(2, 3))
        print(mean_features.shape, std_features.shape, median_features.shape, mad_features.shape, percentile_features.shape)

        # intensity_features = np.concatenate(
        #     (mean_features, std_features, median_features, mad_features, percentile_features), axis=1)
        # return intensity_features

    # TODO: Check whether the calculation is 100% correct
    def glcm_loop_torch(self, discretized_image):
        """
        Adapted from the GLCM matrix implementation of graycomatrix in scikit-image/skimage/feature/texture.py

        Perform co-occurrence matrix accumulation.
        Parameters
        ----------
        discretized_image : torch.tensor of shape (B, C, W, H),
            Integer typed input discretized_image. Only positive valued discretized_images are supported.
            If type is other than uint8, the argument `levels` needs to be set.

        levels : int
            The input discretized_image should contain integers in [0, `levels`-1],
            where levels indicate the number of gray-levels counted
            (typically 256 for an 8-bit discretized_image).
        returns
        out : torch.tensor
            On input a 6D tensor of shape (B, C, levels-1, levels-1, aa, dd) and integer values
            that returns the results of the GLCM computation.

        The modified/reworked function assumes the discretized_image is priorly discretized correctly!!!!
        """
        # The following check is not necessary anymore as we already have converted
        # the image to discretized_image properly:
        # if torch.sum((discretized_image >= 0) & (discretized_image < self.n_levels)).item() < 1:
        #     raise ValueError("discretized_image values cannot exceed levels and also must be positive!!")
        batch_size = discretized_image.size(0)
        glcm = torch.zeros(
            (batch_size, self.n_channels_calc, self.n_levels, self.n_levels, self.n_angles, self.n_distances),
            dtype=torch.int8)

        for aa in self.range_angles:
            for dd in self.range_distances:
                rs0 = self.start_row[aa, dd]
                re0 = self.end_row[aa, dd]
                cs0 = self.start_col[aa, dd]
                ce0 = self.end_col[aa, dd]

                rs1 = rs0 + self.offset_row[aa, dd]
                re1 = re0 + self.offset_row[aa, dd]
                cs1 = cs0 + self.offset_col[aa, dd]
                ce1 = ce0 + self.offset_col[aa, dd]

                glcm[:,
                     :,
                     discretized_image[:, :, rs0:re0, cs0:ce0],
                     discretized_image[:, :, rs1:re1, cs1:ce1],
                     aa,
                     dd] += 1

        glcm = glcm.type(torch.FloatTensor)
        # normalize the GLCM
        glcm_sums = torch.sum(glcm, dim=(2, 3), keepdim=True)
        glcm_sums[:, glcm_sums == 0] = 1
        glcm /= glcm_sums
        return glcm

    def get_haralick_texture_features(self, glcm):
        """
        Adapted from the haralick features implementation of graycoprops in scikit-image/skimage/feature/texture.py
        """
        # print(glcm.shape)
        glcm = glcm.transpose((2, 3, 4, 5, 0, 1))
        (n_levels, n_levels, num_dist, num_angle, batch_size, channels) = glcm.size()

        # contrast = torch.sum(torch.mul(glcm, self.weights_contrast), dim=self.reduction_dims)
        # dissimilarity = torch.sum(torch.mul(glcm, self.weights_dissimilarity), dim=self.reduction_dims)
        # homogeneity = torch.sum(torch.mul(glcm, self.weights_homogeneity), dim=self.reduction_dims)
        # energy = torch.sqrt(torch.sum(torch.pow(glcm, 2), dim=self.reduction_dims))
        einsum_instruction = "ijklmn, ijklmn -> klmn"
        energy = LA.matrix_norm(glcm, dim=self.reduction_dims)
        contrast = torch.einsum(einsum_instruction, glcm, self.weights_contrast)
        dissimilarity = torch.einsum(einsum_instruction, glcm, self.weights_dissimilarity)
        homogeneity = torch.einsum(einsum_instruction, glcm, self.weights_homogeneity)

        ##############################################
        # correlation
        correlation = torch.zeros((num_dist, num_angle, batch_size, channels,), dtype=torch.float32)
        diff_i = self.i_corr - torch.sum(torch.mul(self.i_corr, glcm), dim=self.reduction_dims)
        diff_j = self.j_corr - torch.sum(torch.mul(self.j_corr, glcm), dim=self.reduction_dims)
        std_i = torch.sqrt(torch.sum(torch.mul(glcm, torch.pow(diff_i, 2)), dim=self.reduction_dims))
        std_j = torch.sqrt(torch.sum(torch.mul(glcm, torch.pow(diff_j, 2)), dim=self.reduction_dims))
        cov = torch.sum(glcm * (diff_i * diff_j), dim=self.reduction_dims)
        # diff_i = self.i_corr - torch.einsum("ijklmn, ijklmn -> klmn", self.i_corr, glcm)
        # diff_j = self.j_corr - torch.einsum("ijklmn, ijklmn -> klmn", self.j_corr, glcm)
        # std_i = torch.sqrt(torch.einsum("ijklmn, ijklmn -> klmn", glcm, torch.pow(diff_i, 2)))
        # std_j = torch.sqrt(torch.einsum("ijklmn, ijklmn -> klmn", glcm, torch.pow(diff_j, 2)))
        # cov = torch.einsum("ijklmn, ijklmn, ijklmn -> klmn", glcm, diff_i, diff_j)

        # handle the special case of standard deviations near zero
        mask_0 = std_i < self.std_tolerance
        mask_0[std_j < self.std_tolerance] = True
        correlation[mask_0] = 1
        # handle the standard case
        mask_1 = ~mask_0
        correlation[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
        #############################################################################
        new_order = (2, 3, 0, 1)
        energy = torch.permute(energy, new_order)
        contrast = torch.permute(contrast, new_order)
        dissimilarity = torch.permute(dissimilarity, new_order)
        homogeneity = torch.permute(homogeneity, new_order)
        correlation = torch.permute(correlation, new_order)

        return energy, contrast, dissimilarity, homogeneity, correlation

############################################################################
# n_levels = 16
# P = np.random.randint(0, 3, (n_levels, n_levels, 2, 3, 5, 6))
# I, J = np.ogrid[0:n_levels, 0:n_levels]
# weight_np = (I - J) ** 2
# weights = weight_np.reshape((n_levels, n_levels, 1, 1, 1, 1))
# results1 = np.sum(P * weights, axis=(0, 1))
# results2 = np.einsum("ijklmn, ijklmn -> klmn", P, weights)
# print(I.shape, J.shape, weight_np.shape, weights.shape,
#       results1.shape, results2.shape, np.array_equal(results1, results2))
#
# P = torch.as_tensor(P).type(torch.LongTensor)
# I, J = torch.meshgrid(torch.arange(0, n_levels).type(torch.LongTensor),
#                       torch.arange(0, n_levels).type(torch.LongTensor))
# weight_torch = (I - J) ** 2
# weights = weight_torch.view((n_levels, n_levels, 1, 1, 1, 1))
# results1 = torch.sum(P * weights, dim=(0, 1))
# results2 = torch.einsum("ijklmn, ijklmn -> klmn", P, weights)
# print(I.size(), J.size(), weight_torch.size(), weights.size(),
#       results1.size(), results2.size(), torch.equal(results1, results2))

###################################################################################
# from kymatio.torch import Scattering2D
# img = torch.randint(-4, 4, (10, 11, 32, 32)).type(torch.FloatTensor)
# scattering = Scattering2D(J=2, shape=(32, 32))
# features = scattering(img)
# # print(scattering)
# print(features.size())
####################################################################################
# x = torch.randn((10, 5))
# x = (x-torch.amin(x))/(torch.amax(x)-torch.amin(x))
# boundaries = torch.linspace(0, 1, 10)
# y = torch.bucketize(x, boundaries)
# # print(x)
# # print('\n')
# # print(y)
#
# for ii in range(len(x)):
#     print(x[ii])
#     print(y[ii])
#     print('\n')
#########################################################################################


###############################################################################
main_path = WindowsPath(r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid")
exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"

args = Args(experiment=exp_fold, main_path=main_path, mode="full").args
myclass = FeatureExtractorSingular(args)

img_paths = (
    main_path/exp_fold/args.img_folder/"AssayPlate_PerkinElmer_CellCarrier-384_A01_T0001F001L01A01Z01C01.tif",
    main_path/exp_fold/args.img_folder/"AssayPlate_PerkinElmer_CellCarrier-384_A01_T0001F001L01A02Z01C02.tif",
    main_path/exp_fold/args.img_folder/"AssayPlate_PerkinElmer_CellCarrier-384_A01_T0001F001L01A04Z01C03.tif",
    main_path/exp_fold/args.img_folder/"AssayPlate_PerkinElmer_CellCarrier-384_A01_T0001F001L01A03Z01C04.tif",
    main_path/exp_fold/args.img_folder/"AssayPlate_PerkinElmer_CellCarrier-384_A01_T0001F001L01A05Z01C05.tif",
)
mask_paths = (
    main_path/exp_fold/args.step2_save_path/"w0_A01_F001.png",
    main_path/exp_fold/args.step2_save_path/"w1_A01_F001.png",
    main_path/exp_fold/args.step2_save_path/"w2_A01_F001.png",
    main_path/exp_fold/args.step2_save_path/"w4_A01_F001.png",
)
img, mask = myclass.step0_preprocessing(img_paths, mask_paths)
myclass.step1_get_features(img, mask)
##############################################
# info = torch.iinfo(torch.int8)
# print(info)
