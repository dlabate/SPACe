import numpy as np
from numpy import linalg as LA
from scipy import ndimage as ndi
from scipy.stats import median_abs_deviation

from skimage.measure import _moments, find_contours
from skimage.measure._regionprops import RegionProperties, _cached, only2d
from skimage.feature import graycomatrix
from skimage.segmentation import find_boundaries
from pyefd import elliptic_fourier_descriptors
from radiomics import base, cMatrices

from functools import cached_property


def safe_log_10_v0(value):
    """https://stackoverflow.com/questions/21610198/runtimewarning-divide-by-zero-encountered-in-log"""
    value = np.abs(value)
    result = np.where(value > 1e-12, value, -12)
    # print(result)
    res = np.log10(result, out=result, where=result > 1e-12)
    return res


def safe_log_10_v1(value):
    """Pankaj"""
    return -np.log(1 + np.abs(value))


def discretize_intensity_global(img, intensity_level_thresholds):
    """Assuming Img is a 2d/3d np array, here img refers to the entire field of view"""
    # every intensity value about the intensity_level_thresholds[-1] will be set to intensity_level_thresholds[-1]
    # discrete_img = np.clip(img, None, intensity_level_thresholds[-1]).astype(np.uint8)
    discrete_img = np.zeros(img.shape, dtype=np.uint8)
    n_levels = len(intensity_level_thresholds)
    # print("num levels inside discrete global", n_levels)
    for ii in range(1, n_levels):
        # i0 = intensity_level_thresholds[ii - 1]
        # i2 = intensity_level_thresholds[ii]
        # chopping the image
        # discrete_img[(i0 <= img) & (img < i1)] = ii
        # discrete_img += ((i0 <= img) * (img < i1) * (ii - 1)).astype(np.uint8)
        #########################################################################
        # # round clipping the image
        # i1 = i0 + (i2 - i0) // 2
        # discrete_img[(i0 <= img) & (img < i1)] = ii - 1
        # discrete_img[(i1 <= img) & (img < i2)] = ii
        ###################################
        # hard clipping (chopping) the image
        discrete_img[(intensity_level_thresholds[ii - 1] <= img) &
                     (img < intensity_level_thresholds[ii])] = ii - 1
    discrete_img[img == intensity_level_thresholds[-1]] = n_levels - 1

    return discrete_img


def discretize_intensity_local(img, n_levels=8):
    """
    Assuming Img is a 2d/3d np array, here img refers an object within a bounding box
    within a field of view."""
    min_ = np.amin(img)
    max_ = np.max(img)
    levels = np.linspace(min_, max_, n_levels)
    discrete_img = np.zeros(img.shape, dtype=np.uint8)
    # print("num levels inside discrete local", n_levels)
    for ii in range(1, n_levels):
        # i0 = levels[ii - 1]
        # i2 = levels[ii]
        # chop clipping the image
        # discrete_img[(i0 <= img) & (img < i1)] = ii
        # discrete_img += ((i0 <= img) * (img < i1) * (ii - 1)).astype(np.uint8)
        #########################################################################
        # # round clipping the image
        # i1 = i0 + (i2 - i0) // 2
        # discrete_img[(i0 <= img) & (img < i1)] = ii - 1
        # discrete_img[(i1 <= img) & (img < i2)] = ii
        # ###################################
        # hard clipping (chopping) the image
        discrete_img[(levels[ii - 1] <= img) & (img < levels[ii])] = ii - 1
    discrete_img[img == max_] = n_levels - 1
    return discrete_img


def regionprops(w0_mask, w1_mask, w2_mask, w4_mask, img,
                n_levels, intensity_percentiles, distances, angles, ):
    N = len(np.unique(w0_mask)) - 1
    regions = np.zeros((5, N), dtype=object)
    has_nucleoli = np.zeros((N, 1), dtype=np.uint8)

    max_ = np.amax(w0_mask)
    w0_objects = ndi.find_objects(w0_mask, max_label=max_)
    w1_objects = ndi.find_objects(w1_mask, max_label=max_)
    w2_objects = ndi.find_objects(w2_mask, max_label=max_)
    w4_objects = ndi.find_objects(w4_mask, max_label=max_)
    cnt = 0
    for ii in range(max_):
        if w0_objects[ii] is None:
            continue
        label = ii + 1
        w0_props = RegionPropertiesExtension(w0_objects[ii], label, w0_mask, img[0])
        w1_props = RegionPropertiesExtension(w1_objects[ii], label, w1_mask, img[1])
        w3_props = RegionPropertiesExtension(w1_objects[ii], label, w1_mask, img[3])
        w4_props = RegionPropertiesExtension(w4_objects[ii], label, w4_mask, img[4])
        if w2_objects[ii] is not None:
            w2_props = RegionPropertiesExtension(w2_objects[ii], label, w2_mask, img[2], n_levels)
            has_nucleoli[cnt] = 1
        else:
            w2_props = None
            has_nucleoli[cnt] = 0

        regions[0, cnt] = w0_props
        regions[1, cnt] = w1_props
        regions[2, cnt] = w2_props
        regions[3, cnt] = w3_props
        regions[4, cnt] = w4_props

        cnt += 1

    return regions, has_nucleoli


class RegionPropertiesExtension(RegionProperties):
    """Please refer to `skimage.measure.regionprops` for more information
    on the available region properties.
    """
    ndim = 2
    bd_val = 10
    bd_padding = [(bd_val, bd_val), (bd_val, bd_val)]

    n_levels = 16
    corr_tolerance = 1e-8
    distances = np.arange(1, 21)
    angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    intensity_percentiles = (5, 10, 25, 75, 90, 95)

    glszm_eps = np.spacing(1)
    # P_glszm = None
    # P_glszm_coeffs_Np = None
    # P_glszm_coeffs_Nz = None
    # P_glszm_coeffs_ps = None
    # P_glszm_coeffs_pg = None
    # P_glszm_coeffs_ivector = None
    # P_glszm_coeffs_jvector = None
    P_glszm = {
        'P_glszm': None,
        'Np': 0,
        'Nz': 0,
        'ps': 0,
        'pg': 0,
        'ivector': None,
        'jvector': None,
    }

    P_glrlm = {
        'P_glrlm': None,
        'Nr_glrlm': 0,
        'pr_glrlm': 0,
        'pg_glrlm': 0,
        'ivector_glrlm': None,
        'jvector_glrlm': None,
    }

    P_ngtdm = {
        'P_ngtdm': None,
        'Nvp_ngtdm': 0,
        'p_i_ngtdm': 0,
        's_i_ngtdm': 0,
        'i_ngtdm': None,
        'Ngp_ngtdm': 0,
        'p_zero_ngtdm': 0,
    }

    P_gldm = {
        'P_gldm': None,
        'Nz_gldm': 0,
        'pd_gldm': 0,
        'pg_gldm': 0,
        'i_vector_gldm': None,
        'j_vector_gldm': None,
    }

    def __init__(self, slice, label, label_image, intensity_image,
                 cache_active=True, ):
        super().__init__(slice, label, label_image, intensity_image, cache_active)

        # The number of levels found within an image may not be the same as
        # the total number of globally available levels predefined/preset by us.
        self.I, self.J = self.haralick_ij()
        self.corr_I, self.corr_J = self.haralick_corr_ij()

    @property
    @_cached
    def einsum_instruct_2(self):
        return "ijkm,ijkm->km"

    @property
    @_cached
    def einsum_instruct_3(self):
        return "ijkm,ijkm,ijkm->km"

    @_cached
    def haralick_ij(self):
        # create weights for specified property
        I, J = np.ogrid[0:self.n_levels, 0:self.n_levels]
        return I, J

    @_cached
    def haralick_corr_ij(self):
        I = np.array(range(self.n_levels)).reshape((self.n_levels, 1, 1, 1))
        J = np.array(range(self.n_levels)).reshape((1, self.n_levels, 1, 1))
        return I, J

    @property
    @_cached
    def weights0(self):
        weights0 = (self.I - self.J) ** 2
        weights0 = weights0.reshape((self.n_levels, self.n_levels, 1, 1))
        return weights0

    @property
    @_cached
    def weights1(self):
        weights1 = np.abs(self.I - self.J)
        weights1 = weights1.reshape((self.n_levels, self.n_levels, 1, 1))
        return weights1

    @property
    @_cached
    def weights2(self):
        return 1. / (1. + self.weights0)

    @property
    @_cached
    def bins(self):
        return np.linspace(self.intensity_min, self.intensity_max, self.n_levels)

    @property
    @_cached
    def image_intensity_discrete(self):
        return np.int32(np.digitize(self.image_intensity, self.bins, right=True))

    @property
    @_cached
    def image_int32(self):
        return np.int32(self.image)

    @property
    @only2d
    @_cached
    def moments_hu(self):
        mh = _moments.moments_hu(self.moments_normalized)
        return -1 * np.sign(mh) * safe_log_10_v0(mh)

    @property
    @only2d
    @_cached
    def moments_weighted_hu(self):
        # nu = self.moments_weighted_normalized
        mhw = _moments.moments_hu(self.moments_weighted_normalized)
        return -1 * np.sign(mhw) * safe_log_10_v0(mhw)

    @property
    @_cached
    def moments_weighted_normalized(self):
        # mu = self.moments_weighted_central
        mwn = _moments.moments_normalized(self.moments_weighted_central, order=3)
        # print("mwn")
        # print(mwn)
        # print("Stackoverflow", -1 * np.sign(mwn) * safe_log_10_v0(mwn))
        # print('\n')
        return -1 * np.sign(mwn) * safe_log_10_v0(mwn)

    @property
    @_cached
    def image_intensity_vec(self):
        return self.image_intensity[self.image_intensity > 0]

    @property
    @_cached
    def intensity_statistics(self, ):
        percentiles = np.percentile(self.image_intensity_vec, self.intensity_percentiles)
        intensity_median, intensity_mad, intensity_mean, intensity_std = \
            np.median(self.image_intensity_vec), median_abs_deviation(self.image_intensity_vec), \
                np.mean(self.image_intensity_vec), np.std(self.image_intensity_vec)
        return tuple(percentiles) + (intensity_median, intensity_mad, intensity_mean, intensity_std,)

    @cached_property
    def voxel_coordinates(self):
        # labelledVoxelCoordinates = np.array(np.where(self.image))
        return np.array(np.where(self.image))

    @property
    @_cached
    def glcm(self, ):  # gray-comatrix
        # remove level-zero (pixels with value zero identify the background and have to be removed from the analysis)
        P = graycomatrix(
            self.image_intensity_discrete,
            distances=self.distances,
            angles=self.angles,
            levels=self.n_levels,
            symmetric=False, normed=False)

        # normalize each GLCM
        P = P.astype(np.float32)
        glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums
        return P

    @cached_property
    def glcm_features(self, ):
        """
        Calculate texture properties of a GLCM.
        Compute a feature of a gray level co-occurrence matrix to serve as
        a compact summary of the matrix. The properties are computed as
        follows:
        - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
        - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1}P_{i,j}|i-j|`
        - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
        - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`
        - 'energy': :math:`\\sqrt{ASM}`
        - 'correlation':
        .. math:: \\sum_{i,j=0}^{levels-1} P_{i,j}\\left[\\frac{(i-\\mu_i) \\
                  (j-\\mu_j)}{\\sqrt{(\\sigma_i^2)(\\sigma_j^2)}}\\right]
        Each GLCM is normalized to have a sum of 1 before the computation of
        texture properties.
        .. versionchanged:: 0.19
           `greycoprops` was renamed to `graycoprops` in 0.19.
        Parameters
        ----------
        P : ndarray
        Input array. `P` is the gray-level co-occurrence histogram
        for which to compute the specified property. The value
        `P[i,j,d,theta]` is the number of times that gray-level j
        occurs at a distance d and at an angle theta from
        gray-level i.
        prop : {'contrast', 'dissimilarity', 'homogeneity', 'energy', \
            'correlation', 'ASM'}, optional
        The property of the GLCM to compute. The default is 'contrast'.
        Returns
        -------
        results : 2-D ndarray
        2-dimensional array. `results[d, a]` is the property 'prop' for
        the d'th distance and the a'th angle.
        References
        ----------
        .. [1] M. Hall-Beyer, 2007. GLCM Texture: A Tutorial v. 1.0 through 3.0.
           The GLCM Tutorial Home Page,
           https://prism.ucalgary.ca/handle/1880/51900
           DOI:`10.11575/PRISM/33280`
        Examples
        --------
        Compute the contrast for GLCMs with distances [1, 2] and angles
        [0 degrees, 90 degrees]
        # >>> image = np.array([[0, 0, 1, 1],
        # ...                   [0, 0, 1, 1],
        # ...                   [0, 2, 2, 2],
        # ...                   [2, 2, 3, 3]], dtype=np.uint8)
        # >>> g = graycomatrix(image, [1, 2], [0, np.pi/2], levels=4,
        # ...                  normed=True, symmetric=True)
        """
        # print(gmat.shape)
        (num_level, num_level2, num_dist, num_angle) = self.glcm.shape

        # contrast, dissimilarity, homogeneity
        # contrast = np.sum(self.glcm * self.weights0, axis=(0, 1))
        # dissimilarity = np.sum(self.glcm * self.weights1, axis=(0, 1))
        # homogeneity = np.sum(self.glcm * self.weights2, axis=(0, 1))
        contrast = np.einsum(self.einsum_instruct_2, self.glcm, self.weights0)
        dissimilarity = np.einsum(self.einsum_instruct_2, self.glcm, self.weights1)
        homogeneity = np.einsum(self.einsum_instruct_2, self.glcm, self.weights2)
        ###################################################
        # asm and energy
        # asm = np.sum(self.glcm ** 2, axis=(0, 1))
        # energy = np.sqrt(asm)
        energy = LA.norm(self.glcm, ord='fro', axis=(0, 1))
        ##############################################
        # correlation
        correlation = np.zeros((num_dist, num_angle), dtype=np.float32)
        # diff_i = self.corr_I - np.sum(self.corr_I * self.glcm, axis=(0, 1))
        # diff_j = self.corr_J - np.sum(self.corr_J * self.glcm, axis=(0, 1))
        # std_i = np.sqrt(np.sum(self.glcm * (diff_i ** 2), axis=(0, 1)))
        # std_j = np.sqrt(np.sum(self.glcm * (diff_j ** 2), axis=(0, 1)))
        # cov = np.sum(self.glcm * (diff_i * diff_j), axis=(0, 1))

        diff_i = self.corr_I - np.einsum(self.einsum_instruct_2, self.corr_I, self.glcm)
        diff_j = self.corr_J - np.einsum(self.einsum_instruct_2, self.corr_J, self.glcm)
        std_i = np.sqrt(np.einsum(self.einsum_instruct_2, self.glcm, diff_i ** 2))
        std_j = np.sqrt(np.einsum(self.einsum_instruct_2, self.glcm, diff_j ** 2))
        cov = np.einsum(self.einsum_instruct_3, self.glcm, diff_i, diff_j)

        # handle the special case of standard deviations near zero
        mask_0 = std_i < self.corr_tolerance
        mask_0[std_j < self.corr_tolerance] = True
        correlation[mask_0] = 1
        # handle the standard case
        mask_1 = ~mask_0
        correlation[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
        ###############################################
        haralick_features = tuple(np.vstack((contrast, dissimilarity, homogeneity, energy, correlation)).reshape(-1))
        # return contrast.reshape(-1), dissimilarity.reshape(-1), homogeneity.reshape(-1), \
        #     energy.reshape(-1), correlation.reshape(-1)
        #######################################################
        # If summary of these features is preferable because of redundancy
        # print("contrast:", contrast.shape)
        # print(
        # np.median(contrast, axis=0).shape,
        # median_abs_deviation(contrast, axis=0).shape)

        # haralick_features = np.vstack([
        #     np.median(contrast, axis=0),
        #     median_abs_deviation(contrast, axis=0),
        #
        #     np.median(dissimilarity, axis=0),
        #     median_abs_deviation(dissimilarity, axis=0),
        #
        #     np.median(homogeneity, axis=0),
        #     median_abs_deviation(homogeneity, axis=0),
        #
        #     np.median(energy, axis=0),
        #     median_abs_deviation(energy, axis=0),
        #
        #     np.median(correlation, axis=0),
        #     median_abs_deviation(correlation, axis=0)])
        # print(contrast.shape)
        # print(np.median(contrast, axis=0).shape)
        # print(haralick_features.shape)
        # return haralick_features.reshape(-1)
        return haralick_features

    @property
    @_cached
    def efc_ratio(self, ):
        bd = find_boundaries(np.pad(self.image, self.bd_padding, 'constant', constant_values=(0, 0)))
        bd_contours = find_contours(bd, .1)[0]
        efc = elliptic_fourier_descriptors(bd_contours,
                                           normalize=True,
                                           order=15)
        # N = efc.shape[0]
        efcs = np.sqrt(efc[:, 0] ** 2 + efc[:, 1] ** 2) + np.sqrt(efc[:, 2] ** 2 + efc[:, 3] ** 2)
        ratio = efcs[0] / np.sum(efcs[1:])
        return ratio

    @property
    @_cached
    def grayLevels(self, ):
        return np.unique(self.image_intensity_discrete + 1)

    @property
    @_cached
    def glszm_ng(self, ):
        return int(np.amax(self.grayLevels))

    @property
    @_cached
    def circularity(self, ):
        if self.perimeter > 1e-6:
            return (4 * np.pi * self.area) / self.perimeter ** 2
        else:
            return np.nan

    @property
    def glszm(self, ):
        """
        Number of times a region with a
        gray level and voxel count occurs in an image. P_glszm[level, voxel_count] = # occurrences
    
        For 3D-images this concerns a 26-connected region, for 2D an 8-connected region
        """
        # self.imgArray is self.image_intensity converted to self.image_intensity_discrete
        # self.maskArray is self.img
        # Ng is the number of gray levels wich is n_levels for us
        # Ns is be the number of discrete zone sizes in the image
        # N_p be the number of voxels in the image,
        # self.logger.debug('Calculating GLSZM matrix in C')
        Ng = self.glszm_ng
        Ns = np.sum(self.image_int32)
        # matrix_args = [
        #     self.imageArray,
        #     self.maskArray,
        #     Ng,
        #     Ns,
        #     self.settings.get('force2D', False),
        #     self.settings.get('force2Ddimension', 0)
        # ]
        # if self.voxelBased:
        #     matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]

        print(type(self.image_intensity_discrete),
              self.image_intensity_discrete.dtype,
              type(self.image_int32), self.image_int32.dtype, Ng, Ns)
        P_glszm = cMatrices.calculate_glszm(
            self.image_intensity_discrete + 1,
            self.image_int32,
            Ng,
            Ns,
            False,
            0
        )  # shape (Nvox, Ng, Ns)
        print(P_glszm)

        # Delete rows that specify gray levels not present in the ROI
        NgVector = range(1, Ng + 1)  # All possible gray values
        GrayLevels = self.grayLevels  # Gray values present in ROI
        emptyGrayLevels = np.array(list(set(NgVector) - set(GrayLevels)),
                                   dtype=int)  # Gray values NOT present in ROI

        P_glszm = np.delete(P_glszm, emptyGrayLevels - 1, 1)

        #######################################
        # calculate_glszm_coefficients
        ps = np.sum(P_glszm, 1)  # shape (Nvox, Ns)
        pg = np.sum(P_glszm, 2)  # shape (Nvox, Ng)

        ivector = GrayLevels.astype(float)  # shape (Ng,)
        jvector = np.arange(1, P_glszm.shape[2] + 1, dtype=np.float64)  # shape (Ns,)

        # Get the number of zones in this GLSZM
        Nz = np.sum(P_glszm, (1, 2))  # shape (Nvox,)
        Nz[Nz == 0] = 1  # set sum to np.spacing(1) if sum is 0?

        # Get the number of voxels represented by this GLSZM: Multiply the zones by their size and sum them
        Np = np.sum(ps * jvector[None, :], 1)  # shape (Nvox, )
        Np[Np == 0] = 1

        # Delete columns that specify zone sizes not present in the ROI
        emptyZoneSizes = np.where(np.sum(ps, 0) == 0)
        P_glszm = np.delete(P_glszm, emptyZoneSizes, 2)
        jvector = np.delete(jvector, emptyZoneSizes)
        ps = np.delete(ps, emptyZoneSizes, 1)
        #
        # print(ps)
        P_glszm_dict = {
            'P_glszm': P_glszm,
            'Np': Np,
            'Nz': Nz,
            'ps': ps,
            'pg': pg,
            'ivector': ivector,
            'jvector': jvector,
        }
        # # Return the properties as a dictionary
        return P_glszm_dict

    @property
    def glrlm(self, ):
        """
        A Gray Level Run Length Matrix (GLRLM) quantifies gray level runs, 
        which are defined as the length in number of pixels, of consecutive pixels 
        that have the same gray level value. 
       
        """
        # self.imgArray is self.image_intensity converted to self.image_intensity_discrete
        # self.maskArray is self.img
        # N_r is the number of discrete run lengths in the image
        # N_g is the number of gray levels wich is n_levels for us
        # self.logger.debug('Calculating GLRLM matrix in C')
        Ng_glrlm = self.glszm_ng
        iid_1 = self.image_intensity_discrete + 1
        Nr_glrlm = np.max(iid_1.shape)

        # matrix_args = [
        #     self.imageArray,
        #     self.maskArray,
        #     Ng,
        #     Nr,
        #     self.settings.get('force2D', False),
        #     self.settings.get('force2Ddimension', 0)
        # ]
        # if self.voxelBased:
        #     matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]

        P_glrlm, angles = cMatrices.calculate_glrlm(
            self.image_intensity_discrete + 1,
            self.image_int32,
            Ng_glrlm,
            Nr_glrlm,
            False,
            0
        )  # shape (Nvox, Ng, Ns)
        print(P_glrlm, angles)

        # Delete rows that specify gray levels not present in the ROI
        NgVector = range(1, Ng_glrlm + 1)  # All possible gray values
        GrayLevels = self.grayLevels  # Gray values present in ROI
        emptyGrayLevels = np.array(list(set(NgVector) - set(GrayLevels)),
                                   dtype=int)  # Gray values NOT present in ROI

        P_glrlm = np.delete(P_glrlm, emptyGrayLevels - 1, 1)

        Nr_glrlm = np.sum(P_glrlm, (1, 2))

        if P_glrlm.shape[3] > 1:
            emptyAngles = np.where(np.sum(Nr_glrlm, 0) == 0)
            if len(emptyAngles[0]) > 0:  # One or more angles are 'empty'
                P_glrlm = np.delete(P_glrlm, emptyAngles, 3)
                Nr_glrlm = np.delete(Nr_glrlm, emptyAngles, 1)

        Nr_glrlm[Nr_glrlm == 0] = np.nan
        #######################################
        # calculate_glrlm coefficients
        pr_glrlm = np.sum(P_glrlm, 1)  # shape (Nvox, Nr, Na)
        pg_glrlm = np.sum(P_glrlm, 2)  # shape (Nvox, Ng, Na)

        ivector_glrlm = GrayLevels.astype(float)  # shape (Ng,)
        jvector_glrlm = np.arange(1, P_glrlm.shape[2] + 1, dtype=np.float64)  # shape (Nr,)

        # Delete columns that run lengths not present in the ROI for GLRLM
        emptyRunLenghts = np.where(np.sum(pr_glrlm, (0, 2)) == 0)
        P_glrlm = np.delete(P_glrlm, emptyRunLenghts, 2)
        jvector_glrlm = np.delete(jvector_glrlm, emptyRunLenghts)
        pr_glrlm = np.delete(pr_glrlm, emptyRunLenghts, 1)

        P_glrlm_dict = {
            'P_glrlm': P_glrlm,
            'Nr_glrlm': Nr_glrlm,
            'pr_glrlm': pr_glrlm,
            'pg_glrlm': pg_glrlm,
            'ivector_glrlm': ivector_glrlm,
            'jvector_glrlm': jvector_glrlm,
        }
        # # Return the properties as a dictionary
        return P_glrlm_dict

    @property
    def ngtdm(self, ):
        """
        A Neighbouring Gray Tone Difference Matrix quantifies the difference between a gray value and the average gray value
        of its neighbours within distance :math:`\delta`.
        """
        # self.imgArray is self.image_intensity converted to self.image_intensity_discrete
        # self.maskArray is self.img
        # n_i is the number of voxels in :math:`X_{gl}` with gray level :math:`i`
        # p_i is the gray level probability and equal to :math:`n_i/N_v`
        # N_g is the number of gray levels wich is n_levels for us
        # self.logger.debug('Calculating NGTDM matrix in C')

        Ng_ngtdm = self.glszm_ng
        # matrix_args = [
        #   self.imageArray,
        #   self.maskArray,
        #   numpy.array(self.settings.get('distances', [1])),
        #   self.coefficients['Ng'],
        #   self.settings.get('force2D', False),
        #   self.settings.get('force2Ddimension', 0)
        # ]
        # if self.voxelBased:
        #     matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]

        P_ngtdm = cMatrices.calculate_ngtdm(
            self.image_intensity_discrete + 1,
            self.image_int32,
            np.array(self.settings.get("distances", [1])),  # ???????
            Ng_ngtdm,
            False,
            0)  # shape (Nvox, Ng, 3)

        # Delete empty grey levels
        emptyGrayLevels = np.where(np.sum(P_ngtdm[:, :, 0], 0) == 0)
        P_ngtdm = np.delete(P_ngtdm, emptyGrayLevels, 1)

        #######################################
        # calculate_ngtdm_coefficients
        # No of voxels that have a valid region, lesser equal to Np
        Npv_ngtdm = np.sum(P_ngtdm[:, :, 0], 1)  # shape (Nvox,)

        # Normalize P_ngtdm[:, 0] (= n_i) to obtain p_i
        p_i_ngtdm = P_ngtdm[:, :, 0] / Npv_ngtdm[:, None]
        s_i_ngtdm = P_ngtdm[:, :, 1]
        i_ngtdm = P_ngtdm[:, :, 2]

        # Ngp = number of graylevels, for which p_i > 0
        Ngp_ngtdm = np.sum(P_ngtdm[:, :, 0] > 0, 1)

        p_zero_ngtdm = np.where(p_i_ngtdm == 0)

        P_ngtdm_dict = {
            'P_ngtdm': P_ngtdm,
            'Npv_ngtdm': Npv_ngtdm,
            'p_i_ngtdm': p_i_ngtdm,
            's_i_ngtdm': s_i_ngtdm,
            'i_ngtdm': i_ngtdm,
            'Ngp_ngtdm': Ngp_ngtdm,
            'p_zero_ngtdm': p_zero_ngtdm,
        }
        # # Return the properties as a dictionary
        return P_ngtdm_dict

    @property
    def gldm(self, ):
        """
        A Gray Level Dependence Matrix (GLDM) quantifies gray level dependencies in an image.
        A gray level dependency is defined as a the number of connected voxels within distance :math:`\delta` that are
        dependent on the center voxel.
        """
        # self.imgArray is self.image_intensity converted to self.image_intensity_discrete
        # self.maskArray is self.img
        # N_g is the number of discrete intensity values in the image
        # N_d is the number of discrete dependency sizes in the image
        # N_z is the number of dependency zones in the image
        # distances [[1]]: List of integers. This specifies the distances between the center voxel and the neighbor, for which
        #       angles should be generated.
        # gldm_a [0]: float, :math:`\alpha` cutoff value for dependence. A neighbouring voxel with gray level :math:`j` is
        #       considered dependent on center voxel with gray level :math:`i` if :math:`|i-j|\le\alpha`
        # self.logger.debug('Calculating GLDM matrix in C')

        Nz_gldm = self.glszm_ng
        # matrix_args = [
        #   self.imageArray,
        #   self.maskArray,
        #   numpy.array(self.settings.get('distances', [1])),
        #   Ng,
        #   self.gldm_a,
        #   self.settings.get('force2D', False),
        #   self.settings.get('force2Ddimension', 0)
        # ]
        # if self.voxelBased:
        #     matrix_args += [self.settings.get('kernelRadius', 1), voxelCoordinates]

        P_gldm = cMatrices.calculate_gldm(
            self.image_intensity_discrete + 1,
            self.image_int32,
            np.array(self.settings.get('distances', [1])),  # ????
            Nz_gldm,
            self.gldm_a,  # ????
            False,
            0
        )  # shape (Nvox, Ng, Ns)
        print(P_gldm)

        # Delete rows that specify gray levels not present in the ROI
        NgVector = range(1, Nz_gldm + 1)  # All possible gray values
        GrayLevels = self.grayLevels  # Gray values present in ROI
        emptyGrayLevels = np.array(
            list(set(NgVector) - set(GrayLevels)),
            dtype=int)  # Gray values NOT present in ROI

        P_gldm = np.delete(P_gldm, emptyGrayLevels - 1, 1)
        jvector_gldm = np.arange(1, P_gldm.shape[2] + 1, dtype='float64')

        # shape (Nv, Nd)
        pd_gldm = np.sum(P_gldm, 1)
        # shape (Nv, Ng)
        pg_gdlm = np.sum(P_gldm, 2)

        # Delete columns that dependence sizes not present in the ROI
        empty_sizes = np.sum(pd_gldm, 0)
        P_gldm = np.delete(P_gldm, np.where(empty_sizes == 0), 2)
        jvector_gldm = np.delete(jvector_gldm, np.where(empty_sizes == 0))
        pd_gldm = np.delete(pd_gldm, np.where(empty_sizes == 0), 1)

        Nz_gldm = np.sum(pd_gldm, 1)  # Nz per kernel, shape (Nv, )
        Nz_gldm[Nz_gldm == 0] = 1  # set sum to numpy.spacing(1) if sum is 0?

        ivector_gldm = self.grayLevels.astype(float)

        P_gldm_dict = {
            'P_gldm': P_gldm,
            'Nz_gldm': Nz_gldm,
            'pd_gldm': pd_gldm,
            'pg_gdlm': pg_gdlm,
            'ivector_gldm': ivector_gldm,
            'jvector_gldm': jvector_gldm,
        }
        # # Return the properties as a dictionary
        return P_gldm_dict

    # @cached_property
    # def glszm_features(self):
    #     
    #     print(self.glszm)
    #     ps = self.glszm['ps']
    #     jvector = self.glszm['jvector']
    #     ivector = self.glszm['ivector']
    #     Nz = self.glszm['Nz']
    #     pg = self.glszm['pg']
    #     Np = self.glszm['Np']
    #     P_glszm = self.glszm['P_glszm']
    #
    #     # getSmallAreaEmphasisFeatureValue
    #     sae = np.sum(ps / (jvector[None, :] ** 2), 1) / Nz
    #     # getLargeAreaEmphasisFeatureValue
    #     lae = np.sum(ps * (jvector[None, :] ** 2), 1) / Nz
    #     # getGrayLevelNonUniformityFeatureValue
    #     iv = np.sum(pg ** 2, 1) / Nz
    #     # getGrayLevelNonUniformityNormalizedFeatureValue
    #     ivn = np.sum(pg ** 2, 1) / Nz ** 2
    #     # getSizeZoneNonUniformityFeatureValue
    #     szv = np.sum(ps ** 2, 1) / Nz
    #     # getSizeZoneNonUniformityNormalizedFeatureValue
    #     szvn = np.sum(ps ** 2, 1) / Nz ** 2
    #     # getZonePercentageFeatureValue
    #     zp = Nz / Np
    #
    #     # getGrayLevelVarianceFeatureValue
    #     pg_1 = pg / Nz[:, None]  # divide by Nz to get the normalized matrix
    #     u_i = np.sum(pg_1 * ivector[None, :], 1, keepdims=True)
    #     glv = np.sum(pg_1 * (ivector[None, :] - u_i) ** 2, 1)
    #
    #     # getZoneVarianceFeatureValue
    #     ps_1 = ps / Nz[:, None]  # divide by Nz to get the normalized matrix
    #     u_j = np.sum(ps_1 * jvector[None, :], 1, keepdims=True)
    #     zv = np.sum(ps_1 * (jvector[None, :] - u_j) ** 2, 1)
    #
    #     # getZoneEntropyFeatureValue
    #     p_glszm_1 = P_glszm / Nz[:, None, None]  # divide by Nz to get the normalized matrix
    #     ze = -np.sum(p_glszm_1 * np.log2(p_glszm_1 + self.glszm_eps), (1, 2))
    #
    #     # getLowGrayLevelZoneEmphasisFeatureValue
    #     lie = np.sum(pg / (ivector[None, :] ** 2), 1) / Nz
    #     # getHighGrayLevelZoneEmphasisFeatureValue
    #     hie = np.sum(pg * (ivector[None, :] ** 2), 1) / Nz
    #     # getSmallAreaLowGrayLevelEmphasisFeatureValue
    #     lisae = np.sum(P_glszm / ((ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2)), (1, 2))
    #     # getSmallAreaHighGrayLevelEmphasisFeatureValue
    #     hisae = np.sum(P_glszm * (ivector[None, :, None] ** 2) / (jvector[None, None, :] ** 2), (1, 2))
    #     # getLargeAreaLowGrayLevelEmphasisFeatureValue
    #     lilae = np.sum(P_glszm * (jvector[None, None, :] ** 2) / (ivector[None, :, None] ** 2), (1, 2)) / Nz
    #     # getLargeAreaHighGrayLevelEmphasisFeatureValue
    #     hilae = np.sum(P_glszm * (ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2), (1, 2)) / Nz
    #     return sae, lae, iv, ivn, szv, szvn, zp, glv, zv, ze, lie, hie, lisae, hisae, lilae, hilae

    @cached_property
    def glrlm_features(self):

        """
        Reference: https://pyradiomics.readthedocs.io/en/latest/features.html#module-radiomics.glrlm
        """

        pr = self.glrlm['pr_glrlm']
        pg = self.glrlm['pg_glrlm']
        jvector = self.glrlm['jvector_glrlm']
        ivector = self.glrlm['ivector_glrlm']
        Nr = self.glrlm['Nr_glrlm']

        P_glrlm = self.glrlm['P_glrlm']

        # getShortRunEmphasisFeatureValue
        sre = np.sum((pr / (jvector[None, :, None] ** 2)), 1) / Nr
        sre = np.nanmean(sre, 1)
        # getLongRunEmphasisFeatureValue
        lre = np.sum((pr * (jvector[None, :, None] ** 2)), 1) / Nr
        lre = np.nanmean(lre, 1)
        # getGrayLevelNonUniformityFeatureValue
        gln = np.sum((pg ** 2), 1) / Nr
        gln = np.nanmean(gln, 1)
        # getGrayLevelNonUniformityNormalizedFeatureValue
        glnn = np.sum(pg ** 2, 1) / (Nr ** 2)
        glnn = np.nanmean(glnn, 1)
        # getRunLengthNonUniformityFeatureValue
        rln = np.sum((pr ** 2), 1) / Nr
        rln = np.nanmean(rln, 1)
        # getRunLengthNonUniformityNormalizedFeatureValue
        rlnn = np.sum((pr ** 2), 1) / Nr ** 2
        rlnn = np.nanmean(rlnn, 1)
        # getRunPercentageFeatureValue
        Np = np.sum(pr * jvector[None, :, None], 1)  # shape (Nvox, Na)
        rp = Nr / Np
        rp = np.nanmean(rp, 1)
        # getGrayLevelVarianceFeatureValue
        u_i = np.sum(pg * ivector[None, :, None], 1, keepdims=True)
        glv = np.sum(pg * (ivector[None, :, None] - u_i) ** 2, 1)
        glv = np.nanmean(glv, 1)
        # getRunVarianceFeatureValue
        u_j = np.sum(pr * jvector[None, :, None], 1, keepdims=True)
        rv = np.sum(pr * (jvector[None, :, None] - u_j) ** 2, 1)
        rv = np.nanmean(rv, 1)
        # getRunEntropyFeatureValue
        p_glrlm = P_glrlm / Nr[:, None, None, :]  # divide by Nr to get the normalized matrix
        re = -np.sum(p_glrlm * np.log2(p_glrlm + self.glszm_eps), (1, 2))
        re = np.nanmean(re, 1)
        # getLowGrayLevelRunEmphasisFeatureValue
        lglre = np.sum((pg / (ivector[None, :, None] ** 2)), 1) / Nr
        lglre = np.nanmean(lglre, 1)
        # getHighGrayLevelRunEmphasisFeatureValue
        hglre = np.sum((pg * (ivector[None, :, None] ** 2)), 1) / Nr
        hglre = np.nanmean(hglre, 1)
        # getShortRunLowGrayLevelEmphasisFeatureValue
        srlgle = np.sum((P_glrlm / ((ivector[None, :, None, None] ** 2) * (jvector[None, None, :, None] ** 2))),
                        (1, 2)) / Nr
        srlgle = np.nanmean(srlgle, 1)
        # getShortRunHighGrayLevelEmphasisFeatureValue
        srhgle = np.sum((P_glrlm * (ivector[None, :, None, None] ** 2) / (jvector[None, None, :, None] ** 2)),
                        (1, 2)) / Nr
        srhgle = np.nanmean(srhgle, 1)
        # getLongRunLowGrayLevelEmphasisFeatureValue
        lrlgle = np.sum((P_glrlm * (jvector[None, None, :, None] ** 2) / (ivector[None, :, None, None] ** 2)),
                        (1, 2)) / Nr
        lrlgle = np.nanmean(lrlgle, 1)
        # getLongRunHighGrayLevelEmphasisFeatureValue
        lrhgle = np.sum((P_glrlm * ((jvector[None, None, :, None] ** 2) * (ivector[None, :, None, None] ** 2))),
                        (1, 2)) / Nr
        lrhgle = np.nanmean(lrhgle, 1)

        return sre, lre, gln, glnn, rln, rlnn, rp, glv, rv, re, lglre, hglre, srlgle, srhgle, lrlgle, lrhgle

    @cached_property
    def ngtdm_features(self):

        """
        Reference: https://pyradiomics.readthedocs.io/en/latest/_modules/radiomics/ngtdm.html#RadiomicsNGTDM
        
        """
        p_i = self.ngtdm['p_i_ngtdm']
        s_i = self.ngtdm['s_i_ngtdm']
        i = self.ngtdm['i_ngtdm']
        Npv = self.ngtdm['Npv_ngtdm']
        p_zero = self.ngtdm['p_zero_ngtdm']

        P_ngtdm = self.ngtdm['P_ngtdm']

        # getCoarsenessFeatureValue
        sum_coarse = np.sum(p_i * s_i, 1)
        sum_coarse[sum_coarse != 0] = 1 / sum_coarse[sum_coarse != 0]
        sum_coarse[sum_coarse == 0] = 1e6

        # getContrastFeatureValue
        div = Ngp * (Ngp - 1)
        contrast = (np.sum(p_i[:, :, None] * p_i[:, None, :] * (i[:, :, None] - i[:, None, :]) ** 2, (1, 2)) *
                    np.sum(s_i, 1) / Nvp)
        contrast[div != 0] /= div[div != 0]
        contrast[div == 0] = 0

        # getBusynessFeatureValue
        i_pi = i * p_i
        absdiff = np.abs(i_pi[:, :, None] - i_pi[:, None, :])
        absdiff[p_zero[0], :, p_zero[1]] = 0
        absdiff[p_zero[0], p_zero[1], :] = 0
        absdiff = np.sum(absdiff, (1, 2))
        busyness = np.sum(p_i * s_i, 1)
        busyness[absdiff != 0] = busyness[absdiff != 0] / absdiff[absdiff != 0]
        busyness[absdiff == 0] = 0

        # getComplexityFeatureValue
        pi_si = p_i * s_i
        numerator = pi_si[:, :, None] + pi_si[:, None, :]
        numerator[p_zero[0], :, p_zero[1]] = 0
        numerator[p_zero[0], p_zero[1], :] = 0
        divisor = p_i[:, :, None] + p_i[:, None, :]
        divisor[divisor == 0] = 1  # Prevent division by 0 errors. (Numerator is 0 at those indices too)
        complexity = np.sum(np.abs(i[:, :, None] - i[:, None, :]) * numerator / divisor, (1, 2)) / Nvp

        # getStrengthFeatureValue
        sum_s_i = np.sum(s_i, 1)
        strength = (p_i[:, :, None] + p_i[:, None, :]) * (i[:, :, None] - i[:, None, :]) ** 2
        strength[p_zero[0], :, p_zero[1]] = 0
        strength[p_zero[0], p_zero[1], :] = 0
        strength = np.sum(strength, (1, 2))
        strength[sum_s_i != 0] /= sum_s_i[sum_s_i != 0]
        strength[sum_s_i == 0] = 0

        return sum_coarse, contrast, complexity, strength

    @cached_property
    def gldm_features(self):

        """
        Reference: https://pyradiomics.readthedocs.io/en/latest/_modules/radiomics/gldm.html#RadiomicsGLDM
        
        """

        pd = self.gldm['pd_gldm']
        pg = self.gldm['pg_gdlm']
        jvector = self.gldm['jvector_gldm']
        ivector = self.gldm['ivector_gldm']
        Nz = self.gldm['Nz_gldm']

        P_gldm = self.gldm['P_gldm']

        # getSmallDependenceEmphasisFeatureValue
        sde = np.sum(pd / (jvector[None, :] ** 2), 1) / Nz
        # getLargeDependenceEmphasisFeatureValue
        lre = np.sum(pd * (jvector[None, :] ** 2), 1) / Nz
        # getGrayLevelNonUniformityFeatureValue
        gln = np.sum(pg ** 2, 1) / Nz
        # getDependenceNonUniformityFeatureValue
        dn = np.sum(pd ** 2, 1) / Nz
        # getDependenceNonUniformityNormalizedFeatureValue
        dnn = np.sum(pd ** 2, 1) / Nz ** 2
        # getGrayLevelVarianceFeatureValue
        pg_1 = pg / Nz[:, None]  # divide by Nz to get the normalized matrix
        u_i = np.sum(pg_1 * ivector[None, :], 1, keepdims=True)
        glv = np.sum(pg_1 * (ivector[None, :] - u_i) ** 2, 1)
        # getDependenceVarianceFeatureValue
        pd_1 = pd / Nz[:, None]  # divide by Nz to get the normalized matrix
        u_j = np.sum(pd_1 * jvector[None, :], 1, keepdims=True)
        dv = np.sum(pd_1 * (jvector[None, :] - u_j) ** 2, 1)
        # getDependenceEntropyFeatureValue
        p_gldm = P_gldm / Nz[:, None, None]  # divide by Nz to get the normalized matrix
        de = -np.sum(p_gldm * np.log2(p_gldm + self.glszm_eps), (1, 2))
        # getLowGrayLevelEmphasisFeatureValue
        lgle = np.sum(pg / (ivector[None, :] ** 2), 1) / Nz
        # getHighGrayLevelEmphasisFeatureValue
        hgle = np.sum(pg * (ivector[None, :] ** 2), 1) / Nz
        # getSmallDependenceLowGrayLevelEmphasisFeatureValue
        sdlgle = np.sum(P_gldm / ((ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2)), (1, 2)) / Nz
        # getSmallDependenceHighGrayLevelEmphasisFeatureValue
        sdhgle = np.sum(P_gldm * (ivector[None, :, None] ** 2) / (jvector[None, None, :] ** 2), (1, 2)) / Nz
        # getLargeDependenceLowGrayLevelEmphasisFeatureValue
        ldlgle = np.sum(P_gldm * (jvector[None, None, :] ** 2) / (ivector[None, :, None] ** 2), (1, 2)) / Nz
        # getLargeDependenceHighGrayLevelEmphasisFeatureValue
        ldhgle = np.sum(P_gldm * ((jvector[None, None, :] ** 2) * (ivector[None, :, None] ** 2)), (1, 2)) / Nz

        return sde, lre, gln, dn, dnn, glv, dv, de, lgle, hgle, sdlgle, sdhgle, ldlgle, ldhgle

    # @cached_property
    # def getLargeAreaEmphasisFeatureValue(self):
    #     r"""
    #     **2. Large Area Emphasis (LAE)**
    #
    #     .. math::
    #       \textit{LAE} = \frac{\sum^{N_g}_{i=1}\sum^{N_s}_{j=1}{\textbf{P}(i,j)j^2}}{N_z}
    #
    #     LAE is a measure of the distribution of large area size zones, with a greater value indicative of more larger size
    #     zones and more coarse textures.
    #     """
    #     ps = self.coefficients['ps']
    #     jvector = self.coefficients['jvector']
    #     Nz = self.coefficients['Nz']
    #
    #     lae = np.sum(ps * (jvector[None, :] ** 2), 1) / Nz
    #     return lae
    #
    # @cached_property
    # def getGrayLevelNonUniformityFeatureValue(self):
    #     r"""
    #     **3. Gray Level Non-Uniformity (GLN)**
    #
    #     .. math::
    #       \textit{GLN} = \frac{\sum^{N_g}_{i=1}\left(\sum^{N_s}_{j=1}{\textbf{P}(i,j)}\right)^2}{N_z}
    #
    #     GLN measures the variability of gray-level intensity values in the image, with a lower value indicating more
    #     homogeneity in intensity values.
    #     """
    #     pg = self.coefficients['pg']
    #     Nz = self.coefficients['Nz']
    #
    #     iv = np.sum(pg ** 2, 1) / Nz
    #     return iv
    #
    # @cached_property
    # def getGrayLevelNonUniformityNormalizedFeatureValue(self):
    #     r"""
    #     **4. Gray Level Non-Uniformity Normalized (GLNN)**
    #
    #     .. math::
    #       \textit{GLNN} = \frac{\sum^{N_g}_{i=1}\left(\sum^{N_s}_{j=1}{\textbf{P}(i,j)}\right)^2}{N_z^2}
    #
    #     GLNN measures the variability of gray-level intensity values in the image, with a lower value indicating a greater
    #     similarity in intensity values. This is the normalized version of the GLN formula.
    #     """
    #     pg = self.coefficients['pg']
    #     Nz = self.coefficients['Nz']
    #
    #     ivn = np.sum(pg ** 2, 1) / Nz ** 2
    #     return ivn
    #
    # @cached_property
    # def getSizeZoneNonUniformityFeatureValue(self):
    #     r"""
    #     **5. Size-Zone Non-Uniformity (SZN)**
    #
    #     .. math::
    #       \textit{SZN} = \frac{\sum^{N_s}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j)}\right)^2}{N_z}
    #
    #     SZN measures the variability of size zone volumes in the image, with a lower value indicating more homogeneity in
    #     size zone volumes.
    #     """
    #     ps = self.coefficients['ps']
    #     Nz = self.coefficients['Nz']
    #
    #     szv = np.sum(ps ** 2, 1) / Nz
    #     return szv
    #
    # @cached_property
    # def getSizeZoneNonUniformityNormalizedFeatureValue(self):
    #     r"""
    #     **6. Size-Zone Non-Uniformity Normalized (SZNN)**
    #
    #     .. math::
    #       \textit{SZNN} = \frac{\sum^{N_s}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j)}\right)^2}{N_z^2}
    #
    #     SZNN measures the variability of size zone volumes throughout the image, with a lower value indicating more
    #     homogeneity among zone size volumes in the image. This is the normalized version of the SZN formula.
    #     """
    #     ps = self.coefficients['ps']
    #     Nz = self.coefficients['Nz']
    #
    #     szvn = np.sum(ps ** 2, 1) / Nz ** 2
    #     return szvn
    #
    # @cached_property
    # def getZonePercentageFeatureValue(self):
    #     r"""
    #     **7. Zone Percentage (ZP)**
    #
    #     .. math::
    #       \textit{ZP} = \frac{N_z}{N_p}
    #
    #     ZP measures the coarseness of the texture by taking the ratio of number of zones and number of voxels in the ROI.
    #
    #     Values are in range :math:`\frac{1}{N_p} \leq ZP \leq 1`, with higher values indicating a larger portion of the ROI
    #     consists of small zones (indicates a more fine texture).
    #     """
    #     Nz = self.coefficients['Nz']
    #     Np = self.coefficients['Np']
    #
    #     zp = Nz / Np
    #     return zp
    #
    # @cached_property
    # def getGrayLevelVarianceFeatureValue(self):
    #     r"""
    #     **8. Gray Level Variance (GLV)**
    #
    #     .. math::
    #       \textit{GLV} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_s}_{j=1}{p(i,j)(i - \mu)^2}
    #
    #     Here, :math:`\mu = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_s}_{j=1}{p(i,j)i}`
    #
    #     GLV measures the variance in gray level intensities for the zones.
    #     """
    #     ivector = self.coefficients['ivector']
    #     Nz = self.coefficients['Nz']
    #     pg = self.coefficients['pg'] / Nz[:, None]  # divide by Nz to get the normalized matrix
    #
    #     u_i = np.sum(pg * ivector[None, :], 1, keepdims=True)
    #     glv = np.sum(pg * (ivector[None, :] - u_i) ** 2, 1)
    #     return glv
    #
    # @cached_property
    # def getZoneVarianceFeatureValue(self):
    #     r"""
    #     **9. Zone Variance (ZV)**
    #
    #     .. math::
    #       \textit{ZV} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_s}_{j=1}{p(i,j)(j - \mu)^2}
    #
    #     Here, :math:`\mu = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_s}_{j=1}{p(i,j)j}`
    #
    #     ZV measures the variance in zone size volumes for the zones.
    #     """
    #     jvector = self.coefficients['jvector']
    #     Nz = self.coefficients['Nz']
    #     ps = self.coefficients['ps'] / Nz[:, None]  # divide by Nz to get the normalized matrix
    #
    #     u_j = np.sum(ps * jvector[None, :], 1, keepdims=True)
    #     zv = np.sum(ps * (jvector[None, :] - u_j) ** 2, 1)
    #     return zv
    #
    # @cached_property
    # def getZoneEntropyFeatureValue(self):
    #     r"""
    #     **10. Zone Entropy (ZE)**
    #
    #     .. math::
    #       \textit{ZE} = -\displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_s}_{j=1}{p(i,j)\log_{2}(p(i,j)+\epsilon)}
    #
    #     Here, :math:`\epsilon` is an arbitrarily small positive number (:math:`\approx 2.2\times10^{-16}`).
    #
    #     ZE measures the uncertainty/randomness in the distribution of zone sizes and gray levels. A higher value indicates
    #     more heterogeneneity in the texture patterns.
    #     """
    #     eps = np.spacing(1)
    #     Nz = self.coefficients['Nz']
    #     p_glszm = self.P_glszm / Nz[:, None, None]  # divide by Nz to get the normalized matrix
    #
    #     ze = -np.sum(p_glszm * np.log2(p_glszm + eps), (1, 2))
    #     return ze
    #
    # @cached_property
    # def getLowGrayLevelZoneEmphasisFeatureValue(self):
    #     r"""
    #     **11. Low Gray Level Zone Emphasis (LGLZE)**
    #
    #     .. math::
    #       \textit{LGLZE} = \frac{\sum^{N_g}_{i=1}\sum^{N_s}_{j=1}{\frac{\textbf{P}(i,j)}{i^2}}}{N_z}
    #
    #     LGLZE measures the distribution of lower gray-level size zones, with a higher value indicating a greater proportion
    #     of lower gray-level values and size zones in the image.
    #     """
    #     pg = self.coefficients['pg']
    #     ivector = self.coefficients['ivector']
    #     Nz = self.coefficients['Nz']
    #
    #     lie = np.sum(pg / (ivector[None, :] ** 2), 1) / Nz
    #     return lie
    #
    # def getHighGrayLevelZoneEmphasisFeatureValue(self):
    #     r"""
    #     **12. High Gray Level Zone Emphasis (HGLZE)**
    #
    #     .. math::
    #       \textit{HGLZE} = \frac{\sum^{N_g}_{i=1}\sum^{N_s}_{j=1}{\textbf{P}(i,j)i^2}}{N_z}
    #
    #     HGLZE measures the distribution of the higher gray-level values, with a higher value indicating a greater proportion
    #     of higher gray-level values and size zones in the image.
    #     """
    #     pg = self.coefficients['pg']
    #     ivector = self.coefficients['ivector']
    #     Nz = self.coefficients['Nz']
    #
    #     hie = np.sum(pg * (ivector[None, :] ** 2), 1) / Nz
    #     return hie
    #
    # def getSmallAreaLowGrayLevelEmphasisFeatureValue(self):
    #     r"""
    #     **13. Small Area Low Gray Level Emphasis (SALGLE)**
    #
    #     .. math::
    #       \textit{SALGLE} = \frac{\sum^{N_g}_{i=1}\sum^{N_s}_{j=1}{\frac{\textbf{P}(i,j)}{i^2j^2}}}{N_z}
    #
    #     SALGLE measures the proportion in the image of the joint distribution of smaller size zones with lower gray-level
    #     values.
    #     """
    #     ivector = self.coefficients['ivector']
    #     jvector = self.coefficients['jvector']
    #     Nz = self.coefficients['Nz']
    #
    #     lisae = np.sum(self.P_glszm / ((ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2)), (1, 2)) / Nz
    #     return lisae
    #
    # def getSmallAreaHighGrayLevelEmphasisFeatureValue(self):
    #     r"""
    #     **14. Small Area High Gray Level Emphasis (SAHGLE)**
    #
    #     .. math::
    #       \textit{SAHGLE} = \frac{\sum^{N_g}_{i=1}\sum^{N_s}_{j=1}{\frac{\textbf{P}(i,j)i^2}{j^2}}}{N_z}
    #
    #     SAHGLE measures the proportion in the image of the joint distribution of smaller size zones with higher gray-level
    #     values.
    #     """
    #     ivector = self.coefficients['ivector']
    #     jvector = self.coefficients['jvector']
    #     Nz = self.coefficients['Nz']
    #
    #     hisae = np.sum(self.P_glszm * (ivector[None, :, None] ** 2) / (jvector[None, None, :] ** 2), (1, 2)) / Nz
    #     return hisae
    #
    # def getLargeAreaLowGrayLevelEmphasisFeatureValue(self):
    #     r"""
    #     **15. Large Area Low Gray Level Emphasis (LALGLE)**
    #
    #     .. math::
    #       \textit{LALGLE} = \frac{\sum^{N_g}_{i=1}\sum^{N_s}_{j=1}{\frac{\textbf{P}(i,j)j^2}{i^2}}}{N_z}
    #
    #     LALGLE measures the proportion in the image of the joint distribution of larger size zones with lower gray-level
    #     values.
    #     """
    #     ivector = self.coefficients['ivector']
    #     jvector = self.coefficients['jvector']
    #     Nz = self.coefficients['Nz']
    #
    #     lilae = np.sum(self.P_glszm * (jvector[None, None, :] ** 2) / (ivector[None, :, None] ** 2), (1, 2)) / Nz
    #     return lilae
    #
    # def getLargeAreaHighGrayLevelEmphasisFeatureValue(self):
    #     r"""
    #     **16. Large Area High Gray Level Emphasis (LAHGLE)**
    #
    #     .. math::
    #       \textit{LAHGLE} = \frac{\sum^{N_g}_{i=1}\sum^{N_s}_{j=1}{\textbf{P}(i,j)i^2j^2}}{N_z}
    #
    #     LAHGLE measures the proportion in the image of the joint distribution of larger size zones with higher gray-level
    #     values.
    #     """
    #     ivector = self.coefficients['ivector']
    #     jvector = self.coefficients['jvector']
    #     Nz = self.coefficients['Nz']
    #
    #     hilae = np.sum(self.P_glszm * (ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2), (1, 2)) / Nz
    #     return hilae
