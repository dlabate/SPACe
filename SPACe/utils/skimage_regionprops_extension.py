import numpy as np
from numpy import linalg as LA
from scipy import ndimage as ndi
from scipy.stats import median_abs_deviation

from skimage.measure import _moments, find_contours
from skimage.measure._regionprops import RegionProperties, _cached, only2d
from skimage.feature import graycomatrix
from skimage.segmentation import find_boundaries
from pyefd import elliptic_fourier_descriptors

from functools import cached_property

TEXTURE_PERCENTILES = (25, 50, 75)
TEXTURE_CATEGORIES = ["Contrast", "Dissimilarity", "Homogeneity", "Energy", "Correlation"]
TEXTURE_SUMM_STATISTICS = [f"{it}%" for it in TEXTURE_PERCENTILES] + ["mean", "std", "mad"]
TEXTURE_FEATURE_NAMES = [f"{it0}_{it1}" for it0 in TEXTURE_CATEGORIES for it1 in TEXTURE_SUMM_STATISTICS]


def safe_log_10_v0(value):
    """https://stackoverflow.com/questions/21610198/runtimewarning-divide-by-zero-encountered-in-log"""
    value = np.abs(value)
    result = np.where(value > 1e-12, value, -12)
    # print(result)
    res = np.log10(result, out=result, where=result > 1e-12)
    return res


def safe_log_10_v1(value):
    """Pankaj"""
    return -np.log(1+np.abs(value))


def regionprops(w0_mask, w1_mask, w2_mask, w4_mask, img, n_levels):
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
            w2_props = RegionPropertiesExtension(w2_objects[ii], label, w2_mask, img[2],n_levels)
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

    n_levels = 8
    n_pos_pixels_lb = 10
    corr_tolerance = 1e-8
    distances = np.arange(1, 21)
    angles = np.array([0, np.pi/2])
    angles_str = [0, "pi/2"]
    intensity_percentiles = (10, 25, 75, 90)

    def __init__(self, slice, label, label_image, intensity_image, channel_name,
                 cache_active=True, ):
        super().__init__(slice, label, label_image, intensity_image, cache_active)

        # The number of levels found within an image may not be the same as
        # the total number of globally available levels predefined/preset by us.
        self.channel_name = channel_name
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
        I = np.array(range(0, self.n_levels)).reshape((self.n_levels, 1, 1, 1))
        J = np.array(range(0, self.n_levels)).reshape((1, self.n_levels, 1, 1))
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
        if len(self.image_intensity_vec) < self.n_pos_pixels_lb:
            # print(self.channel_name, self.image_intensity.shape, len(self.image_intensity_vec))
            return (0, )*(len(self.intensity_percentiles)+4)
        percentiles = np.nanpercentile(self.image_intensity_vec, self.intensity_percentiles)
        intensity_median, intensity_mad, intensity_mean, intensity_std = \
            np.nanmedian(self.image_intensity_vec), median_abs_deviation(self.image_intensity_vec), \
            np.nanmean(self.image_intensity_vec), np.nanstd(self.image_intensity_vec)
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
    def glcm_features(self,):
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
        contrast = np.sum(self.glcm * self.weights0, axis=(0, 1))
        dissimilarity = np.sum(self.glcm * self.weights1, axis=(0, 1))
        homogeneity = np.sum(self.glcm * self.weights2, axis=(0, 1))
        # contrast = np.einsum(self.einsum_instruct_2, self.glcm, self.weights0)
        # dissimilarity = np.einsum(self.einsum_instruct_2, self.glcm, self.weights1)
        # homogeneity = np.einsum(self.einsum_instruct_2, self.glcm, self.weights2)
        ###################################################
        # asm and energy
        # asm = np.sum(self.glcm ** 2, axis=(0, 1))
        # energy = np.sqrt(asm)
        energy = LA.norm(self.glcm, ord='fro', axis=(0, 1))
        ##############################################
        # correlation
        correlation = np.zeros((num_dist, num_angle), dtype=np.float32)
        diff_i = self.corr_I - np.sum(self.corr_I * self.glcm, axis=(0, 1))
        diff_j = self.corr_J - np.sum(self.corr_J * self.glcm, axis=(0, 1))
        std_i = np.sqrt(np.sum(self.glcm * (diff_i ** 2), axis=(0, 1)))
        std_j = np.sqrt(np.sum(self.glcm * (diff_j ** 2), axis=(0, 1)))
        cov = np.sum(self.glcm * (diff_i * diff_j), axis=(0, 1))

        # diff_i = self.corr_I - np.einsum(self.einsum_instruct_2, self.corr_I, self.glcm)
        # diff_j = self.corr_J - np.einsum(self.einsum_instruct_2, self.corr_J, self.glcm)
        # std_i = np.sqrt(np.einsum(self.einsum_instruct_2, self.glcm, diff_i**2))
        # std_j = np.sqrt(np.einsum(self.einsum_instruct_2, self.glcm, diff_j**2))
        # cov = np.einsum(self.einsum_instruct_3, self.glcm, diff_i, diff_j)

        # handle the special case of standard deviations near zero
        mask_0 = std_i < self.corr_tolerance
        mask_0[std_j < self.corr_tolerance] = True
        correlation[mask_0] = 1
        # handle the standard case
        mask_1 = ~mask_0
        correlation[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
        ####################################################################################
        # These are way too many features:
        # 1) They take a long time to load
        # 2) They can't fit into a small GPU
        # haralick_features = tuple(np.vstack((contrast, dissimilarity, homogeneity, energy, correlation)).reshape(-1))
        # return haralick_features
        ####################################################################################
        # I needed to manage the program,
        # so instead I decided to use and store the haralick summary statistics instead!!!!
        #####################################################################################
        contrast = tuple(np.percentile(contrast, q=TEXTURE_PERCENTILES)) + \
                   (np.mean(contrast), np.std(contrast), median_abs_deviation(contrast, axis=None), )
        dissimilarity = tuple(np.percentile(dissimilarity, q=TEXTURE_PERCENTILES)) + \
                        (np.mean(dissimilarity), np.std(dissimilarity), median_abs_deviation(dissimilarity, axis=None),)
        homogeneity = tuple(np.percentile(homogeneity, q=TEXTURE_PERCENTILES)) + \
                      (np.mean(homogeneity), np.std(homogeneity), median_abs_deviation(homogeneity, axis=None),)
        energy = tuple(np.percentile(energy, q=TEXTURE_PERCENTILES)) + \
                 (np.mean(energy), np.std(energy), median_abs_deviation(energy, axis=None),)
        correlation = tuple(np.percentile(correlation, q=TEXTURE_PERCENTILES)) + \
                      (np.mean(correlation), np.std(correlation), median_abs_deviation(correlation, axis=None),)
        return contrast+dissimilarity+homogeneity+energy+correlation

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
    def circularity(self, ):
        if self.perimeter > 1e-6:
            return (4 * np.pi * self.area) / self.perimeter ** 2
        else:
            return np.nan