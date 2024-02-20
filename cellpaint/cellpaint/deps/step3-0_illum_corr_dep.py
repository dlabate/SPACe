import time
import tifffile
from tqdm import tqdm
from pathlib import WindowsPath
import matplotlib.pyplot as plt
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import median
from skimage.measure._regionprops import _cached

import multiprocessing as mp
from cellpaint.steps_single_plate.step0_args import Args


"""
Given a 384 well plate, We would like to correct the illumination of the image across all wells.
So here is my work follow:
    1) Find the average image
    2) Find the dilated image  (Does not apply to our case)
    3) Find the smoothed image
    4) Find the scale image    (Does not apply to our case)
"""


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


class IlluminationCorrection:
    """Taken from
    https://github.com/CellProfiler/CellProfiler/blob/master/cellprofiler/modules/correctilluminationcalculate.py
    https://github.com/CellProfiler/CellProfiler/blob/master/cellprofiler/modules/correctilluminationapply.py
    """
    block_size = 60
    object_width = 60
    smoothing_filter_size = object_width / 3.5
    border_size = 50
    num_processes = min(14, mp.cpu_count())
    # sigma = self.smoothing_filter_size
    filter_sigma = max(1, int(smoothing_filter_size + 0.5))
    # print(self.smoothing_filter_size, filter_sigma)
    uint8_max = 256
    uint16_max = 65535

    def __init__(self, args):
        # self.get_mask_img = memory.cache(self.get_mask_img)
        self.args = args
        self.N = len(self.args.img_filepaths)
        # For background, we create a labels image using the block
        # size and find the minimum within each block.
        self.img_dims = tifffile.imread(self.args.img_filepaths[0]).shape[:2]
        self.labels, self.indexes = self.block_fn(self.img_dims, (self.block_size, self.block_size))
        self.strel = self.strel_disk(self.filter_sigma)

    def run_single(self, img_path):
        s_time = time.time()
        orig_image = tifffile.imread(img_path)
        # mask_img = self.get_mask_img(orig_image)
        bg_img = self.step0_preprocess_image_for_averaging(orig_image)
        illum_function_img = self.step1_apply_smoothing(bg_img)
        output_img = self.step2_apply(orig_image, illum_function_img)

        print(f"Time take for illum correction in seconds: {time.time() - s_time}")
        print("orig_img", orig_image.shape, orig_image.dtype, np.amin(orig_image), np.amax(orig_image))
        print("bg_img", bg_img.shape, bg_img.dtype, np.amin(bg_img), np.amax(bg_img))
        print("illum_function_img", illum_function_img.shape, illum_function_img.dtype,
              np.amin(illum_function_img),
              np.amax(illum_function_img))
        print("output_img", output_img.dtype, np.amin(output_img), np.amax(output_img))

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        axes[0, 0].imshow(orig_image, cmap="gray")
        axes[0, 1].imshow(bg_img, cmap="gray")
        axes[1, 0].imshow(illum_function_img, cmap="gray")
        axes[1, 1].imshow(output_img, cmap="gray")
        plt.show()
        _, well_id, fov, channel = Args.sort_key_for_imgs(img_path, "to_sort_channels", self.args.plate_protocol)
        np.save(self.args.step1_save_path/f"{well_id}_{fov}_{channel}.npy", illum_function_img)

    def run_multi(self, ):
        with mp.Pool(processes=self.num_processes) as pool:
            for _ in tqdm(pool.imap_unordered(self.run_single, self.args.img_filepaths), total=self.N):
                pass

    def get_mask_img(self, img):
        mask_img = np.zeros_like(img, dtype=bool)
        mask_img[:self.border_size, :] = 1
        mask_img[:, :self.border_size] = 1
        mask_img[img.shape[0] - self.border_size:, :] = 1
        mask_img[:, img.shape[0] - self.border_size:] = 1
        return mask_img

    def step0_preprocess_image_for_averaging(self, orig_image):
        # Here we are assuming the original image has mask!!!
        """Taken from """

        """Create a version of the image appropriate for averaging"""

        # if orig_image.has_mask:
        #     labels[~orig_image.mask] = -1

        min_block = np.zeros(orig_image.shape, dtype=np.float32)
        minima = self.fixup_scipy_ndimage_result(ndi.minimum(orig_image, self.labels, self.indexes))
        min_block[self.labels != -1] = minima[self.labels[self.labels != -1]]

        # print(labels.dtype, labels.shape, np.unique(labels))
        # print(indexes.dtype, indexes.shape)
        # print(minima.dtype, minima.shape)
        print(min_block.dtype, min_block.shape, np.amin(min_block), np.amax(min_block),
              np.percentile(min_block, (25, 50, 65, 75, 90, 95, 99)))
        return min_block

    def step1_apply_smoothing(self, pixel_data, ):

        # rescaled_pixel_data = pixel_data * 65535
        # rescaled_pixel_data = rescaled_pixel_data.astype(np.uint16)
        # rescaled_pixel_data *= pixel_data.mask
        # output_pixels = median(rescaled_pixel_data, self.strel, behavior="rank")
        pixel_data = pixel_data.astype(np.uint16)
        output_pixels = median(pixel_data, self.strel, behavior="rank")
        return output_pixels

    def step2_apply(self, orig_image, illum_function_pixel_data):
        # rescaled_pixel_data = (illum_function_pixel_data / self.uint8_max) * self.uint16_max
        rescaled_pixel_data = illum_function_pixel_data * self.uint16_max
        rescaled_pixel_data = rescaled_pixel_data.astype(np.uint16)
        print("rescaled_pixel_data: ", rescaled_pixel_data.dtype,
              np.amin(rescaled_pixel_data), np.amax(rescaled_pixel_data))
        output_pixels = orig_image - rescaled_pixel_data
        print("output_pixels before: ", output_pixels.dtype, np.amin(output_pixels), np.amax(output_pixels),
              np.percentile(output_pixels, (10, 25, 50, 75, 90, 95, 99)))
        output_pixels[output_pixels < 0] = 0
        print("output_pixels after: ", output_pixels.dtype, np.amin(output_pixels), np.amax(output_pixels),
              np.percentile(output_pixels, (10, 25, 50, 75, 90, 95, 99)))
        return output_pixels

    @staticmethod
    def fixup_scipy_ndimage_result(whatever_it_returned):
        """
        Taken from https://github.com/CellProfiler/centrosome/blob/master/centrosome/cpmorphology.py

            Convert a result from scipy.ndimage to a numpy array

            scipy.ndimage has the annoying habit of returning a single, bare
            value instead of an array if the indexes passed in are of length 1.
            For instance:
            scind.maximum(image, labels, [1]) returns a float
            but
            scind.maximum(image, labels, [1,2]) returns a list
        """
        if getattr(whatever_it_returned, "__getitem__", False):
            return np.array(whatever_it_returned)
        else:
            return np.array([whatever_it_returned])

    @staticmethod
    @lru_cache(maxsize=1)
    def block_fn(shape, block_shape):
        """
        Taken from https://github.com/CellProfiler/centrosome/blob/master/centrosome/cpmorphology.py

            Create a labels image that divides the image into blocks

            shape - the shape of the image to be blocked
            block_shape - the shape of one block

            returns a labels matrix and the indexes of all labels generated

            The idea here is to block-process an image by using SciPy label
            routines. This routine divides the image into blocks of a configurable
            dimension. The caller then calls scipy.ndimage functions to process
            each block as a labeled image. The block values can then be applied
            to the image via indexing. For instance:

            labels, indexes = block(image.shape, (60,60))
            minima = scind.minimum(image, labels, indexes)
            img2 = image - minima[labels]
        """
        shape = np.array(shape)
        block_shape = np.array(block_shape)
        i, j = np.mgrid[0: shape[0], 0: shape[1]]
        ijmax = (shape.astype(float) / block_shape.astype(float)).astype(int)
        ijmax = np.maximum(ijmax, 1)
        multiplier = ijmax.astype(float) / shape.astype(float)
        i = (i * multiplier[0]).astype(int)
        j = (j * multiplier[1]).astype(int)
        labels = i * ijmax[1] + j
        indexes = np.array(list(range(np.product(ijmax))))
        return labels, indexes

    @staticmethod
    @lru_cache(maxsize=1)
    def strel_disk(radius):
        """
        Taken from https://github.com/CellProfiler/centrosome/blob/master/centrosome/cpmorphology.py

            Create a disk structuring element for morphological operations

            radius - radius of the disk
        """
        iradius = int(radius)
        x, y = np.mgrid[-iradius: iradius + 1, -iradius: iradius + 1]
        radius2 = radius * radius
        strel = np.zeros(x.shape)
        strel[x * x + y * y <= radius2] = 1
        # np.set_printoptions(linewidth=200)
        # cond = x * x + y * y <= radius2
        # print(f"iradius: {iradius}")
        # print(f"x: {x}")
        # print(f"y: {y}")
        # print(f"radius2: {radius2}")
        # print(cond)
        # print(strel)
        return strel


if __name__ == "__main__":
    camii_server_flav = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid"
    camii_server_seema = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Seema"
    camii_server_jump_cpg0012 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0012"
    camii_server_jump_cpg0001 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0001"

    main_path = WindowsPath(camii_server_flav)
    exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"

    args = Args(experiment=exp_fold, main_path=main_path, mode="full").args
    args = set_mancini_datasets_hyperparameters(args)
    myclass = IlluminationCorrection(args)
    for ii in range(20):
        print(myclass.args.img_filepaths[ii].stem)
        myclass.run_single(myclass.args.img_filepaths[ii])
        print('\n')
    # myclass.run_multi()
    # IlluminationCorrection.strel_disk(radius=9)
