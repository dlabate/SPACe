import time

# import torch
import tifffile
import pandas as pd
from tqdm import tqdm
from pathlib import WindowsPath
from functools import lru_cache, partial
import multiprocessing as mp

import cv2
import numpy as np
from skimage.exposure import rescale_intensity, equalize_adapthist
# from skimage.filters import median
# from skimage.measure._regionprops import _cached
# from skimage.restoration import rolling_ball, ball_kernel
# from skimage.measure import label
# from skimage.morphology import remove_small_holes, remove_small_objects, \
#     binary_dilation, binary_erosion, binary_closing, isotropic_closing, dilation, erosion, flood_fill,\
#     disk, square, white_tophat
# from scipy import ndimage as ndi

from cellpaint.steps_single_plate.step0_args import BaseConstructor, Args
from cellpaint.utils.shared_memory import MyBaseManager, TestProxy, create_shared_np_num_arr
# from cellpose import models

import matplotlib.pyplot as plt


class IntensityPreProfiler(BaseConstructor):
    analysis_step = 1

    ball_size = 75
    intensity_percentiles = (1, 10, 25, 50, .75, .9, .95, .99, .999)
    rescale_intensity_range = (5, 99)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ball_size, ball_size))
    meta_cols = ["channel", "exp-id", "well-id", "fov", "treatment", "cell-line", "density", "dosage", "other"]
    stages = ["0-raw-image", "1-rescaled-image", "2-bgsub-image"]
    num_workers = min(mp.cpu_count(), 15)

    def __init__(self, args):
        BaseConstructor.__init__(self)
        self.args = args
        self.n_images = len(self.args.img_filepaths)
        self.feature_cols = ["mean-intensity", "std-intensity",] + [f"{it}-th%" for it in self.intensity_percentiles]
        self.feature_cols = [f"{it1}_{it2}" for it1 in self.stages for it2 in self.feature_cols]
        self.all_cols = self.meta_cols + self.feature_cols

        self.n_feat_cols = len(self.feature_cols)
        self.n_meta_cols = len(self.meta_cols)

        self.metadata_profile = np.zeros((self.n_images, self.n_meta_cols), dtype=object)
        self.feature_profile = create_shared_np_num_arr((self.n_images, self.n_feat_cols), c_dtype="c_float")

    @lru_cache(maxsize=10)
    def get_features(self, filepath):
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        profile1 = self.get_intensity_profile(img)

        prcs = tuple(np.percentile(img, self.rescale_intensity_range))
        img = rescale_intensity(img, in_range=prcs)
        profile2 = self.get_intensity_profile(img)

        img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, self.kernel)
        profile3 = self.get_intensity_profile(img)
        return profile1 + profile2 + profile3

    @lru_cache(maxsize=10)
    def get_metadata_and_features(self, filepath):
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        profile1 = self.get_intensity_profile(img)

        prcs = tuple(np.percentile(img, self.rescale_intensity_range))
        img = rescale_intensity(img, in_range=prcs)
        profile2 = self.get_intensity_profile(img)

        img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, self.kernel)
        profile3 = self.get_intensity_profile(img)
        metadata = self.get_metadata(filepath, self.args)

        return metadata, profile1 + profile2 + profile3

    def get_intensity_profile(self, intensity_image, ):
        mean_intensity = np.mean(intensity_image)
        std_intensity = np.std(intensity_image)
        quantiles = np.percentile(intensity_image, q=self.intensity_percentiles)
        return (mean_intensity, std_intensity, ) + tuple(quantiles)

    def run_single_process(self):
        for ii, filepath in tqdm(enumerate(self.args.img_filepaths[0:40])):
            out = self.get_metadata_and_features(filepath)
            # print(type(out), type(out[0]), type(out[1]))
            # print(len(out), len(out[0]), len(out[1]))
            self.metadata_profile[ii], self.feature_profile[ii] = out

        # saving
        self.profiles = np.concatenate((self.metadata_profile, self.feature_profile), axis=1)
        self.profiles = pd.DataFrame(self.profiles, columns=self.all_cols)
        self.profiles.to_csv(self.args.step1_save_path/"preprocessing-intensity-profile.csv",
                             index=False, float_format="%.2f")

    def run_multi_process(self):
        with mp.Pool(processes=self.num_workers) as pool:
            for ii, (meta_profile, feat_profile) in tqdm(
                    enumerate(pool.imap(self.get_metadata_and_features, self.args.img_filepaths)),
                    total=self.n_images):
                self.metadata_profile[ii], self.feature_profile[ii] = meta_profile, feat_profile
        self.profiles = np.concatenate((self.metadata_profile, self.feature_profile), axis=1)
        self.profiles = pd.DataFrame(self.profiles, columns=self.all_cols)
        self.profiles.to_csv(
            self.args.step1_save_path/"preprocessing-intensity-profile.csv",
            index=False,
            float_format="%.2f")


def run_single_process(args):
    inst = IntensityPreProfiler(args)
    for ii, filepath in tqdm(enumerate(inst.args.img_filepaths[0:40])):
        out = inst.get_features(filepath)
        # print(type(out), type(out[0]), type(out[1]))
        # print(len(out), len(out[0]), len(out[1]))
        inst.metadata_profile[ii], inst.feature_profile[ii] = out

    # saving
    inst.profiles = np.concatenate((inst.metadata_profile, inst.feature_profile), axis=1)
    inst.profiles = pd.DataFrame(inst.profiles, columns=inst.all_cols)
    inst.profiles.to_csv(inst.args.step1_save_path/"preprocessing-intensity-profile.csv",
                         index=False, float_format="%.2f")


def run_multi_process(args):
    MyManager = MyBaseManager()
    # register the custom class on the custom manager
    MyManager.register(IntensityPreProfiler.__name__, IntensityPreProfiler, TestProxy)
    # create a new manager instance
    with MyManager as manager:
        inst = getattr(manager, IntensityPreProfiler.__name__)(args)
        # get metadata and features together (pool.imap is a must!!!)
        with mp.Pool(processes=inst.num_workers) as pool:
            for ii, (meta_prof, feature_prof) in tqdm(
                    enumerate(pool.imap(inst.get_metadata_and_features, inst.args.img_filepaths)),
                    total=inst.n_images):
                inst.metadata_profile[ii], inst.feature_profile[ii] = meta_prof, feature_prof

        # save both to a dataframe
        inst.profiles = np.concatenate((inst.metadata_profile, inst.feature_profile), axis=1)
        inst.profiles = pd.DataFrame(inst.profiles, columns=inst.all_cols)
        inst.profiles.to_csv(
            inst.args.step0_save_path / "preprocessing-intensity-profile.csv",
            index=False,
            float_format="%.2f")