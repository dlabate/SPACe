import re
import argparse
import warnings
from sys import platform
from pathlib import WindowsPath, PosixPath, Path
from string import ascii_lowercase, ascii_uppercase

import cv2
import math
import xlrd
import string
import random
import numbers
import tifffile
import itertools
import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity
import skimage.io as sio

# TODO: Add proper documentation for all the different class
# TODO: Add a README file that explains the steps,
#  the purpose behind each step, as well as the arguments that can be changed


def sort_key_for_imgs(file_path, sort_purpose, plate_protocol):
    """
    Get sort key from the img filename.
    The function is used to sort image filenames for a specific experiment taken with a specific plate_protocol.
    """
    # plate_protocol = plate_protocol.lower()
    if plate_protocol == "greiner" or plate_protocol == "perkinelmer":
        """img filename example:
        .../AssayPlate_PerkinElmer_CellCarrier-384/
        AssayPlate_PerkinElmer_CellCarrier-384_A02_T0001F001L01A01Z01C02.tif
        .../AssayPlate_Greiner_#781896/
        AssayPlate_Greiner_#781896_A04_T0001F001L01A01Z01C01.tif
        """
        folder = file_path.parents[1].stem  # "AssayPlate_PerkinElmer_CellCarrier-384"
        filename = file_path.stem  # AssayPlate_PerkinElmer_CellCarrier-384_A02_T0001F001L01A01Z01C02.tif
        split = filename.split("_")
        well_id = split[-2]  # A02
        fkey = split[-1]  # T0001F001L01A01Z01C02
        inds = [fkey.index(ll, 0, len(fkey)) for ll in fkey if ll.isalpha()]
        inds.append(None)
        timestamp, fov, id2, x1, zslice, channel = \
            [fkey[inds[ii]:inds[ii + 1]] for ii in range(len(inds) - 1)]
        # print(well_id, parts)

    elif plate_protocol == "combchem":
        """img filename example:
        .../P000025-combchem-v3-U2OS-24h-L1-copy1/
        P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s5_w3C00578CF-AD4A-4A29-88AE-D2A2024F9A92.tif"""
        folder = file_path.parents[1].stem
        filename = file_path.stem
        split = filename.split("_")
        well_id = split[1]
        fov = split[2][1]
        channel = split[3][1]
        # print(well_id, parts)

    elif "cpg0012" in plate_protocol:
        """img filename example:
        .../P000025-combchem-v3-U2OS-24h-L1-copy1/
        P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s5_w3C00578CF-AD4A-4A29-88AE-D2A2024F9A92.tif"""
        folder = file_path.parents[1].stem
        filename = file_path.stem
        split = filename.split("_")
        well_id = split[1].upper()
        fov = split[2].upper()
        channel = split[3][0:2]

    elif "cpg0001" in plate_protocol:
        """img filename example:
        .../P000025-combchem-v3-U2OS-24h-L1-copy1/
        P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s5_w3C00578CF-AD4A-4A29-88AE-D2A2024F9A92.tif"""
        folder = file_path.parents[1].stem
        filename = file_path.stem
        split = filename.split("-")
        row, col = int(split[0][1:3]) - 1, split[0][4:6]
        well_id = f"{ascii_uppercase[row]}{col}"
        fov = split[0][6:9]
        channel = split[1][2]
        
    elif "cpgmoa" in plate_protocol:
        """img filename example:
        .../BR00115125/
        BR00115125_A01_T0001F001L01A01Z01C05.tif"""
        folder = file_path.parents[1].stem
        
        filename = file_path.stem
        split = filename.split("_")
        well_id = split[-2]  
        fkey = split[-1]  # T0001F001L01A01Z01C02
        inds = [fkey.index(ll, 0, len(fkey)) for ll in fkey if ll.isalpha()]
        
        inds.append(None)
        timestamp, fov, id2, x1, zslice, channel = \
            [fkey[inds[ii]:inds[ii + 1]] for ii in range(len(inds) - 1)]
        print("folder, well_id, fov, channel", folder, well_id, fov, channel)
        # print(well_id, parts)
    
        
    elif "cimini" in plate_protocol:
        """img filename example:
        BR00120531__2021-02-26T09_23_20-Measurement2/rXXcXXfXXp01-chXXsk1fk1fl1.tiff
            rXX is the row number of the well that was imaged. rXX ranges from r01 to r16.
            cXX is the column number of the well that was imaged. cXX ranges from c01 to c24.
            fXX corresponds to the site that was imaged. fXX ranges from f01 to f09.
            chXX corresponds to the fluorescent channels imaged """
        
        filename = file_path.stem
        number_channel = int(filename[15:16])  
        
        folder = file_path.parents[1].stem  
        well_info = filename[0:6] 
        well_row = chr(int(well_info[1:3]) + 64)  # Convert row number to letter
        well_column = well_info[4:6]  # Extract column number
        well_id = well_row + well_column  
        
        channel = "C0" + str(number_channel)
        number_fov = filename[7:9]  
        fov = "F" + number_fov
        
    else:
        raise NotImplementedError(f"{plate_protocol} is not implemented yet!!!")

    if sort_purpose == "to_sort_channels":
        return folder, well_id, fov, channel

    elif sort_purpose == "to_group_channels":
        # print(folder, well_id, fov)
        return folder, well_id, fov

    elif sort_purpose == "to_match_it_with_mask_path":
        return f"{well_id}_{fov}"
    elif sort_purpose == "to_get_well_id":
        return well_id
    elif sort_purpose == "to_get_well_id_and_fov":
        return well_id, fov
    else:
        raise ValueError(f"sort purpose {sort_purpose} does not exist!!!")


def sort_key_for_masks(mask_path):
    """example: .../w0_A02_F001.png"""
    split = mask_path.stem.split("_")
    well_id = split[0]
    fov = split[1]
    # return f"{well_id}_{fov}"
    return well_id, fov


def set_mask_save_name(well_id, fov, channel_index):
    return f"{well_id}_{fov}_W{channel_index+1}.png"


def get_img_channel_groups(args):
    """
    sort all tiff file in the args.main_path/args.experiment/args.plate_protocol folder.
     Then group them in such a way that all the tiff file which belong to the same image,
     the 4/5 channels of the same image, go to the same group.
     """
    # ################################################################
    # group files that are channels of the same image together.
    # f"{ASSAY}_A02_T0001F002L01A02Z01C01.tif"
    # to group channels we need key = (folder, well_id, fov)
    filename_keys, img_path_groups = [], []
    for key, grp in itertools.groupby(
            args.img_filepaths,
            key=lambda x: sort_key_for_imgs(x, "to_group_channels", args.plate_protocol)):
        filename_keys.append(key)
        img_path_groups.append(list(grp))
    # filename_keys = np.array(filename_keys, dtype=object)
    # img_path_groups = np.array(img_path_groups, dtype=object)

    N = len(img_path_groups)
    return filename_keys, img_path_groups, N


def containsLetterAndNumber(input):
    """https://stackoverflow.com/questions/64862663/how-to-check-if-a-string-is-strictly-
    contains-both-letters-and-numbers"""
    return input.isalnum() and not input.isalpha() and not input.isdigit()


def shorten_str(name, key, max_chars):
    name = name.lower()[0:max_chars]
    n = 7
    if key == "treatment":
        return name if len(name) <= n else '\n'.join([name[i:i+n] for i in range(0, len(name), n)])
    elif key == "cell-line":
        return '\n'.join(name.split('-'))
    else:
        NotImplementedError("")


def get_metadata(self, image_filename):

    # extract metadata
    folder, well_id, fov, channel = sort_key_for_imgs(
        image_filename,
        sort_purpose="to_sort_channels",
        plate_protocol=self.args.plate_protocol)
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

    return channel, self.args.experiment, well_id, fov, treatment, cell_line, density, dosage, other


def load_img(img_path_group, args):
    """Load all image channels from img_path_group and apply necessary the pre-processing steps.
        img_path_group: list of 5 pathlib for w1/w2/w3/w4/w5 channels.

        returns:
        img: a numpy uint16 array of shape (5, height, width) containing the 5 image channels
    """
    # read images from file
    w1_img = tifffile.imread(img_path_group[args.nucleus_idx])[np.newaxis]
    w2_img = tifffile.imread(img_path_group[args.cyto_idx])[np.newaxis]
    w3_img = tifffile.imread(img_path_group[args.nucleoli_idx])[np.newaxis]
    w4_img = tifffile.imread(img_path_group[args.actin_idx])[np.newaxis]
    w5_img = tifffile.imread(img_path_group[args.mito_idx])[np.newaxis]

    # get the rescale intensity percentiles
    w1_in_range = tuple(np.percentile(w1_img, args.rescale_intensity_bounds["w1"]))
    w2_in_range = tuple(np.percentile(w2_img, args.rescale_intensity_bounds["w2"]))
    w3_in_range = tuple(np.percentile(w3_img, args.rescale_intensity_bounds["w3"]))
    w4_in_range = tuple(np.percentile(w4_img, args.rescale_intensity_bounds["w4"]))
    w5_in_range = tuple(np.percentile(w5_img, args.rescale_intensity_bounds["w5"]))

    if args.bg_sub:
        # apply background subtraction using a tophat filter
        w1_img = cv2.morphologyEx(w1_img, cv2.MORPH_TOPHAT, args.bgsub_kernels["w1"])
        w2_img = cv2.morphologyEx(w2_img, cv2.MORPH_TOPHAT, args.bgsub_kernels["w2"])
        w3_img = cv2.morphologyEx(w3_img, cv2.MORPH_TOPHAT, args.bgsub_kernels["w3"])
        w4_img = cv2.morphologyEx(w4_img, cv2.MORPH_TOPHAT, args.bgsub_kernels["w4"])
        w5_img = cv2.morphologyEx(w5_img, cv2.MORPH_TOPHAT, args.bgsub_kernels["w5"])

    # apply the rescale intensity using percentiles
    w1_img = rescale_intensity(w1_img, in_range=w1_in_range)
    w2_img = rescale_intensity(w2_img, in_range=w2_in_range)
    w3_img = rescale_intensity(w3_img, in_range=w3_in_range)
    w4_img = rescale_intensity(w4_img, in_range=w4_in_range)
    w5_img = rescale_intensity(w5_img, in_range=w5_in_range)

    img = np.concatenate([w1_img, w2_img, w3_img, w4_img, w5_img], axis=0)
    return img


class Args(object):
    """
    creates self.args namespace that takes in all the constants necessary to perform the cell-paint analysis
    """
    analysis_step = 0

    # TODO: remove dependency on the OS for pathlib path objects.
    def __init__(
            self,
            experiment,
            main_path="F:\\CellPainting",
            mode="preview",
            step2_segmentation_algorithm="w1=cellpose_w2=cellpose",
            analysis_part=1,
            organelles=("Nucleus", "Cyto", "Nucleoli", "Actin", "Mito"),  # the name assigned to each channel
            ####################################################################
            # hyperparameters/constants used in Cellpaint Step 0
            #######################################################################
            nucleus_idx=0,
            cyto_idx=1,
            nucleoli_idx=2,
            actin_idx=3,
            mito_idx=4,
            max_chars=21,
            n_fovs_per_well=6,
            #######################################################################
            w1_intensity_bounds=(20, 99.9),
            w2_intensity_bounds=(20, 99.9),
            w3_intensity_bounds=(20, 99.9),
            w4_intensity_bounds=(20, 99.9),
            w5_intensity_bounds=(20, 99.9),
            ########################################################################
            w1_bg_rad=50,
            w2_bg_rad=100,
            w3_bg_rad=50,
            w4_bg_rad=100,
            w5_bg_rad=100,
            # hyperparameters/constants used in Cellpaint Step 1
            #######################################################################
            cellpose_nucleus_diam=100,
            cellpose_cyto_diam=150,
            cellpose_batch_size=64,
            cellpose_model_type="cyto2",
            w1_min_size=600,
            w2_min_size=700,
            w3_min_size=40,
            w5_min_size=200,
            # hyperparameters/constants used in Cellpaint Step 2
            #######################################################
            multi_nucleus_dist_thresh=40,
            min_nucleoli_size_multiplier=.005,
            max_nucleoli_size_multiplier=.3,
            nucleoli_bd_area_to_nucleoli_area_threshold=.2,
            w3_local_rescale_intensity_ub=99.2,
            w5_local_rescale_intensity_ub=99.9,
            #######################################################################
            # hyperparameters/constants used in Cellpaint Step 5
            min_fov_cell_count=1,
            distmap_batch_size=64,
    ):
        """
            experiment:
                The name of the experiment folder to perform cellpaint analysis on
            main_path:
                The name of the mother folder where the experiment data (images and platemap) are located
            mode:
                whether to run the analysis steps_single_plate (.py files) in test mode, or preview mode,
                or full mode.
                args.mode="test" or args.mode="preview" or
                args.mode="full" should be used when the code is already debugged
                and we can use the multiproccessing pool during segmentation/step2
                and feature extraction/step3 for speed up

            nucleus_idx:   Nucleus channel index in the tif image(~w1)
            cyto_idx:      Cyto channel index in the tif image(~w2)
            nucleoli_idx:  Nucleoli channel index in the tif image(~w3)
            actin_idx:     Actin channel index in the tif image(~w4)
            mito_idx:      Mito channel index in the tif image(~w5)
        """

        # # TODO: Fix this because can have args passing in through terminal, with the current implementation.
        # for all the steps_single_plate
        parser = argparse.ArgumentParser(description='Cell-paint')
        # # the order of the channel for images taken by fabio at baylor is as follows:
        # # w1: Nucleus channel, w2: cyto channel, w3: nucleoli channel, w4: actin channel, w5: mito channel
        """
        https://stackoverflow.com/questions/48796169/
        how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter
        This way of using argparse parser works with jupyter-notebook as well: self.args, _ = parser.parse_known_args()
        """
        self.args, _ = parser.parse_known_args()
        self.args.mode = mode
        self.args.experiment = experiment
        self.args.main_path = WindowsPath(main_path)
        self.args.step2_segmentation_algorithm = step2_segmentation_algorithm
        self.args.show_masks = True
        # Cellpaint Step 0) that can be tuned to the plate and image dimensions
        #######################################################################
        self.args.analysis_part = analysis_part
        self.args.nucleus_idx = nucleus_idx
        self.args.cyto_idx = cyto_idx
        self.args.nucleoli_idx = nucleoli_idx
        self.args.actin_idx = actin_idx
        self.args.mito_idx = mito_idx
        self.args.n_fovs_per_well = n_fovs_per_well
        self.args.max_chars = max_chars
        #######################################################################
        self.args.rescale_intensity_bounds = {
            "w1": w1_intensity_bounds,
            "w2": w2_intensity_bounds,
            "w3": w3_intensity_bounds,
            "w4": w4_intensity_bounds,
            "w5": w5_intensity_bounds,}
        self.args.bg_sub = False
        self.args.bgsub_kernels = {
            "w1": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w1_bg_rad, w1_bg_rad)),
            "w2": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w2_bg_rad, w2_bg_rad)),
            "w3": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w3_bg_rad, w3_bg_rad)),
            "w4": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w4_bg_rad, w4_bg_rad)),
            "w5": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w5_bg_rad, w5_bg_rad)),}
        ########################################################################
        # Cellpaint Step 2) that can be tuned to the plate and image dimensions
        self.args.cellpose_nucleus_diam = cellpose_nucleus_diam
        self.args.cellpose_cyto_diam = cellpose_cyto_diam
        self.args.cellpose_batch_size = cellpose_batch_size
        self.args.cellpose_model_type = cellpose_model_type
        self.args.min_sizes = {
            "w1": w1_min_size,
            "w2": w2_min_size,
            "w3": w3_min_size,
            "w5": w5_min_size,}
        #######################################################################
        # Cellpaint Step 3) arguments that can be tuned to the plate
        # w2/cyto hyperparameters
        self.args.multi_nucleus_dist_thresh = multi_nucleus_dist_thresh
        # w3/nucleoli segmentation hyperparameters
        self.args.min_nucleoli_size_multiplier = min_nucleoli_size_multiplier
        self.args.max_nucleoli_size_multiplier = max_nucleoli_size_multiplier
        self.args.nucleoli_bd_area_to_nucleoli_area_threshold = nucleoli_bd_area_to_nucleoli_area_threshold
        self.args.w3_local_rescale_intensity_ub = w3_local_rescale_intensity_ub
        # w5/cyto segmentation hyperparameters
        self.args.w5_local_rescale_intensity_ub = w5_local_rescale_intensity_ub
        #######################################################################
        # Cellpaint Step 5) hyperparameters
        self.args.distmap_batch_size = distmap_batch_size
        self.args.min_fov_cell_count = min_fov_cell_count

        # In case we want to redo analysis after Cell-paint step 3 and the image folder is missing or
        # it has not been transferred!!!
        if self.args.analysis_part == 1:
            self.args.imgs_fold = list(filter(
                lambda x: "AssayPlate" in str(x),
                (self.args.main_path / self.args.experiment).iterdir()))[0].stem
            print("self.args.imgs_fold:", self.args.imgs_fold)
            self.args.imgs_dir = self.args.main_path / self.args.experiment / self.args.imgs_fold

            # Get generic width and height dimensions of the image, in the specific experiment.
            # We assume height and width are the same for every single images, in the same experiment!!!
            if "perkinelmer" in self.args.imgs_fold.lower() or \
                    "greiner" in self.args.imgs_fold.lower() or \
                    "combchem" in self.args.imgs_fold.lower():
                self.args.img_suffix = "tif"
                self.args.plate_protocol = self.args.imgs_fold.split("_")[1].lower()
                self.args.img_filepaths = list(self.args.imgs_dir.rglob(
                    f"{self.args.imgs_fold}_*.{self.args.img_suffix}"))

            elif "cpg0012" in self.args.imgs_fold.lower():
                self.args.img_suffix = "tif"
                self.args.plate_protocol = self.args.imgs_fold
                self.args.img_filepaths = list(self.args.imgs_dir.rglob(f"*.{self.args.img_suffix}"))

            elif "cpg0001" in self.args.imgs_fold.lower():
                self.args.img_suffix = "tiff"
                self.args.plate_protocol = self.args.imgs_fold
                self.args.img_filepaths = list(self.args.imgs_dir.rglob(f"*.{self.args.img_suffix}"))
                
            elif "cpgmoa" in self.args.imgs_fold.lower():
                self.args.img_suffix = "tif"
                self.args.plate_protocol = self.args.imgs_fold
                self.args.img_filepaths = list(self.args.imgs_dir.rglob(f"*.{self.args.img_suffix}"))
                
            elif "cimini" in self.args.imgs_fold.lower():
                self.args.img_suffix = "tiff"
                self.args.plate_protocol = self.args.imgs_fold
                self.args.img_filepaths = list(self.args.imgs_dir.rglob(f"*.{self.args.img_suffix}"))
                
                
            else:
                raise NotImplementedError("")

        if self.args.plate_protocol.lower() in ["perkinelmer", "greiner"]:
            # sometimes there are other tif files in the experiment folder that are not an image, so we have to
            # remove them from img_paths, so that they do not mess-up the analysis.
            self.args.img_filepaths = list(
                filter(lambda x: x.stem.split("_")[-1][0:5] == "T0001", self.args.img_filepaths))
        self.args.img_filepaths = sorted(
            self.args.img_filepaths,
            key=lambda x: sort_key_for_imgs(x, "to_sort_channels", self.args.plate_protocol))
        self.args.img_filename_keys, self.args.img_channels_filepaths, self.args.N = get_img_channel_groups(self.args)
        self.args.height, self.args.width = tifffile.imread(self.args.img_filepaths[0]).shape

        self.args.platemap_filepath = list((self.args.main_path / self.args.experiment).rglob("platemap*.xlsx"))[0]

        # self.add_save_path_args()
        # # for Step 3) feature extraction
        self.add_platemap_args()
        # # for Step 4) Distance-map calculation
        self.add_platemap_anchor_args()
        # # for Step 5) Quality Control
        self.add_platemap_control_args()
        # for debug mode
        self.add_other_args()  # for coloring cells when displaying images in test mode

        self.args.channel_dies = {
            "C1": "DAPI",  # nucleus
            "C2": "Concanavalin A",  # cyto
            "C3": "Syto14",  # nucleoli
            "C4": "WGA+Phalloidin",  # actin
            "C5": "MitoTracker",  # mito
        }

    def add_platemap_args(self, ):
        """
        This step is a must for the analysis to work.
        It extracts all the necessary metadata self.args from all the sheets of the plate-map excel file,
        except the last one.
        """
        if self.args.experiment in [
            "2021-CP007",
            "2021-CP008",
            "20220607-CP-Fabio-U2OS-density-10x",
            "20220623-CP-Fabio-Seema-Transfected",
            "20220810-CP-Fabio-WGA-Restain",
            "others"
        ]:
            raise ValueError("Experiment not implemented or is in discard pile!")
        # TODO: check sheetnames and raise value error if sheetname is not present or is incorrect
        # xls = xlrd.open_workbook(self.platemap_filepath, on_demand=True)
        # print(xls.sheet_names())
        # xls.sheet_names()  # <- remeber: xlrd sheet_names is a function, not a property

        self.args.wellid2treatment = self.get_wellid2meta(sheetname="Treatment")
        self.args.wellid2cellline = self.get_wellid2meta(sheetname="CellLine")
        self.args.wellid2dosage = self.get_wellid2meta(sheetname="Dosage")
        self.args.wellid2density = self.get_wellid2meta(sheetname="Density")
        self.args.wellid2other = self.get_wellid2meta(sheetname="Other")

        wkeys = list(self.args.wellid2treatment.keys())
        vals = list(self.args.wellid2treatment.values())
        self.args.wellids = np.unique(wkeys)
        self.args.treatments = np.unique(vals)
        self.args.celllines = np.unique([str(it) for it in list(self.args.wellid2cellline.values())])

        if self.args.wellid2dosage is None:
            self.args.wellid2dosage = {key: 0 for key in wkeys}
            self.args.dosages = np.array([0])
        else:
            self.args.dosages = np.unique(list(self.args.wellid2dosage.values()))

        if self.args.wellid2density is None:
            self.args.wellid2density = {key: 0 for key in wkeys}
            self.args.densities = np.array([0])
        else:
            self.args.densities = np.unique(list(self.args.wellid2density.values()))

        if self.args.wellid2other is None:
            self.args.wellid2other = {key: 0 for key in wkeys}
            self.args.others = np.array([0])
        else:
            # print(list(self.args.wellid2other.values()))
            self.args.others = np.unique(list(self.args.wellid2other.values()))

        # print("treatments: ", self.args.treatments)
        # print("cell-lines: ", self.args.celllines)
        # print("densities: ", self.args.densities)
        # print("dosages: ", self.args.dosages)

        N = len(self.args.wellids)
        cols = ["well-id", "treatment", "cell-line", "density", "dosage", "other"]
        self.args.platemap = np.zeros((N, len(cols)), dtype=object)
        for ii, it in enumerate(self.args.wellids):
            self.args.platemap[ii, :] = \
                (it, self.args.wellid2treatment[it], self.args.wellid2cellline[it],
                 self.args.wellid2density[it], self.args.wellid2dosage[it], self.args.wellid2other[it])
        self.args.platemap = pd.DataFrame(self.args.platemap, columns=cols)
        # print(self.args.wellids)
        return self.args

    def add_platemap_anchor_args(self, ):
        """This step is necessary to automate step IV of the cellpaint Analysis"""
        # anchors
        meta = pd.read_excel(self.args.platemap_filepath, sheet_name="Anchor")
        self.args.anchor_treatment = str(meta["Treatment"].values[0])
        self.args.anchor_cellline = meta["CellLine"].values[0]#str(meta["CellLine"].values[0])
        self.args.anchor_density = meta["Density"].values[0]
        self.args.anchor_dosage = meta["Dosage"].values[0]
        self.args.anchor_other = meta["Other"].values[0]

        # fix the entries of the 5 columns in the "Anchor" sheet
        if self.args.anchor_treatment == np.nan:
            raise ValueError("self.args.anchor_treatment value cannot be Nan. Please specify a value for it.")
        elif self.args.anchor_treatment.lower() != "dmso":
            warnings.warn("self.args.anchor_treatment is not set to DMSO. "
                          "If it is set correctly, ignore this warning!")
        assert isinstance(self.args.anchor_treatment, str), "Anchor treatment must be a string!"

        self.args.anchor_treatment = self.args.anchor_treatment.lower()
        self.args.anchor_cellline = self.fix_anchor_val(self.args.anchor_cellline, "CellLine")
        self.args.anchor_density = self.fix_anchor_val(self.args.anchor_density, "Density")
        self.args.anchor_dosage = self.fix_anchor_val(self.args.anchor_dosage, "Dosage")
        self.args.anchor_other = self.fix_anchor_val(self.args.anchor_other, "Other")
        print(
            f"anchor treatment: {self.args.anchor_treatment}   "
            f"anchor cell-line: {self.args.anchor_cellline}   "
            f"anchor density: {self.args.anchor_density}   "
            f"anchor dosage: {self.args.anchor_dosage}   "
            f"anchor other: {self.args.anchor_other}")

        return self.args

    def add_platemap_control_args(self, ):
        """This step is necessary to automate step IV of the cellpaint Analysis"""
        # anchors
        meta = pd.read_excel(self.args.platemap_filepath, sheet_name="Control")["Treatment"].to_numpy()
        controls = [self.fix_entry(it, "Treatment") for it in meta]
        for it in controls:
            assert np.isin(it, self.args.treatments), \
                f"Control treatment {it} has to be present in list of treatments, but it is not!!!"
        # making sure control treatments always starts with self.args.anchor_treatment == "dmso"
        """https://stackoverflow.com/questions/1014523/
        simple-syntax-for-bringing-a-list-element-to-the-front-in-python"""
        controls.insert(0, controls.pop(controls.index(self.args.anchor_treatment)))
        # print("controls", controls)
        self.args.control_treatments = np.array(controls, dtype=object)
        print("control_treatments: ", self.args.control_treatments)

    def add_other_args(self, ):
        self.args.colors = [
            "red", "green", "blue", "orange", "purple", "lightgreen", "yellow", "pink",
            "khaki", "lime", "olivedrab", "azure", "orchid", "darkslategray",
            "peru", "tan"]
        self.args.csfont = {"fontname": "Comic Sans MS", "fontsize": 16}
        return self.args

    def get_wellid2meta(self, sheetname):
        platemap_filepath = list((self.args.main_path / self.args.experiment).rglob("platemap*.xlsx"))[0]
        meta = pd.read_excel(platemap_filepath, sheet_name=sheetname)
        meta.index = meta["Unnamed: 0"]
        meta.index.name = "row_id"
        meta.drop(["Unnamed: 0", ], axis=1, inplace=True)
        if meta.isnull().values.all():
            return None
        meta = meta.loc[~np.all(meta.isna(), axis=1), ~np.all(meta.isna(), axis=0)]
        wellid2meta = {}
        for row_id in meta.index:
            for col_id in meta.columns:
                wellid2meta[f"{row_id}{str(col_id).zfill(2)}"] = self.fix_entry(meta.loc[row_id, col_id], sheetname)
        return wellid2meta

    @staticmethod
    def fix_entry(entry, belongs_to):
        # print("fix_entry", entry)
        if belongs_to == "Dosage":
            assert isinstance(entry, numbers.Number), "Must be int or float"
            return entry
        elif belongs_to == "Density":
            assert isinstance(entry, numbers.Number), "Must be int or float"
            return np.uint64(entry)

        elif belongs_to in ["Treatment", "CellLine"]:
            if isinstance(entry, str):
                entry = entry.lstrip().strip()
                entry = entry.replace(" + ", '+')
                entry = entry.replace(' ', '-').replace('_', '-')
                return entry.lower()
            elif isinstance(entry, numbers.Number):
                return str(entry)
            else:
                raise ValueError("Data type not supported!!!")
        elif belongs_to == "Other":
            if isinstance(entry, str):
                entry = entry.lstrip().strip()
                entry = entry.replace(" + ", '+')
                entry = entry.replace(' ', '-').replace('_', '-')
                return entry.lower()
            elif isinstance(entry, numbers.Number):
                return entry
            else:
                raise ValueError("Data type not supported!!!")

    def fix_anchor_val(self, val, belongs_to):
        return 0 if isinstance(val, numbers.Number) and math.isnan(val) else self.fix_entry(val, belongs_to)


class RandomSampler(object):
    num_test_wells_per_condition = 2
    num_test_fovs_per_condition = 2

    def __init__(self, args):
        self.args = args

    def get_random_samples(self, ):
        # pick random indices
        random_idxs = random.sample(range(self.N), self.args.num_debug_images)
        # update
        self.filename_keys = self.filename_keys[random_idxs]
        self.img_channels_filepaths = self.img_channels_filepaths[random_idxs]
        self.N = len(self.filename_keys)

    def get_test_samples(self, ):
        """
        Crazy things can happen which might lead into a sample out of range situation.
        BUT, 99% of the time we should be able to successfully perform this sampling algorithm.

        samples self.num_test_wells_per_condition
         well-ids for each unique ("cell-line", "density", "dosage", "other").
         Also, includes DMSO treatment in the sampling, as it is our anchor during DistMap calculation in
         StepIV of Cellpaint.

        Note, for this work,
        the sampled well_ids must be present in both the plate_map excel-file and the image_folder.
        """
        #############################
        # Step I) First things first, one should restrict our well-ids sample pool to the ones that are present
        # in both the platemap, and, the image folder:
        wellids_from_plt = self.args.wellids
        wellids_from_imgs = np.unique(self.filename_keys[:, 1])
        allowed_wellids = np.intersect1d(wellids_from_plt, wellids_from_imgs)
        # print(len(wellids_from_plt), '\n', len(wellids_from_imgs), '\n', len(allowed_wellids))
        platemap = self.args.platemap.loc[np.isin(self.args.platemap["well-id"].values, allowed_wellids)]
        ##############################################
        # Step II) sample well-ids per condition from self.args.plate-map
        if len(self.args.treatments) == 1:  # there is only one treatment (most likely just DMSO)
            conds = [np.ones(len(platemap), dtype=bool)]
        else:  # otherwise, we need to sample DMSO well-ids + sample other compounds well-ids
            cond0 = platemap["treatment"] == self.args.anchor_treatment
            cond1 = platemap["treatment"] != self.args.anchor_treatment
            conds = [cond0, cond1]

        wellids = []
        for ii in range(len(conds)):
            groups = platemap[conds[ii]].groupby(["cell-line", "density", "dosage", "other"])
            for key, grp in groups:
                grp = list(grp['well-id'])
                wellids += random.sample(grp, self.num_test_wells_per_condition)
        # for it in wellids:
        #     print(platemap[platemap["well-id"] == it].values[0])
        # print('\n')
        ################################################################
        # Step III) use sample well-ids + a random fov to sample from our pool of image filepaths
        cond = np.zeros((self.N,), dtype=bool)
        for wid in wellids:
            cond1 = self.filename_keys[:, 1] == wid
            fovs = self.filename_keys[cond1, 2]

            # fov = fovs[random.randint(0, len(fovs)-1)]  # select a random fov within this well
            # cond2 = self.filename_keys[:, 2] == fov
            fovs = random.sample(list(fovs), self.num_test_fovs_per_condition)
            cond2 = np.isin(self.filename_keys[:, 2], fovs)

            # print(wid, np.sum(cond1), len(fovs), fovs, fov)
            cond |= (cond1 & cond2)
        # print(self.N, np.sum(cond))
        ###########################################################
        # Step IV) finally, apply the update
        self.filename_keys = self.filename_keys[cond]
        self.img_channels_filepaths = self.img_channels_filepaths[cond]
        self.N = len(self.filename_keys)
