import time
from tqdm import tqdm

import cv2
import random
import numpy as np
import multiprocessing as mp

from pathlib import WindowsPath
from cellpaint.steps_single_plate.step0_args import Args, sort_key_for_imgs, get_img_channel_groups
from cellpaint.steps_single_plate._segmentation import SegmentationPartI, SegmentationPartII
from cellpaint.utils.shared_memory import MyBaseManager, TestProxy


def get_wellid(x, args):
    return sort_key_for_imgs(x, "to_get_well_id", args.plate_protocol)


def set_custom_datasets_hyperparameters(args):
    # Apply the changes from preview.ipynb file to this function
    ##############################################################################
    # intensity rescaling hyperparameters
    args.w1_intensity_bounds = (0.1, 99.99)
    args.w2_intensity_bounds = (0.1, 99.99)
    args.w3_intensity_bounds = (0.1, 99.99)
    args.w4_intensity_bounds = (0.1, 99.99)
    args.w5_intensity_bounds = (0.1, 99.99)
    ##########################################################################
    # background correction hyperparameters
    """Set args.bg_sub to True first if you decide to do background subtraction."""
    args.bg_sub = False
    args.w1_bg_rad = 50
    args.w2_bg_rad = 100
    args.w3_bg_rad = 50
    args.w4_bg_rad = 100
    args.w5_bg_rad = 100
    #######################################################################
    # image channels order/index during data acquisition set by the investigator/microscope
    args.nucleus_idx = 0
    args.cyto_idx = 1
    args.nucleoli_idx = 2
    args.actin_idx = 3
    args.mito_idx = 4
    #######################################################################
    # hyperparameters/constants used in Cellpaint Step 2
    args.step2_segmentation_algorithm = "w1=pycle_w2=pycle"
    args.cellpose_nucleus_diam = 30
    args.cellpose_cyto_diam = 30
    args.cellpose_batch_size = 64
    args.cellpose_model_type = "cyto2"
    args.w1_min_size = 200
    args.w2_min_size = 300
    args.w3_min_size = 5
    args.w5_min_size = 30
    #######################################################
    # hyperparameters/constants used in Cellpaint Step 3
    args.multi_nucleus_dist_thresh = 20
    args.min_nucleoli_size_multiplier = .000001
    args.max_nucleoli_size_multiplier = .999
    args.nucleoli_bd_area_to_nucleoli_area_threshold = .01
    args.w3_local_rescale_intensity_ub = 99.99
    args.w5_local_rescale_intensity_ub = 99.99
    return args


def preview_run_loop(args, num_wells=None, sample_wellids=None):
    """
    Main function for cellpaint step 2:
        It performs segmentation of nucleus and cytoplasm channels on a few images,
        (99% of the time,they are the first and the second channel of each image)
        using the cellpose python package.

        It saves the two masks as separate png files into:
        args.main_path / args.experiment / "Step0_MasksP1-Preview"
        args.main_path / args.experiment / "Step0_MasksP2-Preview"
    """
    print("Cellpaint Step 2: Cellpose segmentation of Nucleus and Cytoplasm ...")

    args.mode = "preview"
    s_time = time.time()
    args = set_custom_datasets_hyperparameters(args)
    ######################################################################
    # get img paths from wellids
    if num_wells is not None:
        sample_wellids = random.sample(list(args.wellids), num_wells)
    elif (num_wells is not None) and (sample_wellids is not None):
        raise ValueError("num_wells and sample_wellids both can't be None!")
    else:
        sample_wellids = [wellid.upper() for wellid in sample_wellids]
    args.img_filepaths = sorted(filter(lambda x: np.isin(get_wellid(x, args), sample_wellids), args.img_filepaths))
    args.img_filename_keys, args.img_channels_filepaths, args.N = get_img_channel_groups(args)
    ############################################################################
    seg_class1 = SegmentationPartI(args)
    ranger = tqdm(np.arange(args.N), total=args.N)
    for ii in ranger:
        seg_class1.run_single(args.img_channels_filepaths[ii], args.img_filename_keys[ii])
    print(f"Finished Cellpaint step 2 for {args.N} sample 5-channel-images  in {(time.time() - s_time) / 3600} hours\n")
    ##########################################################################################
    """
    We have to Register the ThresholdingSegmentation class object as well as its attributes as shared using:
    https://stackoverflow.com/questions/26499548/accessing-an-attribute-of-a-multiprocessing-proxy-of-a-class
    """
    MyManager = MyBaseManager()
    MyManager.register("SegmentationPartII", SegmentationPartII, TestProxy)
    with MyManager as manager:
        inst = manager.SegmentationPartII(args)
        with mp.Pool(processes=inst.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(inst.run_multi, np.arange(args.N)), total=args.N):
                pass
    print(f"Finished Cellpaint step 3 in: {(time.time() - s_time) / 3600} hours\n")


if __name__ == "__main__":
    camii_server_flav = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid"
    camii_server_seema = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Seema"
    camii_server_jump_cpg0012 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0012"
    camii_server_jump_cpg0001 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0001"

    main_path = WindowsPath(camii_server_flav)
    exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"

    args = Args(experiment=exp_fold, main_path=main_path, mode="preview").args
    preview_run_loop(args, num_wells=1)
    # We have to recreate the args after preview anyways.
