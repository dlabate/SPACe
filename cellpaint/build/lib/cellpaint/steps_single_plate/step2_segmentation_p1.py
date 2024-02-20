import os
import time
from tqdm import tqdm
from pathlib import WindowsPath

import numpy as np
from cellpaint.steps_single_plate.step0_args import Args
from cellpaint.steps_single_plate._segmentation import SegmentationPartI

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def step2_main_run_loop(args):
    """
    Main function for cellpaint step 2:
        It performs segmentation of nucleus and cytoplasm channels,
        (99% of the time,they are the first and the second channel of each image)
        using the cellpose python package.

        It saves the two masks as separate png files into:
        self.args.step1_save_path = args.main_path / args.experiment / "Step1_MasksP1"
    """
    print("Cellpaint Step 2: Cellpose segmentation of Nucleus and Cytoplasm ...")
    seg_class = SegmentationPartI(args)
    s_time = time.time()
    N = seg_class.args.N
    # ranger = np.arange(N)
    ranger = tqdm(np.arange(N), total=N)

    for ii in ranger:
        seg_class.run_single(seg_class.args.img_channels_filepaths[ii], seg_class.args.img_filename_keys[ii])
    print(f"Finished Cellpaint Step 2 for {N} images  in {(time.time() - s_time) / 3600} hours\n")


if __name__ == "__main__":
    camii_server_flav = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid"
    camii_server_seema = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Seema"
    camii_server_jump_cpg0012 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0012"
    camii_server_jump_cpg0001 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0001"

    main_path = WindowsPath(camii_server_flav)
    exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"

    args = Args(experiment=exp_fold, main_path=main_path, mode="full").args
    step2_main_run_loop(args)
