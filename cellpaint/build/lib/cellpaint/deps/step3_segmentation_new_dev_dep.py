import os
import time
from tqdm import tqdm
from pathlib import WindowsPath

import tifffile
import numpy as np
import skimage.io as sio
from scipy.ndimage import find_objects

from cellpaint.utils.img_files_dep import sort_key_for_imgs, get_img_paths
from cellpaint.steps_single_plate.step0_args import Args, BaseConstructor

import torch
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as F

camii_server_flav = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid"
camii_server_seema = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Seema"
camii_server_jump_cpg0012 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0012"
camii_server_jump_cpg0001 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0001"

main_path = WindowsPath(camii_server_flav)
exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"

args = Args(experiment=exp_fold, main_path=main_path).args
mask_filename = "A01_F001_W1.png"
img_filename = "AssayPlate_PerkinElmer_CellCarrier-384_A01_T0001F001L01A01Z01C01.tif"
# img_path = args.main_path/args.experiment/args.imgs_fold/img_filename
# mask_path = args.step2_save_path/mask_filename
mask_filepaths = list(args.step2_save_path.rglob("*.png"))
img_paths = sorted(list(filter(lambda x: "C01" in x.stem, args.img_filepaths)))
mask_paths = sorted(list(filter(lambda x: "W1" in x.stem, mask_filepaths)))

for path1, path2 in zip(img_paths, mask_paths):
    print(path1.stem)
    print(path2.stem)
    img = tifffile.imread(path1)
    mask = sio.imread(path2)

    mask_tensor = torch.as_tensor(np.int16(mask[np.newaxis]))
    obj_ids = torch.unique(mask_tensor)[1:]
    all_masks = mask_tensor == obj_ids[:, None, None]

    s_time = time.time()
    bbox_objs = find_objects(mask)
    print("scipy.ndimage.find_object cpu: ", time.time()-s_time)

    s_time = time.time()
    boxes = masks_to_boxes(all_masks)
    print("torchvision.ops.masks_to_boxes cpu: ", time.time()-s_time)


    s_time = time.time()
    all_masks = all_masks.to("cuda:0")
    print("transfer to gpu time: ", time.time()-s_time)

    s_time = time.time()
    boxes = masks_to_boxes(all_masks)
    print("torchvision.ops.masks_to_boxes gpu first time: ", time.time()-s_time)

    # s_time = time.time()
    # boxes_2 = masks_to_boxes(all_masks)
    # print("torchvision.ops.masks_to_boxes gpu second time: ", time.time()-s_time)
    print('\n')
