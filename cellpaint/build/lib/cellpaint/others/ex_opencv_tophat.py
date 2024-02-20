from pathlib import WindowsPath

import cv2
import numpy as np
import tifffile
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

from cellpaint.steps_single_plate.step0_args import Args

camii_server_flav = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid"
camii_server_seema = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Seema"
camii_server_jump_cpg0012 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0012"
camii_server_jump_cpg0001 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0001"

main_path = WindowsPath(camii_server_flav)
exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"

args = Args(experiment=exp_fold, main_path=main_path, mode="full").args
# args = set_mancini_datasets_hyperparameters(args)

# Getting the kernel to be used in Top-Hat
filterSize = (100, 100)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize)

# Reading the image
input_image = cv2.imread(str(args.img_filepaths[1]), cv2.IMREAD_UNCHANGED)
in_range = tuple(np.percentile(input_image, (5, 99.8)))
input_image = rescale_intensity(input_image, in_range=in_range)
# Applying the Top-Hat operation
tophat_img = cv2.morphologyEx(input_image, cv2.MORPH_TOPHAT, kernel)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
axes[0].imshow(input_image, cmap="gray")
axes[1].imshow(tophat_img, cmap="gray")
plt.show()