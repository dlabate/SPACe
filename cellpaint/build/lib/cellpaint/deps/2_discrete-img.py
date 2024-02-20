import numpy as np
import tifffile
from utils.args import args
from utils.helpers import get_img_paths
import matplotlib.pyplot as plt

args.experiment = "20220831-cellpainting-benchmarking-DRC_20220831_173200"
args.lab = "baylor"
args.batch_size = 64

img_paths = get_img_paths(args.main_path, args.experiment, args.lab)
img = tifffile.imread(img_paths[0])
print(img_paths[0])
print(np.percentile(img, [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 96, 97, 98, 99]))
# plt.imshow(img, cmap="gray")
# plt.show()