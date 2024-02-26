import random
import itertools
import numpy as np
import pandas as pd

from utils.helpers import get_img_paths, sort_key_for_imgs
from utils.args import CellPaintArgs
np.set_printoptions(linewidth=200)


class TestSampler():

    def __init__(self, args):
        self.args = args

    def group_img_channels(self):
        """
        sort all tiff file in the self.args.main_path/self.args.experiment/self.args.plate_protocol folder.
         Then group them in such a way that all the tiff file which belong to the same image,
         the 4/5 channels of the same image, go to the same group.
         """

        img_paths = get_img_paths(self.args.main_path, self.args.experiment, self.args.plate_protocol)
        # ################################################################
        # group files that are channels of the same image together.
        # f"{ASSAY}_A02_T0001F002L01A02Z01C01.tif"
        filename_keys, img_paths_groups = [], []
        for item in itertools.groupby(
                img_paths,
                key=lambda x: sort_key_for_imgs(x, self.args.plate_protocol, sort_purpose="to_group_channels")):
            filename_keys.append(item[0])
            img_paths_groups.append(list(item[1]))

        filename_keys = np.array(filename_keys, dtype=object)
        img_paths_groups = np.array(img_paths_groups, dtype=object)
        N = len(img_paths_groups)
        return filename_keys, img_paths_groups, N


        print(metadata)
        # cond1 = meta[1] == treatment
        # treat_wellids = meta[0, cond1]
        unix = np.unique(metadata[:, 2:].astype("<U200"), axis=0)
        print(unix)
        for it in unix:
            cond = (metadata[:, 1] == treatment) & \
                   (metadata[:, 2] == it[0]) &\
                   (metadata[:, 2] == it[1]) &\
                   (metadata[:, 2] == it[2]) &\
                   (metadata[:, 2] == it[3]) &\
                   (metadata[:, 2] == it[4])
            print(it, cond0.shape, cond1.shape, np.sum(cond0), np.sum(cond1))


args = CellPaintArgs(
    experiment="20221024-CP-Bolt-MCF7",
    testing=False,
    num_test_images=4,
).args
sampler = TestSampler(args)
sampler.sample_wellids()