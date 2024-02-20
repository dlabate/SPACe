import numpy as np

from skimage.exposure import rescale_intensity

import seaborn as sns
import cmcrameri as cmc

from cellpaint.utils.args import CellPaintArgs
from cellpaint.utils.helpers import load_img, get_img_paths, sort_key_for_imgs, ind2sub
import matplotlib.pyplot as plt


class show_images():
    fov = "F004"
    sort_purpose = "to_match_it_with_mask_path"
    channels = ["Nucleus", "Cytoplasm", "Nucleoli", "Actin", "Mitochondria"]

    def __init__(self, args):
        self.args = args
        self.img_paths = get_img_paths(self.args.main_path, self.args.experiment, self.args.plate_protocol)
        # print(img_paths)
        # 1) Compare DMSO across the cell-line
        # 1-1) find and load a DMSO image per cell-line
        self.N1, self.M, self.H, self.W = \
            len(self.args.celllines), len(self.channels), self.args.height, self.args.width
        assert self.N1 == 6
        self.compare_treatments = ["dmso", "imipramine"]
        self.N2 = len(self.compare_treatments)
        # print(self.N1, self.M, self.H, self.W)

        self.save_path = self.args.main_path/self.args.experiment/"Figures"
        self.save_path.mkdir(exist_ok=True, parents=True)

    def compare_celllines_figure(self):

        images = np.zeros((self.N1, self.M, self.H, self.W), dtype=np.uint16)
        for ii in range(self.N1):
            for jj in range(self.M):
                cond = (self.args.platemap["cell-line"] == self.args.celllines[ii]) & \
                       (self.args.platemap["treatment"] == "dmso")
                well_id = self.args.platemap[cond]["well-id"].to_numpy()[-1]
                tmp_paths = list(filter(
                    lambda x: sort_key_for_imgs(x, self.args.plate_protocol, self.sort_purpose) ==
                              f"{well_id}_{self.fov}",
                    self.img_paths))
                images[ii] = load_img(tmp_paths, self.M, self.H, self.W)
                # print(well_id)
                # print(self.args.celllines[ii], well_id)
                # print(tmp_paths)
                # print('\n')

        # normalize the images per channel
        bounds = np.zeros((self.M, 2), dtype=np.float32)
        for jj in range(self.M):
            bounds[jj] = np.percentile(images[:, jj], [40, 99.99])
            images[:, jj] = rescale_intensity(images[:, jj], in_range=tuple(bounds[jj]))
            # images[:, jj] = (images[:, jj]-np.amin(images[:, jj]))/(np.amax(images[:, jj])-np.amin(images[:, jj]))

        for jj in range(self.M):
            fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
            fig.set_size_inches(22, 15)
            fig.suptitle(self.channels[jj], fontname='Comic Sans MS', fontsize=25)
            for ii in range(self.N1):
                kk, ll = ind2sub((2, 3), ii)
                axes[kk, ll].imshow(images[ii, jj], cmap="gray")
                axes[kk, ll].set_title(self.args.celllines[ii], fontname='Comic Sans MS', fontsize=17)
                axes[kk, ll].set_xticks([])
                axes[kk, ll].set_yticks([])
            plt.savefig(self.save_path / f"dmso-c{jj+1}-{self.channels[jj]}.png", bbox_inches='tight', dpi=300)
            # plt.show()

    def compare_treatments_figure(self, cellline):

        images = np.zeros((self.N2, self.M, self.H, self.W), dtype=np.uint16)
        for ii in range(self.N2):
            cond = (self.args.platemap["cell-line"] == cellline) & \
                   (self.args.platemap["treatment"] == self.compare_treatments[ii])
            well_id = self.args.platemap[cond]["well-id"].to_numpy()[-1]
            print(well_id)
            tmp_paths = list(filter(
                lambda x: sort_key_for_imgs(x, self.args.plate_protocol, self.sort_purpose) ==
                          f"{well_id}_{self.fov}",
                self.img_paths))
            images[ii] = load_img(tmp_paths, self.M, self.H, self.W)
            # print(self.compare_treatments[ii], well_id)
            # print(tmp_paths)
            # print('\n')

        # normalize the images per channel
        bounds = np.zeros((self.M, 2), dtype=np.float32)
        for jj in range(self.M):
            bounds[jj] = np.percentile(images[:, jj], [40, 99.99])
            images[:, jj] = rescale_intensity(images[:, jj], in_range=tuple(bounds[jj]))
            # images[:, jj] = (images[:, jj]-np.amin(images[:, jj]))/(np.amax(images[:, jj])-np.amin(images[:, jj]))
        fig, axes = plt.subplots(self.N2, self.M, sharex=True, sharey=True)
        fig.set_size_inches(self.M*4, self.N2*4+1)
        fig.suptitle(f"{cellline} Comparison", fontname='Comic Sans MS', fontsize=25)
        for ii in range(self.N2):
            for jj in range(self.M):
                axes[ii, jj].imshow(images[ii, jj], cmap="gray")
                axes[ii, jj].set_xticks([])
                axes[ii, jj].set_yticks([])

        for jj in range(self.M):
            axes[0, jj].set_title(self.channels[jj], fontname='Comic Sans MS', fontsize=17)
        for ii in range(self.N2):
            axes[ii, 0].set_ylabel(self.compare_treatments[ii], fontname='Comic Sans MS', fontsize=17)
        plt.subplots_adjust(hspace=-.1, wspace=.05)
        plt.savefig(self.save_path/f"{cellline}.png", bbox_inches='tight', dpi=300)
        # plt.show()


def main():
    for experiment in [
        # "20220831-CP-Fabio-DRC-BM-R01",
        # "20220908-CP-Fabio-DRC-BM-R02",

        # "20221102-CP-Fabio-DRC-BM-P01",
        # "20221102-CP-Fabio-DRC-BM-P02",

        # "20221109-CP-Fabio-DRC-BM-P01",
        # "20221109-CP-Fabio-DRC-BM-P02",

        # "20221116-CP-Fabio-DRC-BM-P01",
        # "20221116-CP-Fabio-DRC-BM-P02",


        # "20220920-CP-Bolt-Seema",
        # "20220930-CP-Bolt-Seema",
        "20221021-CP-Bolt-Seema",

        # "20220912-CP-Bolt-MCF7",
        # "20220929-CP-Bolt-MCF7",
        # "20221024-CP-Bolt-MCF7",

        # "20221207-CP-CCandler-Exp2244-1",
        # "20221208-CP-CCandler-Exp2244-2",


    ]:
        args = CellPaintArgs(experiment=experiment).args
        myclass = show_images(args)
        myclass.compare_treatments_figure(args.anchor_cellline)
        myclass.compare_celllines_figure()
        # 2) Compare emiprimin and DMSO in one cell-line (anchor one preferably)


if __name__ == "__main__":

    main()
