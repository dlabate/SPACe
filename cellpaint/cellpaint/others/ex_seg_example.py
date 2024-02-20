import numpy as np
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
from cellpaint.utils.helpers import move_figure, get_overlay
from skimage.exposure import rescale_intensity

csfont = {'fontname': 'Comic Sans MS', 'fontsize': 18}

colors = ["red", "green", "blue", "orange", "purple", "lightgreen", "yellow", "pink",
          "khaki", "lime", "olivedrab", "azure", "orchid",
          "peru", "tan"]
img_path = r"F:\CellPainting\20220908-CP-Fabio-DRC-BM-R02\AssayPlate_PerkinElmer_CellCarrier-384"
mask_path = r"F:\CellPainting\20220908-CP-Fabio-DRC-BM-R02\MasksP2"
img_filenames = [
    "AssayPlate_PerkinElmer_CellCarrier-384_E04_T0001F007L01A05Z01C01.tif",
    "AssayPlate_PerkinElmer_CellCarrier-384_E04_T0001F007L01A04Z01C02.tif",
    "AssayPlate_PerkinElmer_CellCarrier-384_E04_T0001F007L01A03Z01C03.tif",
    "AssayPlate_PerkinElmer_CellCarrier-384_E04_T0001F007L01A02Z01C04.tif",
    "AssayPlate_PerkinElmer_CellCarrier-384_E04_T0001F007L01A01Z01C05.tif",
]
mask_filenames = [f"w{ii}_E04_F007.png" for ii in [0, 1, 2, 4]]


def show_all_masks():
    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
    fig.set_size_inches(20, 10)
    move_figure(fig, 30, 30)

    img = np.concatenate([tifffile.imread(f"{img_path}/{img_filenames[ii]}")[np.newaxis]
                          for ii in [0, 1, 2, 3, 4]], axis=0)
    w0_mask = np.array(Image.open(f"{mask_path}/{mask_filenames[0]}"))
    w1_mask = np.array(Image.open(f"{mask_path}/{mask_filenames[1]}"))
    w2_mask = np.array(Image.open(f"{mask_path}/{mask_filenames[2]}"))
    w4_mask = np.array(Image.open(f"{mask_path}/{mask_filenames[3]}"))

    for ii in range(0, 5):
        img[ii] = rescale_intensity(img[ii], in_range=tuple(np.percentile(img[ii], (5, 99.9))))

    axes[0, 0].imshow(img[0], cmap="gray")
    axes[0, 1].imshow(img[1], cmap="gray")
    axes[0, 2].imshow(img[2], cmap="gray")
    axes[0, 3].imshow(img[4], cmap="gray")

    axes[1, 0].imshow(get_overlay(img[0], w0_mask, colors), cmap="gray")
    axes[1, 1].imshow(get_overlay(img[1], w1_mask, colors), cmap="gray")
    axes[1, 2].imshow(get_overlay(img[2], w2_mask, colors), cmap="gray")
    axes[1, 3].imshow(get_overlay(img[4], w4_mask, colors), cmap="gray")

    axes[0, 0].set_title("Nucleus Channel", **csfont)
    axes[0, 1].set_title("Cytoplasm Channel", **csfont)
    axes[0, 2].set_title("Nucleoli Channel", **csfont)
    axes[0, 3].set_title("Mito Channel", **csfont)

    for ii, ax in enumerate(axes.flatten()):
        ax.set_axis_off()

    axes[0, 0].set_ylabel("Image", **csfont)
    axes[1, 0].set_ylabel("Mask Overlay", **csfont)

    plt.subplots_adjust(hspace=.01, wspace=.01)
    plt.show()


show_all_masks()