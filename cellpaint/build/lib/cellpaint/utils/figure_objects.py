import numpy as np
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb

import matplotlib.pyplot as plt


def get_overlay(img, labeled_mask, colors):
    for ii, it in enumerate(colors):
        labeled_mask[ii, 0] = ii+1

    mask_bd = find_boundaries(
        labeled_mask,
        connectivity=2,
        mode="inner").astype(np.uint16)
    mask_bd[mask_bd != 0] = labeled_mask[mask_bd != 0]
    mask_bd = label2rgb(mask_bd, colors=colors, bg_label=0)

    img = img.copy()
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img) + 1e-6)
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    img[mask_bd != 0] = mask_bd[mask_bd != 0]
    # print(mask_bd.shape, img.shape)

    return img


def move_figure(f, x, y):
    import matplotlib
    # Bottom vertical alignment for more space, size=14)
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def creat_segmentation_example_fig(img, mask0, mask1, mask2, mask_path, args):
    import matplotlib.pyplot as plt
    well_id = mask_path.stem.split("_")[2]
    fov = mask_path.stem.split("_")[3]
    treat = args.wellid2treat[well_id]
    celline = args.wellid2celline[well_id]

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    fig.suptitle(f"{celline:15}{treat:12}{well_id:8}{fov}", **args.csfont)
    fig.set_size_inches(12, 13)
    move_figure(fig, 30, 30)
    #############################################################################
    # Enhance the contrast in case the images look too dim.
    # for shelton2 these values seem to work
    if args.rescale_image:
        img[0] = rescale_intensity(img[0], in_range=tuple(np.percentile(img[0], (40, 99))))
        img[1] = rescale_intensity(img[1], in_range=tuple(np.percentile(img[1], (50, 95))))
        img[2] = rescale_intensity(img[2], in_range=tuple(np.percentile(img[2], (50, 95))))
        img[3] = rescale_intensity(img[3], in_range=tuple(np.percentile(img[3], (50, 95))))
    ############################################################################
    # img1 = np.abs(img[3]-img[1])
    for idx, (it, ax) in enumerate(zip(["Nucleus", "Cyto", "Nucleoli", "Actin"], axes.flatten())):
        ax.set_ylabel(it, **args.csfont)
        ax.set_title(f"Channel {idx + 1}", **args.csfont)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].imshow(get_overlay(img[0], mask0, args.colors), cmap="gray")
    axes[0, 1].imshow(get_overlay(img[1], mask1, args.colors), cmap="gray")
    axes[1, 0].imshow(get_overlay(img[2], mask2, args.colors), cmap="gray")
    axes[1, 1].imshow(get_overlay(img[3], mask2, args.colors), cmap="gray")
    plt.subplots_adjust(hspace=0.1, wspace=.02)
    plt.show()