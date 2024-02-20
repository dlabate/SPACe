import sys, os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from tqdm import tqdm
import numpy as np
import itertools
from cellpose import models

import tifffile

from utils.args import args
from utils.helpers import sort_fn, get_img_paths


from skimage.measure._regionprops import RegionProperties
from skimage import exposure
from skimage.io import imread, imsave
from skimage.measure import label
from skimage.segmentation import find_boundaries, clear_border, mark_boundaries
from skimage.morphology import disk, erosion, dilation, \
    binary_dilation, binary_erosion, binary_closing, remove_small_objects
from skimage.segmentation import watershed, expand_labels
from skimage.filters import threshold_triangle, threshold_otsu, gaussian
from skimage import io


import SimpleITK as sitk
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import find_objects


def get_w0_and_w1_masks_single_img(MODEL, img_path_group, key):
    # get nucleus mask
    et = time.time()
    w0_mask, _, _, _ = MODEL.eval(
        # tifffile.imread(img_paths[1]),
        tifffile.imread(img_path_group[args.nucleus_idx]),
        diameter=args.cellpose_nucleus_diam,
        channels=[0, 0],
        batch_size=args.batch_size,
        z_axis=None,
        channel_axis=None,
        resample=False,
    )

    # get cyto mask
    w1_mask, _, _, _ = MODEL.eval(
        # tifffile.imread(img_paths[1]),
        tifffile.imread(img_path_group[args.cyto_idx]),
        diameter=args.cellpose_nucleus_diam,
        channels=[0, 0],
        batch_size=args.batch_size,
        z_axis=None,
        channel_axis=None,
        resample=False,
    )
    print(f"w0 w1 mask gen {time.time() - et}")

    if args.testing:
        import matplotlib.pyplot as plt
        from skimage.exposure import rescale_intensity
        from skimage.color import label2rgb
        # from skimage.measure import label
        # from skimage.segmentation import find_boundaries
        print(img_path_group[0])
        print(img_path_group[1])
        img0 = tifffile.imread(img_path_group[0])
        img1 = tifffile.imread(img_path_group[1])
        # img0 = rescale_intensity(img0, in_range=tuple(np.percentile(img0, (90, 99.99))))
        # img1 = rescale_intensity(img1, in_range=tuple(np.percentile(img1, (50, 99.9))))
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        axes[0, 0].imshow(img0, cmap="gray")
        axes[0, 1].imshow(img1, cmap="gray")
        axes[1, 0].imshow(label2rgb(w0_mask, bg_label=0), cmap="gray")
        axes[1, 1].imshow(label2rgb(w1_mask, bg_label=0), cmap="gray")
        plt.show()

    # If the total number of segmented cells in both DAPI and CYTO channels is small then skip that image
    if (len(np.unique(w1_mask)) + len(np.unique(w0_mask))) / 2 <= args.min_cell_count + 1:
        return
    ####################################################################################################
    # Create a savename choosing a name for the experiment name, and also using the well_id, and fov.
    # ('2022-0817-CP-benchmarking-density_20220817_120119', 'H24', 'F009')
    exp_id, well_id, fov = key[0], key[1], key[2]
    save_name = f"{args.expid}_{well_id}_{fov}"
    ####################################################################################################
    # Save the masks into disk
    # et = time.time()
    io.imsave(args.cellpose_masks_path / f"w0_{save_name}.png", w0_mask, check_contrast=False)
    io.imsave(args.cellpose_masks_path / f"w1_{save_name}.png", w1_mask, check_contrast=False)
    # print(f"saving {time.time() - et}")


def get_w0w1_masks(w0_mask, w1_mask, img, args):
    w1_mask = clear_border(w1_mask)
    w1_mask[remove_small_objects(
        w1_mask.astype(bool),
        min_size=args.min_cyto_size,
        connectivity=8).astype(int) == 0] = 0
    w0_mask = clear_border(w0_mask)
    w0_mask[remove_small_objects(
        w0_mask.astype(bool),
        min_size=args.min_nucleus_size,
        connectivity=8).astype(int) == 0] = 0

    # remove nuclei with no cytoplasm
    # w0_mask[w1_mask == 0] = 0
    w0_mask = w0_mask * (w1_mask != 0)
    w1_max = int(np.amax(w1_mask))

    w1_objects = find_objects(w1_mask, max_label=w1_max)
    unix0 = np.unique(w0_mask)
    unix1 = np.unique(w1_mask)
    counter0 = unix0[-1]
    counter1 = unix1[-1]

    # TODO: Remove bad nuclues and cyto around edges using circularity
    for ii in range(w1_max):
        if w1_objects[ii] is None:
            continue
        obj_label = ii + 1
        w10_props = RegionProperties(
            slice=w1_objects[ii], label=obj_label, label_image=w1_mask,
            intensity_image=img[args.nucleus_idx],
            cache_active=True, )
        # if the area of cytoplasm mask for this specific cell (obj_label) is small,
        # then remove the cell.
        if w10_props.area < args.min_nucleus_size:
            w1_mask[w1_mask == obj_label] = 0
            w0_mask[w1_mask == obj_label] = 0
            continue

        # within current cell, pieces of w0_mask within w1_mask
        w0_pieces = RegionProperties(
            slice=w1_objects[ii], label=obj_label, label_image=w1_mask,
            intensity_image=w0_mask,
            cache_active=True, ).intensity_image
        # # removing edge artifact, so that the two masks do not touch on the bd
        w0_pieces = label(w0_pieces.astype(bool), background=0)
        # tmp0_props = regionprops_table(w0_pieces, properties=["label", "area"])
        # idxs = tmp0_props["label"][tmp0_props["area"] < args.min_nucleus_size]
        # w0_pieces[np.isin(w0_pieces, idxs)] = 0
        w0_pieces[remove_small_objects(
            w0_pieces.astype(bool),
            min_size=args.min_nucleus_size,
            connectivity=8).astype(int) == 0] = 0
        y0, x0, y1, x1 = w10_props.bbox
        lw0_unix = np.unique(w0_pieces)
        if len(lw0_unix) == 0:  # remove bad cells
            w1_mask[w1_mask == obj_label] = 0
            w0_mask[w1_mask == obj_label] = 0
            continue

        if len(lw0_unix) == 2:  # there is exactly one nucleus in the cyto mask
            # cnt1 += 1
            continue

        # there is cytoplasm but no nuclei inside
        # segment inside of it in nuclei channel
        elif len(lw0_unix) == 1:
            # y0, x0, y1, x1 = w10_props.bbox
            w0_intensity = w10_props.intensity_image

            w0_lmask = get_sitk_img_thresholding_mask_v1(w0_intensity, F1)
            w0_lmask = binary_closing(w0_lmask, disk(8))
            w0_lmask = binary_fill_holes(w0_lmask, structure=np.ones((2, 2)))
            # # remove small noisy dots
            w0_lmask = label(w0_lmask, connectivity=2, background=0)
            w0_lmask[remove_small_objects(
                w0_lmask.astype(bool),
                min_size=args.min_nucleus_size,
                connectivity=8).astype(int) == 0] = 0
            if np.sum(w0_lmask > 0) < args.min_nucleus_size:
                w1_mask[w1_mask == ii + 1] = 0
                continue

            w0_mask[y0:y1, x0:x1][w0_lmask > 0] = w0_lmask[w0_lmask > 0] + counter0
            # cnt0 += 1
            counter0 += 1

        elif len(lw0_unix) > 2:  # there is cytoplasm but more than one nuclei
            # y0, x0, y1, x1 = w10_props.bbox
            w1_lmask = w10_props.image
            # segment and break down this cyto using nucleus
            tmp1 = watershed(
                w1_lmask,
                markers=w0_pieces,
                connectivity=w0_pieces.ndim,
                mask=w1_lmask)
            shape = tmp1.shape
            tmp_unix, tmp1 = np.unique(tmp1, return_inverse=True)
            tmp1 = tmp1.reshape(shape)

            w1_mask[y0:y1, x0:x1][tmp1 > 0] = tmp1[tmp1 > 0] + counter1
            counter1 += len(np.setdiff1d(tmp_unix, [0]))
            # cnt2 += 1
    # print(total, 100*cnt0/total, 100*cnt1/total, 100*cnt2/total)
    # # remove noisy bd effects
    mask_bd = find_boundaries(
        w1_mask,
        connectivity=2,
        mode="inner").astype(np.uint16)
    mask_bd = binary_dilation(mask_bd, disk(2))
    w0_mask[(w0_mask > 0) & mask_bd] = 0
    w0_mask[remove_small_objects(
        w0_mask.astype(bool),
        min_size=400,
        connectivity=8).astype(int) == 0] = 0

    # match the labels for w1_mask and w0_mask
    _, w1_mask = np.unique(w1_mask, return_inverse=True)
    w1_mask = np.uint16(w1_mask.reshape((args.height, args.width)))
    w0_mask[w0_mask > 0] = w1_mask[w0_mask > 0]

    # remove the leftovers from matching step
    diff = np.setdiff1d(np.unique(w1_mask), np.unique(w0_mask))
    w1_mask[np.isin(w1_mask, diff)] = 0
    assert np.array_equal(np.unique(w0_mask), np.unique(w1_mask))

    return w0_mask, w1_mask


def main():
    """Read all the image tif files for the experiment as a list and sort them, "img_paths".
    Then divide the files into groups each containing the 4/5 channels of a single image, "img_paths_groups".
    for example,
    [[args.main_path\args.experiment\f"{args.assay}"\f"{args.assay}"_B02_T0001F001L01A01Z01C01.tif,
    args.main_path\args.experiment\f"{args.assay}"\f"{args.assay}"_B02_T0001F001L01A02Z01C02.tif,
    args.main_path\args.experiment\f"{args.assay}"\f"{args.assay}"_B02_T0001F001L01A03Z01C03.tif,
    args.main_path\args.experiment\f"{args.assay}"\f"{args.assay}"_B02_T0001F001L01A04Z01C04.tif,
    args.main_path\args.experiment\f"{args.assay}"\f"{args.assay}"_B02_T0001F001L01A05Z01C05.tif,],

    [args.main_path\args.experiment\f"{args.assay}"\f"{args.assay}"_B02_T0001F002L01A01Z01C01.tif,
    args.main_path\args.experiment\f"{args.assay}"\f"{args.assay}"_B02_T0001F002L01A02Z01C02.tif,
    args.main_path\args.experiment\f"{args.assay}"\f"{args.assay}"_B02_T0001F002L01A03Z01C03.tif,
    args.main_path\args.experiment\f"{args.assay}"\f"{args.assay}"_B02_T0001F002L01A04Z01C04.tif,
    args.main_path\args.experiment\f"{args.assay}"\f"{args.assay}"_B02_T0001F002L01A05Z01C05.tif,]

    Note that if the number of cells segmented in an image is smaller than "MIN_CELLS",
    meaning the FOV/image is mostly empty, this function will NOT save the corresponding mask into disk!!!

    First, quality check those segmentation by setting TESTING = True.
    Once satisfied, set the TESTING= False and run the code again.

    In addition make sure to choose a proper unique key for your experiment.
    for example the master folder containing the images.
    At the moment it is set as either "EXPERIMENT" or EXPIDS[your_index]

    Enjoy!!!"""

    MODEL = models.Cellpose(gpu=True, model_type='cyto2', net_avg=False)
    # f"{ASSAY}_A02_T0001F002L01A02Z01C01.tif"
    img_paths = get_img_paths(args.main_path, args.experiment, args.plate_protocol)
    # ################################################################
    # group files that are channels of the same image together.
    keys, img_paths_groups = [], []
    for item in itertools.groupby(
            img_paths, key=lambda x: sort_fn(x, args.plate_protocol, sort_purpose="to_group_channels")):
        keys.append(item[0])
        img_paths_groups.append(list(item[1]))

    keys = np.array(keys, dtype=object)
    img_paths_groups = np.array(img_paths_groups, dtype=object)
    N = len(img_paths_groups)
    # for key, item in tqdm(zip(keys, img_paths_groups)):

    s_time = time.time()
    # for ii in tqdm(range(N), total=N):
    for ii in range(N):
        # s1_time = time.time()
        get_w0_and_w1_masks_single_img(MODEL, img_paths_groups[ii], keys[ii])
        # if ii == 20:
        #     break
    print(f"time taken to finish {2 * (ii + 1)} images: {(time.time() - s_time) / 3600} hours")


if __name__ == "__main__":
    main()
