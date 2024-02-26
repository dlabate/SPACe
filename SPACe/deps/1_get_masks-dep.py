import time
from tqdm import tqdm

import multiprocessing as mp
from functools import partial

import numpy as np

import SimpleITK as sitk
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import find_objects


from skimage.measure._regionprops import RegionProperties
from skimage import exposure
from skimage.io import imread, imsave
from skimage.measure import label
from skimage.segmentation import find_boundaries, clear_border, mark_boundaries
from skimage.morphology import disk, erosion, dilation, \
    binary_dilation, binary_erosion, binary_closing, remove_small_objects
from skimage.segmentation import watershed, expand_labels
from skimage.filters import threshold_triangle, threshold_otsu, gaussian


from utils.args import args, ignore_imgaeio_warning, create_shared_multiprocessing_name_space_object
from utils.helpers import get_matching_img_group_nuc_mask_cyto_mask, \
    load_img, get_sitk_img_thresholding_mask_v1


def get_w0w1_masks_step1(w0_mask, w1_mask, img, args):
    img_filter = sitk.OtsuThresholdImageFilter()
    img_filter.SetInsideValue(0)
    img_filter.SetOutsideValue(1)

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
            cache_active=True,)
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
            cache_active=True,).intensity_image
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

        # if len(lw0_unix) != 2:
        #     print(ii+1, np.unique(w0_pieces))
        #     fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        #     fig.suptitle("w0 and w1 mask crops at the beginning")
        #     axes[0, 0].imshow(w10_props.intensity_image, cmap="gray")
        #     axes[0, 1].imshow(img[1, y0:y1, x0:x1], cmap="gray")
        #     axes[1, 0].imshow(w0_pieces, cmap="gray")
        #     axes[1, 1].imshow(w10_props.image, cmap="gray")
        #
        #     axes[0, 0].set_title("w0 intensity image")
        #     axes[0, 1].set_title("w1 intensity image")
        #     axes[1, 0].set_title(f"w0 mask pieces: {np.unique(w0_pieces)}")
        #     axes[1, 1].set_title("w1 mask pieces")
        #     plt.show()

        if len(lw0_unix) == 2:  # there is exactly one nucleus in the cyto mask
            # cnt1 += 1
            continue

        # there is cytoplasm but no nuclei inside
        # segment inside of it in nuclei channel
        elif len(lw0_unix) == 1:
            # y0, x0, y1, x1 = w10_props.bbox
            w0_intensity = w10_props.intensity_image

            w0_lmask = get_sitk_img_thresholding_mask_v1(w0_intensity, img_filter)
            w0_lmask = binary_closing(w0_lmask, disk(8))
            w0_lmask = binary_fill_holes(w0_lmask, structure=np.ones((2, 2)))
            # # remove small noisy dots
            w0_lmask = label(w0_lmask, connectivity=2, background=0)
            w0_lmask[remove_small_objects(
                w0_lmask.astype(bool),
                min_size=args.min_nucleus_size,
                connectivity=8).astype(int) == 0] = 0
            if np.sum(w0_lmask > 0) < args.min_nucleus_size:
                w1_mask[w1_mask == ii+1] = 0
                continue

            # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            # fig.suptitle(f"inside elif statement when len(lw0_unix)=1")
            # axes[0, 0].imshow(w10_props.intensity_image, cmap="gray")
            # axes[0, 1].imshow(img[1, y0:y1, x0:x1], cmap="gray")
            # axes[1, 0].imshow(w0_lmask, cmap="gray")
            # axes[1, 1].imshow(w10_props.image, cmap="gray")
            # axes[0, 0].set_title("w0 intensity image")
            # axes[0, 1].set_title("w1 intensity image")
            # axes[1, 0].set_title("w0 mask pieces")
            # axes[1, 1].set_title("w1 mask pieces")
            # plt.show()

            w0_mask[y0:y1, x0:x1][w0_lmask > 0] = w0_lmask[w0_lmask > 0]+counter0
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
            # print(w1_mask.shape, w1_mask[y0:y1, x0:x1].shape, tmp1.shape)
            # fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
            # fig.suptitle(f"len(lw0_unix)={len(lw0_unix)}")
            # axes[0, 0].imshow(w10_props.intensity_image, cmap="gray")
            # axes[0, 1].imshow(img[1, y0:y1, x0:x1], cmap="gray")
            # axes[0, 2].imshow(img[1, y0:y1, x0:x1], cmap="gray")
            # axes[1, 0].imshow(label2rgb(w0_pieces, bg_label=0), cmap="gray")
            # axes[1, 1].imshow(label2rgb(w1_lmask, bg_label=0), cmap="gray")
            # axes[1, 2].imshow(label2rgb(tmp1, bg_label=0), cmap="gray")
            # axes[0, 0].set_title("w0 intensity image")
            # axes[0, 1].set_title("w1 intensity image")
            # axes[0, 2].set_title("w1 intensity image")
            # axes[1, 0].set_title("w0 mask pieces")
            # axes[1, 1].set_title("w1 mask pieces")
            # axes[1, 2].set_title("w1 mask after watershed")
            # plt.show()

            w1_mask[y0:y1, x0:x1][tmp1 > 0] = tmp1[tmp1 > 0]+counter1
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
    w1_mask = w1_mask.reshape((args.height, args.width))
    w1_mask = np.uint16(w1_mask)
    w0_mask[w0_mask > 0] = w1_mask[w0_mask > 0]

    # remove the leftovers from matching step
    diff = np.setdiff1d(np.unique(w1_mask), np.unique(w0_mask))
    w1_mask[np.isin(w1_mask, diff)] = 0
    assert np.array_equal(np.unique(w0_mask), np.unique(w1_mask))

    # if args.testing and args.show_intermediate_steps:
    #     if args.rescale_image:
    #         img[args.nucleus_idx] = exposure.rescale_intensity(img[
    #         args.nucleus_idx],
    #         in_range=tuple(np.percentile(img[args.nucleus_idx], (70, 99.99))))
    #         img[args.cyto_channel_index] = exposure.rescale_intensity(
    #         img[args.cyto_channel_index],
    #         in_range=tuple(np.percentile(img[args.cyto_channel_index], (50, 99.9))))
    #     fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #     axes[0].imshow(get_overlay(img[args.nucleus_idx], w0_mask, args.colors), cmap="gray")
    #     axes[1].imshow(get_overlay(img[args.cyto_idx], w0_mask, args.colors), cmap="gray")
    #     plt.show()
    return w0_mask, w1_mask


def get_nucleoli_mask(w0_mask, img, args):
    # nucleoli
    img_filter = sitk.YenThresholdImageFilter()
    img_filter.SetInsideValue(0)
    img_filter.SetOutsideValue(1)

    mask2 = np.zeros_like(w0_mask)
    w0_max = int(np.amax(w0_mask))
    w0_objects = find_objects(w0_mask, max_label=w0_max)

    for ii in range(w0_max):
        if w0_objects[ii] is None:
            continue
        obj_label = ii + 1
        w02_props = RegionProperties(
            slice=w0_objects[ii], label=obj_label, label_image=w0_mask,
            intensity_image=img[args.nucleoli_idx],
            cache_active=True,)
        l_img = w02_props.intensity_image
        lmask = w02_props.image
        y0, x0, y1, x1 = w02_props.bbox

        val = threshold_otsu(l_img)
        lb = np.sum(l_img < val) / np.size(l_img)
        l_img = gaussian(l_img, sigma=1)
        prc = tuple(np.percentile(l_img, (lb, args.nucleoli_channel_rescale_intensity_percentile_ub)))
        l_img = exposure.rescale_intensity(l_img, in_range=prc)  # (40, 88), (30, 95)
        min_nucleoli_size = args.min_nucleoli_size_multiplier * (np.sum(lmask))
        max_nucleoli_size = args.max_nucleoli_size_multiplier * (np.sum(lmask))
        ##############################################################################
        lmask_padded = np.pad(lmask, ((5, 5), (5, 5)), constant_values=(0, 0))
        bd = find_boundaries(lmask_padded, connectivity=2)
        bd = bd[5:-5, 5:-5]
        bd = binary_dilation(bd, disk(3))
        #################################################################################
        tmp = get_sitk_img_thresholding_mask_v1(l_img, img_filter).astype(np.uint16)
        # tmp1 = tmp.copy()
        tmp = binary_erosion(tmp, disk(1))
        # tmp2 = tmp.copy()
        tmp = label(tmp, connectivity=2)
        tmax = int(np.amax(tmp))
        ##################################################################################
        tmp_objects = find_objects(tmp, max_label=tmax)
        tmp_bd_objects = find_objects(tmp*bd, max_label=tmax)
        for jj in range(tmax):
            if tmp_objects[jj] is None:
                continue
            tmp_label = jj+1
            tmp_props = RegionProperties(
                slice=tmp_objects[jj], label=tmp_label, label_image=tmp,
                intensity_image=None, cache_active=True,)
            tmp_bd_props = RegionProperties(
                slice=tmp_bd_objects[jj], label=tmp_label, label_image=tmp*bd,
                intensity_image=None, cache_active=True,)
            if tmp_bd_props.area/tmp_props.area > args.nucleoli_bd_area_to_nucleoli_area_threshold or \
                    tmp_props.area < min_nucleoli_size or \
                    tmp_props.area > max_nucleoli_size:
                tmp[tmp == tmp_label] = 0
        mask2[y0:y1, x0:x1][tmp > 0] = obj_label
        # mask2[y0:y1, x0:x1] += (tmp > 0)+np.uint16(obj_label)-1

        # fig, axes = plt.subplots(1, 7, sharex=True, sharey=True)
        # axes[0].imshow(img[args.nucleus_idx][y0:y1, x0:x1], cmap="gray")
        # axes[1].imshow(l_img, cmap="gray")
        # axes[2].imshow(lmask, cmap="gray")
        # axes[3].imshow(tmp1, cmap="gray")
        # axes[4].imshow(tmp2, cmap="gray")
        # axes[5].imshow(bd, cmap="gray")
        # axes[6].imshow(tmp, cmap="gray")
        # plt.show()
    return mask2


def get_mito_mask(w0_mask, w1_mask, img):
    f1 = sitk.OtsuThresholdImageFilter()
    f1.SetInsideValue(0)
    f1.SetOutsideValue(1)

    tmp_img = img[-1].copy()
    tmp_img[(dilation(w1_mask, disk(6)) == 0) | (erosion(w0_mask, disk(4)) > 0)] = 0
    global_mask = get_sitk_img_thresholding_mask_v1(tmp_img, f1)

    mask_local = np.zeros_like(w1_mask)
    cyto_mask = w1_mask.copy()
    cyto_mask[w0_mask > 0] = 0
    # cyto_mask = w1_mask * (1-w0_mask)
    # props = regionprops_table(cyto_mask, tmp_img, properties=["label", "bbox", "intensity_image"])
    w4_max = int(np.amax(cyto_mask))
    w4_objects = find_objects(cyto_mask, max_label=w4_max)
    for ii in range(w4_max):
        if w4_objects[ii] is None:
            continue
        w4_props = RegionProperties(
            slice=w4_objects[ii], label=ii+1, label_image=cyto_mask,
            intensity_image=tmp_img,
            cache_active=True,)
        y0, x0, y1, x1 = w4_props.bbox
        # scaled_img = rescale_intensity(limg, in_range=tuple(np.percentile(limg, [5, 99])))
        tmp_mask = get_sitk_img_thresholding_mask_v1(w4_props.intensity_image, f1).astype(bool)
        # mask_local[y0:y1, x0:x1][tmp_mask > 0] = tmp_mask[tmp_mask > 0]
        mask_local[y0:y1, x0:x1] += tmp_mask
        del tmp_mask

    w4_mask = np.uint16(np.logical_or(global_mask, mask_local))
    # w4_mask[w4_mask > 0] = cyto_mask[w4_mask > 0]
    w4_mask = w4_mask * cyto_mask
    assert np.array_equal(np.unique(w4_mask), np.unique(cyto_mask))
    return w4_mask


def get_masks(idx, w0_mask_paths, w1_mask_paths, img_path_groups, args):
    # make sure w0_mask and w1_mask are both cast to uint16
    ss_time = time.time()
    # s_time = time.time()
    img = load_img(img_path_groups[idx], args.num_channels, args.height, args.width)
    # print(f"img load time {time.time()-s_time}")

    # s_time = time.time()
    w0_mask = imread(str(w0_mask_paths[idx])).astype(np.uint16)
    # print(f"w0_mask load time {time.time()-s_time}")

    # s_time = time.time()
    w1_mask = imread(str(w1_mask_paths[idx])).astype(np.uint16)
    # print(f"w1_mask load time {time.time()-s_time}")

    # s_time = time.time()
    w0_mask, w1_mask = get_w0w1_masks_step1(w0_mask, w1_mask, img, args)
    # print(f"w0 and w1 mask generation {time.time()-s_time}")

    s_time = time.time()
    w2_mask = get_nucleoli_mask(w0_mask, img, args)
    w4_mask = get_mito_mask(w0_mask, w1_mask, img)
    print(f"w2 and w4 mask generation {time.time()-s_time}")
    # if args.testing and args.show_intermediate_steps:
    #     import matplotlib.pyplot as plt
    #     # import matplotlib.font_manager as font_manager
    #     from skimage.exposure import rescale_intensity
    #     from utils.image_and_mask_files import get_overlay, move_figure
    #     fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    #     fig.set_size_inches(13, 9)
    #     move_figure(fig, 30, 30)
    #     axes[0, 0].imshow(get_overlay(img[args.nucleus_idx], w0_mask, args.colors), cmap="gray")
    #     axes[0, 1].imshow(get_overlay(img[args.cyto_idx], w1_mask, args.colors), cmap="gray")
    #     axes[1, 0].imshow(get_overlay(img[args.nucleoli_idx], w2_mask, args.colors), cmap="gray")
    #     axes[1, 1].imshow(get_overlay(img[args.mito_idx], w4_mask, args.colors), cmap="gray")
    #     axes[0, 0].set_title("Nucleus Channel", **args.csfont)
    #     axes[0, 1].set_title("Cytoplasm Channel", **args.csfont)
    #     axes[1, 0].set_title("Nucleoli Channel", **args.csfont)
    #     axes[1, 1].set_title("Mito Channel", **args.csfont)
    #     for ii, ax in enumerate(axes.flatten()):
    #         ax.set_axis_off()
    #     plt.show()

    assert w0_mask.dtype == w1_mask.dtype == w2_mask.dtype == np.uint16

    # s_time = time.time()
    w0_filename = f"{w0_mask_paths[idx].stem}.png"
    w1_filename = f"{w1_mask_paths[idx].stem}.png"
    w2_filename = f"w2_{'_'.join(w0_mask_paths[idx].stem.split('_')[1:])}.png"
    w4_filename = f"w4_{'_'.join(w0_mask_paths[idx].stem.split('_')[1:])}.png"
    imsave(args.final_masks_path / w0_filename, w0_mask, check_contrast=False)
    imsave(args.final_masks_path / w1_filename, w1_mask, check_contrast=False)
    imsave(args.final_masks_path / w2_filename, w2_mask, check_contrast=False)
    imsave(args.final_masks_path / w4_filename, w4_mask, check_contrast=False)
    # e_time = time.time()
    # print(f"saving masks {e_time-s_time}    total {e_time-ss_time} \n")


def main_worker(args):
    # TODO: Get number of channels as an independent separate step
    # TODO: Check nucleus area at every single stage.
    img_path_groups, w0_mask_paths, w1_mask_paths, args.num_channels = \
        get_matching_img_group_nuc_mask_cyto_mask(args.main_path, args.experiment, args.lab)

    if args.testing:
        N = len(w0_mask_paths)
        for ii in range(N):
            # print(w0_mask_paths[ii].stem)
            # choose some specific well-ids you like to check for quality control
            # if well_id == "A01" and fov == "F004":
            get_masks(ii, w0_mask_paths, w1_mask_paths, img_path_groups, args,)
            # creat_segmentation_example_fig(img, w0_mask, w1_mask, mask2, m0)

    else:
        manager = mp.Manager()
        w0_mask_paths = manager.list(w0_mask_paths)
        w1_mask_paths = manager.list(w1_mask_paths)
        img_path_groups = manager.list(img_path_groups)

        args = create_shared_multiprocessing_name_space_object(args)
        N = len(w0_mask_paths)
        my_func = partial(
            get_masks,
            w0_mask_paths=w0_mask_paths,
            w1_mask_paths=w1_mask_paths,
            img_path_groups=img_path_groups,
            args=args,
        )
        with mp.Pool(processes=mp.cpu_count(),) as pool:
            for _ in tqdm(pool.imap(my_func, np.arange(N)), total=N):
                pass


if __name__ == "__main__":
    ignore_imgaeio_warning()
    mp.freeze_support()
    main_worker(args)


