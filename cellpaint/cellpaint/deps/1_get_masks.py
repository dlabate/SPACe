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


F1 = sitk.OtsuThresholdImageFilter()
F1.SetInsideValue(0)
F1.SetOutsideValue(1)
F2 = sitk.YenThresholdImageFilter()
F2.SetInsideValue(0)
F2.SetOutsideValue(1)


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
                w1_mask[w1_mask == ii+1] = 0
                continue

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
    w1_mask = np.uint16(w1_mask.reshape((args.height, args.width)))
    w0_mask[w0_mask > 0] = w1_mask[w0_mask > 0]

    # remove the leftovers from matching step
    diff = np.setdiff1d(np.unique(w1_mask), np.unique(w0_mask))
    w1_mask[np.isin(w1_mask, diff)] = 0
    assert np.array_equal(np.unique(w0_mask), np.unique(w1_mask))

    return w0_mask, w1_mask


def get_nucleoli_and_mito_masks(w0_mask, w1_mask, img, args):
    # mito mask
    tmp_img4 = img[-1].copy()
    tmp_img4[(dilation(w1_mask, disk(6)) == 0) | (erosion(w0_mask, disk(4)) > 0)] = 0
    global_mask4 = get_sitk_img_thresholding_mask_v1(tmp_img4, F1)
    local_mask4 = np.zeros_like(w1_mask)
    cyto_mask = w1_mask.copy()
    # TODO: check if the two are equivalent and whether the product is faster
    cyto_mask[w0_mask > 0] = 0  # 1)
    # cyto_mask = w1_mask * (~w0_mask).astype(np.uint16)  # 2)
    max_ = int(np.amax(cyto_mask))
    w4_objects = find_objects(cyto_mask, max_label=max_)
    ########################################################
    # nucleoli mask
    w2_mask = np.zeros_like(w0_mask)
    w0_objects = find_objects(w0_mask, max_label=max_)
    ########################################################
    for ii in range(max_):
        ##############################################################
        if w4_objects[ii] is None:
            continue
        obj_label = ii+1
        w4_props = RegionProperties(
            slice=w4_objects[ii], label=obj_label, label_image=cyto_mask,
            intensity_image=tmp_img4,
            cache_active=True,)
        w2_props = RegionProperties(
            slice=w0_objects[ii], label=obj_label, label_image=w0_mask,
            intensity_image=img[args.nucleoli_idx],
            cache_active=True, )
        ####################################################################
        # local mito mask calculation
        y0, x0, y1, x1 = w4_props.bbox
        # scaled_img = rescale_intensity(limg, in_range=tuple(np.percentile(limg, [5, 99])))
        tmp4 = get_sitk_img_thresholding_mask_v1(w4_props.intensity_image, F1).astype(bool)
        # local_mask4[y0:y1, x0:x1][tmp_mask > 0] = tmp_mask[tmp_mask > 0]
        local_mask4[y0:y1, x0:x1] += tmp4.astype(np.uint16)
        del tmp4
        #####################################################################
        # local nucleoli mask calculation
        l_img2 = w2_props.intensity_image
        lmask2 = w2_props.image
        y0, x0, y1, x1 = w2_props.bbox
        val = threshold_otsu(l_img2)
        lb = np.sum(l_img2 < val) / np.size(l_img2)
        l_img2 = gaussian(l_img2, sigma=1)
        prc = tuple(np.percentile(l_img2, (lb, args.nucleoli_channel_rescale_intensity_percentile_ub)))
        l_img2 = exposure.rescale_intensity(l_img2, in_range=prc)  # (40, 88), (30, 95)
        min_nucleoli_size = args.min_nucleoli_size_multiplier * (np.sum(lmask2))
        max_nucleoli_size = args.max_nucleoli_size_multiplier * (np.sum(lmask2))
        ##############################################################################
        lmask2_padded = np.pad(lmask2, ((5, 5), (5, 5)), constant_values=(0, 0))
        bd = find_boundaries(lmask2_padded, connectivity=2)
        bd = bd[5:-5, 5:-5]
        bd = binary_dilation(bd, disk(3))
        #################################################################################
        tmp2 = get_sitk_img_thresholding_mask_v1(l_img2, F2).astype(np.uint16)
        tmp2 = binary_erosion(tmp2, disk(1))
        tmp2 = label(tmp2, connectivity=2)
        tmax = int(np.amax(tmp2))
        ##################################################################################
        tmp2_objects = find_objects(tmp2, max_label=tmax)
        tmp2_bd_objects = find_objects(tmp2 * bd, max_label=tmax)
        for jj in range(tmax):
            if tmp2_objects[jj] is None:
                continue
            tmp2_label = jj + 1
            tmp2_props = RegionProperties(
                slice=tmp2_objects[jj], label=tmp2_label, label_image=tmp2,
                intensity_image=None, cache_active=True, )
            tmp2_bd_props = RegionProperties(
                slice=tmp2_bd_objects[jj], label=tmp2_label, label_image=tmp2 * bd,
                intensity_image=None, cache_active=True, )
            if tmp2_bd_props.area / tmp2_props.area > args.nucleoli_bd_area_to_nucleoli_area_threshold or \
                    tmp2_props.area < min_nucleoli_size or \
                    tmp2_props.area > max_nucleoli_size:
                tmp2[tmp2 == tmp2_label] = 0
        w2_mask[y0:y1, x0:x1] += (tmp2 != 0).astype(np.uint16)
        # w2_mask[y0:y1, x0:x1][tmp2 > 0] = obj_label
    # mito mask
    w4_mask = np.logical_or(global_mask4, local_mask4).astype(np.uint16)
    w4_mask *= cyto_mask
    w2_mask *= w0_mask
    # print(len(np.unique(w0_mask)),
    #       len(np.unique(w1_mask)),
    #       len(np.unique(w2_mask)),
    #       len(np.unique(w4_mask)),
    #       len(np.unique(cyto_mask)))
    # w4_mask[w4_mask > 0] = cyto_mask[w4_mask > 0]
    # assert np.array_equal(np.unique(w4_mask), np.unique(cyto_mask))
    return w2_mask, w4_mask


def get_masks(idx, w0_mask_paths, w1_mask_paths, img_path_groups, args):
    # make sure w0_mask and w1_mask are both cast to uint16
    # ss_time = time.time()
    img = load_img(img_path_groups[idx], args.num_channels, args.height, args.width)
    w0_mask = imread(str(w0_mask_paths[idx])).astype(np.uint16)
    w1_mask = imread(str(w1_mask_paths[idx])).astype(np.uint16)
    # print(f"loading {time.time()-ss_time}")
    # s_time = time.time()
    w0_mask, w1_mask = get_w0w1_masks(w0_mask, w1_mask, img, args)
    # print(f"w0 and w1 mask generation {time.time()-s_time}")
    # s_time = time.time()
    w2_mask, w4_mask = get_nucleoli_and_mito_masks(w0_mask, w1_mask, img, args)
    # print(f"w2 and w4 mask generation {time.time()-s_time}")

    if args.testing and args.show_intermediate_steps:
        import matplotlib.pyplot as plt
        # import matplotlib.font_manager as font_manager
        from skimage.exposure import rescale_intensity
        from utils.helpers import get_overlay, move_figure
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.set_size_inches(13, 9)
        move_figure(fig, 30, 30)
        axes[0, 0].imshow(get_overlay(img[args.nucleus_idx], w0_mask, args.colors), cmap="gray")
        axes[0, 1].imshow(get_overlay(img[args.cyto_idx], w1_mask, args.colors), cmap="gray")
        axes[1, 0].imshow(get_overlay(img[args.nucleoli_idx], w2_mask, args.colors), cmap="gray")
        axes[1, 1].imshow(get_overlay(img[args.mito_idx], w4_mask, args.colors), cmap="gray")
        axes[0, 0].set_title("Nucleus Channel", **args.csfont)
        axes[0, 1].set_title("Cytoplasm Channel", **args.csfont)
        axes[1, 0].set_title("Nucleoli Channel", **args.csfont)
        axes[1, 1].set_title("Mito Channel", **args.csfont)
        for ii, ax in enumerate(axes.flatten()):
            ax.set_axis_off()
        plt.show()

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
    # TODO: Check nucleus area at every single stage.
    img_path_groups, w0_mask_paths, w1_mask_paths, args.num_channels = \
        get_matching_img_group_nuc_mask_cyto_mask(args.main_path, args.experiment, args.plate_protocol)


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
    args = create_shared_multiprocessing_name_space_object(args)
    main_worker(args)


