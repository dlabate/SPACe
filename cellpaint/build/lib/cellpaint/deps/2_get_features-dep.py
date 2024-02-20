import re
import time
import itertools
from tqdm import tqdm

import numpy as np
import pandas as pd

import tifffile
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops_table
from skimage.color import label2rgb
import matplotlib.pyplot as plt

from utils.args import args, ignore_imgaeio_warning, create_shared_multiprocessing_name_space_object
from utils.helpers import get_key_from_img_path, get_mask_path_key, \
    get_matching_img_group_nuc_mask_cyto_mask, load_img, containsLetterAndNumber
from functools import partial
from scipy import stats
########################
from scipy.ndimage import find_objects
from skimage.measure._regionprops import RegionProperties
from operator import attrgetter
#########################
import multiprocessing as mp
import numba as nb
# import ctypes
# ctypes.py_object
##################################


def get_intensity_features(regionmask, intensity):
    intensity_vec = intensity[regionmask > 0]
    quartiles = np.percentile(intensity_vec, args.intensity_percentiles)
    intensity_median, intensity_mad, intensity_mean, intensity_std = \
        np.median(intensity_vec), \
        stats.median_abs_deviation(intensity_vec), \
        np.mean(intensity_vec), \
        np.std(intensity_vec)
    return tuple(quartiles) + (intensity_median, intensity_mad, intensity_mean, intensity_std,)


def get_haralick_features(regionmask, intensity_image):
    glcm = graycomatrix(
        intensity_image,
        distances=args.distances,
        angles=args.angles,
        levels=len(args.intensity_level_thresholds) + 1,
        symmetric=True, normed=True)
    glcm = glcm[1:, 1:]
    # print(glcm.shape)
    return graycoprops(glcm, prop='contrast'), \
           graycoprops(glcm, prop='dissimilarity'), \
           graycoprops(glcm, prop='homogeneity'),\
           graycoprops(glcm, prop='ASM'), \
           graycoprops(glcm, prop='energy'), \
           graycoprops(glcm, prop='correlation')


def get_my_props(index,
                 args,
                 img_path_groups,
                 w0_mask_paths,
                 w1_mask_paths,
                 w2_mask_paths,
                 w4_mask_paths,
                 ):
    """
    w0_mask_path = .../w0_P000025-combchem-v3-U2OS-24h-L1-copy1_B02_1.png
    w1_mask_path = .../w1_P000025-combchem-v3-U2OS-24h-L1-copy1_B02_1.png
    m2 = .../w2_P000025-combchem-v3-U2OS-24h-L1-copy1_B02_1.png
    img_channels_group:
    [
    .../P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s1_w1DCEB3369-8F24-4915-B0F6-B543ADD85297.tif,
    .../P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s1_w2C3AF00C2-E9F2-406A-953F-2ACCF649F58B.tif,
    .../P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s1_w3524F4D75-8D83-4DDC-828F-136E6A520E5D.tif,
    .../P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s1_w4568AFB8E-781D-4841-8BC8-8FD870A3147F.tif,
    .../P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s1_w5D9A405BD-1C0C-45E4-A335-CEE88A9AD244.tif,
    ]
    """

    # read masks from hard drive
    # try:
    #     w0_mask = np.array(Image.open(w0_mask_paths[index])).astype(np.uint16)  # nuclei mask
    # except:
    #     return None

    w0_mask = np.array(Image.open(w0_mask_paths[index])).astype(np.uint16)  # nuclei mask
    if len(np.unique(w0_mask)[1:]) < args.min_cell_count:
        return None
    w1_mask = np.array(Image.open(w1_mask_paths[index])).astype(np.uint16)  # w1 mask
    w2_mask = np.array(Image.open(w2_mask_paths[index])).astype(np.uint16)  # w2 mask
    w4_mask = np.array(Image.open(w4_mask_paths[index])).astype(np.uint16)  # w4 mask

    # read image from hard drive
    # img of the fov with 4 channels: (DAPI, CYTOPLASM, WGA, MITOCONDRIA)
    # img of the fov with 5 channels: (DAPI, CYTO, w2, actin, MITO)
    img = load_img(img_path_groups[index], args.num_channels, args.height, args.width)
    if args.experiment == "others":
        # others = nucleus, mito, actin, nucleoli, cyto
        # ours = ["Nucleus", "Cyto", "Nucleoli", "Actin", "Mito"]
        img = img[np.ix_([0, 4, 3, 2, 1])]
    img = img.transpose((1, 2, 0))

    ####################################################################################
    # remove the w0 from the w1plasm mask
    w1_mask[w0_mask > 0] = 0
    # relabel w2_mask with the nuclei/w1plasm masks (already matching) labels
    w2_mask[w2_mask > 0] = w0_mask[w2_mask > 0]
    cell_mask = w0_mask + w1_mask  # actin mask

    # print(len(np.unique(cell_mask)), len(np.unique(w1_mask)), len(np.unique(w0_mask)))
    # assert np.array_equal(np.unique(w1_mask), np.unique(w0_mask))
    assert np.array_equal(cell_mask[w0_mask > 0], w0_mask[w0_mask > 0])

    # unix = np.setdiff1d(np.unique(cell_mask), [0])  # cell ids excluding background
    # pos_idx = np.unique(cell_mask[w2_mask > 0])  # with w2
    # neg_idx = np.setdiff1d(unix, pos_idx)  # without w2
    # # divide cells into 2 separate populations : with w2 and without w2
    # has_nucleoli = cell_mask.copy()
    # has_nucleoli[np.isin(cell_mask, pos_idx)] = 1
    # has_nucleoli[np.isin(cell_mask, neg_idx)] = -1
    ####################################################################################################
    discrete_img = np.clip(np.uint16(img), None, args.intensity_thresh_ub)
    for ii, (it0, it1) in enumerate(zip(
            args.intensity_level_thresholds[0:-1], args.intensity_level_thresholds[1:])):
        # print(ii)
        discrete_img[(it0 <= img) & (img < it1)] = ii
    discrete_img[img >= args.intensity_thresh_ub] = len(args.intensity_level_thresholds) - 1
    # TODO: Fix channel ids for different experiments

    # # initialize img_props for cells "with w2", w2_flag==1
    # # and cells "without w2", w2_flag==2
    # # w2_flag = 1, num_cells = N1
    # # nchannels = NUM_ORGANELLES if w2_flag == 1 else NUM_ORGANELLES - 1
    # # N1, N2 = len(np.unique(w0_mask * (has_nucleoli == 1)))-1, \
    # #          len(np.unique(w0_mask * (has_nucleoli == -1))) - 1
    # # Analyze only the cells the have nucleoli for now (has_nucleoli == 1)
    # cell_mask = cell_mask* (has_nucleoli == 1)
    # w0_mask = w0_mask* (has_nucleoli == 1)
    # w1_mask = w1_mask* (has_nucleoli == 1)
    # w2_mask = w2_mask* (has_nucleoli == 1)
    # w4_mask = w4_mask* (has_nucleoli == 1)

    lcnt = 0  # local counter to assign a count to cells within a single image
    N1 = len(np.unique(w0_mask)) - 1
    n1, n2, n3, n4, n5 = \
        len(args.bbox), \
        len(args.shape_keys), \
        len(args.extra_props_keys_intensity), \
        len(args.extra_prop_keys_haralick), \
        len(args.has_nucleoli)
    max_obj_ind = int(np.amax(cell_mask))

    cell_objects = find_objects(cell_mask, max_label=max_obj_ind)
    # print(type(cell_objects), type(cell_objects[0]))
    # print(cell_objects[0], '\n')

    w0_objects = find_objects(w0_mask, max_label=max_obj_ind)
    w1_objects = find_objects(w1_mask, max_label=max_obj_ind)
    w2_objects = find_objects(w2_mask, max_label=max_obj_ind)
    w4_objects = find_objects(w1_mask, max_label=max_obj_ind)

    w0_features = np.zeros(shape=(N1, n1 + n2 + n3 + n4), dtype=object)
    w1_features = np.zeros(shape=(N1, n1 + n2 + n3 + n4), dtype=object)
    w2_features = np.zeros(shape=(N1, n1 + n2 + n3 + n4), dtype=object)
    w3_features = np.zeros(shape=(N1, n1 + n2 + n3 + n4), dtype=object)
    w4_features = np.zeros(shape=(N1, n1 + n2 + n3 + n4), dtype=object)
    has_nucleoli_col = np.zeros(shape=(N1, 1), dtype=np.uint8)

    range_ = tqdm(range(max_obj_ind), total=max_obj_ind) if args.testing else range(max_obj_ind)
    for jjjj in range_:
        if cell_objects[jjjj] is None:
            # print(jjjj, "cell object is none")
            continue
        obj_label = jjjj + 1
        has_nucleoli_label = 0 if w2_objects[jjjj] is None else 1
        cell_haralick_props = RegionProperties(
            slice=cell_objects[jjjj], label=obj_label, label_image=cell_mask,
            intensity_image=discrete_img[:, :, [1, 3, 4]],
            cache_active=True,
            extra_properties=[get_haralick_features, ])
        nuc_haralick_props = RegionProperties(
            w0_objects[jjjj], obj_label, w0_mask,
            intensity_image=discrete_img[:, :, [0, 2]],
            cache_active=True,
            extra_properties=[get_haralick_features, ])

        w0_intensity_and_shape_props = RegionProperties(
            w0_objects[jjjj], obj_label, w0_mask,
            intensity_image=img[:, :, 0],
            cache_active=True,
            extra_properties=[get_intensity_features, ])
        w1_intensity_and_shape_props = RegionProperties(
            w1_objects[jjjj], obj_label, w1_mask,
            intensity_image=img[:, :, 1],
            cache_active=True,
            extra_properties=[get_intensity_features, ])

        if w2_objects[jjjj] is not None:
            w2_intensity_and_shape_props = RegionProperties(
                w2_objects[jjjj], obj_label, w2_mask,
                intensity_image=img[:, :, 2],
                cache_active=True,
                extra_properties=[get_intensity_features, ])
            w2_features[lcnt, 0:n1] = w2_intensity_and_shape_props.bbox
            w2_features[lcnt, n1:n1 + n2] = attrgetter(*args.shape_keys)(w2_intensity_and_shape_props)
            w2_features[lcnt, n1 + n2:n1 + n2 + n3] = w2_intensity_and_shape_props.get_intensity_features

        w3_intensity_and_shape_props = RegionProperties(
            w1_objects[jjjj], obj_label, w1_mask,
            intensity_image=img[:, :, 3],
            cache_active=True,
            extra_properties=[get_intensity_features, ])
        w4_intensity_and_shape_props = RegionProperties(
            w4_objects[jjjj], obj_label, w4_mask,
            intensity_image=img[:, :, 4],
            cache_active=True,
            extra_properties=[get_intensity_features, ])

        w0_features[lcnt, 0:n1] = w0_intensity_and_shape_props.bbox
        w1_features[lcnt, 0:n1] = w1_intensity_and_shape_props.bbox
        w3_features[lcnt, 0:n1] = w3_intensity_and_shape_props.bbox
        w4_features[lcnt, 0:n1] = w4_intensity_and_shape_props.bbox

        # get shape features for each channel
        # print(w0_intensity_and_shape_props.bbox)
        w0_features[lcnt, n1:n1 + n2] = attrgetter(*args.shape_keys)(w0_intensity_and_shape_props)
        w1_features[lcnt, n1:n1 + n2] = attrgetter(*args.shape_keys)(w1_intensity_and_shape_props)
        w3_features[lcnt, n1:n1 + n2] = attrgetter(*args.shape_keys)(w3_intensity_and_shape_props)
        w4_features[lcnt, n1:n1 + n2] = attrgetter(*args.shape_keys)(w4_intensity_and_shape_props)

        # get intensity features for each channel
        w0_features[lcnt, n1 + n2:n1 + n2 + n3] = w0_intensity_and_shape_props.get_intensity_features
        w1_features[lcnt, n1 + n2:n1 + n2 + n3] = w1_intensity_and_shape_props.get_intensity_features
        w3_features[lcnt, n1 + n2:n1 + n2 + n3] = w3_intensity_and_shape_props.get_intensity_features
        w4_features[lcnt, n1 + n2:n1 + n2 + n3] = w4_intensity_and_shape_props.get_intensity_features

        # get haralick features for each channel
        nuc_haralick_feature_groups = getattr(nuc_haralick_props, "get_haralick_features")
        cell_haralick_feature_groups = getattr(cell_haralick_props, "get_haralick_features")
        # print(nuc_haralick_feature_groups.shape)
        w0_features[lcnt, n1 + n2 + n3:n1 + n2 + n3 + n4] = nuc_haralick_feature_groups[:, :, :, 0].reshape(-1)
        w2_features[lcnt, n1 + n2 + n3:n1 + n2 + n3 + n4] = nuc_haralick_feature_groups[:, :, :, 1].reshape(-1)

        w1_features[lcnt, n1 + n2 + n3:n1 + n2 + n3 + n4] = cell_haralick_feature_groups[:, :, :, 0].reshape(-1)
        w3_features[lcnt, n1 + n2 + n3:n1 + n2 + n3 + n4] = cell_haralick_feature_groups[:, :, :, 1].reshape(-1)
        w4_features[lcnt, n1 + n2 + n3:n1 + n2 + n3 + n4] = cell_haralick_feature_groups[:, :, :, 2].reshape(-1)

        has_nucleoli_col[lcnt, ] = has_nucleoli_label
        lcnt += 1
    # # args.metadata_cols = ["well_id", "treatment", "celline", "density", "dosage", "fov",]
    return w0_features, w1_features, w2_features, w3_features, w4_features, has_nucleoli_col


def warn_user_about_missing_wellids(img_path_groups):
    wellids_from_img_files = [
        get_key_from_img_path(it[0], args.lab, key_purpose="to_get_well_id") for it in img_path_groups]
    wellids_from_platemap = args.wellids

    missig_wells_1 = np.setdiff1d(wellids_from_img_files, wellids_from_platemap)
    missig_wells_2 = np.setdiff1d(wellids_from_platemap, wellids_from_img_files)
    if len(missig_wells_1) > 0:
        raise ValueError(
            f"The following well-ids are in the image-folder  {args.experiment},\n"
            f" but are missing from the platemap file:\n"
            f"{missig_wells_1}")
    elif len(missig_wells_2) > 0:
        raise ValueError(
            f"The following well-ids are in the platemap file,\n"
            f" but are missing from the image-folder  {args.experiments}:\n"
            f"{missig_wells_1}")
    else:
        print("no well-id is missing!!! Enjoy!!!")


def check_paths_matching(img_channels_groups_, w0_mask_paths_, w1_mask_paths_, w2_mask_paths_, w4_mask_paths_):
    """ match experiment_id, well_id, and fov between
    img_channels_group, w0_mask_path, w1_mask_path, and w2_mask_path
    """

    # check if the number items in each file group match
    N = len(img_channels_groups_)
    assert N == len(w0_mask_paths_) == len(w1_mask_paths_) == len(w2_mask_paths_) == len(w4_mask_paths_)

    # check whether all images have the same number of channels
    M = len(img_channels_groups_[0])
    for ii in range(1, N):
        assert M == len(img_channels_groups_[ii])

    # check whether keys between mask files and image (per channel) match
    for ii in range(N):
        key = get_mask_path_key(w0_mask_paths_[ii])
        assert key == \
               get_mask_path_key(w1_mask_paths_[ii]) == \
               get_mask_path_key(w2_mask_paths_[ii]) == \
               get_mask_path_key(w4_mask_paths_[ii])

        for jj in range(M):
            assert get_key_from_img_path(
                img_channels_groups_[ii][jj],
                args.lab,
                key_purpose="to_match_it_with_mask_path") == key


def get_metadata(w0_mask_path):
    ###############################################################################
    w0_mask = np.array(Image.open(w0_mask_path)).astype(np.uint16)
    unix = np.unique(w0_mask)[1:]
    num_cells = len(unix)
    if num_cells < args.min_cell_count:
        return None
    # assert num_cells >= args.min_cell_count

    # extract metadata
    well_id = w0_mask_path.stem.split("_")[2]
    fov = w0_mask_path.stem.split("_")[3]
    if containsLetterAndNumber(fov):
        fov = int(re.findall(r'\d+', fov)[0])
    elif fov.isdigit:
        # print(fov)
        fov = int(fov)
    else:
        raise ValueError(f"FOV value {fov} is unacceptable!")

    dosage = args.wellid2dosage[well_id]
    treatment = args.wellid2treatment[well_id]
    cell_line = args.wellid2cellline[well_id]
    density = args.wellid2density[well_id]

    return num_cells, well_id, treatment, cell_line, density, dosage, fov


def get_all_meta_data(w0_mask_paths_):
    N = len(w0_mask_paths_)
    meta_data = np.zeros((N * 700, 6), dtype=object)
    cnt = 0  # my counter
    cnt2 = 0
    slices = np.zeros((N, 2), dtype=np.int64)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for ii, out in tqdm(enumerate(pool.imap(get_metadata, w0_mask_paths_)), total=N):
            if out is not None:
                n, well_id, treatment, cell_line, density, dosage, fov = out
                meta_data[cnt:cnt + n, 0] = well_id
                meta_data[cnt:cnt + n, 1] = treatment
                meta_data[cnt:cnt + n, 2] = cell_line
                meta_data[cnt:cnt + n, 3] = density
                meta_data[cnt:cnt + n, 4] = dosage
                meta_data[cnt:cnt + n, 5] = fov
                slices[cnt2] = (cnt, cnt + n)
                cnt += n
                cnt2 += 1
    meta_data = meta_data[0:cnt]
    slices = slices[0:cnt2]
    return meta_data, slices


def main_test_worker(num_images):
    save_dir = args.main_path / args.experiment / "test"
    save_dir.mkdir(exist_ok=True, parents=True,)
    # some images might have been removed during cellpose segmentation because they had very few cell,
    # those image groups have to be removed.
    img_path_groups, w0_mask_paths, w1_mask_paths, args.num_channels = \
        get_matching_img_group_nuc_mask_cyto_mask(args.main_path, args.experiment, args.lab, mask_folder="Masks")
    w2_mask_paths = [it.parents[0] / f"w2_{'_'.join(it.stem.split('_')[1:])}.png" for it in w0_mask_paths]
    w4_mask_paths = [it.parents[0] / f"w1_{'_'.join(it.stem.split('_')[1:])}.png" for it in w0_mask_paths]
    # Each image group will match with its corresponding mask.
    check_paths_matching(img_path_groups, w0_mask_paths, w1_mask_paths, w2_mask_paths, w4_mask_paths)
    # sometimes some well_ids coming from image files might be missing from the provided platemap,
    # on the other hand, sometimes some well_ids from the platemap might have no corresponding image file
    # in the experiment folder, in both cases the user should be warned by the program.
    warn_user_about_missing_wellids(img_path_groups)

    img_path_groups = img_path_groups[0:num_images]
    w0_mask_paths = w0_mask_paths[0:num_images]
    w1_mask_paths = w1_mask_paths[0:num_images]
    w2_mask_paths = w2_mask_paths[0:num_images]
    w4_mask_paths = w4_mask_paths[0:num_images]

    N = len(w0_mask_paths)
    meta_data, slices = get_all_meta_data(w0_mask_paths)
    w0_features = np.zeros((N * 600, args.num_feat_cols), dtype=object)
    w1_features = np.zeros((N * 600, args.num_feat_cols), dtype=object)
    w2_features = np.zeros((N * 600, args.num_feat_cols), dtype=object)
    w3_features = np.zeros((N * 600, args.num_feat_cols), dtype=object)
    w4_features = np.zeros((N * 600, args.num_feat_cols), dtype=object)
    has_nucleoli = np.zeros((N * 600, 1), dtype=np.uint8)
    nrows = slices[-1, 1]

    my_func = partial(get_my_props,
                      args=args,
                      img_path_groups=img_path_groups,
                      w0_mask_paths=w0_mask_paths,
                      w1_mask_paths=w1_mask_paths,
                      w2_mask_paths=w2_mask_paths,
                      w4_mask_paths=w4_mask_paths,
                      )
    cnt = 0
    for iiii in tqdm(range(num_images)):
        out = my_func(iiii)
        # print("out[0] shape", out[0].shape)
        ncells = len(out[0])
        w0_features[cnt:cnt + ncells] = out[0]
        w1_features[cnt:cnt + ncells] = out[1]
        w2_features[cnt:cnt + ncells] = out[2]
        w3_features[cnt:cnt + ncells] = out[3]
        w4_features[cnt:cnt + ncells] = out[4]
        has_nucleoli[cnt:cnt + ncells] = out[5]
        cnt += ncells
    w0_features = w0_features[0:cnt]
    w1_features = w1_features[0:cnt]
    w2_features = w2_features[0:cnt]
    w3_features = w3_features[0:cnt]
    w4_features = w4_features[0:cnt]
    has_nucleoli = has_nucleoli[0:cnt]
    # ###############################################################################
    # Saving all_props1 as a numpy array
    # all_props1 = np.core.records.fromarrays(all_props1.T, names=cols1)
    # print(all_props1.dtype.names)
    np.save(args.main_path / args.experiment / "test_w0_features.npy", w0_features)
    np.save(args.main_path / args.experiment / "test_w1_features.npy", w1_features)
    np.save(args.main_path / args.experiment / "test_w2_features.npy", w2_features)
    np.save(args.main_path / args.experiment / "test_w3_features.npy", w3_features)
    np.save(args.main_path / args.experiment / "test_w4_features.npy", w4_features)
    np.save(args.main_path / args.experiment / "test_has_nucleoli.npy", has_nucleoli)
    np.save(args.main_path / args.experiment / "test_metadata.npy", meta_data)
    print(f"cnt: {cnt}  nrow: {nrows}")
    assert cnt == nrows
    ################################################################################
    # saving "features" as a csv file
    print("loading the 'features' numpy array ....")
    meta_data = np.load(args.main_path / args.experiment / "test_metadata.npy", allow_pickle=True)
    has_nucleoli = np.load(args.main_path / args.experiment / "test_has_nucleoli.npy", allow_pickle=True)
    w0_features = np.load(args.main_path / args.experiment / "test_w0_features.npy", allow_pickle=True)
    w1_features = np.load(args.main_path / args.experiment / "test_w1_features.npy", allow_pickle=True)
    w2_features = np.load(args.main_path / args.experiment / "test_w2_features.npy", allow_pickle=True)
    w3_features = np.load(args.main_path / args.experiment / "test_w3_features.npy", allow_pickle=True)
    w4_features = np.load(args.main_path / args.experiment / "test_w4_features.npy", allow_pickle=True)
    print(w0_features.shape, w1_features.shape, meta_data.shape)
    w0_cols = [f"Nucleus-{it}" for it in args.feature_cols]
    w1_cols = [f"Cyto-{it}" for it in args.feature_cols]
    w2_cols = [f"Nucleoli-{it}" for it in args.feature_cols]
    w3_cols = [f"Actin-{it}" for it in args.feature_cols]
    w4_cols = [f"Mito-{it}" for it in args.feature_cols]

    print("converted the 'features' numpy array to a pandas dataframe and saving it as a csv file....")
    # features = pd.DataFrame(
    #     np.concatenate(
    #         [meta_data,
    #          has_nucleoli[:, np.newaxis],
    #          w0_features,
    #          w1_features,
    #          w2_features,
    #          w3_features,
    #          w4_features],
    #         axis=1),
    #     columns=args.metadata_cols+["has nucleoli"]+w0_cols+w1_cols+w2_cols+w3_cols+w4_cols)
    # features.to_csv(args.main_path / args.experiment / "features.csv", index=False, float_format="%.2f")
    ###########################################################################################################
    w0_features = pd.DataFrame(
        np.concatenate([meta_data, has_nucleoli, w0_features], axis=1),
        columns=args.metadata_cols + ["has nucleoli"] + w0_cols)
    w1_features = pd.DataFrame(
        np.concatenate([meta_data, has_nucleoli, w1_features], axis=1),
        columns=args.metadata_cols + ["has nucleoli"] + w1_cols)
    w2_features = pd.DataFrame(
        np.concatenate([meta_data, has_nucleoli, w2_features], axis=1),
        columns=args.metadata_cols + ["has nucleoli"] + w2_cols)
    w3_features = pd.DataFrame(
        np.concatenate([meta_data, has_nucleoli, w3_features], axis=1),
        columns=args.metadata_cols + ["has nucleoli"] + w3_cols)
    w4_features = pd.DataFrame(
        np.concatenate([meta_data, has_nucleoli, w4_features], axis=1),
        columns=args.metadata_cols + ["has nucleoli"] + w4_cols)
    w0_features.to_csv(save_dir / "test_w0_features.csv", index=False, float_format="%.2f")
    w1_features.to_csv(save_dir / "test_w1_features.csv", index=False, float_format="%.2f")
    w2_features.to_csv(save_dir / "test_w2_features.csv", index=False, float_format="%.2f")
    w3_features.to_csv(save_dir / "test_w3_features.csv", index=False, float_format="%.2f")
    w4_features.to_csv(save_dir / "test_w4_features.csv", index=False, float_format="%.2f")
    assert cnt == nrows


def main(args):
    save_dir = args.main_path / args.experiment / "features"
    save_dir.mkdir(exist_ok=True, parents=True,)

    manager = mp.Manager()
    # some images might have been removed during cellpose segmentation because they had very few cell,
    # those image groups have to be removed.
    img_path_groups, w0_mask_paths, w1_mask_paths, args.num_channels = \
        get_matching_img_group_nuc_mask_cyto_mask(args.main_path, args.experiment, args.lab, mask_folder="Masks")
    w2_mask_paths = [it.parents[0] / f"w2_{'_'.join(it.stem.split('_')[1:])}.png" for it in w0_mask_paths]
    w4_mask_paths = [it.parents[0] / f"w4_{'_'.join(it.stem.split('_')[1:])}.png" for it in w0_mask_paths]
    # Each image group will match with its corresponding mask.
    check_paths_matching(img_path_groups, w0_mask_paths, w1_mask_paths, w2_mask_paths, w4_mask_paths)
    # sometimes some well_ids coming from image files might be missing from the provided platemap,
    # on the other hand, sometimes some well_ids from the platemap might have no corresponding image file
    # in the experiment folder, in both cases the user should be warned by the program.
    warn_user_about_missing_wellids(img_path_groups)

    N = len(w0_mask_paths)
    meta_data, slices = get_all_meta_data(w0_mask_paths)
    w0_features = np.zeros((N * 600, args.num_feat_cols), dtype=object)
    w1_features = np.zeros((N * 600, args.num_feat_cols), dtype=object)
    w2_features = np.zeros((N * 600, args.num_feat_cols), dtype=object)
    w3_features = np.zeros((N * 600, args.num_feat_cols), dtype=object)
    w4_features = np.zeros((N * 600, args.num_feat_cols), dtype=object)
    has_nucleoli = np.zeros((N * 600, 1), dtype=np.uint8)
    nrows = slices[-1, 1]

    w0_mask_paths = manager.list(w0_mask_paths)
    w1_mask_paths = manager.list(w1_mask_paths)
    w2_mask_paths = manager.list(w2_mask_paths)
    w4_mask_paths = manager.list(w4_mask_paths)
    img_path_groups = manager.list(img_path_groups)
    my_func = partial(get_my_props,
                      args=args,
                      img_path_groups=img_path_groups,
                      w0_mask_paths=w0_mask_paths,
                      w1_mask_paths=w1_mask_paths,
                      w2_mask_paths=w2_mask_paths,
                      w4_mask_paths=w4_mask_paths,)

    cnt = 0
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # for out in pool.imap(my_func, np.arange(N)):
        for out in tqdm(pool.imap(my_func, np.arange(N)), total=N):
            if out is not None:
                ncells = len(out[0])
                w0_features[cnt:cnt + ncells] = out[0]
                w1_features[cnt:cnt + ncells] = out[1]
                w2_features[cnt:cnt + ncells] = out[2]
                w3_features[cnt:cnt + ncells] = out[3]
                w4_features[cnt:cnt + ncells] = out[4]
                has_nucleoli[cnt:cnt + ncells] = out[5]
                cnt += ncells
    w0_features = w0_features[0:cnt]
    w1_features = w1_features[0:cnt]
    w2_features = w2_features[0:cnt]
    w3_features = w3_features[0:cnt]
    w4_features = w4_features[0:cnt]
    has_nucleoli = has_nucleoli[0:cnt]
    # ###############################################################################
    # Saving all_props1 as a numpy array
    # all_props1 = np.core.records.fromarrays(all_props1.T, names=cols1)
    # print(all_props1.dtype.names)
    np.save(save_dir / "w0_features.npy", w0_features)
    np.save(save_dir / "w1_features.npy", w1_features)
    np.save(save_dir / "w2_features.npy", w2_features)
    np.save(save_dir / "w3_features.npy", w3_features)
    np.save(save_dir / "w4_features.npy", w4_features)
    np.save(save_dir / "has_nucleoli.npy", has_nucleoli)
    np.save(save_dir / "metadata.npy", meta_data)
    print(f"cnt: {cnt}  nrow: {nrows}")
    assert cnt == nrows
    ################################################################################
    # saving "features" as a csv file
    print("loading the 'features' numpy array ....")
    meta_data = np.load(save_dir / "metadata.npy", allow_pickle=True)
    has_nucleoli = np.load(save_dir / "has_nucleoli.npy", allow_pickle=True)
    w0_features = np.load(save_dir / "w0_features.npy", allow_pickle=True)
    w1_features = np.load(save_dir / "w1_features.npy", allow_pickle=True)
    w2_features = np.load(save_dir / "w2_features.npy", allow_pickle=True)
    w3_features = np.load(save_dir / "w3_features.npy", allow_pickle=True)
    w4_features = np.load(save_dir / "w4_features.npy", allow_pickle=True)
    print(w0_features.shape, w1_features.shape, meta_data.shape)
    w0_cols = [f"Nucleus-{it}" for it in args.feature_cols]
    w1_cols = [f"Cyto-{it}" for it in args.feature_cols]
    w2_cols = [f"Nucleoli-{it}" for it in args.feature_cols]
    w3_cols = [f"Actin-{it}" for it in args.feature_cols]
    w4_cols = [f"Mito-{it}" for it in args.feature_cols]

    print("converted the 'features' numpy array to a pandas dataframe and saving it as a csv file....")
    # features = pd.DataFrame(
    #     np.concatenate(
    #         [meta_data,
    #          has_nucleoli[:, np.newaxis],
    #          w0_features,
    #          w1_features,
    #          w2_features,
    #          w3_features,
    #          w4_features],
    #         axis=1),
    #     columns=args.metadata_cols+["has nucleoli"]+w0_cols+w1_cols+w2_cols+w3_cols+w4_cols)
    # features.to_csv(save_dir / "features.csv", index=False, float_format="%.2f")
    ###########################################################################################################
    w0_features = pd.DataFrame(
        np.concatenate([meta_data, has_nucleoli, w0_features], axis=1),
        columns=args.metadata_cols + ["has nucleoli"] + w0_cols)
    w1_features = pd.DataFrame(
        np.concatenate([meta_data, has_nucleoli, w1_features], axis=1),
        columns=args.metadata_cols + ["has nucleoli"] + w1_cols)
    w2_features = pd.DataFrame(
        np.concatenate([meta_data, has_nucleoli, w2_features], axis=1),
        columns=args.metadata_cols + ["has nucleoli"] + w2_cols)
    w3_features = pd.DataFrame(
        np.concatenate([meta_data, has_nucleoli, w3_features], axis=1),
        columns=args.metadata_cols + ["has nucleoli"] + w3_cols)
    w4_features = pd.DataFrame(
        np.concatenate([meta_data, has_nucleoli, w4_features], axis=1),
        columns=args.metadata_cols + ["has nucleoli"] + w4_cols)
    w0_features.to_csv(save_dir / "w0_features.csv", index=False, float_format="%.2f")
    w1_features.to_csv(save_dir / "w1_features.csv", index=False, float_format="%.2f")
    w2_features.to_csv(save_dir / "w2_features.csv", index=False, float_format="%.2f")
    w3_features.to_csv(save_dir / "w3_features.csv", index=False, float_format="%.2f")
    w4_features.to_csv(save_dir / "w4_features.csv", index=False, float_format="%.2f")
    ###################################################################################################


if __name__ == "__main__":
    st_time = time.time()
    args = create_shared_multiprocessing_name_space_object(args)
    ignore_imgaeio_warning()
    if args.testing:
        main_test_worker(num_images=3)
    else:
        # run the feature extraction function on the entire set of images using multiprocessing pool
        main(args)
    print(f"Time taken to complete feature extraction: {time.time() - st_time} seconds.")

    # img_path_groups, nucleus_mask_paths, w1_mask_paths, args.num_channels = \
    #     get_matching_img_group_nuc_mask_cyto_mask(args.main_path, args.experiment, args.lab, mask_folder="Masks")
    # img_path = img_path_groups[0][0]
    # img = tifffile.imread((img_path))
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(img, origin="upper", cmap="gray")
    # axes[1].imshow(img, origin="upper", cmap="gray")
    # axes[0].text(500, 500, 'KP', bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=30, color='blue')
    # axes[1].text(500, 500, 'KP', bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=30, color='blue')
    # plt.show()
