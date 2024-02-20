import re
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image

from utils.args import args, ignore_imgaeio_warning, create_shared_multiprocessing_name_space_object
from utils.helpers import sort_key_for_imgs, sort_key_for_masks, remove_img_path_groups_with_no_masks,\
    check_paths_matching, warn_user_about_missing_wellids, containsLetterAndNumber

from functools import partial
import multiprocessing as mp

from utils.extensions_and_mods import get_features


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
        remove_img_path_groups_with_no_masks(args.main_path, args.experiment, args.plate_protocol, mask_folder="Masks")
    w2_mask_paths = [it.parents[0] / f"w2_{'_'.join(it.stem.split('_')[1:])}.png" for it in w0_mask_paths]
    w4_mask_paths = [it.parents[0] / f"w4_{'_'.join(it.stem.split('_')[1:])}.png" for it in w0_mask_paths]
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
    w0_features = np.zeros((N * 600, args.num_feat_cols), dtype=np.float32)
    w1_features = np.zeros((N * 600, args.num_feat_cols), dtype=np.float32)
    w2_features = np.zeros((N * 600, args.num_feat_cols), dtype=np.float32)
    w3_features = np.zeros((N * 600, args.num_feat_cols), dtype=np.float32)
    w4_features = np.zeros((N * 600, args.num_feat_cols), dtype=np.float32)
    has_nucleoli = np.zeros((N * 600, 1), dtype=bool)
    nrows = slices[-1, 1]

    my_func = partial(get_features,
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
        ncells = len(out[0][0])
        w0_features[cnt:cnt + ncells] = out[0][0]
        w1_features[cnt:cnt + ncells] = out[0][1]
        w2_features[cnt:cnt + ncells] = out[0][2]
        w3_features[cnt:cnt + ncells] = out[0][3]
        w4_features[cnt:cnt + ncells] = out[0][4]
        has_nucleoli[cnt:cnt + ncells] = out[1]
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
    img_path_groups, w0_mask_paths, w1_mask_paths, args.num_channels = remove_img_path_groups_with_no_masks(
        args.main_path, args.experiment, args.plate_protocol, mask_folder="Masks")
    w2_mask_paths = [it.parents[0] / f"w2_{'_'.join(it.stem.split('_')[1:])}.png" for it in w0_mask_paths]
    w4_mask_paths = [it.parents[0] / f"w4_{'_'.join(it.stem.split('_')[1:])}.png" for it in w0_mask_paths]
    # Each image group will match with its corresponding mask.
    check_paths_matching(img_path_groups, w0_mask_paths, w1_mask_paths, w2_mask_paths, w4_mask_paths)
    # sometimes some well_ids coming from image files might be missing from the provided platemap,
    # on the other hand, sometimes some well_ids from the platemap might have no corresponding image file
    # in the experiment folder, in both cases the user should be warned by the program.
    warn_user_about_missing_wellids(img_path_groups)

    N = len(w0_mask_paths)
    #############################################################
    # getting meta data
    meta_data, slices = get_all_meta_data(w0_mask_paths)
    #############################################################

    w0_features = np.zeros((N * 600, args.num_feat_cols), dtype=np.float32)
    w1_features = np.zeros((N * 600, args.num_feat_cols), dtype=np.float32)
    w2_features = np.zeros((N * 600, args.num_feat_cols), dtype=np.float32)
    w3_features = np.zeros((N * 600, args.num_feat_cols), dtype=np.float32)
    w4_features = np.zeros((N * 600, args.num_feat_cols), dtype=np.float32)
    has_nucleoli = np.zeros((N * 600, 1), dtype=bool)
    nrows = slices[-1, 1]

    w0_mask_paths = manager.list(w0_mask_paths)
    w1_mask_paths = manager.list(w1_mask_paths)
    w2_mask_paths = manager.list(w2_mask_paths)
    w4_mask_paths = manager.list(w4_mask_paths)
    img_path_groups = manager.list(img_path_groups)
    my_func = partial(get_features,
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
                ncells = len(out[0][0])
                w0_features[cnt:cnt + ncells] = out[0][0]
                w1_features[cnt:cnt + ncells] = out[0][1]
                w2_features[cnt:cnt + ncells] = out[0][2]
                w3_features[cnt:cnt + ncells] = out[0][3]
                w4_features[cnt:cnt + ncells] = out[0][4]
                has_nucleoli[cnt:cnt + ncells] = out[1]
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
    ###########################################################################################################
    # creating dataframes and saving
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
    mp.freeze_support()
    args = create_shared_multiprocessing_name_space_object(args)
    ignore_imgaeio_warning()
    if args.testing:
        main_test_worker(num_images=3)
    else:
        # run the feature extraction function on the entire set of images using multiprocessing pool
        main(args)
    print(f"Time taken to complete feature extraction: {time.time() - st_time} seconds.")

    # img_path_groups, nucleus_mask_paths, w1_mask_paths, args.num_channels = \
    #     get_matching_img_group_nuc_mask_cyto_mask(args.main_path, args.experiment, args.plate_protocol,
    #     mask_folder="Masks")
    # img_path = img_path_groups[0][0]
    # img = tifffile.imread((img_path))
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(img, origin="upper", cmap="gray")
    # axes[1].imshow(img, origin="upper", cmap="gray")
    # axes[0].text(500, 500, 'KP', bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=30, color='blue')
    # axes[1].text(500, 500, 'KP', bbox=dict(fill=False, edgecolor='red', linewidth=2), fontsize=30, color='blue')
    # plt.show()
