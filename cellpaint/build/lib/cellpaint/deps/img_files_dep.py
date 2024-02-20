import warnings
import imageio.core.util

import itertools

import tifffile
import numpy as np
from string import ascii_lowercase, ascii_uppercase


class FoundItem(Exception):
    pass


def ignore_imgaeio_warning():

    def ignore_warnings(*args, **kwargs):
        pass

    warnings.filterwarnings('ignore')
    imageio.core.util._precision_warn = ignore_warnings


def sort_key_for_imgs(file_path, sort_purpose, plate_protocol):
    """
    Get sort key from the img filename.
    The function is used to sort image filenames for a specific experiment taken with a specific plate_protocol.
    """
    # plate_protocol = plate_protocol.lower()
    if plate_protocol == "greiner" or plate_protocol == "perkinelmer":
        """img filename example:
        .../AssayPlate_PerkinElmer_CellCarrier-384/
        AssayPlate_PerkinElmer_CellCarrier-384_A02_T0001F001L01A01Z01C02.tif
        .../AssayPlate_Greiner_#781896/
        AssayPlate_Greiner_#781896_A04_T0001F001L01A01Z01C01.tif
        """
        folder = file_path.parents[1].stem  # "AssayPlate_PerkinElmer_CellCarrier-384"
        filename = file_path.stem  # AssayPlate_PerkinElmer_CellCarrier-384_A02_T0001F001L01A01Z01C02.tif
        split = filename.split("_")
        well_id = split[-2]  # A02
        fkey = split[-1]  # T0001F001L01A01Z01C02
        inds = [fkey.index(ll, 0, len(fkey)) for ll in fkey if ll.isalpha()]
        inds.append(None)
        timestamp, fov, id2, x1, zslice, channel = \
            [fkey[inds[ii]:inds[ii + 1]] for ii in range(len(inds) - 1)]
        # print(well_id, parts)

    elif plate_protocol == "combchem":
        """img filename example:
        .../P000025-combchem-v3-U2OS-24h-L1-copy1/
        P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s5_w3C00578CF-AD4A-4A29-88AE-D2A2024F9A92.tif"""
        folder = file_path.parents[1].stem
        filename = file_path.stem
        split = filename.split("_")
        well_id = split[1]
        fov = split[2][1]
        channel = split[3][1]
        # print(well_id, parts)

    elif "cpg0012" in plate_protocol:
        """img filename example:
        .../P000025-combchem-v3-U2OS-24h-L1-copy1/
        P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s5_w3C00578CF-AD4A-4A29-88AE-D2A2024F9A92.tif"""
        folder = file_path.parents[1].stem
        filename = file_path.stem
        split = filename.split("_")
        well_id = split[1].upper()
        fov = split[2].upper()
        channel = split[3][0:2]

    elif "cpg0001" in plate_protocol:
        """img filename example:
        .../P000025-combchem-v3-U2OS-24h-L1-copy1/
        P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s5_w3C00578CF-AD4A-4A29-88AE-D2A2024F9A92.tif"""
        folder = file_path.parents[1].stem
        filename = file_path.stem
        split = filename.split("-")
        row, col = int(split[0][1:3]) - 1, split[0][4:6]
        well_id = f"{ascii_uppercase[row]}{col}"
        fov = split[0][6:9]
        channel = split[1][2]
    else:
        raise NotImplementedError(f"{plate_protocol} is not implemented yet!!!")

    if sort_purpose == "to_sort_channels":
        return folder, well_id, fov, channel

    elif sort_purpose == "to_group_channels":
        # print(folder, well_id, fov)
        return folder, well_id, fov

    elif sort_purpose == "to_match_it_with_mask_path":
        return f"{well_id}_{fov}"
    elif sort_purpose == "to_get_well_id":
        return well_id
    elif sort_purpose == "to_get_well_id_and_fov":
        return well_id, fov
    else:
        raise ValueError(f"sort purpose {sort_purpose} does not exist!!!")


def sort_key_for_masks(mask_path):
    """example: .../w0_A02_F001.png"""
    split = mask_path.stem.split("_")
    well_id = split[-2]
    fov = split[-1]
    return f"{well_id}_{fov}"


def get_img_paths(main_path, experiment, plate_protocol, suffix="tif"):
    """AssayPlate_Greiner_#781896_A04_T0001F001L01A01Z01C01"""
    """AssayPlate_PerkinElmer_CellCarrier-384_A02_T0001F001L01A01Z01C02.tif"""
    """P000025-combchem-v3-U2OS-24h-L1-copy1_B02_s5_w3C00578CF-AD4A-4A29-88AE-D2A2024F9A92.tif"""

    img_paths = list((main_path / experiment).rglob(f"*.{suffix}"))
    if len(img_paths) < 500:
        raise ValueError(f"Not enough image files found!!! "
                         f"please check the image folder {main_path/experiment/plate_protocol} "
                         f"and make sure your images files were actually copied properly!!!")

    if plate_protocol.lower() in ['greiner', 'perkinelmer']:
        # sometimes there are other tif files in the experiment folder that are not an image, so we have to
        # remove them from img_paths, so that they do not mess-up the analysis.
        img_paths = [item for item in img_paths if item.stem.split("_")[-1][0:5] == "T0001"]
    img_paths = sorted(
        img_paths,
        key=lambda x: sort_key_for_imgs(x, plate_protocol=plate_protocol, sort_purpose="to_sort_channels"))
    return img_paths


def get_all_file_paths(
        main_path, experiment, analysis_save_path,
        plate_protocol="PerkinElmer", mask_folder="MasksP1", nucleus_idx=0, cyto_idx=1):
    img_paths = get_img_paths(main_path, experiment, plate_protocol)
    # print(nucleus_idx, cyto_idx, analysis_save_path / mask_folder)
    mask0_paths = list((analysis_save_path / mask_folder).rglob(f"w{nucleus_idx}_*.png"))  # Nucleus Masks
    mask1_paths = list((analysis_save_path / mask_folder).rglob(f"w{cyto_idx}_*.png"))  # Cytoplasm Masks

    # if len(mask0_paths) < 100 or len(mask1_paths) < 100:
    #     raise ValueError(
    #         f"There are not enough mask files found in either of {mask0_paths} folder or {mask1_paths} folder!!!"
    #         f"Make sure that the Cellpaint Step I completed!!!")

    # print(len(img_paths), len(mask0_paths), len(mask1_paths))
    mask0_paths = sorted(mask0_paths, key=sort_key_for_masks)
    mask1_paths = sorted(mask1_paths, key=sort_key_for_masks)

    #  grouping channels of each image together!
    keys, img_path_groups = [], []
    for item in itertools.groupby(
            img_paths, key=lambda x: sort_key_for_imgs(
                x, sort_purpose="to_group_channels", plate_protocol=plate_protocol)):
        keys.append(item[0])
        img_path_groups.append(list(item[1]))
    num_channels = len(img_path_groups[0])
    # making sure there is a one-to-one correspondence/matching between img_paths, mask0_paths, and mask1_paths.
    for it0, it1, it2 in zip(mask0_paths, mask1_paths, keys):
        split0, split1 = it0.stem.split("_"), it1.stem.split("_")
        # print(split0, split1, it2)
        assert "_".join(split0[-2:]) == "_".join(split0[-2:]) == "_".join(it2[1:]), \
            f"mask and image file names must match!!! but found:" \
            f"mask0-key={split0},  mask1-key={split1}  image-key={it2}"
    # img_path_groups = np.vstack(img_path_groups)
    return img_path_groups, mask0_paths, mask1_paths, num_channels


def load_img(file_paths, num_channels, height, width):
    """load channel tiff files belonging to a single image into a single tiff file.
    file_paths: list of channel tiff files
    example:
    [.../AssayPlate_PerkinElmer_CellCarrier-384_A02_T0001F001L01A01Z01C01.tiff,
     .../AssayPlate_PerkinElmer_CellCarrier-384_A02_T0001F001L01A01Z01C02.tiff,
     .../AssayPlate_PerkinElmer_CellCarrier-384_A02_T0001F001L01A01Z01C03.tiff,
     .../AssayPlate_PerkinElmer_CellCarrier-384_A02_T0001F001L01A01Z01C04.tiff,
     .../AssayPlate_PerkinElmer_CellCarrier-384_A02_T0001F001L01A01Z01C05.tiff,]
     num_channels: The number of channels contained in the image,
     height: height of each image (It is always fixed for a single experiment.),
     width: width of each image (It is always fixed for a single experiment.),
    """
    if len(file_paths) != num_channels:
        return np.zeros((num_channels, height, width), dtype=np.float32)

    img = np.zeros((num_channels, height, width), dtype=np.float32)
    for jj in range(num_channels):
        img[jj] = tifffile.imread(file_paths[jj])
    return img


def find_first_occurance(string_list, substring):
    N = len(string_list)
    for ii in range(N):
        if substring in string_list[ii]:
            return ii


def unique_with_preservered_order(mylist):
    a = np.array(mylist)
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]



