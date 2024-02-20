import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
# from pathlib import Path, WindowsPath
import itertools
import tifffile
from cellpose import models
from skimage import io

import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
# from skimage.measure import label
# from skimage.color import label2rgb
# from skimage.segmentation import find_boundaries

from utils.args import args
from utils.helpers import get_key_from_img_path, get_img_paths, \
    get_sitk_img_thresholding_mask_v1, get_overlay
from utils.dynamics import compute_masks

from cellpose import transforms
from cellpose.resnet_torch import CPnet
from cellpose.utils import get_masks_unet, fill_holes_and_remove_small_masks
import cv2

import torch
from tqdm import tqdm


# TODO: Add comments and functions and explanations
# TODO: NUM_CHANNELS calculation needs to be fixed!?
if args.testing:
    args.batch_size = 6


def get_batches(X, Y, batch_size):
    # batch generator
    n_samples = len(X)
    indices = np.arange(n_samples)
    # # Shuffle at the start of epoch
    # np.random.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        end = np.minimum(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]


def main():
    img_paths = get_img_paths(args.main_path, args.experiment, args.lab)
    # img = np.concatenate([
    #     tifffile.imread(img_paths[0])[np.newaxis],
    #     tifffile.imread(img_paths[1])[np.newaxis]], axis=0)
    img = tifffile.imread(img_paths[1])
    img = np.concatenate([img[np.newaxis], np.zeros_like(img)[np.newaxis]], axis=0)
    print("loaded image shape", img.shape)

    nbase = [2, 32, 64, 128, 256]
    # cyto2torch_2
    # cyto_2
    tile = True
    diam_mean = 30
    model_path = r"C:\Users\safaripoorfatide\.cellpose\models\cyto2torch_2"

    model = CPnet(
        nbase,
        nout=3,
        sz=3,
        residual_on=True,
        style_on=True,
        concatenation=False,
        mkldnn=False,
    )
    model.load_model(filename=model_path, cpu=False)
    model.to("cuda")
    model.eval()
    # print(model)

    rescale = diam_mean / args.cellpose_nucleus_diam

    img = img.transpose((1, 2, 0))
    if img.ndim < 4:
        img = img[np.newaxis, ...]

    img = transforms.normalize_img(img, invert=False)
    img = transforms.resize_image(img, rsz=rescale)
    print(img.shape)
    # make image nchan x Ly x Lx for net
    img = np.transpose(img, (0, 3, 1, 2))
    img = img[0]
    print(img.shape)
    detranspose = (1, 2, 0)

    ######################################################################
    # _run_net
    #######
    # pad image for net so Ly and Lx are divisible by 4
    img, ysub, xsub = transforms.pad_image_ND(img)
    slc = [slice(0, img.shape[n] + 1) for n in range(img.ndim)]
    slc[-3] = slice(0, 3 + 1)
    slc[-2] = slice(ysub[0], ysub[-1] + 1)
    slc[-1] = slice(xsub[0], xsub[-1] + 1)
    slc = tuple(slc)

    #############################
    #  _run_tiled
    #############
    IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(
        img, bsize=224, augment=False, tile_overlap=.1)
    ny, nx, nchan, ly, lx = IMG.shape
    print(IMG.shape)
    IMG = np.reshape(IMG, (ny * nx, nchan, ly, lx))
    batch_size = 32
    niter = int(np.ceil(IMG.shape[0] / batch_size))
    nout = 3

    y = np.zeros((IMG.shape[0], nout, ly, lx))
    for k in tqdm(range(niter)):
        irange = np.arange(batch_size * k, min(IMG.shape[0], batch_size * k + batch_size))

        with torch.no_grad():
            x = torch.from_numpy(np.float32(IMG[irange]))
            x = x.to("cuda:0")
            y0, _ = model(x)
            y0 = y0.cpu().numpy()
            y[irange] = y0.reshape(len(irange), y0.shape[-3], y0.shape[-2], y0.shape[-1])

    yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
    yf = yf[:, :img.shape[1], :img.shape[2]]
    print(f"run_tiled yf {yf.shape}")
    #############
    #  _run_tiled
    ###############################

    # slice out padding
    yf = yf[slc]
    # transpose so channels axis is last again
    yf = np.transpose(yf, detranspose)
    print(f"_run_net yf {yf.shape}")
    #######################################################################
    # _run_net
    ##################################################################

    # cell_threshold = 2.0
    # boundary_threshold = 0.5
    # mask = get_masks_unet(yf, cell_threshold, boundary_threshold)
    # mask = fill_holes_and_remove_small_masks(mask, min_size=15)
    # mask = transforms.resize_image(
    #     mask,
    #     1930, 1930,
    #     interpolation=cv2.INTER_NEAREST)

    # resample = True
    yf = transforms.resize_image(yf, 1930, 1930)
    print(f"resample yf {yf.shape} {yf.dtype}")
    print(np.unique(yf))

    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    axes[0, 0].imshow(tifffile.imread(img_paths[0]), cmap="gray")
    axes[0, 1].imshow(tifffile.imread(img_paths[1]), cmap="gray")
    axes[0, 2].imshow(tifffile.imread(img_paths[2]), cmap="gray")

    axes[1, 0].imshow(yf[:, :,  0], cmap="gray")
    axes[1, 1].imshow(yf[:, :,  1], cmap="gray")
    axes[1, 2].imshow(yf[:, :,  2], cmap="gray")
    plt.show()

    # compute the mask
    cellprob = yf[:, :, 2]
    dP = yf[:, :, :2].transpose((2, 0, 1))

    niter = 1 / rescale * 200
    print(f"niter {niter}")
    mask, _ = compute_masks(
        dP, cellprob,
        niter=niter,
        cellprob_threshold=0.0,
        flow_threshold=.4,
        interp=True,
        resize=None,
        use_gpu=True, device='cuda')

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    axes[0].imshow(tifffile.imread(img_paths[0]))
    axes[1].imshow(tifffile.imread(img_paths[1]))
    axes[2].imshow(tifffile.imread(mask))
    plt.show()

    # with torch.no_grad():
    #     for k in range(niter):
    #         irange = np.arange(batch_size * k, min(IMG.shape[0], batch_size * k + batch_size))
    #         y0, style = model(IMG[irange])






    # # f"{ASSAY}_A02_T0001F002L01A02Z01C01.tif"
    # img_paths = get_img_paths(args.main_path, args.experiment, args.lab)
    #
    # # ################################################################
    # # group files that are channels of the same image together.
    # keys, img_paths_groups = [], []
    # for item in itertools.groupby(
    #         img_paths, key=lambda x: get_key_from_img_path(x, args.lab, key_purpose="to_group_channels")):
    #     keys.append(item[0])
    #     img_paths_groups.append(list(item[1]))
    # keys = np.array(keys, dtype=object)
    # args.num_channels = len(img_paths_groups[0])
    # print("num_channels: ", args.num_channels)
    # assert args.batch_size > args.num_channels
    #
    # img_paths_groups = np.array(img_paths_groups, dtype=object)
    # # create the generator object my_iter to read groups of images in batches
    # # which significantly speeds up Cellpose's segmentation compared to reading one image at a time
    # my_iter = get_batches(keys, img_paths_groups, batch_size=args.batch_size)
    # ######################################################################
    #
    # for iii, (batch_keys, batch_paths) in enumerate(my_iter):
    #     # if batch_keys[0][1] not in ["O05", "O06", "P05", "P06"]:
    #     #     continue
    #     N = len(batch_paths)
    #     batch_imgs = np.zeros((N, args.num_channels, args.height, args.width), dtype=np.float32)
    #
    #     # print(batch_keys[0])
    #     for jjj, pgroup in enumerate(batch_paths):
    #         for kkk, channel in enumerate(pgroup):
    #             batch_imgs[jjj, kkk] = tifffile.imread(channel)



if __name__ == "__main__":
    main()
