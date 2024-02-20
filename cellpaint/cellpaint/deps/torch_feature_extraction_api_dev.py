import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.multiprocessing import cpu_count

from tqdm import tqdm

import numpy as np
from PIL import Image
import scipy.ndimage as ndi

from cellpaint.steps_single_plate.step0_args import Args
from cellpaint.utils.img_files_dep import remove_img_path_groups_with_no_masks, load_img, sort_key_for_imgs


class CellPaintDataset(TensorDataset):
    sort_purpose = "to_group_channels"
    def __init__(self,
                 args,
                 img_path_groups,
                 w0_mask_paths,
                 w1_mask_paths,
                 w2_mask_paths,
                 w4_mask_paths,
                 transforms
                 ):
        super().__init__()
        self.args = args
        self.img_path_groups = img_path_groups
        self.w0_mask_paths = w0_mask_paths
        self.w1_mask_paths = w1_mask_paths
        self.w2_mask_paths = w2_mask_paths
        self.w4_mask_paths = w4_mask_paths

        self.img_ids = ["_".join(sort_key_for_imgs(it[0], self.args.plate_protocol, self.sort_purpose))
                        for it in self.img_path_groups]
        self.img_numids = np.arange(len(self.img_ids))
        self.id2num = dict(zip(self.img_ids, self.img_numids))
        self.num2id = dict(zip(self.img_numids, self.img_ids))
        self.transforms = transforms
        print("init done!!!")

    def get_mask(self, index):
        w0_mask = np.array(Image.open(self.w0_mask_paths[index])).astype(np.uint16)  # nuclei mask
        w1_mask = np.array(Image.open(self.w1_mask_paths[index])).astype(np.uint16)  # w1 mask
        w2_mask = np.array(Image.open(self.w2_mask_paths[index])).astype(np.uint16)  # w2 mask
        w4_mask = np.array(Image.open(self.w4_mask_paths[index])).astype(np.uint16)  # w4 mask

        return np.concatenate([w0_mask[np.newaxis],
                               w1_mask[np.newaxis],
                               w2_mask[np.newaxis],
                               w1_mask[np.newaxis],
                               w4_mask[np.newaxis]], axis=0)

    def __getitem__(self, index):
        # img of the fov with 5 channels: (DAPI, CYTO, w2, actin, MITO)
        img = load_img(self.img_path_groups[index],
                       self.args.num_channels,
                       self.args.height,
                       self.args.width)
        mask = self.get_mask(index)
        img_id = self.id2num[self.img_ids[index]]

        if self.transforms:
            img, mask, img_id = self.transforms(img, mask, img_id)

        return img, mask, img_id

    def __len__(self):
        # TODO: maybe change this total number of bounding boxes in all images.
        # maybe not! we'll see!
        return len(self.img_path_groups)


class GetPaddedObjects(object):
    def __init__(self, cin=5, side=600):
        self.side = side
        self.cin = cin

    def get_pad_tuple(self, mask_crop):
        hh, ww = mask_crop.shape[1], mask_crop.shape[2]
        padh, padw = (self.side - hh) // 2, (self.side - ww) // 2
        return (0, 0), (padh, self.side-hh-padh), (padw, self.side-ww-padw)

    def __call__(self, img, mask, img_id):
        """
        img: a (cin, h, w) uint16 np.array
        mask: a (cin, h, w) uint16 np.array
        img_id: an integer, identifying the image
        """
        # print(img.shape, mask.shape)
        # for each bbox we need img_crop, mask_crop, obj_label, and img_id
        # to be able retrieve its metadata later
        max_ = np.amax(mask[1])
        img_batch = np.zeros((max_, self.cin, self.side, self.side), dtype=np.uint16)
        mask_batch = np.zeros((max_, self.cin, self.side, self.side), dtype=np.uint16)
        objects = ndi.find_objects(mask[1], max_label=max_)
        # TODO: Do this for loop as function in C/C++ for a significant speed up!!!
        #  This might require a lot of work!
        #  So, as Pnakaj always says: "Let's leave it FOR THE TIME BEING!!!"
        cnt = 0
        for ii in tqdm(range(max_)):  # loops over each roi/bbox
            if objects[ii] is None:  # if there is no cell skip it (this cell was removed in preprocessing).
                continue
            obi_label = ii + 1
            y0, x0, y1, x1 = objects[ii][0].start, objects[ii][1].start, \
                             objects[ii][0].stop, objects[ii][1].stop
            # get the mask crops
            mask_crop = mask[:, y0:y1, x0:x1] == obi_label
            img_crop = img[:, y0:y1, x0:x1] * mask_crop
            # pad them while keeping them almost centered
            padding = self.get_pad_tuple(mask_crop)
            # print(ii, mask_crop.shape, img_crop.shape, padding)
            mask_crop = np.pad(mask_crop, padding, 'constant', constant_values=(0, 0))
            img_crop = np.pad(img_crop, padding, 'constant', constant_values=(0, 0))
            # print(ii, mask_crop.shape, img_crop.shape, (y1-y0, x1-x0))
            mask_batch[cnt] = mask_crop
            img_batch[cnt] = img_crop
            cnt += 1
        mask_batch = mask_batch[0:cnt]
        img_batch = img_batch[0:cnt]
        # print(2*'\n')

        imgid_batch = np.repeat(img_id, repeats=cnt)
        return img_batch, mask_batch, imgid_batch


class ComposeTriplet(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, imgid):
        for t in self.transforms:
            img, mask, imgid = t(img, mask, imgid)
        return img, mask, imgid


class ToTensorTriplet(object):
    def __call__(self, img, mask, imgid):
        img = torch.as_tensor(np.float32(img))
        mask = torch.as_tensor(np.float32(mask))
        imgid = torch.tensor(imgid,)

        return img, mask, imgid


def custom_collate(batch):
    imgs = torch.cat([item[0] for item in batch], dim=0)
    masks = torch.cat([item[1] for item in batch], dim=0)
    imgids = torch.cat([item[2] for item in batch], dim=0)

    return imgs, masks, imgids


def area(mask):
    """mask: torch.tensor(N, 5, H, W)"""
    return torch.sum(mask, dim=(2,3))


def area_convex(mask):
    """mask: torch.tensor(N, 5, H, W)"""
    mask_convex = convex_hull_image(mask)
    return torch.sum(mask_convex, dim=(2,3))


def convex_hull_image(image, offset_coordinates=True, tolerance=1e-10):
    return


def main():
    experiment = "20221109-CP-Fabio-DRC-BM-P02"
    args = Args(experiment=experiment, mode="full").args

    img_path_groups, w0_mask_paths, w1_mask_paths, args.num_channels = \
        remove_img_path_groups_with_no_masks(
            args.main_path, args.experiment, args.analysis_save_path,
            args.plate_protocol, mask_folder=args.masks_path_p2.stem,
            nucleus_idx=args.nucleus_idx, cyto_idx=args.cyto_idx)
    w2_mask_paths = [it.parents[0] / f"w2_{'_'.join(it.stem.split('_')[1:])}.png" for it in w0_mask_paths]
    w4_mask_paths = [it.parents[0] / f"w4_{'_'.join(it.stem.split('_')[1:])}.png" for it in w0_mask_paths]
    print("creating dataset ...")
    dataset = CellPaintDataset(
        args,
        img_path_groups,
        w0_mask_paths,
        w1_mask_paths,
        w2_mask_paths,
        w4_mask_paths,
        transforms=ComposeTriplet([GetPaddedObjects(cin=5, side=600), ToTensorTriplet()]))
    print("creating dataloader ...")
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
        collate_fn=custom_collate)
    print("begin iterating ...")
    for ii, sample in enumerate(data_loader):
        print(ii, sample[0].size())


if __name__ == "__main__":
    main()