import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T_
from torchvision.transforms import functional as F_
import elasticdeform

"""
Taken from:
https://github.com/pytorch/vision/issues/230
https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py"""


def pad_if_smaller(img, size, fill=0):
    min_size = np.minimum(img.size(0), img.size(1))
    if min_size < size:
        ow, oh = img.size(0), img.size(1)
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F_.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F_.resize(image, size)
        target = F_.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F_.hflip(image)
            target = F_.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T_.RandomCrop.get_params(image, (self.size, self.size))
        image = F_.crop(image, *crop_params)
        target = F_.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F_.center_crop(image, self.size)
        target = F_.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = torch.from_numpy(np.float32(image))
        target = torch.as_tensor(np.float32(target))
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F_.normalize(image, mean=self.mean, std=self.std)
        return image, target


class NormalizeV2(object):

    def __call__(self, image, target):
        """Assuming target values are in {0, 1}"""
        max_, min_ = np.amax(image), np.amin(image)
        image = (image-min_)/(max_-min_)
        return image, target


"""My own implementation"""
# class ZeroPad(object):
#
#     def __init__(self, side, shape):
#         self.shape = shape
#         self.side = side
#
#     def __call__(self, sample):
#         image, bbox, label = sample[0], sample[1], sample[2]
#         L, R, T, B = 0, image.shape[1], 0, image.shape[0]  # image dimensions
#         t, h, l, w = bbox['top'], bbox['height'], bbox['left'], bbox['width']  # bbox inside image dimensions
#         b, r = t + h, l + w
#
#         if l == L and w < self.side:  # left pad
#             pad = ((0, 0), (0, 0), (self.side - w, 0))
#             image = np.lib.pad(image, pad, mode='constant', constant_values=0)
#
#         if r == R and w < self.side:  # right pad
#             pad = ((0, 0), (0, 0), (0, self.side - w))
#             image = np.lib.pad(image, pad, mode='constant', constant_values=0)
#
#         if t == T and h < self.side:  # top pad
#             pad = ((0, 0), (self.side - h, 0), (0, 0))
#             image = np.lib.pad(image, pad, mode='constant', constant_values=0)
#
#         if b == B and h < self.side:  # bottom pad
#             pad = ((0, 0), (0, self.side - h), (0, 0))
#             image = np.lib.pad(image, pad, mode='constant', constant_values=0)
#
#         return image, label


class UnsqueezeV1(object):
    """
    Add a fourth dimension for 3D CNN.
    Meaning, make 4D data
    (Batch x (Planes) x Channels x Width x Height).
    Add a second dimension for 1D CNN, making 2D data
    (Batch x (Planes) x Channels).
    """

    def __call__(self, image):
        return torch.unsqueeze(image, 0)


class ToTensorV1(object):
    """
    """

    def __call__(self, image):
        return torch.from_numpy(np.float32(image))


class Unsqueeze(object):
    """
    Add a fourth dimension for 3D CNN.
    Meaning, make 4D data
    (Batch x (Planes) x Channels x Width x Height).
    Add a second dimension for 1D CNN, making 2D data
    (Batch x (Planes) x Channels).
    """

    def __call__(self, image, target):
        return torch.unsqueeze(image, 0), torch.unsqueeze(target, 0)


class ElasticDeform(object):
    def __init__(self, sigma, points):
        self.sigma = sigma
        self.points = points

    def __call__(self, image, target):
        image = elasticdeform.deform_random_grid(image, sigma=self.sigma, points=self.points)
        return image, target
