# pylint: disable-all

import torch
from torch.utils.data import Dataset
from skimage.color import rgb2lab
from skimage.io import imread
from skimage.transform import resize
import torchvision.datasets as dsets
from os import listdir, walk
from os.path import join, isfile, isdir
import numpy as np
from colorutils import NNEncode, uint_color2tanh_range
from sklearn.model_selection import train_test_split
import time


class CustomImages(Dataset):
    def __init__(self, root, train=True, val=False, color_space='lab', transform=None, test_size=0.9, val_size=0.125, location='cpu'):
        """
            color_space: 'yub' or 'lab'
        """
        self.root_dir = root
        all_files = []
        for r, _, files in walk(self.root_dir):
            for f in files:
                if f.endswith('.jpg'):
                    all_files.append(join(r, f))
        train_val_files, test_files = train_test_split(
            all_files, test_size=test_size, random_state=69)
        train_files, val_files = train_test_split(train_val_files,
                                                  test_size=val_size, random_state=69)
        if (train and val):
            self.filenames = val_files
        elif train:
            self.filenames = train_files
        else:
            self.filenames = test_files

        self.color_space = color_space
        if (self.color_space not in ['rgb', 'lab']):
            raise(NotImplementedError)
        self.transform = transform
        self.location = location
        self.nnenc = NNEncode(location=self.location)
        self.train = train

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = imread(self.filenames[idx])
        if self.color_space == 'lab':
            img = rgb2lab(img)
        if self.transform is not None:
            img = self.transform(img)
        bwimg = img[:, :, 0:1].transpose(2, 0, 1)
        bwimg = torch.from_numpy(bwimg).float()
        abimg = img[:, :, 1:].transpose(2, 0, 1)    # abimg dim: 2, h, w
        abimg = torch.from_numpy(abimg).float()
        label = -1
        if (self.train):
            if ('cuda' in self.location):
                label = self.nnenc.imgEncodeTorch(abimg)
            else:
                label = self.nnenc.imgEncode(abimg)

        return (bwimg, label, abimg)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = resize(image, (new_h, new_w))

        return img


class GANDataset(Dataset):
    def __init__(self, root, train=True, val=False, transform=None, test_size=0.1, val_size=0.125):
        """
            color_space: 'yub' or 'lab'
        """
        self.root_dir = root
        all_files = []
        for r, _, files in walk(self.root_dir):
            for f in files:
                if f.endswith('.jpg'):
                    all_files.append(join(r, f))
        train_val_files, test_files = train_test_split(
            all_files, test_size=test_size, random_state=69)
        train_files, val_files = train_test_split(train_val_files,
                                                  test_size=val_size, random_state=69)
        if (train and val):
            self.filenames = val_files
        elif train:
            self.filenames = train_files
        else:
            self.filenames = test_files

        self.transform = transform
        self.train = train

    def __getitem__(self, idx):
        bwimg = imread(self.filenames[idx], as_grey=True)
        realimg = imread(self.filenames[idx])
        bwimg = resize(bwimg, (256, 256))
        realimg = resize(realimg, (256, 256))

        input_img = uint_color2tanh_range(bwimg)
        label_img = uint_color2tanh_range(realimg)

        ip = torch.from_numpy(np.array([input_img])).float()
        label = torch.from_numpy(label_img.transpose(2, 0, 1)).float()

        return (ip, label)

    def __len__(self):
        return len(self.filenames)
