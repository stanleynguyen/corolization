# pylint: disable-all

import torch
from torch.utils.data import Dataset
from skimage.color import rgb2yuv, rgb2lab
from skimage.io import imread
import torchvision.datasets as dsets
from os import listdir
from os.path import join, isfile


class CIFAR10ForResEncoder(Dataset):
    def __init__(self, root, train=True, transform=None, color_space='yuv'):
        self.dataset = dsets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.color_space = color_space
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = self.dataset[index][0]
        if self.color_space == 'lab':
            data = rgb2lab(img)
        else:
            data = rgb2yuv(img)

        bwimg = data[:, :, 0:1].transpose(2, 0, 1)
        bwimg = torch.from_numpy(bwimg).float()
        label = data[:, :, 1:].transpose(2, 0, 1)
        label = torch.from_numpy(label).float()

        if self.transform is not None:
            bwimg = self.transform(bwimg)

        return (bwimg, label)


class CustomImages(Dataset):
    def __init__(self, root, train=True, color_space='yub', transform=None):
        """
            color_space: 'yub' or 'lab'
        """
        self.root_dir = root
        self.root_dir += join(self.root_dir,
                              '/train') if train else join(self.root_dir, '/test')
        self.filenames = [f for f in listdir(
            self.root_dir) if isfile(join(self.root_dir, f))]
        self.color_space = color_space
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = imread(join(self.root_dir, self.filenames[idx]))
        if self.color_space == 'lab':
            img = rgb2lab(img)
        else:
            img = rgb2yuv(img)

        bwimg = img[:, :, 0:1].transpose(2, 0, 1)
        bwimg = torch.from_numpy(bwimg).float()
        label = img[:, :, 1:].transpose(2, 0, 1)
        label = torch.from_numpy(label).float()

        if self.transform is not None:
            bwimg = self.transform(bwimg)

        return (bwimg, label)
