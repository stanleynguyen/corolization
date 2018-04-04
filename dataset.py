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
from colorutils import color2bin
from sklearn.model_selection import train_test_split


class CustomImages(Dataset):
    def __init__(self, root, train=True, val=False, color_space='lab', transform=None):
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
            all_files, test_size=0.1, random_state=69)
        train_files, val_files = train_test_split(train_val_files,
                test_size=0.1, random_state=69)
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
        abimg = img[:, :, 1:].transpose(2, 0, 1)
        label = np.zeros((441, abimg.shape[1], abimg.shape[2]))
        for h in range(label.shape[1]):
            for w in range(label.shape[2]):
                binidx = color2bin(abimg[:, h, w])
                label[binidx, h, w] = 1
        label = torch.from_numpy(label).float()
        # label = label.view(-1)
        abimg = torch.from_numpy(abimg).float()
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
