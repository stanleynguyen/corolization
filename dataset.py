# pylint: disable-all

import torch
from torch.utils.data import Dataset
from skimage.color import rgb2yuv
import torchvision.datasets as dsets


class CIFAR10ForResEncoder(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.dataset = dsets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = rgb2yuv(self.dataset[index][0])
        bwimg = data[:, :, 0:1].transpose(2, 0, 1)
        bwimg = torch.from_numpy(bwimg).float()
        label = data[:, :, 1:].transpose(2, 0, 1)
        label = torch.from_numpy(label).float()

        if self.transform is not None:
            bwimg = self.transform(bwimg)

        return (bwimg, label)
