from skimage.color import rgb2yuv, rgb2lab

import torch
import torchvision.datasets as dsets
import numpy as np
import sklearn.neighbors as nn
import os

# compute index of colour bin


class NNEncode():
    def __init__(self, NN=5, sigma=5, km_filepath=os.curdir + os.path.join(os.sep, 'static', 'pts_in_hull.npy')):
        self.cc = np.load(km_filepath)
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)

    def imgEncode(self, abimg):
        label = np.zeros((abimg.shape[1]*abimg.shape[2],313))

        (dists,indexes) = self.nbrs.kneighbors(abimg.reshape(abimg.shape[0],-1).transpose(), self.NN)

        weights = np.exp(-dists**2/(2*self.sigma**2))
        weights = weights/np.sum(weights,axis=1).reshape(-1,1)

        pixel_indexes = np.arange(0,abimg.shape[1]*abimg.shape[2],dtype='int')[:,np.newaxis]
        label[pixel_indexes, indexes] = weights

        label = label.transpose().reshape(313,abimg.shape[1],abimg.shape[2])
        return label

    def imgEncodeTorch(self, abimg):
        label = torch.zeros((abimg.shape[1]*abimg.shape[2],313))
        label = label.cuda()

        (dists,indexes) = self.nbrs.kneighbors(abimg.view(abimg.shape[0],-1).t(), self.NN)
        dists = torch.from_numpy(dists).float().cuda()
        indexes = torch.from_numpy(indexes).cuda()

        weights = torch.exp(-dists**2/(2*self.sigma**2)).cuda()
        weights = weights/torch.sum(weights,dim=1).view(-1,1)

        pixel_indexes = torch.Tensor.long(torch.arange(start=0,end=abimg.shape[1]*abimg.shape[2])[:,np.newaxis])
        pixel_indexes = pixel_indexes.cuda()
        label[pixel_indexes, indexes] = weights
        label = label.t().contiguous().view(313,abimg.shape[1],abimg.shape[2])
        return label


    def bin2color(self, idx):
        return self.cc[idx]


# for finding the distribution of values of L, a, b in the dataset
'''
    Ranges of L, a, b in CIFAR10 dataset:
    lmin: 0, lmax: 100
    amin: -73, amax: 82
    bmin: -103, bmax: 94 '''


def count_color_stats():
    dataset = dsets.CIFAR10(root='./data', train=False, download=False)
    lmin, lmax, amin, amax, bmin, bmax = [0]*6

    for index in range(len(dataset)):
        img = dataset[index][0]
        data = rgb2lab(img)
        data = np.round(data).astype(int)

        l = data[:, :, 0]
        mi = np.min(l)
        ma = np.max(l)
        lmin = min(lmin, mi)
        lmax = max(lmax, ma)
        # unique, counts = np.unique(l, return_counts=True)
        # ldict = dict(zip(unique, counts))

        a = data[:, :, 1]
        mi = np.min(a)
        ma = np.max(a)
        amin = min(amin, mi)
        amax = max(amax, ma)
        # unique, counts = np.unique(a, return_counts=True)
        # adict = dict(zip(unique, counts))

        b = data[:, :, 2]
        mi = np.min(b)
        ma = np.max(b)
        bmin = min(bmin, mi)
        bmax = max(bmax, ma)
        # unique, counts = np.unique(b, return_counts=True)
        # bdict = dict(zip(unique, counts))

    # print("L VALUES\n\n", ldict)
    # print("A VALUES\n\n", adict)
    # print("B VALUES\n\n", bdict)

    print("lmin: %d, lmax: %d" % (lmin, lmax))
    print("amin: %d, amax: %d" % (amin, amax))
    print("bmin: %d, bmax: %d" % (bmin, bmax))
