from skimage.color import rgb2yuv, rgb2lab

import torch
import torchvision.datasets as dsets
import numpy as np
import sklearn.neighbors as nn
import os
import numpy as np
from torch.utils.data import Dataset
from math import sqrt, pi

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


def cal_emp_weights(dset, bins_num, sigma, lamda):
    bins_prob = np.zeros((bins_num, 1))
    for (_, _, ab_tensor) in dset:
        ab_img = ab_tensor.numpy()
        for h in range(ab_img.shape[1]):
            for w in range(ab_img.shape[2]):
                bins_prob[color2bin(ab_img[:, h, w])] += 1

    bins_sum = bins_prob.sum()
    bins_prob /= bins_sum
    # print(bins_prob.sum())

    bins_idx_arr = np.arange(bins_num)

    for i, prob in enumerate(bins_prob):
        bins_prob[i] = (prob * (np.power(i - bins_idx_arr, 2) /
                                (sigma * sigma)) / (sqrt(2 * pi) * sigma)).sum()

    w = 1/((1 - lamda) * bins_prob + lamda / bins_num)
    w /= ((bins_prob * w).sum())
    return w
