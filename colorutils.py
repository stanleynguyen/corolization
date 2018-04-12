from skimage.color import rgb2yuv, rgb2lab

import torch
import torchvision.datasets as dsets
import numpy as np
import sklearn.neighbors as nn
import os
import numpy as np
from torch.utils.data import Dataset
import pdb
import random

class NNEncode():
    def __init__(self, NN=5, sigma=5, km_filepath=os.curdir + os.path.join(os.sep, 'static', 'pts_in_hull.npy'), train=True, location='cpu'):
        self.cc = np.load(km_filepath)
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)
        if train:
            self.weights = torch.load('static/weights')
            if ('cuda' in location):
                self.weights = self.weights.cuda()

    # not in use (too slow) #TODO: make it same as gpu version
    def imgEncode(self, abimg):
        abimg = abimg.numpy()
        label = np.zeros((abimg.shape[1]*abimg.shape[2],313))

        (dists,indexes) = self.nbrs.kneighbors(abimg.reshape(abimg.shape[0],-1).transpose(), self.NN)

        weights = np.exp(-dists**2/(2*self.sigma**2))
        weights = weights/np.sum(weights,axis=1).reshape(-1,1)

        pixel_indexes = np.arange(0,abimg.shape[1]*abimg.shape[2],dtype='int')[:,np.newaxis]
        label[pixel_indexes, indexes] = weights

        label = label.transpose().reshape(313,abimg.shape[1],abimg.shape[2])
        return torch.from_numpy(label).float()

    # computes soft encoding of ground truth ab image, multiplied by weight (for class rebalancing)
    def imgEncodeTorch(self, abimg):
        abimg = abimg.cuda()
        w, h = abimg.shape[1],abimg.shape[2]
        label = torch.zeros((w*h,313))
        label = label.cuda()

        (dists,indexes) = self.nbrs.kneighbors(abimg.view(abimg.shape[0],-1).t(), self.NN)
        dists = torch.from_numpy(dists).float().cuda()
        indexes = torch.from_numpy(indexes).cuda()

        weights = torch.exp(-dists**2/(2*self.sigma**2)).cuda()
        weights = weights/torch.sum(weights,dim=1).view(-1,1)

        pixel_indexes = torch.Tensor.long(torch.arange(start=0,end=abimg.shape[1]*abimg.shape[2])[:,np.newaxis])
        pixel_indexes = pixel_indexes.cuda()
        label[pixel_indexes, indexes] = weights
        label = label.t().contiguous().view(313,w,h)

        rebal_indexes = indexes[:, 0]
        rebal_weights = self.weights[rebal_indexes]
        rebal_weights = rebal_weights.view(w,h)
        rebal_label = rebal_weights * label

        return rebal_label


    def bin2color(self, idx):
        return self.cc[idx]



