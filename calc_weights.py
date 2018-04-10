import numpy as np
import dataset
import sklearn.neighbors as nn
from math import sqrt, pi
import os
import torch

# calculate the weight for each bin based on empirical probability, for class rebalancing
# only needs to be run once
def cal_emp_weights(dset, bins_num=313, sigma=5, lamda=0.5):
    cc = np.load(os.path.join('static', 'pts_in_hull.npy'))
    nbrs = nn.NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cc)

    bins_prob = torch.zeros(bins_num)

    print('dset length: ', len(dset))
    for i in range(len(dset)):
        if (i%100==0):
            print('calc img: ', i)
        _, _, abimg = dset[i]
        _, indexes = nbrs.kneighbors(abimg.view(abimg.shape[0],-1).t(), 1)
        bins_prob[torch.from_numpy(indexes).view(-1)] += 1
    bins_sum = bins_prob.sum()
    bins_prob /= bins_sum

    bins_idx_arr = np.arange(bins_num)

    for i, prob in enumerate(bins_prob):
        bins_prob[i] = (prob * (np.power(i - bins_idx_arr, 2) /
                                (sigma * sigma)) / (sqrt(2 * pi) * sigma)).sum()

    w = 1/((1 - lamda) * bins_prob + lamda / bins_num)
    w /= ((bins_prob * w).sum())
    torch.save(w, 'static/weights')
    return w

entire_dataset = dataset.CustomImages('./SUN2012', train=False, test_size=0.999, val_size=0)
cal_emp_weights(entire_dataset, 313)