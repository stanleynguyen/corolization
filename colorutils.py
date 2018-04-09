from skimage.color import rgb2yuv, rgb2lab
import numpy as np
from torch.utils.data import Dataset
from math import sqrt, pi

# compute index of colour bin


def color2bin(data):
    a, b = data
    return int((round(a/10)+10)*21 + round(b/10)+10)


def bin2color(idx):
    b = (idx % 21 - 10)*10
    a = (np.floor(idx/21) - 10)*10
    return a, b


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

    # bins_prob /= (sqrt(2 * pi) * sigma).sum()
    for i, prob in enumerate(bins_prob):
        bins_prob[i] = (prob * (np.power(i - bins_idx_arr, 2) /
                                (sigma * sigma)) / (sqrt(2 * pi) * sigma)).sum()

    return 1/((1 - lamda) * bins_prob + lamda / bins_num)
