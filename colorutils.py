from skimage.color import rgb2yuv, rgb2lab
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
        label = np.zeros((313, abimg.shape[1], abimg.shape[2]))
        for h in range(label.shape[1]):
            for w in range(label.shape[2]):
                (dists,inds) = self.nbrs.kneighbors(abimg[np.newaxis, :, h, w], self.NN)
                dists = dists[0]
                inds = inds[0]

                wts = np.exp(-dists**2/(2*self.sigma**2))
                wts = wts/np.sum(wts)

                # shape of dists, inds and wts is (5,)
                for i in range(5):
                    label[inds[i], h, w] = wts[i]

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
