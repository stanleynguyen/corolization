from skimage.color import rgb2yuv, rgb2lab
import torchvision.datasets as dsets
import numpy as np

# compute index of colour bin


def color2bin(data):
    a, b = data
    return (round(a/10)+10)*21 + round(b/10)+10


def bin2color(idx):
    b = (idx % 21 - 10)*10
    a = (int(idx/21) - 10)*10
    return (a, b)


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
