# pylint: disable-all

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from skimage.color import lab2rgb
from torch.autograd import Variable
import matplotlib.pyplot as plt
import getopt
from colorutils import NNEncode

import dataset
from corolization import ColorfulColorizer

import pdb

test_dataset = dataset.CustomImages(
    root='./SUN2012', train=False)

batch_size = 100
location = 'cpu'
test_cases = np.floor(np.random.rand(5) * len(test_dataset)).astype(int)
try:
    opts, args = getopt.getopt(sys.argv[1:], 'hl:c:', [
                               'location=', 'testcases='])
except getopt.GetoptError:
    print('python test.py -l <location> -c <testcases>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('python test.py -l <location> -c <testcases>')
        sys.exit(0)
    elif opt in ('-l', '--location'):
        location = arg
    elif opt in ('-c', '--testcases'):
        test_cases = list(map(lambda x: int(x), arg.split(',')))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

encoder = ColorfulColorizer()

if 'cuda' in location:
    print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
    encoder.cuda()
try:
    encoder.load_state_dict(torch.load(
        'colorizer.pkl', map_location=location))
except:
    print('ERROR: please make sure you have a model with name `colorizer.pkl` in your path')

encoder.eval()
# encoder.parameters()

outputs = []
images = []
labels = []
print(test_cases)
for c in test_cases:
    image,_, label = test_dataset[c]
    image = Variable(torch.from_numpy(np.array([image.numpy()])))
    label = Variable(label)
    if 'cuda' in location:
        image = image.cuda()
        label = label.cuda()

    images.append(image)
    labels.append(label)
    output = encoder(image)
    outputs.append(output)

f, axarr = plt.subplots(len(test_cases), 3)

T = 0.38
q = 313  # number of colours
nnenc = NNEncode()
bin_index = np.arange(q)
ab_list = nnenc.bin2color(bin_index)   # q, 2

for i in range(len(test_cases)):
    l_layer = images[i].data[0].numpy()
    bin_probabilities = outputs[i].data[0].numpy()  # bin_probabilities dim: q, h, w
    ab_label = labels[i].data.numpy()

    # convert bin_probab -> ab_pred
    bin_probabilities = np.exp(np.log(bin_probabilities)/T)
    bin_sum = bin_probabilities.sum(0)
    bin_sum = bin_sum.reshape((1, bin_sum.shape[0], bin_sum.shape[1]))
    bin_probabilities /= bin_sum

    # ab_pred dim: 2, h, w
    ab_pred = (bin_probabilities[:, np.newaxis, :, :] * ab_list[:, :, np.newaxis, np.newaxis]).sum(0)

    img_input = l_layer[0]

    img_pred = np.concatenate((l_layer, ab_pred), axis=0)
    img_actual = np.concatenate((l_layer, ab_label), axis=0)

    axarr[i][0].imshow(img_input, cmap='gray')
    axarr[i][1].imshow(lab2rgb(img_pred.transpose(1, 2, 0)))
    axarr[i][2].imshow(lab2rgb(img_actual.transpose(1, 2, 0)))
plt.show()
