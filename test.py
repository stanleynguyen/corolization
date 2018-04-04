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
from colorutils import bin2color

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
bin_index = np.arange(441)
a, b = bin2color(bin_index)
a = torch.FloatTensor(a).cuda()
b = torch.FloatTensor(b).cuda()

for i in range(len(test_cases)):
    l_layer = images[i].data[0]
    bin_probabilities = outputs[i].data[0]
    ab_label = labels[i].data
    # convert bin_probab -> ab_pred
    # 441,256,256
    bin_probabilities = torch.exp(torch.log(bin_probabilities)/T)
    bin_sum = bin_probabilities.sum(0)
    bin_sum = bin_sum.view(1, bin_sum.shape[0], bin_sum.shape[1])
    bin_probabilities /= bin_sum

    a_layer = (bin_probabilities * a.view(a.shape[0],1,1)).sum(0)
    b_layer = (bin_probabilities * b.view(b.shape[0],1,1)).sum(0)
    a_layer = a_layer.view(1, a_layer.shape[0], a_layer.shape[1])
    b_layer = b_layer.view(1, b_layer.shape[0], b_layer.shape[1])
    img_input = l_layer[0]

    img_pred = torch.cat((l_layer, a_layer, b_layer), dim=0)
    img_actual = torch.cat((l_layer, ab_label), dim=0)

    axarr[i][0].imshow(img_input, cmap='gray')
    img_pred = img_pred.cpu().numpy().astype('float64')
    img_actual = img_actual.cpu().numpy().astype('float64')
    axarr[i][1].imshow(lab2rgb(img_pred.transpose(1, 2, 0)))
    axarr[i][2].imshow(lab2rgb(img_actual.transpose(1, 2, 0)))
plt.show()
