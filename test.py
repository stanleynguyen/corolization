# pylint: disable-all

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from skimage.color import yuv2rgb
from torch.autograd import Variable
import matplotlib.pyplot as plt
import getopt

import dataset
from corolization import ResidualEncoder

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

encoder = ResidualEncoder(32)

if 'cuda' in location:
    print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
    encoder.cuda()
try:
    encoder.load_state_dict(torch.load(
        'residual_encoder.pkl', map_location=location))
except:
    print('ERROR: please make sure you have a model with name `residual_encoder.pkl` in your path')

encoder.eval()

outputs = []
images = []
labels = []
for c in test_cases:
    image, label = test_dataset[c]
    image = Variable(torch.from_numpy(np.array([image.numpy()])))
    label = Variable(torch.from_numpy(np.array([label.numpy()])))
    if 'cuda' in location:
        image = image.cuda()
        label = label.cuda()

    images.append(image)
    labels.append(label)
    output = encoder(image)
    outputs.append(output)

f, axarr = plt.subplots(len(test_cases), 3)
for i in range(len(test_cases)):
    y_layer = images[i].data[0].numpy()
    uv_pred = outputs[i].data[0].numpy()
    uv_label = labels[i].data[0].numpy()

    img_input = y_layer[0]
    img_pred = np.concatenate((y_layer, uv_pred), axis=0)
    img_actual = np.concatenate((y_layer, uv_label), axis=0)

    axarr[i][0].imshow(img_input, cmap='gray')
    axarr[i][1].imshow(yuv2rgb(img_pred.transpose(1, 2, 0)))
    axarr[i][2].imshow(yuv2rgb(img_actual.transpose(1, 2, 0)))
plt.show()
