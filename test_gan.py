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

from pix2pix import Generator
from dataset import GANDataset
from colorutils import modelimg2cvimg

test_dataset = GANDataset(
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
                                          shuffle=True)

gan_gen = Generator(1, 3)

if 'cuda' in location:
    print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
    gan_gen.cuda()
try:
    gan_gen.load_state_dict(torch.load(
        'generator.pkl', map_location=location))
except:
    print('ERROR: please make sure you have a model with name `generator.pkl` in your path')

gan_gen.eval()

outputs = []
images = []
labels = []
print(test_cases)
for c in test_cases:
    print('encoding ', c)
    image, label = test_dataset[c]
    image = Variable(torch.from_numpy(
        np.array([image.numpy()])), volatile=True)
    label = Variable(label, volatile=True)
    if 'cuda' in location:
        image = image.cuda()
        label = label.cuda()

    images.append(modelimg2cvimg(np.array(image.cpu().data.numpy())))
    labels.append(modelimg2cvimg(np.array([label.cpu().data.numpy()])))
    output = gan_gen(image)
    outputs.append(modelimg2cvimg(np.array(output.cpu().data.numpy())))

f, axarr = plt.subplots(len(test_cases), 3)

for i in range(len(test_cases)):
    img_input = images[i]
    img_actual = labels[i]
    img_pred = outputs[i]
    axarr[i][0].imshow(img_input[:, :, 0], cmap='gray')
    axarr[i][1].imshow(img_pred)
    axarr[i][2].imshow(img_actual)

plt.show()
