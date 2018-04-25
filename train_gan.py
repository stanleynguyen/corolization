# pylint: disable-all

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os
import getopt
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pix2pix import Discriminator, Generator, weights_init
from dataset import GANDataset
from colorutils import modelimg2cvimg


dset_root = './SUN2012'
batch_size = 32
num_epochs = 100
print_freq = 50
input_channel = 1
output_channel = 3
learning_rate = 0.0001
decay_freq = 50

continue_training = False
location = 'cpu'
try:
    opts, args = getopt.getopt(sys.argv[1:], 'hl:c', [
                               'location=', 'continue='])
except getopt.GetoptError:
    print('python train.py -l <location> -c')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('python train.py -l <location> -c <testcases>')
        sys.exit(0)
    elif opt in ('-l', '--location'):
        location = arg
    elif opt in ('-c', '--continue'):
        continue_training = True

train_dset = GANDataset(root=dset_root, train=True)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dset, batch_size=batch_size, shuffle=True)

generator_G = Generator(input_channel, output_channel)
discriminator_D = Discriminator(input_channel, output_channel)
loss_L1 = nn.L1Loss()
loss_binaryCrossEntropy = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator_G.parameters(
), lr=learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
optimizer_D = torch.optim.Adam(discriminator_D.parameters(
), lr=learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)

if continue_training and os.path.isfile('generator.pkl') and os.path.isfile('discriminator.pkl'):
    generator_G.load_state_dict(torch.load(
        'generator.pkl', map_location=location))
    discriminator_D.load_state_dict(torch.load(
        'discriminator.pkl', map_location=location))
    print('Model loaded!')
else:
    weights_init(generator_G)
    weights_init(discriminator_D)

if 'cuda' in location:
    print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
    generator_G.cuda()
    discriminator_D.cuda()
    loss_L1.cuda()
    loss_binaryCrossEntropy.cuda()

for epoch in range(num_epochs):
    for i, (ip, label) in enumerate(train_loader):
        ip_var = Variable(ip)
        target_var = Variable(label)

        if 'cuda' in location:
            ip_var = ip_var.cuda()
            target_var = target_var.cuda()

        out_generator_G = generator_G.forward(ip_var)
        optimizer_D.zero_grad()
        negative_examples = discriminator_D.forward(
            ip_var.detach(), out_generator_G.detach())
        positive_examples = discriminator_D.forward(ip_var, target_var)

        d_ones_label = Variable(torch.ones(positive_examples.size()))
        d_zeros_label = Variable(torch.zeros(negative_examples.size()))
        if 'cuda' in location:
            d_ones_label = d_ones_label.cuda()
            d_zeros_label = d_zeros_label.cuda()
        loss_dis = 0.5 * (loss_binaryCrossEntropy(positive_examples, d_ones_label) +
                          loss_binaryCrossEntropy(negative_examples, d_zeros_label))
        loss_dis.backward(retain_variables=True)
        optimizer_D.step()

        optimizer_G.zero_grad()
        negative_examples = discriminator_D.forward(ip_var, out_generator_G)
        g_ones_label = Variable(torch.ones(negative_examples.size()))
        if 'cuda' in location:
            g_ones_label = g_ones_label.cuda()
        loss_gen = loss_binaryCrossEntropy(
            negative_examples, g_ones_label) + loss_L1(out_generator_G, target_var) * 100
        loss_gen.backward()
        optimizer_G.step()

        if i % print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}] loss_gen={4} loss_dis={5}'.format(epoch,
                                                                               num_epochs, i, len(train_loader), loss_gen.data[0], loss_dis.data[0]))
        if (i+1) % decay_freq == 0:
            learning_rate /= 2
            optimizer_G = torch.optim.Adam(generator_G.parameters(
            ), lr=learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
            optimizer_D = torch.optim.Adam(discriminator_D.parameters(
            ), lr=learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
            print('upate lr to:', learning_rate)
    out_gen = out_generator_G.cpu().data.numpy()
    cvimg = modelimg2cvimg(out_gen)
    results_path = './data/results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    cv2.imwrite('%s/result_epoch%4d.jpg' % (results_path, epoch), cvimg)

    torch.save(generator_G.state_dict(), 'generator.pkl')
    torch.save(discriminator_D.state_dict(), 'discriminator.pkl')
