# pylint: disable-all

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.color import yuv2rgb
import sys
import os
import getopt

from corolization import ResidualEncoder
import dataset

train_dataset = dataset.CIFAR10ForResEncoder(
    root='./data', train=True)


continue_training = False
location = 'cpu'
try:
    opts, args = getopt.getopt(sys.argv[1:], 'hl:c', [
                               'location=', 'continue='])
except getopt.GetoptError:
    print('python test.py -l <location> -c')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('python test.py -l <location> -c <testcases>')
        sys.exit(0)
    elif opt in ('-l', '--location'):
        location = arg
    elif opt in ('-c', '--continue'):
        continue_training = True

batch_size = 100
num_epochs = 3

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

encoder = ResidualEncoder(32)
if continue_training and os.path.isfile('residual_encoder.pkl'):
    encoder.load_state_dict(torch.load(
        'residual_encoder.pkl', map_location=location))
    print('Model loaded!')
if 'cuda' in location:
    print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
    encoder.cuda()

criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)

        if 'cuda' in location:
            images = images.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = encoder(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Iter [%d/%d] Loss: %.4f' %
              (i+1, len(train_dataset)//batch_size, loss.data[0]))
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

    # lr decay
    learning_rate /= 10
    optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    torch.save(encoder.state_dict(), 'residual_encoder.pkl')