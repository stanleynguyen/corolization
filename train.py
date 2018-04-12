# pylint: disable-all

import time

import torch
from torch.autograd import Variable
import sys
import os
import getopt

from corolization import ColorfulColorizer, MultinomialCELoss
import dataset

def main(dset_root, batch_size, num_epochs, print_freq, encoder, criterion,
         optimizer, scheduler, step_every_iteration=False):
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

    train_dataset = dataset.CustomImages(
        root=dset_root, train=True, location=location)

    val_dataset = dataset.CustomImages(
        root=dset_root, train=True, val=True, location=location)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    if continue_training and os.path.isfile('best_model.pkl'):
        encoder.load_state_dict(torch.load(
            'best_model.pkl', map_location=location))
        print('Model loaded!')


    if 'cuda' in location:
        print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
        encoder.cuda()
        criterion.cuda()

    best_loss = 100
    losses = []

    for epoch in range(num_epochs):
        # train for one epoch
        epoch_losses = train(train_loader, encoder, criterion, optimizer, scheduler, epoch, location, step_every_iteration, num_epochs, print_freq)
        losses.append(epoch_losses)

        save_checkpoint(encoder.state_dict())

        # evaluate on validation set
        val_loss = validate(val_loader, encoder, criterion, location, num_epochs, print_freq)
        if (not step_every_iteration):
            scheduler.step(val_loss.data[0])
        is_best = val_loss.data[0] < best_loss

        if is_best:
            print('new best validation')
            best_loss = val_loss.data[0]
            save_checkpoint(encoder.state_dict(), is_best)
    return losses

def save_checkpoint(state, is_best=False, filename='colorizer.pkl'):
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'best_model.pkl')

def train(train_loader, model, criterion, optimizer, scheduler, epoch,
          location, step_every_iteration,num_epochs, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (image, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        image_var = Variable(image)
        target_var = Variable(target)

        if 'cuda' in location:
            image_var = image_var.cuda()
            target_var = target_var.cuda()

        # compute output
        output = model(image_var)
        loss = criterion(output, target_var)
        losses.update(loss.data[0], image.size(0))

        # step scheduler for lr finder
        if (step_every_iteration):
            scheduler.step()
            for k, param_group in enumerate(optimizer.param_groups):
                print(param_group['lr'])
            # print(optimizer.param_groups.lr)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, num_epochs, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
    return losses

def validate(val_loader, model, criterion, location,num_epochs, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (image, target, _) in enumerate(val_loader):
        image_var = Variable(image, volatile=True)
        target_var = Variable(target, volatile=True)

        if 'cuda' in location:
            image_var = image_var.cuda()
            target_var = target_var.cuda()

        # compute output
        output = model(image_var)
        loss = criterion(output, target_var)
        losses.update(loss.data[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))

    print(' * Val Loss {loss.avg:.3f}'
          .format(loss=losses))

    return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    dset_root = './SUN2012'
    batch_size = 12
    num_epochs = 100
    print_freq = 100
    encoder = ColorfulColorizer()
    criterion = MultinomialCELoss()
    optimizer = torch.optim.SGD(encoder.parameters(),
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
            patience=3, verbose=True)
    main(dset_root, batch_size, num_epochs, print_freq, encoder,
         criterion, optimizer, scheduler)
