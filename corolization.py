# pylint: disable-all

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class MultinomialCELoss(nn.Module):
    def __init__(self):
        super(MultinomialCELoss, self).__init__()

    # x dim: n, q, h, w
    # y dim: n, q, h, w
    # n number of cases
    # h, w height width
    # q number of bins
    # output: loss, as a float
    def forward(self, x, y):
        # softmax 
        # x = torch.exp(x)
        # x_sum = x.sum(1)
        # x_sum = x_sum.view(x_sum.shape[0],1,x_sum.shape[1],x_sum.shape[2])
        # x = x / x_sum
        x = torch.log(x)
        zlogz = y*x
        loss = - zlogz.sum()
        loss /= (x.shape[0] * x.shape[2] * x.shape[3])
        return loss


class ColorfulColorizer(nn.Module):
    def __init__(self):
        super(ColorfulColorizer, self).__init__()

        self.op_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.op_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.op_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.op_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.op_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.op_6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.op_7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.op_8 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 313, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        self.op_9 = nn.Sequential(
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.op_1(x)
        out = self.op_2(out)
        out = self.op_3(out)
        out = self.op_4(out)
        out = self.op_5(out)
        out = self.op_6(out)
        out = self.op_7(out)
        out = self.op_8(out)
        out = self.op_9(out)
        # out = out.view(out.size()[0], out.size()[1],-1)
        return out
