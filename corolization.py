# pylint: disable-all

import torch
import torch.nn as nn
import torch.nn.functional as F
from colorutils import color2bin

class MultinomialCELoss(nn.Module):
    def __init__(self):
        super(MultinomialCELoss, self).__init__()

    # x dim: n, 2, h, w
    # y dim: n, 2, h, w
    # n number of cases
    # h, w height width
    # q number of bins
    # output: loss, as a float
    def forward(self, x, y):

        # encode labels into one-hot vectors
        yenc = torch.FloatTensor(
            x.size(0), 21 * 21, x.size(2), x.size(3)).zero_()
        # should use soft encoding instead of one-hot (footnote 2 of richzhang paper)

        for n in range(y.size(0)):
            for h in range(y.size(2)):
                for w in range(y.size(3)):
                    pixel = y[n:n+1, :, h, w]
                    bin_idx = color2bin(pixel.data[0].numpy())
                    yenc[n][bin_idx][h][w] = 1

        zlogz = yenc * x.log()
        loss = - zlogz.sum(0).sum(0).sum(0)
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

    def forward(self, x):
        out = self.op_1(x)
        out = self.op_2(out)
        out = self.op_3(out)
        out = self.op_4(out)
        out = self.op_5(out)
        out = self.op_6(out)
        out = self.op_7(out)
        out = self.op_8(out)

        return out
