# pylint: disable-all

import torch
import torch.nn as nn
import torch.nn.functional as F
from colorutils import color2bin


class ResidualEncoder(nn.Module):
    def __init__(self, img_size=224):
        super(ResidualEncoder, self).__init__()
        self.img_size = img_size

        self.vgg_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.vgg_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.vgg_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.vgg_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.enc_conv1 = nn.Conv2d(512, 256, 1)
        self.enc_bn1 = nn.BatchNorm2d(512)
        self.enc_deconv1 = nn.ConvTranspose2d(
            256, 256, 3, padding=1, stride=2, output_padding=1)
        self.enc_conv2 = nn.Conv2d(512, 128, 3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(256)
        self.enc_deconv2 = nn.ConvTranspose2d(
            128, 128, 3, padding=1, stride=2, output_padding=1)
        self.enc_conv3 = nn.Conv2d(256, 64, 3, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(128)
        self.enc_deconv3 = nn.ConvTranspose2d(
            64, 64, 3, padding=1, stride=2, output_padding=1)
        self.enc_conv4 = nn.Conv2d(128, 3, 3, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(64)
        self.enc_conv5 = nn.Conv2d(6, 3, 3, padding=1)
        self.enc_bn5 = nn.BatchNorm2d(3)

        self.final_conv = nn.Conv2d(3, 2, 3, padding=1)

        pass

    def forward(self, x):
        out = torch.cat((x, x, x), 1)
        vgg_out1 = F.relu(out)
        vgg_out2 = self.vgg_block1(vgg_out1)
        vgg_out3 = self.vgg_block2(vgg_out2)
        vgg_out4 = self.vgg_block3(vgg_out3)
        vgg_out5 = self.vgg_block4(vgg_out4)

        out = self.enc_bn1(vgg_out5)
        out = self.enc_conv1(out)
        enc_out1 = self.enc_deconv1(out)
        out = self.enc_bn2(vgg_out4)
        out = torch.cat((out, enc_out1), 1)
        out = self.enc_conv2(out)
        enc_out2 = self.enc_deconv2(out)
        out = self.enc_bn3(vgg_out3)
        out = torch.cat((out, enc_out2), 1)
        out = self.enc_conv3(out)
        enc_out3 = self.enc_deconv3(out)
        out = self.enc_bn4(vgg_out2)
        out = torch.cat((out, enc_out3), 1)
        enc_out4 = self.enc_conv4(out)
        out = self.enc_bn5(vgg_out1)
        out = torch.cat((out, enc_out4), 1)
        enc_out5 = self.enc_conv5(out)

        final_out = self.final_conv(enc_out5)

        return final_out


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
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.op_7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
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
            nn.Conv2d(256, 313, kernel_size=1)
            nn.UpsamplingBilinear2d(scale_factor=4);
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

            # TODO: implement the soft encoding and convert to AB layers
