# pylint: disable-all

import torch
import torch.nn as nn
import torch.nn.functional as F


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
