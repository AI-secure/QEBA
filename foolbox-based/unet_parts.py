# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super(MobileConv2d, self).__init__()
        self.conv_dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch)
        self.conv_pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, mobilenet=False):
        super(double_conv, self).__init__()
        if mobilenet:
            conv_mod = MobileConv2d
        else:
            conv_mod = nn.Conv2d

        self.conv = nn.Sequential(
            conv_mod(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            conv_mod(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, mobilenet=False):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, mobilenet=mobilenet)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, mobilenet=False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, mobilenet=mobilenet)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, mobilenet=False):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv(in_ch, out_ch, mobilenet=mobilenet)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, mobilenet=False):
        super(outconv, self).__init__()
        if mobilenet:
            self.conv = MobileConv2d(in_ch, out_ch, 1)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
