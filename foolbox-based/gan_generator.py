import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, nkernel, nstride, npad, transpose=False, leaky=True):
        super(ConvBlock, self).__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(c_in, c_out, nkernel, nstride, npad, bias=False)
        else:
            self.conv = nn.Conv2d(c_in, c_out, nkernel, nstride, npad, bias=False)
        self.bnorm = nn.BatchNorm2d(c_out)
        if leaky:
            self.relu = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.relu(x)
        return x



class GANDiscriminator(nn.Module):
    def __init__(self, n_channels=3, gpu=False):
        super(GANDiscriminator, self).__init__()
        self.gpu = gpu

        self.init_conv = nn.Conv2d(3, 16, 4, 2, 1, bias=False)
        self.init_relu = nn.LeakyReLU(0.2, inplace=True)
        self.cb1 = ConvBlock(16, 32, 4, 2, 1)
        self.cb2 = ConvBlock(32, 32, 4, 2, 1)
        self.cb3 = ConvBlock(32, 64, 4, 2, 1)
        self.cb4 = ConvBlock(64, 64, 4, 2, 1)
        self.cb5 = ConvBlock(64, 128, 3, 2, 1)
        self.outconv = nn.Conv2d(128, 1, 4, 1, 0)

        #self.init_conv = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        #self.init_relu = nn.LeakyReLU(0.2, inplace=True)
        #self.cb1 = ConvBlock(64, 128, 4, 2, 1)
        #self.cb2 = ConvBlock(128, 256, 4, 2, 1)
        #self.cb3 = ConvBlock(256, 512, 4, 2, 1)
        #self.outconv = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

        if self.gpu:
            self.cuda()


    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if self.gpu:
            x = x.cuda()

        x = self.init_relu(self.init_conv(x))
        x = self.cb1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.cb4(x)
        x = self.cb5(x)
        x = self.outconv(x)
        score = x.mean(0)
        return score.view(1)

class GANGenerator(nn.Module):
    def __init__(self, n_z, n_channels=3, preprocess=None, gpu=False):
        super(GANGenerator, self).__init__()
        self.gpu = gpu
        self.n_z = n_z
        self.preprocess = preprocess

        self.init_cb = ConvBlock(self.n_z, 128, 4, 1, 0, transpose=True)
        self.cb1 = ConvBlock(128, 64, 3, 2, 1, transpose=True, leaky=False)
        self.cb2 = ConvBlock(64, 64, 4, 2, 1, transpose=True, leaky=False)
        self.cb3 = ConvBlock(64, 32, 4, 2, 1, transpose=True, leaky=False)
        self.cb4 = ConvBlock(32, 32, 4, 2, 1, transpose=True, leaky=False)
        self.cb5 = ConvBlock(32, 16, 4, 2, 1, transpose=True, leaky=False)
        self.outconv = nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False)
        self.outnorm = nn.Tanh()
        #self.init_cb = ConvBlock(self.n_z, 512, 4, 1, 0, transpose=True, leaky=False)
        #self.cb1 = ConvBlock(512, 256, 4, 2, 1, transpose=True, leaky=False)
        #self.cb2 = ConvBlock(256, 128, 4, 2, 1, transpose=True, leaky=False)
        #self.cb3 = ConvBlock(128, 64, 4, 2, 1, transpose=True, leaky=False)
        #self.outconv = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        #self.outnorm = nn.Tanh()

        if self.gpu:
            self.cuda()

    def forward(self, x):
        x = torch.FloatTensor(x)
        if self.gpu:
            x = x.cuda()
        x = x.unsqueeze(2).unsqueeze(2)

        x = self.init_cb(x)
        x = self.cb1(x)
        x = self.cb2(x)
        x = self.cb3(x)
        x = self.cb4(x)
        x = self.cb5(x)
        out = self.outnorm(self.outconv(x))
        return out

        raise NotImplementedError()

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std

        Z = torch.FloatTensor(N, self.n_z).normal_(0,1)
        ps = self.forward(Z).cpu().detach().numpy()

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps

    def calc_rho(self, gt, inp):
        return np.array([0.01])[0]

