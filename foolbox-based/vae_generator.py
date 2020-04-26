import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class VAEGenerator(nn.Module):
    def __init__(self, n_channels, lam=1e-11, preprocess=None, gpu=False):
        super(VAEGenerator, self).__init__()
        self.gpu = gpu
        self.preprocess = preprocess
        self.lam = lam
        #self.ENC_SHAPE = (10, 14, 14)
        #C,H,W = self.ENC_SHAPE

        self.inc = inconv(n_channels, 24)
        self.down1 = down(24, 24)
        self.down2 = down(24, 48)
        self.down3 = down(48, 48)
        self.down4_mu = down(48, 48)
        self.down4_std = down(48, 48)
        self.ENC_SHAPE = (48,14,14)
        self.latent_dim = 9408
        #self.v_to_enc = nn.Linear(self.N, C*H*W)
        #self.up1 = up(128+self.ENC_SHAPE[0], 64)
        self.up1 = up(48, 48)
        self.up2 = up(48, 48)
        self.up3 = up(48, 24)
        self.up4 = up(24, 24)
        self.outc = outconv(24, n_channels)
        self.loss_fn = nn.MSELoss()
        #self.loss_fn = nn.CosineSimilarity()
        #self.outc1 = outconv(64, n_channels)
        #self.outc2 = outconv(64, n_channels)
        if self.gpu:
            self.cuda()

    def encode(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        z_mu = self.down4_mu(x)
        z_logstd2 = self.down4_std(x)
        z_std = torch.exp(z_logstd2/2)
        z = torch.zeros_like(z_mu).normal_() * z_std + z_mu
        x_enc = z / torch.sqrt((z**2).sum((1,2,3),keepdim=True))
        return z_mu, z_std, x_enc

    def decode(self, x_enc):
        x = self.up1(x_enc)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        x = x / torch.sqrt((x**2).sum((1,2,3),keepdim=True))
        return x

    def forward(self, x):
        x = torch.FloatTensor(x)
        if self.gpu:
            x = x.cuda()

        x_norm = torch.sqrt((x**2).sum((1,2,3),keepdim=True))
        x = x / x_norm

        z_mu, z_std, x_enc = self.encode(x)
        x = self.decode(x_enc)

        x = x * x_norm
        return x_enc, x, z_mu, z_std

    def loss(self, pred, gt, z_mu, z_std):
        gt_var = torch.FloatTensor(gt)
        if self.gpu:
            gt_var = gt_var.cuda()

        pred = pred / torch.sqrt((pred**2).sum((1,2,3),keepdim=True))
        gt_var = gt_var / torch.sqrt((gt_var**2).sum((1,2,3),keepdim=True))
        l_recon = self.loss_fn(pred, gt_var)
        l_kl = (-0.5 * (self.latent_dim + (torch.log(z_std**2)-z_mu**2-z_std**2).sum((1,2,3)))).mean()
        loss = l_recon + self.lam * l_kl

        return loss, l_recon, l_kl
        #return 1-self.loss_fn(pred, gt_var).mean()

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std

        rv = np.random.randn(N, *self.ENC_SHAPE)
        rv = rv / np.sqrt((rv**2).sum((1,2,3),keepdims=True))
        rv_var = torch.FloatTensor(rv)
        if self.gpu:
            rv_var = rv_var.cuda()
        ps_var = self.decode(rv_var)
        ps = ps_var.detach().cpu().numpy()

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps

    def calc_rho(self, gt, inp):
        return np.array([0.01])[0]

