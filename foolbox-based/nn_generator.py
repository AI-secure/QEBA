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

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class NNGenerator(nn.Module):
    def __init__(self, n_channels, N_b=100, batch_size=32, preprocess=None, gpu=False):
        super(NNGenerator, self).__init__()
        self.N = N_b
        self.gpu = gpu
        self.preprocess = preprocess
        #self.ENC_SHAPE = (10, 14, 14)
        #C,H,W = self.ENC_SHAPE

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 64)
        self.down2 = down(64, 64)
        self.down3 = down(64, 64)
        self.down4 = down(64, 64)
        #self.v_to_enc = nn.Linear(self.N, C*H*W)
        #self.up1 = up(128+self.ENC_SHAPE[0], 64)
        self.up1 = up(128, 64)
        self.up2 = up(128, 64)
        self.up3 = up(128, 64)
        self.up4 = up(128, 64)
        self.outcs = nn.ModuleList([outconv(64, n_channels) for _ in range(self.N)])
        #self.outc1 = outconv(64, n_channels)
        #self.outc2 = outconv(64, n_channels)
        if self.gpu:
            self.cuda()

    def forward(self, x, i_list):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #C,H,W = self.ENC_SHAPE
        #v_enc = self.v_to_enc(v).view(-1,C,H,W)
        #x_enc = torch.cat([x5, v_enc], axis=1)
        x_enc = x5

        x = self.up1(x_enc, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        ret_list = []
        for i in i_list:
            ret = torch.sigmoid(self.outcs[i](x))
            ret_list.append(ret)
        #if i == 0:
        #    x = self.outc1(x)
        #elif i == 1:
        #    x = self.outc2(x)
        return ret_list

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std


        x = torch.FloatTensor(inp).unsqueeze(0)
        if self.gpu:
            x = x.cuda()
        with torch.no_grad():
            vs = self.gen_all_vs(x)[:,0].detach().cpu().numpy()

        #gram-schmidt
        import scipy.linalg as splinalg
        vs_ortho = splinalg.orth(vs.reshape(self.N, -1).transpose(), rcond=0.0)
        #assert 0
        vs_ortho = vs_ortho.transpose().reshape(*vs.shape)
        vs = vs_ortho
        
        ps = []
        for _ in range(N):
            rv = np.random.randn(self.N, 1,1,1)
            pi = (rv * vs).sum(axis=0)
            ps.append(pi)
        ps = np.stack(ps, axis=0)


        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps

    def gen_all_vs(self, x):
        vs = self.forward(x, list(range(self.N)))
        vs = torch.stack(vs, axis=0)
        return vs

    def calc_rho(self, gt, inp):
        x = torch.FloatTensor(inp).unsqueeze(0)
        if self.gpu:
            x = x.cuda()
        with torch.no_grad():
            vs = self.gen_all_vs(x)[:,0].detach().cpu().numpy()
        import scipy.linalg as splinalg
        vs_ortho = splinalg.orth(vs.reshape(self.N, -1).transpose(), rcond=0.0)
        vs_ortho = vs_ortho.transpose().reshape(*vs.shape)

        all_cos2 = 0.0
        for vi in vs_ortho:
            cosi = (vi*gt).sum() / np.sqrt( (vi**2).sum() * (gt**2).sum() )
            all_cos2 += (cosi ** 2)
        rho = np.sqrt(all_cos2)
        return rho
