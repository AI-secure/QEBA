# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import numpy as np
from skimage import transform

from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, batch_size=32, preprocess=None, mobilenet=False):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64, mobilenet=mobilenet)
        self.down1 = down(64, 64, mobilenet=mobilenet)
        self.down2 = down(64, 128, mobilenet=mobilenet)
        self.down3 = down(128, 128, mobilenet=mobilenet)
        self.down4 = down(128, 256, mobilenet=mobilenet)
        self.down5 = down(256, 256, mobilenet=mobilenet)
        self.up1 = up(256, 256, mobilenet=mobilenet)
        self.up2 = up(256, 128, mobilenet=mobilenet)
        self.up3 = up(128, 128, mobilenet=mobilenet)
        self.up4 = up(128, 64, mobilenet=mobilenet)
        self.up5 = up(64, 64, mobilenet=mobilenet)
        self.outc = outconv(64, n_channels, mobilenet=mobilenet)

        # For generating perturb
        self.batch_size = batch_size
        # For preprocessing
        self.preprocess = preprocess

    def forward(self, x):
        x = self.encode(x)
        #print (x.shape)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        return x

    def decode(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.outc(x)
        return torch.sigmoid(x)

    def calc_feature(self, x, level):
        if level == 0:
            return x.view(x.size()[0], -1)

        x = self.inc(x)
        if level == 1:
            return x.view(x.size()[0], -1)

        x = self.down1(x)
        if level == 2:
            return x.view(x.size()[0], -1)

        x = self.down2(x)
        if level == 3:
            return x.view(x.size()[0], -1)

        x = self.down3(x)
        if level == 4:
            return x.view(x.size()[0], -1)

        x = self.down4(x)
        if level == 5:
            return x.view(x.size()[0], -1)

        x = self.down5(x)
        assert level is None or level >= 6
        return x.view(x.size()[0], -1)


    def generate_ps(self, inp, N, level, factor=4):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std
            #inp = (inp+1)/2
        ps = self.generate_ps_by_enc_grad(inp, N, level)
        #ps = self.generate_ps_by_dec(inp, N)
        #ps = self.generate_ps_by_dec_autozoom(inp, N)

        ps_pool = np.array(F.avg_pool2d(torch.FloatTensor(ps), factor))
        ps = transform.resize(ps_pool.transpose(0,2,3,1), ps.transpose(0,2,3,1).shape).transpose(0,3,1,2)
        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            #ps = ps*2-1
            ps = ps.transpose(0, *(rev_transp+1))

        return ps

    def generate_ps_by_enc_grad(self, inp, N, level):
        inp = torch.FloatTensor(inp)
        inp.requires_grad = True

        N_forw = min(N, self.batch_size)
        inp_ext = inp.repeat(N_forw,1,1,1)
        #x = self.encode(inp_ext)
        #feature = x.view(x.size(0), -1)
        feature = self.calc_feature(inp_ext, level)

        rvs = []
        st = 0
        while st < N:
            ed = min(st+self.batch_size, N)
            ws = torch.FloatTensor(np.random.normal(size=(ed-st, feature.size()[1])))
            tgt = (feature[:ed-st] * ws).sum()
            retain_graph = (ed != N)
            gs = torch.autograd.grad(tgt, inp_ext, retain_graph=retain_graph)[0][:ed-st]
            gs_processed = gs.cpu().numpy()

            rvs.append(gs_processed)
            st = ed
        rvs = np.concatenate(rvs, axis=0)

        return rvs

    def generate_ps_by_dec(self, inp, N):
        with torch.no_grad():
            inp = torch.FloatTensor(inp)

            N_forw = min(N, self.batch_size)
            x = self.encode(inp.unsqueeze(0))
            feature = x.view(x.size(0), -1)
            dec_x = self.decode(x)
            print (dec_x.shape)

            rvs = []
            st = 0
            while st < N:
                ed = min(st+self.batch_size, N)
                f_perturb = np.random.normal(size=(ed-st, feature.size()[1]))
                f_perturb = f_perturb / f_perturb.sum(axis=1, keepdims=True)
                f_perturb = torch.FloatTensor(f_perturb)
                new_f = feature + f_perturb
                dec_new_x = self.decode(new_f.view(-1, *x.size()[1:]))
                updates = dec_new_x - dec_x
                updates_processed = updates.cpu().numpy()

                rvs.append(updates_processed)
                st = ed
        rvs = np.concatenate(rvs, axis=0)

        return rvs

    def generate_ps_by_dec_autozoom(self, inp, N):
        with torch.no_grad():
            inp = torch.FloatTensor(inp)
            ref = self.encode(inp.unsqueeze(0)).size()[1:]

            rvs = []
            st = 0
            while st < N:
                ed = min(st+self.batch_size, N)
                rand_f = np.random.normal(size=(ed-st, *ref))
                rand_f = torch.FloatTensor(rand_f)
                rv = self.decode(rand_f)
                rvs.append(rv)
                st = ed
        rvs = np.concatenate(rvs, axis=0)

        return rvs
