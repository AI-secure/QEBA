import numpy as np
from skimage import transform

class ResizeGenerator:
    def __init__(self, batch_size=32, factor=4.0, preprocess=None):
        self.batch_size = batch_size
        self.factor = factor
        self.preprocess = preprocess

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std

        ps = []
        for _ in range(N):
            shape = inp.shape
            assert len(shape)==3 and shape[0] == 3
            p_small = np.random.randn(shape[0], int(shape[1]/self.factor), int(shape[2]/self.factor))
            #if (_ == 0):
            #    print (p_small.shape)
            p = transform.resize(p_small.transpose(1,2,0), inp.transpose(1,2,0).shape).transpose(2,0,1)
            ps.append(p)
        ps = np.stack(ps, axis=0)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))

        return ps

    def calc_rho(self, gt, inp):
        if self.preprocess is not None and inp.shape[2] == 3:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            gt = gt.transpose(*transp)

        C, H, W = gt.shape
        h_use, w_use = int(H/self.factor), int(W/self.factor)
        p = transform.resize(gt.transpose(1,2,0), (h_use,w_use,C))
        proj = transform.resize(p, (H,W,C)).transpose(2,0,1)
        rho = (gt*proj).sum() / np.sqrt( (gt**2).sum() * (proj**2).sum() )
        #rho = np.sqrt( (proj**2).sum() / (gt**2).sum() )
        return rho
