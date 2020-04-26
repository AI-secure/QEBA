import numpy as np
from scipy import fftpack


def get_2d_dct(x):
    return fftpack.dct(fftpack.dct(x.T, norm='ortho').T, norm='ortho')

def get_2d_idct(x):
    return fftpack.idct(fftpack.idct(x.T, norm='ortho').T, norm='ortho')

def RGB_img_dct(img):
    assert len(img.shape) == 3 and img.shape[0] == 3
    signal = np.zeros_like(img)
    for c in range(3):
        signal[c] = get_2d_dct(img[c])
    return signal

def RGB_signal_idct(signal):
    assert len(signal.shape) == 3 and signal.shape[0] == 3
    img = np.zeros_like(signal)
    for c in range(3):
        img[c] = get_2d_idct(signal[c])
    return img

class DCTGenerator:
    def __init__(self, factor, batch_size=32, preprocess=None):
        self.factor = factor
        self.batch_size = batch_size
        self.preprocess = preprocess

    def generate_ps(self, inp, N, level=None):
        # print(inp.shape) # (218, 178, 3)
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std
        # print(inp.shape) # (3, 218, 178)
        
        C, H, W = inp.shape
        h_use, w_use = int(H/self.factor), int(W/self.factor)

        ###
        #print (inp.shape)
        #p1 = np.zeros_like(inp)
        #p1[:,:h_use,:w_use] = np.random.randn(C, h_use, w_use)
        #pimg = RGB_signal_idct(p1)
        #p2 = RGB_img_dct(pimg)
        #import matplotlib.pyplot as plt
        #plt.subplot(1,2,1)
        #plt.imshow(p1.transpose(1,2,0))
        #plt.axis('off')
        #plt.title('Signal')
        #plt.subplot(1,2,2)
        #plt.imshow(pimg.transpose(1,2,0))
        #plt.axis('off')
        #plt.title('Image')
        #plt.show()
        #assert 0
        ###

        ps = []
        for _ in range(N):
            p_signal = np.zeros_like(inp)
            for c in range(C):
                rv = np.random.randn(h_use, w_use)
                rv_ortho, _ = np.linalg.qr(rv, mode='full')
                p_signal[c,:h_use,:w_use] = rv_ortho
            p_img = RGB_signal_idct(p_signal)
            ps.append(p_img)
        ps = np.stack(ps, axis=0)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        # print(ps.shape) # (100, 218, 178, 3)
        return ps

    def calc_rho(self, gt, inp):
        # print(inp.shape) # (218, 178, 3)
        if self.preprocess is not None and inp.shape[2] == 3:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            gt = gt.transpose(*transp)

        C, H, W = inp.shape
        h_use, w_use = int(H/self.factor), int(W/self.factor)

        gt_signal = RGB_img_dct(gt)
        rho = np.sqrt( (gt_signal[:h_use, :w_use]**2).sum() / (gt_signal**2).sum() )
        return rho
