import numpy as np
import os
from tqdm import tqdm

class DiskMatrix:
    def __init__(self, path, N_used=None, N_multi=10):
        self.path = path
        self.N_multi=N_multi
        assert os.path.exists(path+'_0.npy'), "DiskMatrix does not exist"
        if N_used is None:
            i = 0
            used_num = 0
            while os.path.exists(path+'_%d.npy'%i):
                cur_block = np.load(path+'_%d.npy'%i)
                used_num += cur_block.shape[0]
                i += 1
            self.shape = (used_num, cur_block.shape[1])
        else:
            block = np.load(path+'_0.npy')
            self.shape = (N_used, block.shape[1])

    def dot(self, X):
        return self.rightdot(X)

    def rightdot(self, X, verbose=True):
        assert self.shape[1] == X.shape[0]
        M, N = self.shape
        N, R = X.shape

        rets = []
        i = 0
        used_num = 0
        agg_block = []
        #while used_num < M and os.path.exists(self.path+'_%d.npy'%i):
        if verbose:
            pbar = tqdm(total=M)
        while used_num < M:
            #print (used_num)
            cur_block = np.load(self.path+'_%d.npy'%i)
            if used_num + cur_block.shape[0] > M:
                cur_block = cur_block[:M-used_num]
            used_num += cur_block.shape[0]
            if verbose:
                pbar.update(cur_block.shape[0])
            i += 1
            agg_block.append(cur_block)
            if used_num < M and os.path.exists(self.path+'_%d.npy'%i) and len(agg_block) < self.N_multi:
                continue
            agg_block = np.concatenate(agg_block, axis=0)
            rets.append(agg_block.dot(X))
            agg_block = []
        if verbose:
            pbar.close()
        return np.concatenate(rets, axis=0)

    def leftdot(self, X, verbose=True):
        assert self.shape[0] == X.shape[1]
        M, N = self.shape
        R, M = X.shape

        rets = 0.0
        i = 0
        used_num = 0
        agg_block = []
        #while used_num < M and os.path.exists(self.path+'_%d.npy'%i):
        if verbose:
            pbar = tqdm(total=M)
        while used_num < M:
            #print (used_num)
            cur_block = np.load(self.path+'_%d.npy'%i)
            if used_num + cur_block.shape[0] > M:
                cur_block = cur_block[:M-used_num]
            used_num += cur_block.shape[0]
            if verbose:
                pbar.update(cur_block.shape[0])
            i += 1
            agg_block.append(cur_block)
            if used_num < M and os.path.exists(self.path+'_%d.npy'%i) and len(agg_block) < self.N_multi:
                continue
            agg_block = np.concatenate(agg_block, axis=0)
            #print (X[:,used_num-agg_block.shape[0]:used_num].shape)
            #print (agg_block.shape)
            rets = rets + X[:,used_num-agg_block.shape[0]:used_num].dot(agg_block)
            agg_block = []
        if verbose:
            pbar.close()
        return rets

    def norm(self, verbose=True):
        M, N = self.shape

        norm2 = 0.0
        i = 0
        used_num = 0
        agg_block = []
        #while used_num < M and os.path.exists(self.path+'_%d.npy'%i):
        if verbose:
            pbar = tqdm(total=M)
        while used_num < M:
            cur_block = np.load(self.path+'_%d.npy'%i)
            if used_num + cur_block.shape[0] > M:
                cur_block = cur_block[:M-used_num]
            used_num += cur_block.shape[0]
            if verbose:
                pbar.update(cur_block.shape[0])
            i += 1
            agg_block.append(cur_block)
            if used_num < M and os.path.exists(self.path+'_%d.npy'%i) and len(agg_block) < self.N_multi:
                continue
            agg_block = np.concatenate(agg_block, axis=0)
            norm2 = norm2 + np.linalg.norm(agg_block)**2
            agg_block = []
        if verbose:
            pbar.close()

        return np.sqrt(norm2)

class ConcatDiskMat(DiskMatrix):
    def __init__(self, mats):
        self.mats = mats
        M, N = mats[0].shape
        for mat in mats[1:]:
            assert mat.shape[1] == N
            M = M + mat.shape[0]
        self.shape = (M, N)

    def dot(self, X):
        return self.rightdot(X)

    def rightdot(self, X):
        vals = []
        for mat in self.mats:
            vals.append(mat.rightdot(X))
        vals = np.concatenate(vals, axis=0)
        return vals

    def leftdot(self, X):
        vals = 0.0
        st = 0
        for mat in self.mats:
            vals = vals + mat.leftdot(X[:,st:st+mat.shape[0]])
            st = st+mat.shape[0]
        return vals

    def norm(self,):
        vals = 0.0
        for mat in self.mats:
            vals = vals + mat.norm()**2
        return np.sqrt(vals)
