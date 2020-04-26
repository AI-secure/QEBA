import numpy as np
import torch
import torch.utils.data

PATH = '../gen_data/pytorch1/'
TRAIN_RATIO = 0.8

class BAPPDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.labels = torch.FloatTensor(np.load(PATH+'out.npy').astype(np.float32))
        self.train = train
        if train:
            self.index = np.arange(int(len(self.labels)*TRAIN_RATIO))
        else:
            self.index = np.arange(int(len(self.labels)*TRAIN_RATIO), len(self.labels))


    def __len__(self,):
        return len(self.index)

    def __getitem__(self, i):
        idx = self.index[i]
        X = torch.FloatTensor(np.load(PATH+'%d.npy'%idx))
        y = self.labels[idx]
        return X, y
