import torch
import torch.utils.data
import numpy as np
import json
from PIL import Image

TRAIN_NUM_RATIO = 0.8

class DogCatDataset(torch.utils.data.Dataset):
    def __init__(self, train, transform, path='../raw_data/dog_and_cat/train'):
        self.train = train
        self.transform = transform
        self.path = path
        if train:
            self.st = 0
            self.N = int(TRAIN_NUM_RATIO * 12500)*2
        else:
            self.st = int(TRAIN_NUM_RATIO * 12500)
            self.N = 25000 - self.st*2

    def __getitem__(self, idx):
        n = idx // 2 + self.st
        lab = idx % 2
        lab_str = 'cat' if lab == 0 else 'dog'
        path = self.path + '/%s.%d.jpg'%(lab_str, n)
        img = Image.open(path).convert("RGB")
        return self.transform(img), lab

    def __len__(self):
        return self.N
