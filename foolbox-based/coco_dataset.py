import torch
import torch.utils.data
import numpy as np
import json
from PIL import Image

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, train, transform, path='../raw_data/coco'):
        self.train = train
        self.transform = transform
        self.path = path
        if train:
            self.st = 0
            self.N = 160000
        else:
            self.st = 160000
            self.N = 4000

    def __getitem__(self, idx):
        path = self.path + '/%d.jpg'%(self.st+idx)
        img = Image.open(path).convert("RGB")
        return self.transform(img), 0

    def __len__(self):
        return self.N

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, train, transform, path='../raw_data/imagenet'):
        self.train = train
        self.transform = transform
        self.path = path
        if train:
            self.st = 0
            self.N = 280000
        else:
            self.st = 280000
            self.N = 20000

    def __getitem__(self, idx):
        path = self.path + '/%d.JPEG'%(self.st+idx)
        img = Image.open(path).convert("RGB")
        return self.transform(img), 0

    def __len__(self):
        return self.N
