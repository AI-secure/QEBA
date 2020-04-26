import torch
import torch.utils.data
import torchvision.transforms as transforms
import os
import random
from PIL import Image


class TinyImagenet(torch.utils.data.Dataset):
    def __init__(self, transform, train=True):
        self.transform = transform
        self.train = train
        self.classes = os.listdir('../raw_data/tiny-imagenet-200/train')
        if not train:
            self.id_map = {}
            for c in self.classes:
                self.id_map[c] = []
            with open('../raw_data/tiny-imagenet-200/val/val_annotations.txt') as inf:
                for idx,line in enumerate(inf):
                    info = line.strip().split('\t')
                    assert info[0] == 'val_%d.JPEG'%idx
                    self.id_map[info[1]].append(idx)

    def load_img(self, c, i):
        if self.train:
            path = '../raw_data/tiny-imagenet-200/train/%s/images/%s_%d.JPEG'%(c,c,i)
        else:
            idx = self.id_map[c][i]
            path = '../raw_data/tiny-imagenet-200/val/images/val_%d.JPEG'%idx
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        N = 500 if self.train else 50
        c = self.classes[idx//N]
        i = idx % N

        X = self.load_img(c,i)
        X = self.transform(X)
        return X, idx//N

    def __len__(self):
        return 500*200 if self.train else 50*200
