import torch
import torch.utils.data
import torchvision
import numpy as np
import json
from PIL import Image
import os


train_ratio = 0.8


def sort_imgs(root_dir):
    img_id_dict = {}
    with open('%s/identity_CelebA.txt' % (root_dir), 'r') as id_inf:
        for line in id_inf:
            _img, _id = line.strip().split(' ')
            assert _img not in img_id_dict
            img_id_dict[_img] = int(_id)

    id_cnt_dict = {}
    # copy images to the folder that matches their identity (person)
    for _img in os.listdir('%s/img_align_celeba' % (root_dir)):
        if not _img.endswith('.jpg'):
            continue
        _id = img_id_dict[_img]
        if _id not in id_cnt_dict:
            id_cnt_dict[_id] = 0
        id_dir = '%s/processed/%d' % (root_dir, _id)
        if not os.path.isdir(id_dir):
            os.makedirs(id_dir)
        id_cnt_dict[_id] += 1
        os.system('cp %s/img_align_celeba/%s %s/%s' % (root_dir, _img, id_dir, _img))

    # sort celebrity ids with the number of images for that person (descending order)
    _ids = id_cnt_dict.keys()
    _cnts = id_cnt_dict.values()
    _cnts, _ids = zip(*sorted(zip(_cnts, _ids), reverse=True))
    np.save('%s/processed/cnts.npy' % (root_dir), _cnts)
    np.save('%s/processed/ids.npy' % (root_dir), _ids)


def get_dataset(root_dir, n_id, random_sample):
    _cnts = np.load('%s/processed/cnts.npy' % (root_dir))
    _ids = np.load('%s/processed/ids.npy' % (root_dir))

    # get identities of interest
    if random_sample:
        r_ids = np.random.choice(a=np.arange(1, 10177 + 1), size=n_id, replace=False)
    else:
        r_ids = _ids[:n_id]  # if not randomly sample, use the celebrities with the largest number of images
    np.save('%s/processed/random_ids_%d.npy' % (root_dir, n_id), r_ids)

    # train-test split
    trains = []
    tests = []
    _label = 0
    for r_id in r_ids:
        imgs = []
        for _img in os.listdir('%s/processed/%d' % (root_dir, r_id)):
            if not _img.endswith('.jpg'):
                continue
            imgs.append([r_id, _img, _label])
        n_train = np.minimum(len(imgs) - 1, int(train_ratio * len(imgs)))
        trains += imgs[:n_train]
        tests += imgs[n_train:]
        _label += 1
    np.save('%s/processed/trains_%d.npy' % (root_dir, n_id), trains)
    np.save('%s/processed/tests_%d.npy' % (root_dir, n_id), tests)


def preprocess_data(root_dir, n_id, random_sample):
    sort_imgs(root_dir)
    get_dataset(root_dir, n_id, random_sample)


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, is_train, transform, preprocess=False, random_sample=False, n_id=100):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        self.random_sample = random_sample
        self.n_id = n_id

        if preprocess:
            # preprocess to put imgs of each person in one folder
            preprocess_data(root_dir, n_id, random_sample)

        if self.is_train:
            self.data = np.load('%s/processed/trains_%d.npy' %(self.root_dir, self.n_id))
        else:
            self.data = np.load('%s/processed/tests_%d.npy' %(self.root_dir, self.n_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _id, _img, _label = self.data[idx]
        img_path = '%s/processed/%d/%s' %(self.root_dir, int(_id), _img)
        img = Image.open(img_path)#.convert("RGB")
        if self.transform:
            img = self.transform(img)
        # print(img.shape)
        return img, int(_label)



if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    trainset = CelebADataset(root_dir='../..', is_train=True, transform=transform, preprocess=True, random_sample=False, n_id=100)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False)
    for _, (x, y) in enumerate(trainloader):
        print(_)
        print(x)
        print(y)
        # if _ > 2:
        assert 0

    # celebatestset = CelebADataset(root_dir='..', is_train=False, transform=None, preprocess=False, random_sample=False, n_id=100)

