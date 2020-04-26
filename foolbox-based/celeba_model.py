import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.models as models

import celeba_dataset
from celeba_dataset import CelebADataset


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])


class CelebAResNet(nn.Module):
    def __init__(self, num_class, pretrained=True, gpu=False):
        super(CelebAResNet, self).__init__()
        self.pretrained = pretrained
        self.gpu = gpu
        self.num_class = num_class

        self.resnet = models.resnet18(pretrained=pretrained)
        self.output = nn.Linear(1000, self.num_class)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        x = self.resnet(x)
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


# class CelebADenseNet(nn.Module):
#     def __init__(self, num_class, pretrained=True, gpu=False):
#         super(CelebADenseNet, self).__init__()
#         self.pretrained = pretrained
#         self.gpu = gpu
#         self.num_class = num_class
#
#         self.densenet = models.densenet121(pretrained=pretrained)
#         self.output = nn.Linear(1000, self.num_class)
#
#         if gpu:
#             self.cuda()
#
#     def forward(self, x):
#         if self.gpu:
#             x = x.cuda()
#         #x = F.interpolate(x, [224,224])
#         #x = self.resnet(x)
#
#         x = self.densenet.features(x)
#         x = F.relu(x, inplace=True)
#         x = x.view(x.size(0), -1)
#         x = self.densenet.classifier(x)
#
#         x = self.output(x)
#
#         return x
#
#     def loss(self, pred, label):
#         if self.gpu:
#             label = label.cuda()
#         return F.cross_entropy(pred, label)


def epoch_train(model, optimizer, dataloader):
    model.train()

    cum_loss = 0.0
    cum_acc = 0.0
    tot_num = 0.0
    for X, y in dataloader:
        B = X.size()[0]
        if (B==1):
            continue
        pred = model(X)
        loss = model.loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cum_loss += loss.item() * B
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(y)).sum().item()
        tot_num = tot_num + B

    print(cum_loss / tot_num, cum_acc / tot_num)
    return cum_loss / tot_num, cum_acc / tot_num

def epoch_eval(model, dataloader):
    model.eval()

    cum_loss = 0.0
    cum_acc = 0.0
    tot_num = 0.0
    for X, y in dataloader:
        B = X.size()[0]
        with torch.no_grad():
            pred = model(X)
            loss = model.loss(pred, y)

        cum_loss += loss.item() * B
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(y)).sum().item()
        tot_num = tot_num + B

    print(cum_loss / tot_num, cum_acc / tot_num)
    return cum_loss / tot_num, cum_acc / tot_num


if __name__ == '__main__':
    gpu = True
    num_class = 10
    n_e = 500
    batch_size = 128

    root_dir = '../raw_data/celeba'
    do_random_sample = False

    # celeba_dataset.preprocess_data(root_dir, n_id=num_class, random_sample=do_random_sample)
    # celeba_dataset.sort_imgs(root_dir)
    # celeba_dataset.get_dataset(root_dir, n_id=num_class, random_sample=do_random_sample)

    trainset = CelebADataset(root_dir=root_dir, is_train=True, transform=transform, preprocess=False,
                             random_sample=do_random_sample, n_id=num_class)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = CelebADataset(root_dir=root_dir, is_train=False, transform=transform, preprocess=False,
                             random_sample=do_random_sample, n_id=num_class)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    #print (len(trainset), len(testset))
    #assert 0

    resmodel = CelebAResNet(num_class, pretrained=True, gpu=gpu)
    # optimizer = torch.optim.SGD(resmodel.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(resmodel.output.parameters(), lr=1e-4, weight_decay=1e-5)

    for e in range(n_e):
        print("Epoch %d" %(e))
        epoch_train(model=resmodel, optimizer=optimizer, dataloader=trainloader)
        print("Evaluate")
        epoch_eval(model=resmodel, dataloader=testloader)
        torch.save(resmodel.state_dict(), '../models/celeba.model')

