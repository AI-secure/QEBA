import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from cifar10_resnet_model import CifarDNN, CifarResNet, epoch_train, epoch_eval
from tqdm import tqdm

GPU = True
BATCH_SIZE=64

if __name__ == '__main__':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainset = torchvision.datasets.CIFAR10(root='../raw_data/', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='../raw_data/', train=False, download=True, transform=transform)
    #pca_trainset, pca_testset = torch.utils.data.random_split(testset, [8000, 2000])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    #MODEL_TYPE = 'res18'
    for MODEL_TYPE in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
        model = CifarDNN(model_type=MODEL_TYPE, gpu=GPU)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tqdm(range(10)) as pbar:
            for _ in pbar:
                train_loss, train_acc = epoch_train(model, optimizer, trainloader)
                test_loss, test_acc = epoch_eval(model, testloader)
                torch.save(model.state_dict(), '../models/cifar10_%s.model'%MODEL_TYPE)
                pbar.set_description("Train acc %.4f, Test acc %.4f"%(train_acc, test_acc))
