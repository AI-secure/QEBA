import numpy as np
from nn_generator import NNGenerator
from nn_sequential_generator import NNSeqGenerator
from bapp_dataset import BAPPDataset
from tiny_imagenet import TinyImagenet
from dogcat_dataset import DogCatDataset
from coco_dataset import CocoDataset
#import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import math
from tqdm import tqdm
from nngen_utils import epoch_train_2d, epoch_eval_2d, epoch_train_2d_all, epoch_train_reg, epoch_eval_reg, epoch_train_3d, epoch_eval_3d, epoch_train_3d_all, epoch_train_seq, epoch_eval_seq

if __name__ == '__main__':
    GPU = True
    TASK = 'imagenet'

    if TASK == 'imagenet':
        BATCH_SIZE=8
        LMD = 0.0
        #N_b=10
        N_b=30
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        #trainset = DogCatDataset(train=True, transform=transform)
        #testset = DogCatDataset(train=False, transform=transform)
        trainset = CocoDataset(train=True, transform=transform)
        testset = CocoDataset(train=False, transform=transform)
        print (len(trainset), len(testset))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
        #ref_model = models.resnet18(pretrained=True).eval()
        ref_model = models.densenet121(pretrained=True).eval()
    elif TASK == 'cifar':
        BATCH_SIZE = 64
        N_b=768
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        import torchvision
        cifar_testset = torchvision.datasets.CIFAR10(root='../raw_data/', train=False, download=False, transform=transform)
        trainset, testset = torch.utils.data.random_split(cifar_testset, [8000,2000])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
        from cifar10_resnet_model import CifarResNet
        ref_model = CifarResNet(gpu=GPU)
        ref_model.load_state_dict(torch.load('../models/cifar10_resnet18.model'))
        #from cifar10_resnet_model import CifarDenseNet
        #ref_model = CifarDenseNet(gpu=GPU)
        #ref_model.load_state_dict(torch.load('../models/cifar10_densenet.model'))
    else:
        raise NotImplementedError()

    #ref_model = models.resnet18(pretrained=True).eval()
    #ref_model = models.densenet121(pretrained=True).eval()
    model = NNGenerator(n_channels=3, N_b=N_b, gpu=GPU)
    #model = NNSeqGenerator(n_channels=3, N_b=N_b)
    #print (model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    #loss_fn = torch.nn.MSELoss()
    cos_fn = torch.nn.CosineSimilarity(dim=-1)
    model.train()
    if GPU:
        model.cuda()
        ref_model.cuda()
        cos_fn.cuda()

    for epoch in range(20):
        epoch_train_3d_all(model, ref_model, optimizer, trainloader, total=math.ceil(len(trainset)/BATCH_SIZE), N_b=N_b)
        epoch_eval_3d(model, ref_model, testloader, total=math.ceil(len(testset)/BATCH_SIZE), N_b=N_b)
        torch.save(model.state_dict(), 'nn_gen_%d_%s.model'%(N_b,TASK))

        #epoch_train_reg(model, ref_model, optimizer, trainloader, LMD, total=math.ceil(len(trainset)/BATCH_SIZE), N_b=N_b)
        #epoch_eval_reg(model, ref_model, testloader, LMD, total=math.ceil(len(testset)/BATCH_SIZE), N_b=N_b)
        #torch.save(model.state_dict(), 'nn_dirgen.model')

        #epoch_train_seq(model, ref_model, optimizer, trainloader, total=math.ceil(len(trainset)/BATCH_SIZE), N_b=N_b)
        #epoch_eval_seq(model, ref_model, testloader, total=math.ceil(len(testset)/BATCH_SIZE), N_b=N_b)
        #torch.save(model.state_dict(), 'nn_seqgen.model')

