import os
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from cifar10_resnet_model import CifarResNet, CifarDNN
from coco_dataset import ImageNetDataset, CocoDataset

def calc_gt_grad(ref_model, Xs, preprocess_std=(0.229, 0.224, 0.225)):
    X_withg = torch.autograd.Variable(Xs, requires_grad=True)
    #score = ref_model(X_withg).max(1)[0].mean()
    #score.backward()
    scores = ref_model(X_withg)
    labs = scores.max(1)[1]
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    loss = loss_fn(scores, labs)
    loss.backward()
    grad = X_withg.grad.data
    grad = grad / torch.FloatTensor(np.array(preprocess_std)[:,None,None]).cuda()
    return grad

if __name__ == '__main__':
    GPU = True
    N_used = 999999
    TASK = 'cifartrain'
    #REF = 'googlenet'
    for REF in ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
    #for REF in ('res18',):
    #for REF in ('dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet'):
        print ("Task: %s; Ref model: %s"%(TASK, REF))

        if TASK == 'imagenet' or TASK == 'coco':
            BATCH_SIZE = 32
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            if TASK == 'imagenet':
                trainset = ImageNetDataset(train=True, transform=transform)
                testset = ImageNetDataset(train=False, transform=transform)
            else:
                trainset = CocoDataset(train=True, transform=transform)
                testset = CocoDataset(train=False, transform=transform)
            if REF == 'dense121':
                ref_model = models.densenet121(pretrained=True).eval()
            elif REF == 'res18':
                ref_model = models.resnet18(pretrained=True).eval()
            elif REF == 'res50':
                ref_model = models.resnet50(pretrained=True).eval()
            elif REF == 'vgg16':
                ref_model = models.vgg16(pretrained=True).eval()
            elif REF == 'googlenet':
                ref_model = models.googlenet(pretrained=True).eval()
            elif REF == 'wideresnet':
                ref_model = models.wide_resnet50_2(pretrained=True).eval()
            if GPU:
                ref_model.cuda()
        elif TASK == 'cifar' or TASK == 'cifartrain':
            BATCH_SIZE = 64
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            if TASK == 'cifar':
                cifar_testset = torchvision.datasets.CIFAR10(root='../raw_data/', train=False, download=False, transform=transform)
                trainset, testset = torch.utils.data.random_split(cifar_testset, [8000,2000])
            else:
                cifar_testset = torchvision.datasets.CIFAR10(root='../raw_data/', train=True, download=False, transform=transform)
                trainset, testset = torch.utils.data.random_split(cifar_testset, [48000, 2000])

            ref_model = CifarDNN(model_type=REF, gpu=GPU, pretrained=False)
            ref_model.load_state_dict(torch.load('../models/cifar10_%s.model'%REF))
            #if REF == 'dense121':
            #    ref_model = CifarDenseNet(gpu=GPU)
            #    ref_model.load_state_dict(torch.load('../models/cifar10_densenet.model'))
            #elif REF == 'res18':
            #    ref_model = CifarResNet(gpu=GPU)
            #    ref_model.load_state_dict(torch.load('../models/cifar10_resnet18.model'))
            if GPU:
                ref_model.cuda()

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)

        path = '/data/hcli/%s_%s'%(TASK,REF)
        if not os.path.isdir(path):
            os.mkdir(path)
        i = 0
        for Xs, _ in tqdm(trainloader):
            if GPU:
                Xs = Xs.cuda()
            grad_gt = calc_gt_grad(ref_model, Xs)
            grad_gt = grad_gt.reshape(grad_gt.shape[0], -1)
            np.save(path+'/train_batch_%d.npy'%i, grad_gt.cpu().numpy())
            i += 1
            if (i * BATCH_SIZE >= N_used):
                break

        i = 0
        for Xs, _ in tqdm(testloader):
            if GPU:
                Xs = Xs.cuda()
            grad_gt = calc_gt_grad(ref_model, Xs)
            grad_gt = grad_gt.reshape(grad_gt.shape[0], -1)
            np.save(path+'/test_batch_%d.npy'%i, grad_gt.cpu().numpy())
            i += 1
            if (i * BATCH_SIZE >= N_used):
                break
