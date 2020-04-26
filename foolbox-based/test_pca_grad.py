import numpy as np
import os
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from cifar10_resnet_model import CifarResNet, CifarDenseNet, epoch_train, epoch_eval
import argparse
import time
from sklearn.decomposition import PCA
import fbpca
from pca_generator import gen_topK_colspace
from disk_mat import DiskMatrix

#def calc_gt_grad(ref_model, Xs):
#    X_withg = torch.autograd.Variable(Xs, requires_grad=True)
#    score = ref_model(X_withg).max(1)[0].mean()
#    score.backward()
#    grad = X_withg.grad.data
#    return grad

#def gen_all_grads(model, dataloader, GPU):
#    grads = []
#    for Xs, _ in tqdm(dataloader):
#        if GPU:
#            Xs = Xs.cuda()
#        B = Xs.size(0)
#        gradX = calc_gt_grad(model, Xs)
#        grads.append(gradX.detach().cpu().numpy().reshape(B, -1))
#    return np.concatenate(grads, axis=0)

if __name__ == '__main__':
    #GPU = True
    #BATCH_SIZE = 64
    #mean = (0.485, 0.456, 0.406)
    #std = (0.229, 0.224, 0.225)
    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean, std),
    #])
    #testset = torchvision.datasets.CIFAR10(root='../raw_data/', train=False, download=True, transform=transform)
    #pca_trainset, pca_testset = torch.utils.data.random_split(testset, [8000, 2000])
    #pca_trainloader = torch.utils.data.DataLoader(pca_trainset, batch_size=BATCH_SIZE)
    #pca_testloader = torch.utils.data.DataLoader(pca_testset, batch_size=BATCH_SIZE)
    #trainset = torchvision.datasets.CIFAR10(root='../raw_data/', train=True, download=True, transform=transform)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    TASK = 'imagenet'
    REF = 'dense121'
    TRANSF = 'res18'

    ### Load Grad
    #grads_train = load_all_grads(TASK, REF, train=True, N_used=20000)
    #grads_test = load_all_grads(TASK, REF, train=False, N_used=5000)
    #grads_test_transfer = load_all_grads(TASK, TRANSF, train=True, N_used=5000)
    grads_train = load_diskmat(TASK, REF, train=True, N_used=100000)
    grads_test = load_diskmat(TASK, REF, train=False, N_used=5000)
    grads_test_transfer = load_diskmat(TASK, TRANSF, train=True, N_used=5000)
    print (grads_train.shape)
    print (grads_test.shape)
    print (grads_test_transfer.shape)
    #print (grads_train.dot(np.zeros((grads_train.shape[1],2))).shape)
    #print (grads_train.leftdot(np.zeros((2,grads_train.shape[0]))).shape)
    #assert 0

   # for N_red in (20, 50, 100, 192, 200, 500, 768, 1000):
    #for N_red in (100, 300, 500, 1000, 5000, 9408):
    #for N_red in (100, 1000, 5000, 9408):
    N_red = 9408
    for N_multi in (20,50,100,200,500,1000):
        print ("N_multi:", N_multi)
        grads_train.N_multi=N_multi
        # print()

        #time1 = time.time()
        #pca_model = PCA(N_red)
        #pca_model.fit(grads_train)
        #pca_comp = pca_model.components_
        #reduced_train = grads_train.dot(pca_comp.transpose())
        #reduced_test = grads_test.dot(pca_comp.transpose())
        #reduced_test_dense = grads_test_transfer.dot(pca_comp.transpose())
        #time2 = time.time()
        #print("sklearn\t%d\t%.6f\t%.6f\t%.6f\t%.4f" %(
        #N_red,
        #np.linalg.norm(reduced_train) / np.linalg.norm(grads_train),
        #np.linalg.norm(reduced_test) / np.linalg.norm(grads_test),
        #np.linalg.norm(reduced_test_dense) / np.linalg.norm(grads_test_transfer),
        #time2-time1))

        time1 = time.time()
        Vt_train = gen_topK_colspace(A=grads_train, k=N_red)
        approx_train = grads_train.dot(Vt_train.transpose())
        approx_test = grads_test.dot(Vt_train.transpose())
        approx_test_dense = grads_test_transfer.dot(Vt_train.transpose())
        time2 = time.time()
        print("approx\t%d\t%.6f\t%.6f\t%.6f\t%.4f" %(
        N_red,
        norm(approx_train) / norm(grads_train),
        norm(approx_test) / norm(grads_test),
        norm(approx_test_dense) / norm(grads_test_transfer),
        time2-time1))
