import numpy as np
from pca_generator import PCAGenerator
from bapp_dataset import BAPPDataset
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from dogcat_dataset import DogCatDataset
from tiny_imagenet import TinyImagenet
from tqdm import tqdm
from disk_mat import DiskMatrix, ConcatDiskMat
import os
import argparse

def load_all_grads(task, ref, train=True, N_used=None, mounted=False):
    grads = []
    if mounted:
        path = '/home/hcli/data/%s_%s/%s_batch'%(task, ref, 'train' if train else 'test')
    else:
        path = '/data/hcli/%s_%s/%s_batch'%(task, ref, 'train' if train else 'test')
    #path = '../grad_data/%s_%s/%s_batch'%(task, ref, 'train' if train else 'test')
    i = 0
    used_num = 0
    while used_num < N_used and os.path.exists(path+'_%d.npy'%i):
        cur_block = np.load(path+'_%d.npy'%i)
        if used_num + cur_block.shape[0] > N_used:
            cur_block = cur_block[:N_used-used_num]
        used_num += cur_block.shape[0]
        i += 1
        grads.append(cur_block)
    return np.concatenate(grads, axis=0)

def load_diskmat(task, ref, train=True, N_used=None, N_multi=1, mounted=False):
    if mounted:
        path = '/home/hcli/data/%s_%s/%s_batch'%(task, ref, 'train' if train else 'test')
    else:
        path = '/data/hcli/%s_%s/%s_batch'%(task, ref, 'train' if train else 'test')
    #path = '../grad_data/%s_%s/%s_batch'%(task, ref, 'train' if train else 'test')
    return DiskMatrix(path, N_used=N_used, N_multi=N_multi)

def norm(A):
    if isinstance(A, np.ndarray):
        return np.linalg.norm(A)
    else:
        return A.norm()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mounted', action='store_true')
    args = parser.parse_args()

    GPU = True
    TASK = 'imagenet'
    #REFs = ['res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet']
    #REFs = ['res18',]
    #REFs = ['rnd',]
    REF = 'avg'
    #REF = 'res18'
    TRANSF = 'res18'
    print ("REF:", REF)
    print ("TRANSF:", TRANSF)

    if TASK == 'imagenet':
        N_b = 9408
        #N_b = 100
        X_shape = (3,224,224)
        approx = True
    elif TASK == 'cifar' or TASK == 'cifartrain':
        N_b = 1200
        X_shape = (3,32,32)
        approx = False
    #save_path = 'pca_gen_%d_%s_%s.npy'%(N_b,TASK,REF)
    save_path = 'pca_gen_%d_imagenetcoco_%s.npy'%(N_b,REF)
    print ("save path:", save_path)

    if TASK == 'imagenet':
        #grads_train = load_diskmat(TASK, REF, train=True, N_used=280000, N_multi=50, mounted=args.mounted)
        grads_train = ConcatDiskMat([
                load_diskmat(TASK, REF, train=True, N_used=280000, N_multi=50, mounted=args.mounted),
                load_diskmat('coco', REF, train=True, N_used=120000, N_multi=50, mounted=args.mounted),
                ])
        #grads_train = load_all_grads(TASK, REF, train=True, N_used=4000, mounted=args.mounted)
        grads_test = load_all_grads(TASK, REF, train=False, N_used=4000, mounted=args.mounted)
        grads_test_transfer = load_all_grads(TASK, TRANSF, train=False, N_used=4000, mounted=args.mounted)
    elif TASK == 'cifar':
        grads_train = load_all_grads(TASK, REF, train=True, N_used=8000, mounted=args.mounted)
        grads_test = load_all_grads(TASK, REF, train=False, N_used=2000, mounted=args.mounted)
        grads_test_transfer = load_all_grads(TASK, TRANSF, train=False, N_used=2000, mounted=args.mounted)
    elif TASK == 'cifartrain':
        grads_train = load_all_grads(TASK, REF, train=True, N_used=48000, mounted=args.mounted)
        grads_test = load_all_grads(TASK, REF, train=False, N_used=2000, mounted=args.mounted)
        grads_test_transfer = load_all_grads(TASK, TRANSF, train=False, N_used=2000, mounted=args.mounted)
    print (grads_train.shape)
    print (grads_test.shape)
    print (grads_test_transfer.shape)

    model = PCAGenerator(N_b=N_b, X_shape=X_shape, approx=approx)
    model.fit(grads_train)
    model.save(save_path)
    print ("Model Saved")

    #model.load(save_path)
    #print ("Model Loaded")

    #### Test on DCT basis
    #sgn = np.eye(9408).reshape(9408, 3, 56, 56)
    #print (sgn.shape)
    #print (sgn.sum(0))
    #print (sgn.sum((1,2,3)))
    #basis = np.zeros((9408,3,224,224))
    #basis[:,:,:56,:56] = sgn
    #print (basis.shape)
    #from dct_generator import RGB_signal_idct
    #for _ in tqdm(range(9408)):
    #    basis[_] = RGB_signal_idct(basis[_])
    #model.X_shape = basis.shape[1:]
    #model.basis = basis.reshape(basis.shape[0], -1)
    #print (model.basis.shape)
    #### Test on DCT basis

    approx_test = grads_test.dot(model.basis.transpose())
    approx_test_transfer = grads_test_transfer.dot(model.basis.transpose())
    print("Rho: ?\t%.6f\t%.6f" %(
        #norm(approx_train) / norm(grads_train),
        norm(approx_test) / norm(grads_test),
        norm(approx_test_transfer) / norm(grads_test_transfer),
        ))
    approx_train = grads_train.dot(model.basis.transpose())
    print (norm(approx_train) / norm(grads_train))
    assert 0

    if TASK == 'imagenet':
        BATCH_SIZE = 64
        N_b = 5000
        N_used = 100000
        approx = True
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        #trainset = DogCatDataset(train=True, transform=transform)
        #testset = DogCatDataset(train=False, transform=transform)
        trainset = TinyImagenet(train=True, transform=transform)
        testset = TinyImagenet(train=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
        ref_model = models.resnet18(pretrained=True).eval()
    elif TASK == 'cifar':
        BATCH_SIZE = 64
        N_b = 768
        #N_b = 50
        N_used = 10000
        approx = True
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        import torchvision
        cifar_testset = torchvision.datasets.CIFAR10(root='../raw_data/', train=False, download=False, transform=transform)
        trainset, testset = torch.utils.data.random_split(cifar_testset, [8000,2000])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
        #from cifar10_resnet_model import CifarResNet
        #ref_model = CifarResNet(gpu=GPU)
        #ref_model.load_state_dict(torch.load('../models/cifar10_resnet18.model'))
        from cifar10_resnet_model import CifarDenseNet
        ref_model = CifarDenseNet(gpu=GPU)
        ref_model.load_state_dict(torch.load('../models/cifar10_densenet.model'))
    else:
        raise NotImplementedError()

    model = PCAGenerator(N_b=N_b, approx=approx)
    if GPU:
        ref_model.cuda()
    grads = []
    #for Xs, _ in tqdm(trainloader):
    for Xs, _ in trainloader:
    #for Xs, _ in tqdm(testloader):
        if GPU:
            Xs = Xs.cuda()
        grad_gt = calc_gt_grad(ref_model, Xs)
        #grads.append(grad_gt)
        grads.append(grad_gt.cpu().numpy())
        if (len(grads)*BATCH_SIZE > N_used):
            break
    grads = np.concatenate(grads, axis=0)
    model.fit(grads)
    model.save('pca_gen_%d_%s.npy'%(N_b,TASK))
    #model.load('pca_gen_1000_imagenet.npy')

    #X = testset[3001][0].cuda()
    ##print (X)
    ##print (X.shape)
    ##assert 0
    #grad_gt = calc_gt_grad(ref_model, X.unsqueeze(0))[0]
    #print (model.calc_rho(grad_gt.cpu().numpy(), X.cpu().numpy()))

    for Xs, _ in testloader:
        if GPU:
            Xs = Xs.cuda()
        grad_gt = calc_gt_grad(ref_model, Xs)
        print (model.calc_rho(grad_gt[0].cpu().numpy(), Xs[0].cpu().numpy()))
