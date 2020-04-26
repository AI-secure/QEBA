import numpy as np
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
#from tqdm import tqdm
from dogcat_dataset import DogCatDataset
from tiny_imagenet import TinyImagenet
import time
import fbpca
import argparse
from pca_generator import gen_topK_colspace


def calc_gt_grad(ref_model, Xs):
    X_withg = torch.autograd.Variable(Xs, requires_grad=True)
    score = ref_model(X_withg).max(1)[0].mean()
    score.backward()
    grad = X_withg.grad.data
    return grad

def gen_all_grads(model, dataloader, GPU, N_used=100000):
    grads = []
    tot_num = 0
    #for Xs, _ in tqdm(dataloader):
    for Xs, _ in dataloader:
        if GPU:
            Xs = Xs.cuda()
        B = Xs.size(0)
        gradX = calc_gt_grad(model, Xs)
        grads.append(gradX.detach().cpu().numpy().reshape(B, -1))

        tot_num += B
        if (tot_num > N_used):
            break
    return np.concatenate(grads, axis=0)

if __name__ == '__main__':
    GPU = True
    BATCH_SIZE=32
    parser = argparse.ArgumentParser()
    parser.add_argument('--calc_grads', type=int, default=0)
    args = parser.parse_args()

    # ref model: densenet
    model = models.densenet121(pretrained=True).eval()
    if GPU:
        model = model.cuda()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    #trainset = DogCatDataset(train=True, transform=transform)
    #testset = DogCatDataset(train=False, transform=transform)
    trainset = TinyImagenet(train=True, transform=transform)
    testset = TinyImagenet(train=False, transform=transform)

    pca_trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    pca_testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)

    TASK = 'imagenet'
    if args.calc_grads:
        grads_train = gen_all_grads(model, pca_trainloader, GPU)
        #np.save('train_grads_%s.npy'%(TASK), grads_train)
        grads_test = gen_all_grads(model, pca_testloader, GPU)
        #np.save('test_grads_%s.npy' % (TASK), grads_test)

        # victim model: resnet18
        model = models.resnet18(pretrained=True).eval()
        if GPU:
            model = model.cuda()
        grads_test_transfer = gen_all_grads(model, pca_testloader, GPU)
        #np.save('transfer_grads_%s.npy' % (TASK), grads_test_transfer)
    else:
        grads_train = np.load('train_grads_%s.npy' % (TASK))
        grads_test = np.load('test_grads_%s.npy' % (TASK))
        grads_test_transfer = np.load('transfer_grads_%s.npy' % (TASK))

    print (grads_train.shape)
    # print (np.linalg.matrix_rank(grads_train))
    print (grads_test.shape)
    # print (np.linalg.matrix_rank(grads_test))
    print (grads_test_transfer.shape)
    # print (np.linalg.matrix_rank(grads_test_transfer))

    #for N_red in (100, 300, 500, 1000, 3000, 5000, 9408, 12000):
    #for N_red in (100, 300, 500, 1000, 5000):
    #for N_red in (1000, 5000, 9408):
    for N_red in (5000,):
        print ()
        time3 = time.time()
        #U_train, S_train, Vt_train = fbpca.pca(A=grads_train, k=N_red)
        Vt_train = gen_topK_colspace(A=grads_train, k=N_red)
        time4 = time.time()
        approx_train = grads_train @ Vt_train.transpose()
        approx_test = grads_test @ Vt_train.transpose()
        approx_test_dense = grads_test_transfer @ Vt_train.transpose()

        print("approx\t%d\t%.6f\t%.6f\t%.6f\t%.4f" % (
        N_red,
        (np.linalg.norm(approx_train, axis=1) / np.linalg.norm(grads_train, axis=1)).mean(),
        (np.linalg.norm(approx_test, axis=1) / np.linalg.norm(grads_test, axis=1)).mean(),
        (np.linalg.norm(approx_test_dense, axis=1) / np.linalg.norm(grads_test_transfer, axis=1)).mean(),
        time4 - time3))
