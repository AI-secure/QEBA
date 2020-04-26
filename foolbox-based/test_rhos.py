import numpy as np
from pca_generator import PCAGenerator
from bapp_dataset import BAPPDataset
from coco_dataset import CocoDataset
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import argparse
from train_pca_generator import load_all_grads
from gradient_generate import calc_gt_grad

if __name__ == '__main__':
    N_b = 9408
    X_shape = (3,224,224)
    approx = True
    BATCH_SIZE = 32
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    testset = CocoDataset(train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)

    grad_test = load_all_grads('imagenet', 'res18', train=False, N_used=1000, mounted=False)
    p_gen = PCAGenerator(N_b=9408, approx=True)
    p_gen.load('pca_gen_9408_imagenet_avg.npy')
    print ("model loaded")
    from dct_generator import DCTGenerator
    p_gen_dct = DCTGenerator(factor=4.0)
    print ("dct model loaded")

    ref_model = models.resnet18(pretrained=True).eval()
    #grad_pre = np.load('/data/hcli/imagenet_res18/test_batch_0.npy')[0]*32.0
    #grad_cur = calc_gt_grad(ref_model, testset[0][0].unsqueeze(0))[0].reshape(-1).cpu().numpy()
    #print (grad_pre)
    #print (grad_cur)
    #print (grad_pre.shape)
    #print (grad_cur.shape)

    #X, y = testset[0]
    #r1 = p_gen_dct.calc_rho(grad_test[0].reshape(*X_shape), X.numpy())
    #print (r1)
    #sgn = np.eye(9408).reshape(9408, 3, 56, 56)
    #print (sgn.shape)
    #print (sgn.sum(0))
    #print (sgn.sum((1,2,3)))
    #basis = np.zeros((9408,3,224,224))
    #basis[:,:,:56,:56] = sgn
    #from dct_generator import RGB_signal_idct, RGB_img_dct
    ##sgn = RGB_img_dct(grad_test[0].reshape)
    #print (basis.shape)
    #for _ in tqdm(range(9408)):
    #    basis[_] = RGB_signal_idct(basis[_])
    #basis = basis.reshape(basis.shape[0], -1)
    #print (basis.shape)
    #r2 = np.linalg.norm(grad_test[0].dot(basis.transpose())) / np.linalg.norm(grad_test[0])
    #print (r2)
    #assert 0

    #X, y = testset[0]
    #ypred = ref_model(X.unsqueeze(0)).max(1)[1]
    #r1 = p_gen.calc_rho(grad_test[0].reshape(*X_shape), X.numpy())
    #print (r1)
    #assert 0

    #X, _ = testset[0]
    #X = torch.autograd.Variable(X, requires_grad=True)
    #import foolbox
    #from attack_setting import load_imagenet_img
    #mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    #std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    #fmodel = foolbox.models.PyTorchModel(ref_model, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    #src_image = load_imagenet_img('../raw_data/imagenet/280000.JPEG')
    ##print (X)
    ##print ((src_image-mean)/std)
    #score1 = ref_model(X.cuda().unsqueeze(0))[0]
    ##score2 = fmodel.forward_one(src_image)
    #score2 = fmodel.forward_one(src_image)
    ##print (score1)
    ##print (score2)
    #src_l1 = torch.argmax(score1)
    #src_l2 = np.argmax(score2)
    #print (src_l1, src_l2)
    #grad1 = calc_gt_grad(ref_model, X.cuda().unsqueeze(0))[0]
    #print (grad1)
    ##print (grad_test[0].reshape(3,224,224)*32)
    #grad2 = fmodel.gradient_one(src_image, label=src_l2)
    #print (grad2)
    #print (grad1.cpu().numpy() / grad2)
    #assert 0
    ##print (ypred, src_label)
    #print (ref_model(X.cuda().unsqueeze(0))[0])
    #print (fmodel.forward_one(src_image))
    #grad_fool = fmodel.gradient_one(src_image, label=src_label)
    #print (grad_fool.reshape(-1))
    #print (grad_test[0]*32)
    #assert 0

    tot_rho = 0.0
    for i in range(1000):
        X, y = testset[i]

        cur_rho = p_gen.calc_rho(grad_test[i].reshape(*X_shape), X.numpy())
        cur_rho_dct = p_gen_dct.calc_rho(grad_test[i].reshape(*X_shape), X.numpy())
        #cur_rho_norm = np.linalg.norm(grad_test[i].dot(p_gen.basis.transpose())) / np.linalg.norm(grad_test[i])
        tot_rho += cur_rho
        #print ("Cur:%.6f; Avg:%.6f; calc by norm:%.6f; by dct:%.6f"%(cur_rho, tot_rho/(i+1), cur_rho_norm, cur_rho_dct))
        print ("Cur:%.6f; Avg:%.6f; by dct:%.6f"%(cur_rho, tot_rho/(i+1), cur_rho_dct))
        #if cur_rho > 0.65:
        #    print ("At %d, rho is %.6f (avg rho %.6f)"%(i, cur_rho, tot_rho/(i+1)))
