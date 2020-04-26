import foolbox
#import keras
import numpy as np
#import keras
#from keras.applications.resnet50 import ResNet50
from foolbox.criteria import TargetClass
import argparse

import torch
import torchvision.models as models
from perturb_generator import PerturbGenerator
#from biggan_generator import BigGANGenerator
from unet_generator import UNet
from resize_generator import ResizeGenerator
from dct_generator import DCTGenerator
import json



def MSE(x1, x2):
    return ((x1-x2)**2).mean()

def load_imagenet_img(path):
    from PIL import Image
    image = Image.open(path)
    tmp = np.array(image)
    image = image.resize((224,224))
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    ### for pytorch ###
    image = image / 255
    image = image.transpose(2,0,1)
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--atk_level', type=int, required=True)
    args = parser.parse_args()

    resnet18 = models.resnet18(pretrained=True).eval()  # for CPU, remove cuda()
    if args.use_gpu:
        resnet18.cuda()

    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    fmodel = foolbox.models.PyTorchModel(resnet18, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))

    src_image = load_imagenet_img('../raw_data/imagenet_example/bad_joke_eel.png')
    tgt_image = load_imagenet_img('../raw_data/imagenet_example/awkward_moment_seal.png')
    #tgt_image = load_imagenet_img('../raw_data/imagenet_example/example.png')
    src_label = np.argmax(fmodel.forward_one(src_image))
    tgt_label = np.argmax(fmodel.forward_one(tgt_image))
    print (src_image.shape)
    print (tgt_image.shape)
    print ("Source Image Label:", src_label)
    print ("Target Image Label:", tgt_label)
    #mask = None
    mask = np.zeros((3,224,224)).astype(np.float32)
    #mask[:,:56,:56] = 1  #H
    #mask[:,-56:,-56:] = 1  #H
    #mask[:,:56,-56:] = 1  #E
    #mask[:,-56:,:56] = 1  #E
    #mask[:,104:120,104:120]=1  #VH
    #mask[:,100:124,100:124]=1  #M
    #mask[:,96:128,96:128]=1  #VE
    mask[:,90:140,90:140]=1  #VVE

    #p_gen = None
    #p_gen = PerturbGenerator(preprocess=((0,1,2),mean,std))
    #p_gen = PerturbGenerator()
    #p_gen = BigGANGenerator()
    #p_gen = UNet(n_channels=3)
    #p_gen.load_state_dict(torch.load('unet.model', map_location='cpu'))
    #p_gen = ResizeGenerator()
    p_gen = DCTGenerator()
    #rvs = p_gen.generate_ps(src_image, 10, level=3)
    #print (rvs)
    #print (rvs.shape)
    #assert 0

    #attack = foolbox.attacks.BoundaryAttackPlusPlus(fmodel)
    #attack = foolbox.attacks.BoundaryAttackPlusPlus(fmodel, criterion=TargetClass(src_label))
    attack = foolbox.attacks.BAPP_physical(fmodel, criterion=TargetClass(src_label))
    adv = attack(tgt_image, tgt_label, starting_point = src_image, iterations=50, stepsize_search='geometric_progression', verbose=True, unpack=False, max_num_evals=100, initial_num_evals=100, internal_dtype=np.float32, rv_generator = p_gen, atk_level=args.atk_level, mask=mask)

    #Final adv
    adversarial = adv.perturbed
    adv_label = np.argmax(fmodel.forward_one(adversarial))
    ret1 = tgt_image
    ret2 = adversarial
    print ("Total calls:", adv._total_prediction_calls)
    print ("Final MSE between Target and Adv:", MSE(ret1, ret2))
    print ("Source label: %d; Target label: %d; Adv label: %d"%(src_label, tgt_label, adv_label))

    print (attack.logger)
    with open('BAPP_result/attack_%s.log'%args.suffix, 'w') as outf:
        json.dump(attack.logger, outf)

    if mask is not None:
        np.savez('BAPP_result/mask_%s.npz'%args.suffix, mask=mask, perturbed=adv.perturbed)


    # Generate an encoder for operating the image
    #p_gen = PerturbGenerator()
    #tgt_image = load_imagenet_img('../raw_data/imagenet_example/awkward_moment_seal.png')
    #rv = p_gen.generate_ps(tgt_image, N=100)
    #print (rv.shape)

    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #for i in range(10):
    #    plt.subplot(2,5,i+1)
    #    rv_plt = rv[i]
    #    if (rv_plt.max() > rv_plt.min()):
    #        print (i)
    #        rv_plt = (rv_plt-rv_plt.min()) / (rv_plt.max()-rv_plt.min())
    #    plt.imshow(rv_plt.transpose(1,2,0))
    #    plt.axis('off')
    #plt.show()
    #fig.savefig('rvs.png')
    #assert 0


