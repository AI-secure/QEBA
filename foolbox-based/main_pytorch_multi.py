import foolbox
import numpy as np
from foolbox.criteria import TargetClass
import argparse

import torch
import torchvision.models as models
from perturb_generator import PerturbGenerator
#from biggan_generator import BigGANGenerator
from unet_generator import UNet
from resize_generator import ResizeGenerator
from dct_generator import DCTGenerator
from pca_generator import PCAGenerator
from nn_generator import NNGenerator
from nn_sequential_generator import NNSeqGenerator
import json
from attack_setting import load_pgen, imagenet_attack, cifar_attack, celeba_attack
from tqdm import tqdm

def MSE(x1, x2):
    return ((x1-x2)**2).mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--model_discretize', action='store_true')
    parser.add_argument('--attack_discretize', action='store_true')
    parser.add_argument('--atk_level', type=int, default=999)
    args = parser.parse_args()
    #TASK = 'celeba'
    #TASK = 'cifar'
    TASK = 'imagenet'
    N_img = 50
    N_repeat = 1
    #PGEN = 'DCT'

    np.random.seed(0)
    if TASK == 'imagenet':
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = imagenet_attack(args, N_img)
    elif TASK == 'cifar':
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = cifar_attack(args, N_img)
    elif TASK == 'celeba':
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = celeba_attack(args, N_img)
    else:
        raise NotImplementedError()
    print ("Setting loaded")
    print ("Task: %s; Number of Image: %s; Number of repeat: %s"%(TASK, N_img, N_repeat))

    #for PGEN in ['PCA9408basis', 'DCT9408', 'resize9408', 'naive']:
    #for PGEN in ['resize9408']:
    for PGEN in ['GAN128', ]:

    #for PGEN in ['PCA1200train', 'naive', 'DCT1200']:
    #for PGEN in ['resize768']:
    #for PGEN in ['PCA300train', 'PCA1200train', 'DCT300', 'DCT1200']:
    #for PGEN in ['naive', 'DCT768', 'PCA768', 'DCT192', 'PCA192']:
    #for PGEN in ['DCT1200', 'PCA1200', 'PCA2000']:
    #for PGEN in ['PCA768train']:
        p_gen = load_pgen(TASK, PGEN, args)
        if TASK == 'cifar':
            if PGEN == 'naive':
                ITER = 150
                maxN = 30
                initN = 30
            elif PGEN.startswith('DCT') or PGEN.startswith('resize'):
                ITER = 150
                maxN = 30
                initN = 30
            elif PGEN.startswith('PCA') or PGEN.startswith('AE'):
                ITER = 150
                maxN = 30
                initN = 30
            else:
                raise NotImplementedError()
        elif TASK == 'imagenet' or TASK == 'celeba':
            if PGEN == 'naive':
                ITER = 200
                maxN = 100
                initN = 100
            elif PGEN.startswith('PCA') or PGEN.startswith('AE')or PGEN.startswith('VAE'):
                ITER = 200
                maxN = 100
                initN = 100
            elif PGEN.startswith('DCT') or PGEN.startswith('resize'):
                ITER = 200
                maxN = 100
                initN = 100
            elif PGEN == 'NNGen' or PGEN.startswith('GAN'):
                ITER = 500
                maxN = 30
                initN = 30
            else:
                raise NotImplementedError()

        all_logs = []
        print ("PGEN:", PGEN)
        with tqdm(range(N_img)) as pbar:
            for i in pbar:
                src_image, src_label, tgt_image, tgt_label = src_images[i], src_labels[i], tgt_images[i], tgt_labels[i]
                #print (src_image.shape)
                #print (tgt_image.shape)
                #print ("Source Image Label:", src_label)
                #print ("Target Image Label:", tgt_label)

                ### Test generator
                if p_gen is None:
                    rho = 1.0
                else:
                    rvs = p_gen.generate_ps(src_image, 10, level=999)
                    grad_gt = fmodel.gradient_one(src_image, label=src_label)
                    rho = p_gen.calc_rho(grad_gt, src_image).item()
                pbar.set_description("src label: %d; tgt label: %d; rho: %.4f"%(src_label, tgt_label, rho))
                #print (rho)
                #assert 0
                #continue
                ### Test generator

                for _ in range(N_repeat):
                    attack = foolbox.attacks.BAPP_custom(fmodel, criterion=TargetClass(src_label))
                    adv = attack(tgt_image, tgt_label, starting_point = src_image, iterations=ITER, stepsize_search='geometric_progression', unpack=False, max_num_evals=maxN, initial_num_evals=initN, internal_dtype=np.float32, rv_generator = p_gen, atk_level=args.atk_level, mask=mask, batch_size=16, rho_ref = rho, log_every_n_steps=10, discretize=args.attack_discretize, verbose=False, plot_adv=False)
                    #adv = attack(tgt_image, tgt_label, starting_point = src_image, iterations=ITER, stepsize_search='geometric_progression', unpack=False, max_num_evals=maxN, initial_num_evals=initN, internal_dtype=np.float32, rv_generator = p_gen, atk_level=args.atk_level, mask=mask, batch_size=16, rho_ref = rho, log_every_n_steps=1, discretize=args.attack_discretize, verbose=True, plot_adv=False)
                    #assert 0
                    all_logs.append(attack.logger)

        #assert 0
        with open('BAPP_result/attack_multi_%s_%s_%s.log'%(TASK, PGEN, args.suffix), 'w') as outf:
            json.dump(all_logs, outf)
