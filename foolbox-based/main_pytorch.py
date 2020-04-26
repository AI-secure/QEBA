import foolbox
import numpy as np
from foolbox.criteria import TargetClass
import argparse

import torch
import json
from attack_setting import load_imagenet_img, load_pgen, imagenet_attack, cifar_attack

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
    TASK = 'imagenet'

    np.random.seed(0)
    if TASK == 'imagenet':
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = imagenet_attack(args, 50)
    elif TASK == 'cifar':
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = cifar_attack(args, 50)
    elif TASK == 'celeba':
        src_images, src_labels, tgt_images, tgt_labels, fmodel, mask = celeba_attack(args, 50)
    else:
        raise NotImplementedError()
    #
    from PIL import Image
    for pid, img in enumerate(src_images+tgt_images):
        print (img.shape)
        j = Image.fromarray((img.transpose(1,2,0)*255).astype(np.uint8))
        j.save('BAPP_result/img%d.png'%pid)

        grad_gt = fmodel.gradient_one(img, label=(src_labels+tgt_labels)[pid])
        j = Image.fromarray((np.abs(grad_gt).transpose(1,2,0)*255).astype(np.uint8))
        j.save('BAPP_result/grad%d.png'%pid)
    assert 0
    #

    #2, 11, 16, 28, 49
    #src_image, src_label = src_images[28], src_labels[28]
    #tgt_image, tgt_label = tgt_images[28], tgt_labels[28]

    #src_image = load_imagenet_img('../raw_data/imagenet_example/doge.png')
    #src_image = load_imagenet_img('../raw_data/imagenet_example/awkward_moment_seal.png')
    src_image = load_imagenet_img('../raw_data/imagenet_example/393_anemonefish.jpg')
    tgt_image = load_imagenet_img('../raw_data/imagenet_example/282_tigercat.jpg')
    src_label = np.argmax(fmodel.forward_one(src_image))
    tgt_label = np.argmax(fmodel.forward_one(tgt_image))

    print (src_image.shape)
    print (tgt_image.shape)
    print ("Source Image Label:", src_label)
    print ("Target Image Label:", tgt_label)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(src_image.transpose(1,2,0))
    plt.savefig('BAPP_result/%s_%s_src.png'%(TASK, args.suffix))
    plt.close(fig)
    fig = plt.figure()
    plt.imshow(tgt_image.transpose(1,2,0))
    plt.savefig('BAPP_result/%s_%s_tgt.png'%(TASK, args.suffix))
    plt.close(fig)
    np.save('BAPP_result/%s_%s_src.npy'%(TASK, args.suffix), src_image.transpose(1,2,0))
    np.save('BAPP_result/%s_%s_tgt.npy'%(TASK, args.suffix), tgt_image.transpose(1,2,0))

    ### Test generator
    #import time
    #t1 = time.time()
    #rvs = p_gen.generate_ps(src_image, 10, level=3)
    #t2 = time.time()
    #print (rvs.shape)
    #grad_gt = fmodel.gradient_one(src_image, label=src_label)
    #print (grad_gt.shape)
    #print (p_gen.calc_rho(grad_gt, src_image))
    #t3 = time.time()
    #print (t2-t1,t3-t2)
    #assert 0

    #for PGEN in ['PCA9408basis', 'DCT9408', 'naive', 'resize9408']:
    #for PGEN in ['DCT9408', 'naive', 'resize9408']:
    for PGEN in ['naive',]:
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
            elif PGEN.startswith('PCA'):
                ITER = 150
                maxN = 30
                initN = 30
            else:
                raise NotImplementedError()
        elif TASK == 'imagenet' or TASK == 'celeba':
            if PGEN == 'naive':
                ITER = 100
                maxN = 100
                initN = 100
            elif PGEN.startswith('PCA'):
                ITER = 100
                maxN = 100
                initN = 100
            elif PGEN.startswith('DCT') or PGEN.startswith('resize'):
                ITER = 100
                maxN = 100
                initN = 100
            elif PGEN == 'NNGen':
                ITER = 500
                maxN = 30
                initN = 30
            else:
                raise NotImplementedError()
        #ITER = 20
        print ("PGEN: %s"%PGEN)
        if p_gen is None:
            rho = 1.0
        else:
            rvs = p_gen.generate_ps(src_image, 10, level=999)
            grad_gt = fmodel.gradient_one(src_image, label=src_label)
            rho = p_gen.calc_rho(grad_gt, src_image).item()
        print ("rho: %.4f"%rho)
        attack = foolbox.attacks.BAPP_custom(fmodel, criterion=TargetClass(src_label))
        adv = attack(tgt_image, tgt_label, starting_point = src_image, iterations=ITER, stepsize_search='geometric_progression', unpack=False, max_num_evals=maxN, initial_num_evals=initN, internal_dtype=np.float32, rv_generator = p_gen, atk_level=args.atk_level, mask=mask, batch_size=16, rho_ref = rho, log_every_n_steps=1, suffix=args.suffix+PGEN, verbose=False)

        with open('BAPP_result/attack_%s_%s_%s.log'%(TASK, PGEN, args.suffix), 'w') as outf:
            json.dump(attack.logger, outf)

