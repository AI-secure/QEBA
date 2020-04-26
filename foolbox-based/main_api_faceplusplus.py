import foolbox
from foolbox.criteria import TargetClass
import numpy as np
from dct_generator import DCTGenerator
from resize_generator import ResizeGenerator
from pca_generator import PCAGenerator
import json
import argparse


size_224 = 224, 224


def load_img(path):
    from PIL import Image
    image = Image.open(path)
    image = image.resize(size_224)
    image = np.asarray(image, dtype=np.float32)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='facepp')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--gen', type=int, default=0)
    parser.add_argument('--src', type=str)
    parser.add_argument('--tgt', type=str)
    args = parser.parse_args()

    src_img_path = '../%s.jpg' %(args.src)
    tgt_img_path = '../%s.jpg' %(args.tgt)

    fmodel = foolbox.models.FacePlusPlusModel(bounds=(0,255), src_img_path=src_img_path, channel_axis=2, simi_threshold=args.threshold)

    src_image = load_img(src_img_path)
    tgt_image = load_img(tgt_img_path)
    src_label = np.argmax(fmodel.forward_one(src_image))
    tgt_label = np.argmax(fmodel.forward_one(tgt_image))
    assert src_label != tgt_label
    print (src_image.shape)
    print (tgt_image.shape)
    print ("Source Image Label:", src_label)
    print ("Target Image Label:", tgt_label)

    mask = None
    #mask = np.zeros((320,240,3)).astype(np.float32)
    ##mask[200:,:120] = 1
    #mask[160:,:] = 1

    if args.gen == 0:
        p_gen = None
    elif args.gen == 1:
        p_gen = DCTGenerator(factor=4.0, preprocess=((2,0,1),0,1))
    elif args.gen == 2:
        p_gen = ResizeGenerator(factor=4.0, preprocess=((2,0,1),0,1))
    elif args.gen == 3:
        p_gen = PCAGenerator(N_b=9408, approx=True, basis_only=True, preprocess=((2,0,1),0,1))
        p_gen.load('../../pca_gen_9408_imagenet_avg.npy')
    else:
        print("Generator unsupported")
        assert 0
    #p_gen = ResizeGenerator(preprocess=((2,0,1),0,1))
    attack = foolbox.attacks.BAPP_custom(fmodel, criterion=TargetClass(src_label))
    adv = attack(tgt_image, tgt_label, starting_point = src_image, iterations=50, batch_size=9999,
                 stepsize_search='geometric_progression', verbose=True, unpack=False, max_num_evals=100,
                 initial_num_evals=100, internal_dtype=np.float32, rv_generator = p_gen, atk_level=999, mask=mask,
                 discretize=True, suffix='_%s_%d_%s_%s'%(args.suffix,args.gen, args.src, args.tgt))

    print (attack.logger)
    with open('BAPP_result/attack_%s_%d_%s_%s.log'%(args.suffix, args.gen, args.src, args.tgt), 'w') as outf:
        json.dump(attack.logger, outf)

    if mask is not None:
        np.savez('BAPP_result/mask_%s_%d_%s_%s.npz'%(args.suffix, args.gen, args.src, args.tgt), mask=mask, perturbed=adv.perturbed)
