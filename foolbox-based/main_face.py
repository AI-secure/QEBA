import foolbox
from foolbox.criteria import TargetClass
import numpy as np
from dct_generator import DCTGenerator
from resize_generator import ResizeGenerator
import json
import argparse

def load_img(path):
    from PIL import Image
    image = Image.open(path)
    image = np.asarray(image, dtype=np.float32)
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()

    fmodel = foolbox.models.FaceLiveModel(bounds=(0,255))
    #fmodel = foolbox.models.FaceRecogModel(bounds=(0,255))

    src_image = load_img('../raw_data/face/live_a_1.png')
    tgt_image = load_img('../raw_data/face/spoof_a_1.png')
    src_label = np.argmax(fmodel.forward_one(src_image))
    tgt_label = np.argmax(fmodel.forward_one(tgt_image))
    print (src_image.shape)
    print (tgt_image.shape)
    print ("Source Image Label:", src_label)
    print ("Target Image Label:", tgt_label)

    mask = None
    #mask = np.zeros((320,240,3)).astype(np.float32)
    ##mask[200:,:120] = 1
    #mask[160:,:] = 1

    #p_gen = None
    p_gen = DCTGenerator(preprocess=((2,0,1),0,1))
    #p_gen = ResizeGenerator(preprocess=((2,0,1),0,1))
    attack = foolbox.attacks.BAPP_custom(fmodel, criterion=TargetClass(src_label))
    adv = attack(tgt_image, tgt_label, starting_point = src_image, iterations=100, batch_size=9999, stepsize_search='geometric_progression', verbose=True, unpack=False, max_num_evals=100, initial_num_evals=200, internal_dtype=np.float32, rv_generator = p_gen, atk_level=999, mask=mask)

    print (attack.logger)
    with open('BAPP_result/attack_%s.log'%args.suffix, 'w') as outf:
        json.dump(attack.logger, outf)

    if mask is not None:
        np.savez('BAPP_result/mask_%s.npz'%args.suffix, mask=mask, perturbed=adv.perturbed)
