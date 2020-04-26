import tensorflow as tf
import cv2
import numpy as np
import foolbox
from foolbox.criteria import TargetClass
import argparse

from perturb_generator import PerturbGenerator
from mobilenet_generator import MobileNetGenerator
from unet_generator import UNet
import torch
import torchvision.models as models
import json

def MSE(x1, x2):
    return ((x1-x2)**2).mean()

def get_image_data(img_path, input_height=224, input_width=224):
    im = cv2.imread(img_path)
    if len(im.shape) == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    image_shape = np.shape(im)
    image = cv2.resize(im, (input_width, input_height))
    # bgr to rgb
    image = image[:, :, ::-1].astype(np.float32)
    image = (image / 127.5) - 1.0
    return image, image_shape

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='')
    #parser.add_argument('--atk_level', type=int, required=True)
    args = parser.parse_args()

    #graph = load_pb('../raw_data/receipt/receipt.pb')
    graph = load_pb('../raw_data/meizhuang/meizhuang_model_2.pb')
    inp = graph.get_tensor_by_name('input_1:0')
    outp = graph.get_tensor_by_name('softmax_1/Softmax:0')

    ###
    ##for tensor in graph.as_graph_def().node:
    ##    print (tensor.name)
    #for i, op in enumerate(graph.get_operations()):
    #    print (str(op.name))
    #image, _ = get_image_data('../raw_data/meizhuang/test2/A3_makeups/A3_makeupsA2XIvSrORwOcAAAAAAAAAAABjARYnAA.jpg', input_height=448, input_width=448)
    #with tf.Session(graph=graph) as sess:
    #    ret = sess.run(outp, feed_dict={inp: image[None]})
    #    print (ret.shape)
    #assert 0
    ###

    src_image, _ = get_image_data('../raw_data/meizhuang/test2/A3_makeups/A3_makeupsA2XIvSrORwOcAAAAAAAAAAABjARYnAA.jpg', input_height=448, input_width=448)
    tgt_image, _ = get_image_data('../raw_data/meizhuang/test2/B2_logistics/B2_logisticsA4f1MTIVm3pu_KEZOmucZzwBjARYnAA.jpg', input_height=448, input_width=448)
    #tgt_image, _ = get_image_data('../raw_data/imagenet_example/awkward_moment_seal.png', input_height=448, input_width=448)
    with tf.Session(graph=graph) as sess:
        src_pred = sess.run(outp, feed_dict={inp: src_image[None]})
        src_label = np.argmax(src_pred)
        tgt_pred = sess.run(outp, feed_dict={inp: tgt_image[None]})
        tgt_label = np.argmax(tgt_pred)
    print (src_image.shape)
    print (tgt_image.shape)
    print (src_label, src_pred)
    print (tgt_label, tgt_pred)

    p_gen = None
    transp = np.array((2,0,1))
    mean = -1
    std = 2
    #mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)) * 2 - 1
    #std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)) * 2
    #p_gen = PerturbGenerator(preprocess = (transp, mean, std))
    #p_gen = MobileNetGenerator(preprocess = (transp, mean, std))
    p_gen = UNet(n_channels=3, preprocess = (transp, mean, std), batch_size=8)
    p_gen.load_state_dict(torch.load('unet.model', map_location='cpu'))
    ATK_LVL = 999

    with tf.Session(graph=graph) as sess:
        fmodel = foolbox.models.TensorFlowModel(inp, outp, (-1,1))
        attack = foolbox.attacks.BAPP_custom(fmodel, criterion=TargetClass(src_label))
        adv = attack(tgt_image, tgt_label, starting_point = src_image, iterations=200, stepsize_search='geometric_progression', verbose=True, unpack=False, max_num_evals=100, initial_num_evals=100, internal_dtype=np.float32, rv_generator = p_gen, atk_level=ATK_LVL, batch_size=1)

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

