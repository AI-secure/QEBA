import foolbox
import numpy as np
from foolbox.criteria import TargetClass
import argparse

import torch
import torchvision.models as models
from perturb_generator import PerturbGenerator
from unet_generator import UNet
from resize_generator import ResizeGenerator
from dct_generator import DCTGenerator
import json

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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == '__main__':
    resnet18 = models.resnet18(pretrained=True).eval()  # for CPU, remove cuda()
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

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20,4))

    data = np.load('BAPP_result/mask_test1.npz')
    m1, p1 = data['mask'], data['perturbed']
    new_img = m1*p1 + (1-m1)*tgt_image
    pred = softmax(fmodel.forward_one(new_img))
    p_src = pred[src_label]
    p_tgt = pred[tgt_label]
    p_argmax = pred.argmax()
    p_max = pred[p_argmax]
    pred[p_argmax] -= 100
    p_argsec = pred.argmax()
    p_sec = pred[p_argsec]
    print ("M1: %.5f, %.5f, %.5f(%d), %.5f(%d)"%(p_src, p_tgt, p_max, p_argmax, p_sec, p_argsec))
    plt.subplot(1,5,1)
    plt.imshow(new_img.transpose(1,2,0))
    plt.title("Target label prob: %.4f;\nPredicted label prob: %.4f"%(p_src, p_max))
    plt.axis('off')

    #assert 0
    #data = np.load('BAPP_result/mask_test2.npz')
    #m2, p2 = data['mask'], data['perturbed']
    #new_img = m2*p2 + (1-m2)*tgt_image
    #pred = softmax(fmodel.forward_one(new_img))
    #p_src = pred[src_label]
    #p_tgt = pred[tgt_label]
    #p_argmax = pred.argmax()
    #p_max = pred[p_argmax]
    #pred[p_argmax] -= 100
    #p_argsec = pred.argmax()
    #p_sec = pred[p_argsec]
    #print ("M2: %.5f, %.5f, %.5f(%d), %.5f(%d)"%(p_src, p_tgt, p_max, p_argmax, p_sec, p_argsec))
    #plt.subplot(1,5,2)
    #plt.imshow(new_img.transpose(1,2,0))
    #plt.title("Target label prob: %.4f;\nPredicted label prob: %.4f"%(p_src, p_max))
    #plt.axis('off')

    #data = np.load('BAPP_result/mask_test3.npz')
    #m3, p3 = data['mask'], data['perturbed']
    #new_img = m3*p3 + (1-m3)*tgt_image
    #pred = softmax(fmodel.forward_one(new_img))
    #p_src = pred[src_label]
    #p_tgt = pred[tgt_label]
    #p_argmax = pred.argmax()
    #p_max = pred[p_argmax]
    #pred[p_argmax] -= 100
    #p_argsec = pred.argmax()
    #p_sec = pred[p_argsec]
    #print ("M3: %.5f, %.5f, %.5f(%d), %.5f(%d)"%(p_src, p_tgt, p_max, p_argmax, p_sec, p_argsec))
    #plt.subplot(1,5,3)
    #plt.imshow(new_img.transpose(1,2,0))
    #plt.title("Target label prob: %.4f;\nPredicted label prob: %.4f"%(p_src, p_max))
    #plt.axis('off')

    #data = np.load('BAPP_result/mask_test4.npz')
    #m4, p4 = data['mask'], data['perturbed']
    #new_img = m4*p4 + (1-m4)*tgt_image
    #pred = softmax(fmodel.forward_one(new_img))
    #p_src = pred[src_label]
    #p_tgt = pred[tgt_label]
    #p_argmax = pred.argmax()
    #p_max = pred[p_argmax]
    #pred[p_argmax] -= 100
    #p_argsec = pred.argmax()
    #p_sec = pred[p_argsec]
    #print ("M4: %.5f, %.5f, %.5f(%d), %.5f(%d)"%(p_src, p_tgt, p_max, p_argmax, p_sec, p_argsec))
    #plt.subplot(1,5,4)
    #plt.imshow(new_img.transpose(1,2,0))
    #plt.title("Target label prob: %.4f;\nPredicted label prob: %.4f"%(p_src, p_max))
    #plt.axis('off')

    #new_img = tgt_image
    #new_img = m1*p1 + (1-m1)*new_img
    #new_img = m2*p2 + (1-m2)*new_img
    #new_img = m3*p3 + (1-m3)*new_img
    #new_img = m4*p4 + (1-m4)*new_img
    #pred = softmax(fmodel.forward_one(new_img))
    #p_src = pred[src_label]
    #p_tgt = pred[tgt_label]
    #p_argmax = pred.argmax()
    #p_max = pred[p_argmax]
    #pred[p_argmax] -= 100
    #p_argsec = pred.argmax()
    #p_sec = pred[p_argsec]
    #print ("M1+M2+M3+M4: %.5f, %.5f, %.5f(%d), %.5f(%d)"%(p_src, p_tgt, p_max, p_argmax, p_sec, p_argsec))
    #plt.subplot(1,5,5)
    #plt.imshow(new_img.transpose(1,2,0))
    #plt.title("Target label prob: %.4f;\nPredicted label prob: %.4f;\nSecond Highest prob: %.4f"%(p_src, p_max, p_sec))
    #plt.axis('off')
    #plt.show()

    benign_photo = load_imagenet_img('../raw_data/imagenet_example/photo_benign_process.jpg')
    pred = softmax(fmodel.forward_one(benign_photo))
    p_src = pred[src_label]
    p_tgt = pred[tgt_label]
    p_argmax = pred.argmax()
    p_max = pred[p_argmax]
    pred[p_argmax] -= 100
    p_argsec = pred.argmax()
    p_sec = pred[p_argsec]
    print ("Benign photo: %.5f, %.5f, %.5f(%d), %.5f(%d)"%(p_src, p_tgt, p_max, p_argmax, p_sec, p_argsec))

    mal_photo = load_imagenet_img('../raw_data/imagenet_example/photo_mal_process.jpg')
    pred = softmax(fmodel.forward_one(mal_photo))
    p_src = pred[src_label]
    p_tgt = pred[tgt_label]
    p_argmax = pred.argmax()
    p_max = pred[p_argmax]
    pred[p_argmax] -= 100
    p_argsec = pred.argmax()
    p_sec = pred[p_argsec]
    print ("Mal photo: %.5f, %.5f, %.5f(%d), %.5f(%d)"%(p_src, p_tgt, p_max, p_argmax, p_sec, p_argsec))

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(benign_photo.transpose(1,2,0))
    plt.subplot(1,2,2)
    plt.imshow(mal_photo.transpose(1,2,0))
    plt.show()
