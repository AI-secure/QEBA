import foolbox
#import keras
import numpy as np
import keras
from keras.applications.resnet50 import ResNet50
from foolbox.criteria import TargetClass
import argparse


def MSE(x1, x2):
    return ((x1-x2)**2).mean()

def load_imagenet_img(path):
    from PIL import Image
    image = Image.open(path)
    tmp = np.array(image)
    image = image.resize((224,224))
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()
    if args.use_gpu:
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

    keras.backend.set_learning_phase(0)
    kmodel = ResNet50(weights='imagenet')
    preprocessing = (np.array([104, 116, 123]), 1)
    fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

    #image, label = foolbox.utils.imagenet_example()
    #print (image.shape)
    #print (label)
    src_image = load_imagenet_img('../raw_data/imagenet_example/bad_joke_eel.png')
    tgt_image = load_imagenet_img('../raw_data/imagenet_example/awkward_moment_seal.png')
    #tgt_image = load_imagenet_img('../raw_data/imagenet_example/example.png')
    src_label = np.argmax(fmodel.forward_one(src_image[:,:,::-1]))
    tgt_label = np.argmax(fmodel.forward_one(tgt_image[:,:,::-1]))
    print (src_image.shape)
    print (tgt_image.shape)
    print ("Source Image Label:", src_label)
    print ("Target Image Label:", tgt_label)

    #attack = foolbox.attacks.BoundaryAttackPlusPlus(fmodel)
    #attack = foolbox.attacks.BoundaryAttackPlusPlus(fmodel, criterion=TargetClass(src_label))
    attack = foolbox.attacks.BAPP_custom(fmodel, criterion=TargetClass(src_label))
    adv = attack(tgt_image[:,:,::-1], tgt_label, starting_point = src_image[:,:,::-1], iterations=20, stepsize_search='geometric_progression', verbose=True, unpack=False, max_num_evals=100, initial_num_evals=100)

    #attack = foolbox.attacks.BoundaryAttack(fmodel)
    #attack = foolbox.attacks.BoundaryAttack(fmodel, criterion=TargetClass(src_label))
    #adv = attack(tgt_image[:,:,::-1], tgt_label, starting_point = src_image[:,:,::-1], iterations=2000, log_every_n_steps=50, verbose=True, unpack=False)

    #Final adv
    adversarial = adv.perturbed[:,:,::-1]
    adv_label = np.argmax(fmodel.forward_one(adversarial[:,:,::-1]))
    ret1 = tgt_image/255
    ret2 = adversarial/255
    print ("Total calls:", adv._total_prediction_calls)
    print ("Final MSE between Target and Adv:", MSE(ret1, ret2))
    print ("Source label: %d; Target label: %d; Adv label: %d"%(src_label, tgt_label, adv_label))
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(ret1)
    plt.title('Target')
    plt.axis('off')
    plt.subplot(1,4,2)
    plt.imshow(ret2)
    plt.title('Adv')
    plt.axis('off')
    plt.subplot(1,4,3)
    plt.imshow((ret1-ret2)+0.5)
    plt.title('absolute diff')
    plt.axis('off')
    plt.subplot(1,4,4)
    difference = ret1 - ret2
    plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
    plt.title('relative diff')
    plt.axis('off')

    fig.savefig('result.png')
