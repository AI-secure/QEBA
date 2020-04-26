import foolbox
#import keras
import numpy as np

def MSE(x1, x2):
    return ((x1-x2)**2).mean()

def keras_exp(image, label):
    import keras
    from keras.applications.resnet50 import ResNet50

    # instantiate model
    keras.backend.set_learning_phase(0)
    kmodel = ResNet50(weights='imagenet')
    preprocessing = (np.array([104, 116, 123]), 1)
    fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

    # apply attack on source image
    # ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
    attack = foolbox.attacks.PGD(fmodel)
    adversarial = attack(image[:, :, ::-1], label)
    # if the attack fails, adversarial will be None and a warning will be printed
    return adversarial[:,:,::-1]

def tensorflow_exp(image, label):
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras import optimizers
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Lambda

    # instantiate model
    model = ResNet50(weights='imagenet')
    inputs = model.layers[0].input
    def logFunc(x):
        return K.log(x)
    rev_softmax = Lambda(logFunc)
    logits = [layer.output for layer in model.layers][-1]
    logits = rev_softmax(logits)
    used_model = Model(inputs=inputs, outputs=logits)
    
    preprocessing = (np.array([104, 116, 123]), 1)
    fmodel = foolbox.models.TensorFlowModel.from_keras(used_model, bounds=(0, 255), preprocessing=preprocessing)
    
    # apply attack on source image
    # ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
    attack = foolbox.attacks.PGD(fmodel)
    adversarial = attack(image[:, :, ::-1], label)
    # if the attack fails, adversarial will be None and a warning will be printed
    return adversarial[:,:,::-1]

def pytorch_exp(image, label):
    import torchvision.models as models

    model = models.resnet50(pretrained=True)


if __name__ == '__main__':
    # get source image and label
    image, label = foolbox.utils.imagenet_example()

    adversarial = keras_exp(image, label)
    print (MSE(image, adversarial))
    adversarial = tensorflow_exp(image, label)
    print (MSE(image, adversarial))
    #adversarial = pytorch_exp(image, label)
    #print (MSE(image, adversarial))
