import numpy as np
import foolbox
import torch
from perturb_generator import PerturbGenerator
#from biggan_generator import BigGANGenerator
from unet_generator import UNet
from resize_generator import ResizeGenerator
from dct_generator import DCTGenerator
from pca_generator import PCAGenerator
from nn_generator import NNGenerator
from nn_sequential_generator import NNSeqGenerator
from ae_generator import AEGenerator
from gan_generator import GANGenerator
from vae_generator import VAEGenerator

def load_pgen(task, pgen_type, args):
    if task == 'imagenet' or task == 'celeba':
        if pgen_type == 'naive':
            p_gen = None
        elif pgen_type == 'resize9408':
            p_gen = ResizeGenerator(factor=4.0)
        elif pgen_type == 'DCT2352':
            p_gen = DCTGenerator(factor=8.0)
        elif pgen_type == 'DCT4107':
            p_gen = DCTGenerator(factor=6.0)
        elif pgen_type == 'DCT9408':
            p_gen = DCTGenerator(factor=4.0)
        elif pgen_type == 'DCT16428':
            p_gen = DCTGenerator(factor=3.0)
        elif pgen_type == 'NNGen':
            p_gen = NNGenerator(N_b=30, n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('nn_gen_30_imagenet.model'))
        elif pgen_type == 'ENC':
            p_gen = UNet(n_channels=3)
            p_gen.load_state_dict(torch.load('unet.model', map_location='cpu'))
        elif pgen_type == 'PCA1000':
            p_gen = PCAGenerator(N_b=1000, approx=True)
            p_gen.load('pca_gen_1000_imagenet.npy')
        elif pgen_type == 'PCA5000':
            p_gen = PCAGenerator(N_b=5000, approx=True)
            p_gen.load('pca_gen_5000_imagenet.npy')
        elif pgen_type == 'PCA9408':
            p_gen = PCAGenerator(N_b=9408, approx=True)
            p_gen.load('pca_gen_9408_imagenet_avg.npy')
        elif pgen_type == 'PCA2352basis':
            p_gen = PCAGenerator(N_b=2352, approx=True, basis_only=True)
            p_gen.load('pca_gen_2352_imagenet_avg.npy')
        elif pgen_type == 'PCA4107basis':
            p_gen = PCAGenerator(N_b=4107, approx=True, basis_only=True)
            p_gen.load('pca_gen_4107_imagenet_avg.npy')
        elif pgen_type == 'PCA4107basismore':
            p_gen = PCAGenerator(N_b=4107, approx=True, basis_only=True)
            p_gen.load('pca_gen_4107_imagenet_rndavg.npy')
        elif pgen_type == 'PCA9408basis':
            p_gen = PCAGenerator(N_b=9408, approx=True, basis_only=True)
            p_gen.load('pca_gen_9408_imagenet_avg.npy')
            #p_gen.load('pca_gen_9408_imagenet_abc.npy')
        elif pgen_type == 'PCA9408basismore':
            p_gen = PCAGenerator(N_b=9408, approx=True, basis_only=True)
            p_gen.load('pca_gen_9408_imagenet_rndavg.npy')
            #p_gen.load('pca_gen_9408_imagenet_abc.npy')
        elif pgen_type == 'PCA9408basisnormed':
            p_gen = PCAGenerator(N_b=9408, approx=True, basis_only=True)
            p_gen.load('pca_gen_9408_imagenet_normed_avg.npy')
            #p_gen.load('pca_gen_9408_imagenet_abc.npy')
        elif pgen_type == 'AE9408':
            p_gen = AEGenerator(n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('ae_generator.model'))
        elif pgen_type == 'GAN128':
            p_gen = GANGenerator(n_z=128, n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('gan128_generator.model'))
        elif pgen_type == 'VAE9408':
            p_gen = VAEGenerator(n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('vae_generator.model'))
    elif task == 'cifar':
        if pgen_type == 'naive':
            p_gen = None
        elif pgen_type == 'resize768':
            p_gen = ResizeGenerator(factor=2.0)
        elif pgen_type == 'DCT192':
            p_gen = DCTGenerator(factor=4.0)
        elif pgen_type == 'DCT300':
            p_gen = DCTGenerator(factor=3.0)
        elif pgen_type == 'DCT768':
            p_gen = DCTGenerator(factor=2.0)
        elif pgen_type == 'DCT1200':
            p_gen = DCTGenerator(factor=1.6)
        elif pgen_type == 'PCA192':
            p_gen = PCAGenerator(N_b=192)
            p_gen.load('pca_gen_192_cifar_avg.npy')
        elif pgen_type == 'PCA192train':
            p_gen = PCAGenerator(N_b=192)
            p_gen.load('pca_gen_192_cifartrain_avg.npy')
        elif pgen_type == 'PCA300train':
            p_gen = PCAGenerator(N_b=300)
            p_gen.load('pca_gen_300_cifartrain_avg.npy')
        elif pgen_type == 'PCA768':
            p_gen = PCAGenerator(N_b=768)
            p_gen.load('pca_gen_768_cifar_avg.npy')
        elif pgen_type == 'PCA768train':
            p_gen = PCAGenerator(N_b=768)
            p_gen.load('pca_gen_768_cifartrain_avg.npy')
        elif pgen_type == 'PCA1200':
            p_gen = PCAGenerator(N_b=1200)
            p_gen.load('pca_gen_1200_cifar_avg.npy')
        elif pgen_type == 'PCA1200train':
            p_gen = PCAGenerator(N_b=1200)
            p_gen.load('pca_gen_1200_cifartrain_avg.npy')
        elif pgen_type == 'PCA2000':
            p_gen = PCAGenerator(N_b=2000)
            p_gen.load('pca_gen_2000_cifar_avg.npy')
        elif pgen_type == 'NNGen50':
            p_gen = NNGenerator(N_b=50, n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('nn_gen_50_cifar.model'))
        elif pgen_type == 'NNGen768':
            p_gen = NNGenerator(N_b=768, n_channels=3, gpu=args.use_gpu)
            p_gen.load_state_dict(torch.load('nn_gen_768_cifar.model'))
    return p_gen

def load_imagenet_img(path):
    from PIL import Image
    image = Image.open(path).convert('RGB')
    tmp = np.array(image)
    image = image.resize((224,224))
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    ### for pytorch ###
    image = image / 255
    image = image.transpose(2,0,1)
    return image

def imagenet_attack(args, N_img):
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True).eval()  # for CPU, remove cuda()
    if args.use_gpu:
        resnet18.cuda()
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    fmodel = foolbox.models.PyTorchModel(resnet18, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std), discretize=args.model_discretize)

    src_images, src_labels = [], []
    tgt_images, tgt_labels = [], []
    while (len(src_images) < N_img):
        sid = np.random.randint(280000,300000)
        tid = np.random.randint(280000,300000)
        src_image = load_imagenet_img('../raw_data/imagenet/%d.JPEG'%sid)
        tgt_image = load_imagenet_img('../raw_data/imagenet/%d.JPEG'%tid)
        src_label = np.argmax(fmodel.forward_one(src_image))
        tgt_label = np.argmax(fmodel.forward_one(tgt_image))
        if (src_label != tgt_label):
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            src_labels.append(src_label)
            tgt_labels.append(tgt_label)
            #print (sid, tid)

    mask = None
    #mask = np.zeros((3,224,224)).astype(np.float32)
    ##mask[:,:56,:56] = 1  #H
    ##mask[:,-56:,-56:] = 1  #H
    ##mask[:,:56,-56:] = 1  #E
    ##mask[:,-56:,:56] = 1  #E
    ##mask[:,104:120,104:120]=1  #VH
    ##mask[:,100:124,100:124]=1  #M
    #mask[:,96:128,96:128]=1  #VE
    return src_images, src_labels, tgt_images, tgt_labels, fmodel, mask

def cifar_attack(args, N_img):
    from cifar10_resnet_model import CifarDNN
    model = CifarDNN(model_type='res18', gpu=args.use_gpu, pretrained=False, discretize=args.model_discretize).eval()
    model.load_state_dict(torch.load('../models/cifar10_res18.model'))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    #fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=(mean[:,None,None], std[:,None,None]))
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=(mean.reshape((3,1,1)), std.reshape((3,1,1))))

    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean, std),
    ])
    dataset = datasets.CIFAR10(root='../raw_data/', train=False, download=True, transform=transform)
    src_images, src_labels = [], []
    tgt_images, tgt_labels = [], []
    while (len(src_images) < N_img):
        sid = np.random.randint(len(dataset))
        tid = np.random.randint(len(dataset))
        #sid = np.random.randint(2000)
        #tid = np.random.randint(2000)
        src_image, _ = dataset[sid]
        tgt_image, _ = dataset[tid]
        src_image, tgt_image = src_image.numpy(), tgt_image.numpy()
        src_label = np.argmax(fmodel.forward_one(src_image))
        tgt_label = np.argmax(fmodel.forward_one(tgt_image))
        if (src_label != tgt_label):
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            src_labels.append(src_label)
            tgt_labels.append(tgt_label)
    mask = None
    
    return src_images, src_labels, tgt_images, tgt_labels, fmodel, mask

def celeba_attack(args, N_img):
    from celeba_model import CelebAResNet
    #model = CelebAResNet(10, pretrained=False, gpu=args.use_gpu, discretize=args.model_discretize).eval()
    model = CelebAResNet(10, pretrained=False, gpu=args.use_gpu).eval()
    model.load_state_dict(torch.load('../models/celeba.model'))
    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=(mean.reshape((3,1,1)), std.reshape((3,1,1))))

    from celeba_dataset import CelebADataset
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = CelebADataset(root_dir='../raw_data/celeba', is_train=False, transform=transform, preprocess=False)
    src_images, src_labels = [], []
    tgt_images, tgt_labels = [], []
    while (len(src_images) < N_img):
        sid = np.random.randint(len(dataset))
        tid = np.random.randint(len(dataset))
        src_image, _ = dataset[sid]
        tgt_image, _ = dataset[tid]
        src_image, tgt_image = src_image.numpy(), tgt_image.numpy()
        src_label = np.argmax(fmodel.forward_one(src_image))
        tgt_label = np.argmax(fmodel.forward_one(tgt_image))
        if (src_label != tgt_label):
        #if (True):
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            src_labels.append(src_label)
            tgt_labels.append(tgt_label)
    mask = None

    return src_images, src_labels, tgt_images, tgt_labels, fmodel, mask

def imagenet_attack_old(args, pgen_type):
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True).eval()  # for CPU, remove cuda()
    if args.use_gpu:
        resnet18.cuda()

    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    fmodel = foolbox.models.PyTorchModel(resnet18, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))

    #src_image = load_imagenet_img('../raw_data/imagenet_example/bad_joke_eel.png')
    #tgt_image = load_imagenet_img('../raw_data/imagenet_example/awkward_moment_seal.png')
    src_image = load_imagenet_img('../raw_data/imagenet/280020.JPEG')
    tgt_image = load_imagenet_img('../raw_data/imagenet/280030.JPEG')
    src_label = np.argmax(fmodel.forward_one(src_image))
    tgt_label = np.argmax(fmodel.forward_one(tgt_image))

    if pgen_type == 'naive':
        p_gen = None
    elif pgen_type == 'DCT':
        p_gen = DCTGenerator(factor=4.0)
    elif pgen_type == 'NNGen':
        p_gen = NNGenerator(N_b=30, n_channels=3, gpu=args.use_gpu)
        p_gen.load_state_dict(torch.load('nn_gen.model'))
    elif pgen_type == 'ENC':
        p_gen = UNet(n_channels=3)
        p_gen.load_state_dict(torch.load('unet.model', map_location='cpu'))
    elif pgen_type == 'PCA1000':
        p_gen = PCAGenerator(N_b=1000, approx=True)
        p_gen.load('pca_gen_1000_imagenet.npy')
    elif pgen_type == 'PCA5000':
        p_gen = PCAGenerator(N_b=5000, approx=True)
        p_gen.load('pca_gen_5000_imagenet.npy')
    elif pgen_type == 'PCA5000':
        p_gen = PCAGenerator(N_b=5000, approx=True)
        p_gen.load('pca_gen_5000_imagenet.npy')
    elif pgen_type == 'PCA9408':
        p_gen = PCAGenerator(N_b=9408, approx=True)
        p_gen.load('pca_gen_9408_imagenet_avg.npy')
        #p_gen.load('pca_gen_9408_imagenet_abc.npy')
    else:
        raise NotImplementedError()
    return src_image, src_label, tgt_image, tgt_label, fmodel, p_gen, mask
