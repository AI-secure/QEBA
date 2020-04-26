import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from dogcat_dataset import DogCatDataset
from tiny_imagenet import TinyImagenet

def calc_gt_grad(ref_model, Xs):
    X_withg = torch.autograd.Variable(Xs, requires_grad=True)
    score = ref_model(X_withg).max(1)[0].mean()
    score.backward()
    grad = X_withg.grad.data
    return grad

GPU = True

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
#trainset = TinyImagenet(train=True, transform=transform)
#testset = TinyImagenet(train=False, transform=transform)
trainset = DogCatDataset(train=True, transform=transform)
testset = DogCatDataset(train=False, transform=transform)

model = models.resnet18(pretrained=True).eval()
if GPU:
    model = model.cuda()

for i in range(10):
    grad = calc_gt_grad(model, trainset[i][0].cuda().unsqueeze(0)).squeeze(0)
    grad_f = grad.cpu().detach().numpy()
    threshold = np.percentile(np.abs(grad_f), 95)
    print (grad_f.shape)
    grad_f_clip = grad_f.copy()
    grad_f_clip[np.abs(grad_f)<threshold]=0
    print ("Retained percantage:", np.linalg.norm(grad_f_clip)/np.linalg.norm(grad_f))
    print ("Ratio of non-zero:", (grad_f_clip!=0).sum()/(grad_f!=0).sum())
assert 0

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#
#fig = plt.figure()
#plt.imshow(grad_f[0])
#plt.colorbar()
#fig.savefig('grad_sparse-R.pdf')
#
#fig = plt.figure()
#plt.imshow(grad_f[1])
#plt.colorbar()
#fig.savefig('grad_sparse-G.pdf')
#
#fig = plt.figure()
#plt.imshow(grad_f[2])
#plt.colorbar()
#fig.savefig('grad_sparse-B.pdf')
