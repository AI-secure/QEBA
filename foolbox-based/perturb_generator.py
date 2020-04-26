import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class PerturbGenerator:
    def __init__(self, batch_size=32, preprocess=None, gpu=False):
        self.model = models.vgg16(pretrained=True).eval()
        self.fc1 = list(self.model.classifier)[0]
        #self.model = models.resnet18(pretrained=True).eval()

        # For generating perturb
        self.batch_size = batch_size
        # For preprocessing
        self.preprocess = preprocess

        self.gpu = gpu
        if self.gpu:
            self.model.cuda()

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std

        inp = torch.FloatTensor(inp)
        if self.gpu:
            inp = inp.cuda()
        inp.requires_grad = True

        #x = self.model.features(inp.unsqueeze(0))
        #x = self.model.avgpool(x)
        #feature = self.fc1(x.view(x.size(0), -1)).squeeze(0)
        #rvs = []
        #indices = np.random.choice(len(feature), N, replace=False)
        #for i in indices:
        #    test = torch.autograd.grad(feature[i], inp, retain_graph=True)[0]
        #    test = test.cpu().numpy() * (2*np.random.randint(2)-1)
        #    #print (test.shape)
        #    rvs.append(test)
        #rvs = np.array(rvs)

        N_forw = min(N, self.batch_size)
        inp_ext = inp.repeat(N_forw,1,1,1)
        x = self.model.features(inp_ext)
        x = self.model.avgpool(x)
        feature = self.fc1(x.view(x.size(0), -1))
        #x = self.model.conv1(inp_ext)
        #x = self.model.bn1(x)
        #x = self.model.relu(x)
        #x = self.model.maxpool(x)
        #x = self.model.layer1(x)
        #x = self.model.layer2(x)
        #x = self.model.layer3(x)
        ##x = self.model.layer4(x)
        #feature = x.view(x.size(0), -1)
        #print (feature.shape)
        rvs = []
        st = 0
        while st < N:
            ed = min(st+self.batch_size, N)
            #indices = np.random.choice(4096, ed-st)
            #chosen_f = feature[np.arange(ed-st), indices]
            #gs = torch.autograd.grad(chosen_f.sum(), inp_ext, retain_graph=True)[0][:ed-st]
            #gs_processed = gs.cpu().numpy() * (2*np.random.randint(2)-1)
            ws = torch.FloatTensor(np.random.normal(size=(ed-st, feature.size()[1])))
            if self.gpu:
                ws = ws.cuda()
            tgt = (feature[:ed-st] * ws).sum()
            gs = torch.autograd.grad(tgt, inp_ext, retain_graph=True)[0][:ed-st]
            gs_processed = gs.cpu().numpy()

            rvs.append(gs_processed)
            st = ed
        rvs = np.concatenate(rvs, axis=0)

        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            rev_transp = np.argsort(transp)
            rvs = rvs * std + mean
            rvs = rvs.transpose(0, *(rev_transp+1))

        return rvs
