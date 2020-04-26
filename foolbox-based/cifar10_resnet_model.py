import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CifarDNN(nn.Module):
    def __init__(self, model_type, pretrained=True, gpu=False):
        super(CifarDNN, self).__init__()
        self.gpu = gpu
        if model_type == 'dense121':
            self.model = models.densenet121(pretrained=True).eval()
        elif model_type == 'res18':
            self.model = models.resnet18(pretrained=True).eval()
        elif model_type == 'res50':
            self.model = models.resnet50(pretrained=True).eval()
        elif model_type == 'vgg16':
            self.model = models.vgg16(pretrained=True).eval()
        elif model_type == 'googlenet':
            self.model = models.googlenet(pretrained=True).eval()
        elif model_type == 'wideresnet':
            self.model = models.wide_resnet50_2(pretrained=True).eval()
        else:
            raise NotImplementedError()
        self.output = nn.Linear(1000, 10)
        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        x = F.interpolate(x, scale_factor=7)
        x = self.model(x)
        x = self.output(x)
        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


class CifarResNet(nn.Module):
    def __init__(self, pretrained=True, gpu=False):
        super(CifarResNet, self).__init__()
        self.pretrained = pretrained
        self.gpu = gpu

        self.resnet = models.resnet18(pretrained=pretrained)
        self.output = nn.Linear(1000, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        #x = F.interpolate(x, [224,224])
        #x = self.resnet(x)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)

        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


class CifarDenseNet(nn.Module):
    def __init__(self, pretrained=True, gpu=False):
        super(CifarDenseNet, self).__init__()
        self.pretrained = pretrained
        self.gpu = gpu

        self.densenet = models.densenet121(pretrained=pretrained)
        self.output = nn.Linear(1000, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        #x = F.interpolate(x, [224,224])
        #x = self.resnet(x)

        x = self.densenet.features(x)
        x = F.relu(x, inplace=True)
        x = x.view(x.size(0), -1)
        x = self.densenet.classifier(x)

        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


def epoch_train(model, optimizer, dataloader):
    model.train()

    cum_loss = 0.0
    cum_acc = 0.0
    tot_num = 0.0
    for X, y in dataloader:
        B = X.size()[0]
        if (B==1):
            continue
        pred = model(X)
        loss = model.loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cum_loss += loss.item() * B
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(y)).sum().item()
        tot_num = tot_num + B

    return cum_loss / tot_num, cum_acc / tot_num

def epoch_eval(model, dataloader):
    model.eval()

    cum_loss = 0.0
    cum_acc = 0.0
    tot_num = 0.0
    for X, y in dataloader:
        B = X.size()[0]
        with torch.no_grad():
            pred = model(X)
            loss = model.loss(pred, y)

        cum_loss += loss.item() * B
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(y)).sum().item()
        tot_num = tot_num + B

    return cum_loss / tot_num, cum_acc / tot_num
