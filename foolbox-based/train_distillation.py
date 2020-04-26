import numpy as np
from nn_generator import NNGenerator
from bapp_dataset import BAPPDataset
import torch
import torch.nn as nn
import torchvision.models as models
import math
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    BATCH_SIZE=64
    LMD = 0.1
    GPU = True

    trainset = BAPPDataset(train=True)
    testset = BAPPDataset(train=False)
    print (len(trainset), len(testset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)

    model = nn.Sequential(models.resnet18(pretrained=False).eval(), nn.Linear(1000, 1))
    loss_fn = nn.BCEWithLogitsLoss()
    if GPU:
        model.cuda()
        loss_fn.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    from tqdm import tqdm
    for epoch in range(30):
        tot_num = 0.0
        cum_loss = 0.0
        all_pred = []
        all_gt = []
        with tqdm(enumerate(trainloader), total=math.ceil(len(trainset)/BATCH_SIZE)) as pbar:
            for i, (Xs, ys) in pbar:
                if GPU:
                    Xs = Xs.cuda()
                    ys = ys.cuda()
                pred = model(Xs).squeeze(1)
                l = loss_fn(pred, ys)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                tot_num += len(Xs)
                cum_loss += l.item() * len(Xs)
                all_pred += list(pred.detach().cpu().numpy())
                all_gt += list(ys.cpu().numpy())
                try:
                    cur_auc = roc_auc_score(all_gt, all_pred)
                except:
                    cur_auc = 0.5
                pbar.set_description("Cur l: %.6f; Avg l: %.6f; All AUC: %.6f"%(l.item(), cum_loss / tot_num, cur_auc))
                #if (i > 10):
                #    break

        tot_num = 0.0
        cum_loss = 0.0
        all_pred = []
        all_gt = []
        with tqdm(enumerate(testloader), total=math.ceil(len(testset)/BATCH_SIZE)) as pbar:
            for i, (Xs, ys) in pbar:
                if GPU:
                    Xs = Xs.cuda()
                    ys = ys.cuda()
                with torch.no_grad():
                    pred = model(Xs).squeeze(1)
                    l = loss_fn(pred, ys)

                tot_num += len(Xs)
                cum_loss += l.item() * len(Xs)
                all_pred += list(pred.detach().cpu().numpy())
                all_gt += list(ys.cpu().numpy())
                try:
                    cur_auc = roc_auc_score(all_gt, all_pred)
                except:
                    cur_auc = 0.5
                pbar.set_description("Test l: %.6f; Test AUC: %.6f"%(cum_loss / tot_num, cur_auc))
                #if (i > 10):
                #    break

        torch.save(model.state_dict(), 'distillation.model')
