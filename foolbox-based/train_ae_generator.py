import numpy as np
#import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from ae_generator import AEGenerator

def calc_cos_sim(x1, x2, dim=1):
    cos = (x1*x2).sum(dim) / np.sqrt( (x1**2).sum(dim) * (x2**2).sum(dim) )
    return cos

def epoch_train(model, optimizer, BATCH_SIZE = 32):
    N_train = 8750
    data_path = '/data/hcli/imagenet_avg/train_batch_%d.npy'
    #test: 625; train: 8750
    #/data/hcli/imagenet_avg/train_batch_%d.npy
    
    #N_used = 200
    N_used = N_train
    model.train()
    perm = np.random.permutation(N_used)
    tot_num = 0.0
    cum_loss = 0.0
    cum_cos = 0.0
    with tqdm(perm) as pbar:
        for idx in pbar:
            X = np.load(data_path%idx)
            #X = X / np.sqrt((X**2).sum(1, keepdims=True))
            B = X.shape[0]
            assert B <= BATCH_SIZE
            X = np.reshape(X, (B,3,224,224))
            X_enc, X_dec = model(X)
            l = model.loss(X_dec, X)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            cos_sim = calc_cos_sim(X.reshape(B,-1), X_dec.cpu().detach().numpy().reshape(B,-1), dim=1)
            cum_loss += l.item() * B
            cum_cos += cos_sim.sum()
            tot_num += B
            pbar.set_description("Cur loss = %.6f; Avg loss = %.6f; Avg cos = %.4f"%(l.item(), cum_loss/tot_num, cum_cos/tot_num))

    return cum_loss/tot_num, cum_cos/tot_num

def epoch_eval(model, BATCH_SIZE=32):
    N_test = 625
    data_path = '/data/hcli/imagenet_avg/test_batch_%d.npy'

    #N_used = 50
    N_used = N_test
    model.eval()
    perm = np.random.permutation(N_used)
    tot_num = 0.0
    cum_loss = 0.0
    cum_cos = 0.0
    with tqdm(perm) as pbar:
        for idx in pbar:
            X = np.load(data_path%idx)
            #X = X / np.sqrt((X**2).sum(1, keepdims=True))
            B = X.shape[0]
            assert B <= BATCH_SIZE
            X = np.reshape(X, (B,3,224,224))
            with torch.no_grad():
                X_enc, X_dec = model(X)
                l = model.loss(X_dec, X)

            cos_sim = calc_cos_sim(X.reshape(B,-1), X_dec.cpu().detach().numpy().reshape(B,-1), dim=1)
            cum_loss += l.item() * B
            cum_cos += cos_sim.sum()
            tot_num += B
            pbar.set_description("Avg loss = %.6f; Avg cos = %.4f"%(cum_loss/tot_num, cum_cos/tot_num))

    return cum_loss/tot_num, cum_cos/tot_num


if __name__ == '__main__':
    GPU = True

    model = AEGenerator(n_channels=3, gpu=GPU)
    #inp = np.ones((8,3,224,224))
    #X_enc, X_dec = model(inp)
    #print (X_enc.shape)
    #print (X_dec.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(10):
        print (epoch_train(model, optimizer))
        print (epoch_eval(model))
        torch.save(model.state_dict(), 'ae_generator.model')
