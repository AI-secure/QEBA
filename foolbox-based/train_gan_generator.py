import sys
import numpy as np
#import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from gan_generator import GANDiscriminator, GANGenerator
from dcgan_origin import DCGAN_D, DCGAN_G
import matplotlib.pyplot as plt

def epoch_train(modelD, modelG, optimizerD, optimizerG, N_Z, BATCH_SIZE=32, D_ITERS=1, G_ITERS=1, outf=None):
    N_train = 8750
    data_path = '/data/hcli/imagenet_avg/train_batch_%d.npy'
    #test: 625; train: 8750
    #/data/hcli/imagenet_avg/train_batch_%d.npy
    
    #N_used = 200
    N_used = N_train
    modelD.train()
    modelG.train()
    perm = np.random.permutation(N_used)

    tot_num = 0.0
    cum_Dreal = 0.0
    cum_Dfake = 0.0
    cum_G = 0.0
    with tqdm(perm) as pbar:
        for idx in pbar:
            # TODO: different input per iter
            X = np.load(data_path%idx)
            B = X.shape[0]
            assert B <= BATCH_SIZE
            X = np.reshape(X, (B,3,224,224))

            from skimage.transform import resize #origin
            #X = torch.FloatTensor([resize(Xi.transpose(1,2,0),(64,64)).transpose(2,0,1) for Xi in X]).cuda() #resize to 64*64#origin
            #X = np.array([resize(Xi.transpose(1,2,0),(64,64)).transpose(2,0,1) for Xi in X]) #resize to 64*64#origin

            # Train D
            errD_real = 0.0
            errD_fake = 0.0
            for _ in range(D_ITERS):
                for p in modelD.parameters():
                    p.data.clamp_(-0.01,0.01)
                optimizerD.zero_grad()

                # Loss with real
                l_real = modelD(X)

                # Loss with fake
                noise = torch.FloatTensor(B,N_Z).normal_(0,1)
                #noise = noise.resize_(B,N_Z,1,1).cuda()#origin
                fake = modelG(noise).detach()
                l_fake = modelD(fake)

                l = l_real - l_fake
                l.backward()
                optimizerD.step()
                errD_fake += l_fake.item()
                errD_real += l_real.item()
            errD_real = errD_real / D_ITERS
            errD_fake = errD_fake / D_ITERS

            # Train G
            errG = 0.0
            for _ in range(G_ITERS):
                optimizerG.zero_grad()
                noise = torch.FloatTensor(B,N_Z).normal_(0,1)
                #noise = noise.resize_(B,N_Z,1,1).cuda()#origin
                fake = modelG(noise)
                l_G = modelD(fake)
                l_G.backward()
                optimizerG.step()
                errG += l_G.item()
            errG = errG / D_ITERS
            
            # Log result and show
            cum_Dreal += errD_real * B
            cum_Dfake += errD_fake * B
            cum_G += errG * B
            tot_num += B
            pbar.set_description("Cur Dreal/Dfake/G err: %.4f/%.4f/%.4f; Avg: %.4f/%.4f/%.4f"%(errD_real, errD_fake, errG, cum_Dreal/tot_num, cum_Dfake/tot_num, cum_G/tot_num))
            if outf is not None:
                outf.write('%.6f %.6f %.6f\n'%(errD_real, errD_fake, errG))

    return cum_Dreal/tot_num, cum_Dfake/tot_num, cum_G/tot_num

if __name__ == '__main__':
    GPU = True
    N_Z = 128

    modelD = GANDiscriminator(n_channels=3, gpu=GPU)
    #modelD = DCGAN_D(isize=64, nz=N_Z, nc=3, ndf=64, ngpu=1)#origin
    #modelD.cuda()#origin
    #print (modelD)
    #inp = np.ones((8,3,224,224))
    #print (modelD(inp))
    #assert 0

    modelG = GANGenerator(n_z=N_Z, n_channels=3, gpu=GPU)
    #modelG = DCGAN_G(isize=64, nz=N_Z, nc=3, ngf=64, ngpu=1)#origin
    #modelG.cuda()#origin
    #print (modelG)
    #inp = np.ones((8, N_Z))
    #print (modelG.forward(inp).shape)
    #assert 0

    optimizerD = torch.optim.RMSprop(modelD.parameters(), lr=5e-5)
    optimizerG = torch.optim.RMSprop(modelG.parameters(), lr=5e-5)
    fixed_noise = torch.FloatTensor(10, N_Z).normal_()
    #fixed_noise = fixed_noise.resize_(10,N_Z,1,1).cuda()#origin

    with open('loss_curve.txt', 'w') as outf:
        for _ in range(100):
            fake = modelG(fixed_noise)
            fig = plt.figure(figsize=(20,8))
            for i in range(10):
                plt.subplot(2,5,i+1)
                to_plt = fake[i].detach().cpu().numpy().transpose(1,2,0)
                to_plt = (to_plt-to_plt.min()) / (to_plt.max()-to_plt.min())
                plt.imshow(to_plt)
            plt.savefig('gan_eg-%d.pdf'%_)
            plt.close(fig)

            print (epoch_train(modelD, modelG, optimizerD, optimizerG, N_Z, outf=outf))
            torch.save(modelD.state_dict(), 'gan%d_discriminator.model'%N_Z)
            torch.save(modelG.state_dict(), 'gan%d_generator.model'%N_Z)
    #modelD.load_state_dict(torch.load('gan%d_discriminator.model'%N_Z))
    #modelG.load_state_dict(torch.load('gan%d_generator.model'%N_Z))

    fake = modelG(fixed_noise)
    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        plt.subplot(2,5,i+1)
        to_plt = fake[i].detach().cpu().numpy().transpose(1,2,0)
        to_plt = (to_plt-to_plt.min()) / (to_plt.max()-to_plt.min())
        plt.imshow(to_plt)
    plt.savefig('gan_eg-final.pdf')
