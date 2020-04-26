import torch
import numpy as np
from unet_generator import UNet
from tiny_imagenet import TinyImagenet
import torchvision.transforms as transforms
import argparse
import math


def load_imagenet_img(path):
    from PIL import Image
    image = Image.open(path)
    tmp = np.array(image)
    image = image.resize((224,224))
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    image = image / 255
    image = image.transpose(2,0,1)
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--mobilenet', action='store_true')
    args = parser.parse_args()
    BATCH_SIZE=32

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    dataset = TinyImagenet(transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)

    model = UNet(n_channels=3, mobilenet=args.mobilenet)
    #print (model)
    print (sum(p.numel() for p in model.parameters() if p.requires_grad))
    #assert 0
    if args.use_gpu:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    model.train()

    #for i, Xs in enumerate(dataloader):
    #    for _ in range(1000):
    #        if args.use_gpu:
    #            Xs = Xs.cuda()
    #        enc_Xs = model(Xs)
    #        l = loss_fn(Xs, enc_Xs)
    #        print (_, l)
    #        optimizer.zero_grad()
    #        l.backward()
    #        optimizer.step()
    #    break
    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #plt.subplot(1,2,1)
    #plt.imshow(Xs[0].cpu().detach().numpy().transpose(1,2,0))
    #plt.subplot(1,2,2)
    #plt.imshow(enc_Xs[0].cpu().detach().numpy().transpose(1,2,0))
    #fig.savefig('enc.png')
    #assert 0

    from tqdm import tqdm
    for epoch in range(10):
        tot_num = 0.0
        cum_loss = 0.0
        with tqdm(enumerate(dataloader), total=math.ceil(len(dataset)/BATCH_SIZE)) as pbar:
            for i, Xs in pbar:
                if args.use_gpu:
                    Xs = Xs.cuda()
                enc_Xs = model(Xs)
                l = loss_fn(Xs, enc_Xs)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                tot_num += len(Xs)
                cum_loss = cum_loss + l.item() * len(Xs)
                pbar.set_description("Cur loss: %.6f; Avg loss: %.6f"%(l.item(), cum_loss / tot_num))
                #if i > 300:
                #    break
        if args.mobilenet:
            torch.save(model.state_dict(), 'unet_mobile.model')
        else:
            torch.save(model.state_dict(), 'unet.model')
    #model.load_state_dict(torch.load('unet.model'))

    model.eval()
    img = load_imagenet_img('../raw_data/imagenet_example/bad_joke_eel.png')
    if args.use_gpu:
        ret = model(torch.FloatTensor(img[None]).cuda())
    else:
        ret = model(torch.FloatTensor(img[None]))
    print (ret.shape)
    ret_img = ret[0].cpu().detach().numpy()
    print (ret_img.shape)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img.transpose(1,2,0))
    plt.subplot(1,2,2)
    plt.imshow(ret_img.transpose(1,2,0))
    fig.savefig('imgnet_enc.png')
