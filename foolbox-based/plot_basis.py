import numpy as np
from dct_generator import DCTGenerator
from pca_generator import PCAGenerator

if __name__ == '__main__':
    from PIL import Image
    image = Image.open('../raw_data/imagenet/284932.JPEG').convert('RGB').resize((224,224))
    image = np.asarray(image).transpose(2,0,1) / 255
    print (image.shape)

    #p_gen = DCTGenerator(factor=8.0)
    p_gen = PCAGenerator(N_b=2352, approx=True, basis_only=True)
    p_gen.load('pca_gen_2352_imagenet_avg.npy')
    rvs = p_gen.generate_ps(image, 10, level=999)

    tmp = rvs.reshape(10, -1)
    print (tmp@tmp.transpose())
    assert 0

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(24,4))
    plt.subplot(2,6,1)
    plt.imshow(image.transpose(1,2,0))
    for i in range(10):
        plt.subplot(2,6,i+2)
        #rvs_plt = (rvs[i] - rvs[i].min()) / (rvs[i].max() - rvs[i].min())
        rvs_plt = (rvs[i] / np.abs(rvs[i]).max() + 1) / 2
        print (rvs_plt)
        plt.imshow(rvs_plt.transpose(1,2,0))
        #plt.imshow(image.transpose(1,2,0))
    fig.savefig('rvs.pdf')
