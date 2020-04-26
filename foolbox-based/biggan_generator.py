import torch
import numpy as np
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample

class BigGANGenerator:
    def __init__(self, batch_size=32):
        self.model = BigGAN.from_pretrained('biggan-deep-256')
        self.batch_size = batch_size

    def generate_ps(self, inp, N):
        truncation = 0.1

        rvs = []
        st = 0
        while st < N:
            ed = min(st+self.batch_size, N)
            class_v = np.random.dirichlet([1]*1000, size=ed-st)
            noise_v = truncated_noise_sample(truncation=truncation, batch_size=ed-st)
            class_v = torch.FloatTensor(class_v)
            noise_v = torch.FloatTensor(noise_v)
            with torch.no_grad():
                output = self.model(noise_v, class_v, truncation)
            output = (output.cpu().numpy() + 1) / 2
            output = output[:, :, 16:-16, 16:-16] #Using only the 224*224 in the middle
            rvs.append(output)
            st = ed
        rvs = np.concatenate(rvs, axis=0)

        return rvs

