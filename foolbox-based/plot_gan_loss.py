import numpy as np
errs_Dreal = []
errs_Dfake = []
errs_G = []
with open('loss_curve.txt') as inf:
    for line in inf:
        errD_real, errD_fake, errG = map(float, line.strip().split())
        errs_Dreal.append(errD_real)
        errs_Dfake.append(errD_fake)
        errs_G.append(errG)

N = 50

errs_Dreal = np.convolve(errs_Dreal, np.ones((N,))/N, mode='valid')
errs_Dfake = np.convolve(errs_Dfake, np.ones((N,))/N, mode='valid')
errs_G = np.convolve(errs_G, np.ones((N,))/N, mode='valid')

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(errs_Dreal, label='err-Dreal')
plt.plot(errs_Dfake, label='err-Dfake')
plt.plot([x+y for x,y in zip(errs_Dreal, errs_Dfake)], label='err-D')
plt.plot(errs_G, label='err-G')
plt.legend()
fig.savefig('gan-loss.pdf')
