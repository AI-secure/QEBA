import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

def plot_logs(path, style, label, ATTR_ID):
    with open(path) as inf:
        logs = json.load(inf)
    print (len(logs))
    for i, log in enumerate(logs):
        organized_data = zip(*log)
        Xs = [entry[0] for entry in log]
        ys = [entry[1+ATTR_ID] for entry in log]
        if i == 0:
            plt.plot(Xs, ys, style, label=label)
        else:
            plt.plot(Xs, ys, style)

def mean(l, logarithm=False):
    if logarithm:
        l_lg = [np.log(x) for x in l]
        return np.exp(sum(l_lg) / len(l_lg))
    else:
        return sum(l) / len(l)

if __name__ == '__main__':
    TASK = 'imagenet'
    ATTR_NAME = {0:'dist', 1:'cos-est_vs_gt', 2:'rho', 3:'cos-est_vs_dist', 4: 'cos-gt_vs_dist'}
    #N_img = 5
    #N_repeat = 2
    ATTR_IDs = [0]
    if TASK == 'imagenet':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_imagenet_naive_.log', 'b-', 'naive'),
            ('BAPP_result/attack_multi_imagenet_DCT_.log', 'g-', 'DCT'),
            #('BAPP_result/attack_multi_imagenet_PCA9408_.log', 'k-', 'PCA9408'),
            ('BAPP_result/attack_multi_imagenet_PCA9408basis_.log', 'r-', 'PCA9408basis'),
            #('BAPP_result/attack_multi_imagenet_NNGen_.log', 'r-', 'NNGen'),
        ]

    all_logs = []
    for info in PLOT_INFO:
        with open(info[0]) as inf:
            all_logs.append(json.load(inf))

