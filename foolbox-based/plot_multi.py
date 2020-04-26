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

if __name__ == '__main__':
    ATTR_ID = 0
    TASK = 'imagenet'
    for ATTR_ID in range(5):
        ATTR_NAME = {0:'dist', 1:'cos-est_vs_gt', 2:'rho', 3:'cos-est_vs_dist', 4: 'cos-gt_vs_dist'}[ATTR_ID]

        fig = plt.figure()

        if TASK == 'imagenet':
            #plot_logs('BAPP_result/attack_multi_imagenet_NNGen_.log', 'r-', 'NNGen', ATTR_ID)
            plot_logs('BAPP_result/attack_multi_imagenet_DCT_.log', 'g-', 'DCT', ATTR_ID)
            plot_logs('BAPP_result/attack_multi_imagenet_naive_.log', 'b-', 'naive', ATTR_ID)
            plot_logs('BAPP_result/attack_multi_imagenet_PCA9408_.log', 'k-', 'PCA9408', ATTR_ID)
        elif TASK == 'cifar':
            plot_logs('BAPP_result/attack_multi_cifar_NNGen768_.log', 'm-', 'NNGen768', ATTR_ID)
            plot_logs('BAPP_result/attack_multi_cifar_NNGen50_.log', 'y-', 'NNGen50', ATTR_ID)
            plot_logs('BAPP_result/attack_multi_cifar_PCA768_.log', 'r-', 'PCA768', ATTR_ID)
            plot_logs('BAPP_result/attack_multi_cifar_PCA50_.log', 'k-', 'PCA50', ATTR_ID)
            plot_logs('BAPP_result/attack_multi_cifar_DCT_.log', 'g-', 'DCT', ATTR_ID)
            plot_logs('BAPP_result/attack_multi_cifar_naive_.log', 'b-', 'naive', ATTR_ID)
        else:
            raise NotImplementedError()


        if ATTR_ID == 0:
            plt.yscale('log')
        plt.legend()
        plt.title(ATTR_NAME)
        fig.savefig('multi_%s_%s.pdf'%(TASK, ATTR_NAME))
        #plt.xlim([0,5000])
        #fig.savefig('multi-5000_%s_%s.pdf'%(TASK, ATTR_NAME))
