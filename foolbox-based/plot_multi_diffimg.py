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
    if TASK == 'imagenet':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_imagenet_naive_.log', 'k-', 'HSJA'),
            ('BAPP_result/attack_multi_imagenet_resize9408_.log', 'r-.', 'QEBA-S'),
            ('BAPP_result/attack_multi_imagenet_DCT9408_.log', 'r--', 'QEBA-F'),
            ('BAPP_result/attack_multi_imagenet_PCA9408basis_.log', 'r-', 'QEBA-I'),
            ('BAPP_result/attack_multi_imagenet_AE9408_.log', 'b-', 'QEBA-AE'),
            ('BAPP_result/attack_multi_imagenet_VAE9408_.log', 'b--', 'QEBA-VAE'),
            ('BAPP_result/attack_multi_imagenet_GAN128_.log', 'y-', 'QEBA-GAN'),

            #('BAPP_result/attack_multi_imagenet_DCT9408_.log', 'r--', 'DCT9408'),
            #('BAPP_result/attack_multi_imagenet_DCT9408_modeldisc.log', 'b--', 'modeldisc'),
            #('BAPP_result/attack_multi_imagenet_DCT9408_alldisc.log', 'g--', 'alldisc'),

            #('BAPP_result/attack_multi_imagenet_naive_.log', 'k-', 'naive'),
            #('BAPP_result/attack_multi_imagenet_DCT2352_.log', 'g--', 'DCT2352'),
            #('BAPP_result/attack_multi_imagenet_DCT4107_.log', 'b--', 'DCT4107'),
            #('BAPP_result/attack_multi_imagenet_DCT9408_.log', 'r--', 'DCT9408'),
            #('BAPP_result/attack_multi_imagenet_DCT16428_.log', 'y--', 'DCT16428'),
            ##('BAPP_result/attack_multi_imagenet_PCA2352basis_.log', 'g-', 'PCA2352basis'),
            #('BAPP_result/attack_multi_imagenet_PCA4107basis_.log', 'b-', 'PCA4107basis'),
            #('BAPP_result/attack_multi_imagenet_PCA9408basis_.log', 'r-', 'PCA9408basis'),
            #('BAPP_result/attack_multi_imagenet_PCA4107basismore_.log', 'y-', 'PCA4107basismore'),
            #('BAPP_result/attack_multi_imagenet_PCA9408basismore_.log', 'g-', 'PCA9408basis'),
        ]
    elif TASK == 'cifar':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_cifar_naive_.log', 'k-', 'naive'),
            #('BAPP_result/attack_multi_cifar_DCT300_.log', 'b--', 'DCT300'),
            #('BAPP_result/attack_multi_cifar_PCA300train_.log', 'b-', 'PCA300'),
            ('BAPP_result/attack_multi_cifar_resize768_.log', 'r-.', 'Resize768'),
            ('BAPP_result/attack_multi_cifar_DCT768_.log', 'r--', 'DCT768'),
            ('BAPP_result/attack_multi_cifar_PCA768train_.log', 'r-', 'PCA768train'),
            #('BAPP_result/attack_multi_cifar_DCT1200_.log', 'r--', 'DCT1200'),
            #('BAPP_result/attack_multi_cifar_PCA1200train_.log', 'r-', 'PCA1200'),

            #('BAPP_result/attack_multi_cifar_naive_.log', 'k-', 'naive'),
            #('BAPP_result/attack_multi_cifar_DCT192_.log', 'b--', 'DCT192'),
            #('BAPP_result/attack_multi_cifar_PCA192_.log', 'b-', 'PCA192'),
            #('BAPP_result/attack_multi_cifar_DCT768_.log', 'r--', 'DCT768'),
            #('BAPP_result/attack_multi_cifar_PCA768_.log', 'r-', 'PCA768'),
            #('BAPP_result/attack_multi_cifar_DCT1200_.log', 'g--', 'DCT1200'),
            #('BAPP_result/attack_multi_cifar_PCA1200_.log', 'g-', 'PCA1200'),
            #('BAPP_result/attack_multi_cifar_PCA768train_.log', 'y-', 'PCA768train'),
        ]
    elif TASK == 'celeba':
        PLOT_INFO = [
            ('BAPP_result/attack_multi_celeba_naive_.log', 'k-', 'HSJA'),
            ('BAPP_result/attack_multi_celeba_resize9408_.log', 'r-.', 'QEBA-S'),
            ('BAPP_result/attack_multi_celeba_DCT9408_.log', 'r--', 'QEBA-F'),
            ('BAPP_result/attack_multi_celeba_PCA9408basis_.log', 'r-', 'QEBA-I'),
            ('BAPP_result/attack_multi_celeba_AE9408_.log', 'b-', 'QEBA-AE'),
            ('BAPP_result/attack_multi_celeba_VAE9408_.log', 'b--', 'QEBA-VAE'),
        ]
    ATTR_NAME = {0:'dist', 1:'cos-est_vs_gt', 2:'rho', 3:'cos-est_vs_dist', 4: 'cos-gt_vs_dist'}

    all_logs = []
    for info in PLOT_INFO:
        with open(info[0]) as inf:
            all_logs.append(json.load(inf))

    ### Plot mean img
    fig = plt.figure(figsize=(3,3))
    for logs, info in zip(all_logs, PLOT_INFO):
        #print (logs[0])
        print (info[2])
        print (len(logs))
        avg_X = []
        avg_y = []
        for t in range(len(logs[0])):
            avg_X.append(mean([log[t][0] for log in logs]))
            avg_y.append(mean([log[t][1] for log in logs], logarithm=True))
            #avg_y.append(mean([log[t][1] for log in logs], logarithm=False))
        print (avg_X)
        print (avg_y)
        plt.plot(avg_X, avg_y, info[1], label=info[2])
    plt.legend()
    plt.yscale('log')
    if TASK == 'cifar':
        plt.xlim([0,5000])
    else:
        plt.xlim([0,20000])
        plt.xticks([0,5000,10000,15000,20000],['0','5K','10K','15K','20K'])
    plt.xlabel('# Queries')
    plt.ylabel('Mean Square Error')
    plt.savefig('multi_%s_mean.pdf'%TASK, bbox_inches='tight')
    assert 0

    ### Plot mean stat
    stat_id = 2
    fig = plt.figure(figsize=(3,3))
    for logs, info in zip(all_logs, PLOT_INFO):
        #print (logs[0])
        avg_X = []
        avg_y = []
        for t in range(len(logs[0])):
            avg_X.append(mean([log[t][0] for log in logs]))
            avg_y.append(mean([log[t][stat_id+1] for log in logs]))
            #avg_y.append(mean([log[t][1] for log in logs], logarithm=False))
        print (info[2], avg_X, avg_y)
        plt.plot(avg_X, avg_y, info[1], label=info[2])
    plt.legend()
    plt.yscale('log')
    if TASK == 'cifar':
        plt.xlim([0,5000])
    else:
        plt.xlim([0,20000])
        plt.xticks([0,5000,10000,15000,20000],['0','5K','10K','15K','20K'])
    plt.xlabel('# Queries')
    plt.ylabel('Mean Square Error')
    plt.savefig('multi_%s_%s.pdf'%(TASK,ATTR_NAME[stat_id]), bbox_inches='tight')
    assert 0

    ### Plot success rate
    if TASK == 'imagenet':
        thresh=1e-3
    else:
        thresh=1e-5
    fig = plt.figure(figsize=(3,3))
    for logs, info in zip(all_logs, PLOT_INFO):
        all_success = []
        for log in logs:
            qs = [entry[0] for entry in log]
            ds = [entry[1] for entry in log]
            success_nq = None
            for q, d in zip(qs, ds):
                if (d < thresh):
                    success_nq = q
                    break
            if success_nq is None:
                success_nq =499999
            all_success.append(success_nq)
        plt.plot(sorted(all_success)+[999999], list(np.arange(len(all_success)) / len(all_success))+[1.0], info[1], label=info[2])
    plt.legend()
    if TASK == 'cifar':
        plt.xlim([0,5000])
    else:
        plt.xlim([0,20000])
        plt.xticks([0,5000,10000,15000,20000],['0','5K','10K','15K','20K'])
    plt.ylim([0,1])
    plt.xlabel('# Queries')
    plt.ylabel('Success rate at %.0e'%thresh)
    plt.savefig('multi_%s_success.pdf'%TASK, bbox_inches='tight')
    assert 0

    ### success rate table
    #THRESH = [1e-2,1e-3,1e-4]
    #NQUERY = [5000,10000,20000]
    #print ('\\begin{table*}[h]')
    #print ('    \\centering')
    #print ('    \\begin{tabular}{|'+'c|'*(1+len(THRESH))+'}')
    #print ('\t\\hline')
    #print ('\t%s/%s/%s/%s & %.2f & %.3f & %.4f \\\\'%(tuple(info[2] for info in PLOT_INFO)+tuple(THRESH)))
    #print ('\t\\hline')
    #for nq in NQUERY:
    #    print ('\t%d & '%nq,end='')
    #    first1 = True
    #    for dist in THRESH:
    #        if first1:
    #            first1 = False
    #        else:
    #            print (' & ', end='')
    #        first2 = True
    #        for logs, info in zip(all_logs, PLOT_INFO):
    #            if first2:
    #                first2=False
    #            else:
    #                print (' / ', end='')
    #            all_success = []
    #            for log in logs:
    #                qs = [entry[0] for entry in log]
    #                ds = [entry[1] for entry in log]
    #                for q, d in zip(qs, ds):
    #                    if (q > nq):
    #                        all_success.append(0)
    #                        break
    #                    elif (d < dist):
    #                        all_success.append(1)
    #                        break
    #            val = mean(all_success)
    #            print ('%.2f'%val, end='')
    #    print (' \\\\')
    #print ('\t\\hline')
    #print ('    \\end{tabular}')
    #print ('    \\caption{%s}'%( TASK ))
    #print ('    \\label{tab:result-%s}'%TASK)
    #print ('\\end{table*}')
    #assert 0
    #for logs, info in zip(all_logs, PLOT_INFO):
    #    all_success = []
    #    for log in logs:
    #        qs = [entry[0] for entry in log]
    #        ds = [entry[1] for entry in log]
    #        for q, d in zip(qs, ds):
    #            if (q > nq):
    #                all_success.append(0)
    #                break
    #            elif (d < dist):
    #                all_success.append(1)
    #                break
    #    val = mean(all_success)
    #    print ("%s, dist %e, #q %d, success@(#q,dist) = %.2f"%(info[2], dist, nq, val))

    ### Plot some img
    #ATTR_IDs = [0,1,2]
    #N_img = 10
    #N_repeat = 1
    #img_st = 40
    #fig = plt.figure(figsize=(N_img*5,len(ATTR_IDs)*5))
    #for img_id in range(N_img):
    #    for n_attr, attr_id in enumerate(ATTR_IDs):
    #        #plt.subplot(N_img, len(ATTR_IDs), img_id*len(ATTR_IDs)+n_attr+1)
    #        plt.subplot(len(ATTR_IDs), N_img, n_attr*N_img+img_id+1)
    #        for log, info in zip(all_logs, PLOT_INFO):
    #            plt.plot((1000,1000),(1e-3,1e-3), info[1], label=info[2])
    #            for log_id in range((img_st+img_id)*N_repeat, (img_st+img_id+1)*N_repeat):
    #                Xs = [entry[0] for entry in log[log_id]]
    #                ys = [entry[1+attr_id] for entry in log[log_id]]
    #                plt.plot(Xs, ys, info[1])
    #        plt.title(ATTR_NAME[attr_id])
    #        if attr_id == 0:
    #            plt.yscale('log')
    #        plt.legend()
    #fig.savefig('multi_%s_%s-%s.pdf'%(TASK, img_st, img_st+N_img-1))
    #assert 0

    ### Calc success@(#q,dist)
    #dist = 1e-4
    #nq = 10000
    #for logs, info in zip(all_logs, PLOT_INFO):
    #    all_success = []
    #    for log in logs:
    #        qs = [entry[0] for entry in log]
    #        ds = [entry[1] for entry in log]
    #        for q, d in zip(qs, ds):
    #            if (q > nq):
    #                all_success.append(0)
    #                break
    #            elif (d < dist):
    #                all_success.append(1)
    #                break
    #    val = mean(all_success)
    #    print ("%s, dist %e, #q %d, success@(#q,dist) = %.4f"%(info[2], dist, nq, val))


    ### Calc #q@dist
    #dist=1e-5
    #for logs, info in zip(all_logs, PLOT_INFO):
    #    all_nq = []
    #    for log in logs:
    #        qs = [entry[0] for entry in log]
    #        ds = [entry[1] for entry in log]
    #        for q, d in zip(qs, ds):
    #            if (d < dist):
    #                break
    #        all_nq.append(q)
    #    val = mean(all_nq, logarithm=True)
    #    print ("%s, dist %e, #q@dist = %.1f"%(info[2], dist, val))

    ### Calc dist@#q
    #nq = 10000
    #for logs, info in zip(all_logs, PLOT_INFO):
    #    all_dist = []
    #    for log in logs:
    #        qs = [entry[0] for entry in log]
    #        ds = [entry[1] for entry in log]
    #        for q, d in zip(qs, ds):
    #            if (q > nq):
    #                break
    #        all_dist.append(d)
    #    val = mean(all_dist, logarithm=True)
    #    print ("%s, #q %d, dist@#q = %e"%(info[2], nq, val))
