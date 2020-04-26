import matplotlib.pyplot as plt
import json
import os

if __name__ == '__main__':
    ##methods = ['norm', 'vgg', 'unet', 'mix']
    #methods = ['norm', 'vgg', 'unet']
    #lt = {'norm':'--', 'vgg':':', 'unet':'-', 'mix':'-.'}
    #settings = ['20', '20inc', '100', '100inc']
    #lc = {'20':'b', '20inc':'r', '100':'g', '100inc':'k'}
    #fig = plt.figure()
    #for M in methods:
    #    for S in settings:
    #        if not os.path.exists('BAPP_result/attack_%s%s.log'%(M,S)):
    #            print ("%s%s not exists"%(M,S))
    #            continue
    #        with open('BAPP_result/attack_%s%s.log'%(M,S)) as inp:
    #            results = json.load(inp)
    #        Xs, ys = zip(*results)
    #        line_setting = lc[S] + lt[M]
    #        if max(Xs) > 10000:
    #            plt.plot(Xs, ys, line_setting)

    #Orthogonalize
    with open('BAPP_result/attack_norm100inc.log') as inp:
        results = json.load(inp)
    Xs, ys = zip(*results)
    print (len(Xs))
    plt.plot(Xs, ys, label='BAPP')

    with open('BAPP_result/attack_unet20inc.log') as inp:
        results = json.load(inp)
    Xs, ys = zip(*results)
    print (len(Xs))
    plt.plot(Xs, ys, label='BAPP + Encoder Grad')

    with open('BAPP_result/attack_unet20inc_ortho.log') as inp:
        results = json.load(inp)
    Xs, ys = zip(*results)
    print (len(Xs))
    plt.plot(Xs, ys, label='BAPP + Encoder Grad + Ortho')

    #with open('BAPP_result/attack_resize100inc.log') as inp:
    with open('BAPP_result/attack_resize20inc.log') as inp:
        results = json.load(inp)
    Xs, ys = zip(*results)
    print (len(Xs))
    plt.plot(Xs, ys, label='BAPP + Resize')

    #with open('BAPP_result/attack_resize100inc_ortho.log') as inp:
    #    results = json.load(inp)
    #Xs, ys = zip(*results)
    #print (len(Xs))
    #plt.plot(Xs, ys, label='BAPP + Resize + Ortho')

    #with open('BAPP_result/attack_unet20inc_resol8.log') as inp:
    #    results = json.load(inp)
    #Xs, ys = zip(*results)
    #print (len(Xs))
    #plt.plot(Xs, ys, label='BAPP + Dim Reduct + Resize')

    with open('BAPP_result/attack_dct20inc.log') as inp:
        results = json.load(inp)
    Xs, ys = zip(*results)
    print (len(Xs))
    plt.plot(Xs, ys, label='BAPP + DCT')
    #TO RUN: autozoom, bilinear
    plt.yscale('log')
    plt.xlim([0,35000])
    #plt.xlim([0,5000])
    plt.legend()
    plt.xlabel('# queries')
    plt.ylabel('l2-distance')
    plt.show()

    #Different level
    #fig = plt.figure()
    #with open('BAPP_result/attack_difflevel.log') as inp:
    #    results = json.load(inp)
    #Xs, ys = zip(*results)
    #plt.plot(Xs, ys)
    #for i in (70, 120, 170, 220, 270, 320):
    #    plt.axvline(Xs[i], linestyle='--')
    #plt.yscale('log')
    #plt.show()

    #Different level + ortho
    #fig = plt.figure()
    #lc = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    #for lvl in range(7):
    #    with open('BAPP_result/attack_unet20inc_lvl%d.log'%lvl) as inp:
    #        results = json.load(inp)
    #    Xs, ys = zip(*results)
    #    plt.plot(Xs, ys, lc[lvl], label='Attack at layer %d'%(6-lvl))

    #    with open('BAPP_result/attack_unet20inc_ortho_lvl%d.log'%lvl) as inp:
    #        results = json.load(inp)
    #    Xs, ys = zip(*results)
    #    #plt.plot(Xs, ys, '--', label='Attack at layer %d, orthogonalized'%(6-lvl))
    #    plt.plot(Xs, ys, lc[lvl]+'--')
    #plt.legend()
    #plt.yscale('log')
    #plt.show()

    #Receipt
    #fig = plt.figure()
    #with open('BAPP_result/attack_receipt_norm20inc.log') as inp:
    #    results = json.load(inp)
    #Xs, ys = zip(*results)
    #plt.plot(Xs, ys, label='Normal')
    #with open('BAPP_result/attack_receipt_unetmob20inc.log') as inp:
    #    results = json.load(inp)
    #Xs, ys = zip(*results)
    #plt.plot(Xs, ys, label='UNet(mobile)')
    #with open('BAPP_result/attack_receipt_unetmob20inc_ortho.log') as inp:
    #    results = json.load(inp)
    #Xs, ys = zip(*results)
    #plt.plot(Xs, ys, label='UNet(mobile) Ortho')
    #plt.legend()
    #plt.yscale('log')
    #plt.show()

    #meizhuang
    #fig = plt.figure()
    #with open('BAPP_result/attack_meizhuang_norm100.log') as inp:
    #    results = json.load(inp)
    #Xs, ys = zip(*results)
    #plt.plot(Xs, ys, label='BAPP')
    #with open('BAPP_result/attack_meizhuang_unet100.log') as inp:
    #    results = json.load(inp)
    #Xs, ys = zip(*results)
    #plt.plot(Xs, ys, label='BAPP + Dim reduct')
    #with open('BAPP_result/attack_meizhuang_unet100_ortho.log') as inp:
    #    results = json.load(inp)
    #Xs, ys = zip(*results)
    #plt.plot(Xs, ys, label='BAPP + Dim reduct + Ortho')
    #plt.legend()
    #plt.yscale('log')
    #plt.xlabel('# queries')
    #plt.ylabel('l2-distance')
    #plt.show()
