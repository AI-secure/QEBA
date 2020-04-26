import numpy as np
import os
import math
from tqdm import tqdm

if __name__ == '__main__':
    N_train = 48000
    N_test = 2000

    TASK = 'cifartrain'
    if TASK == 'imagenet' or TASK == 'coco':
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = 64
    #REFs = ('res18', 'dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet')
    REFs = ('dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet')
    #REFs = ('res18',)


    #path = '/data/hcli/%s_rndavg'%(TASK)
    #if not os.path.isdir(path):
    #    os.mkdir(path)
    #for i in tqdm(range(math.ceil(N_train/BATCH_SIZE))):
    #    idxes = np.random.choice(5, 3, replace=False)
    #    rset1 = [REFs[i] for i in idxes]
    #    rset2 = [REFs[i] for i in range(5) if i not in idxes]
    #    avg_data = 0.0
    #    for REF in rset1:
    #        avg_data = avg_data + np.load('/data/hcli/%s_%s/train_batch_%d.npy'%(TASK, REF, i))
    #    avg_data = avg_data / len(rset1)
    #    np.save(path+'/train_batch_%d.npy'%(2*i), avg_data)
    #    avg_data = 0.0
    #    for REF in rset2:
    #        avg_data = avg_data + np.load('/data/hcli/%s_%s/train_batch_%d.npy'%(TASK, REF, i))
    #    avg_data = avg_data / len(rset1)
    #    np.save(path+'/train_batch_%d.npy'%(2*i+1), avg_data)
    #for i in tqdm(range(math.ceil(N_test/BATCH_SIZE))):
    #    idxes = np.random.choice(5, 3, replace=False)
    #    rset1 = [REFs[i] for i in idxes]
    #    rset2 = [REFs[i] for i in range(5) if i not in idxes]
    #    avg_data = 0.0
    #    for REF in rset1:
    #        avg_data = avg_data + np.load('/data/hcli/%s_%s/test_batch_%d.npy'%(TASK, REF, i))
    #    avg_data = avg_data / len(rset1)
    #    np.save(path+'/test_batch_%d.npy'%(2*i), avg_data)
    #    avg_data = 0.0
    #    for REF in rset2:
    #        avg_data = avg_data + np.load('/data/hcli/%s_%s/test_batch_%d.npy'%(TASK, REF, i))
    #    avg_data = avg_data / len(rset1)
    #    np.save(path+'/test_batch_%d.npy'%(2*i+1), avg_data)

    #path = '/data/hcli/%s_normed_avg'%(TASK)
    #if not os.path.isdir(path):
    #    os.mkdir(path)
    #for i in tqdm(range(math.ceil(N_train/BATCH_SIZE))):
    #    avg_data = 0.0
    #    for REF in REFs:
    #        cur_data = np.load('/data/hcli/%s_%s/train_batch_%d.npy'%(TASK, REF, i))
    #        avg_data = avg_data + cur_data / np.sqrt((cur_data**2).sum(1, keepdims=True))
    #    avg_data = avg_data / len(REFs)
    #    np.save(path+'/train_batch_%d.npy'%i, avg_data)
    #for i in tqdm(range(math.ceil(N_test/BATCH_SIZE))):
    #    avg_data = 0.0
    #    for REF in REFs:
    #        avg_data = avg_data + np.load('/data/hcli/%s_%s/test_batch_%d.npy'%(TASK, REF, i))
    #    avg_data = avg_data / len(REFs)
    #    np.save(path+'/test_batch_%d.npy'%i, avg_data)

    path = '/data/hcli/%s_avg'%(TASK)
    if not os.path.isdir(path):
        os.mkdir(path)
    for i in tqdm(range(math.ceil(N_train/BATCH_SIZE))):
        avg_data = 0.0
        for REF in REFs:
            avg_data = avg_data + np.load('/data/hcli/%s_%s/train_batch_%d.npy'%(TASK, REF, i))
        avg_data = avg_data / len(REFs)
        np.save(path+'/train_batch_%d.npy'%i, avg_data)
    for i in tqdm(range(math.ceil(N_test/BATCH_SIZE))):
        avg_data = 0.0
        for REF in REFs:
            avg_data = avg_data + np.load('/data/hcli/%s_%s/test_batch_%d.npy'%(TASK, REF, i))
        avg_data = avg_data / len(REFs)
        np.save(path+'/test_batch_%d.npy'%i, avg_data)

    #path = '/data/hcli/%s_rnd'%(TASK)
    #if not os.path.isdir(path):
    #    os.mkdir(path)
    #for i in tqdm(range(math.ceil(N_train/BATCH_SIZE))):
    #    rand = np.random.randint(len(REFs))
    #    REF = REFs[rand]
    #    cur_data = np.load('/data/hcli/%s_%s/train_batch_%d.npy'%(TASK, REF, i))
    #    np.save(path+'/train_batch_%d.npy'%i, cur_data)
    #for i in tqdm(range(math.ceil(N_test/BATCH_SIZE))):
    #    rand = np.random.randint(len(REFs))
    #    REF = REFs[rand]
    #    cur_data = np.load('/data/hcli/%s_%s/test_batch_%d.npy'%(TASK, REF, i))
    #    np.save(path+'/test_batch_%d.npy'%i, cur_data)
