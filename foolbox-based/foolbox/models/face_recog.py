import numpy as np
from .base import Model
import base64
from struct import unpack
import os
from PIL import Image

TMP_DIR = 'face_tmp'
API_DIR = '/home/xiaoyu.ft/zface_algo'
API_COMMAND = 'LD_LIBRARY_PATH=%s/lib_cpu/:%s/lib_common/:$LD_LIBRARY_PATH %s/bin_cpu/ClientZFace 9112 1 %s/ %s/list.txt %s/bin_cpu/config.txt %s/out.txt > %s/log.txt'%(API_DIR,API_DIR,API_DIR,TMP_DIR,TMP_DIR,API_DIR,TMP_DIR,TMP_DIR)
#API_COMMAND = 'LD_LIBRARY_PATH=%s/lib_cpu/:%s/lib_common/:$LD_LIBRARY_PATH %s/bin_cpu/ClientZFace 9112 1 %s/ %s/list.txt %s/bin_cpu/config.txt %s/out.txt'%(API_DIR,API_DIR,API_DIR,TMP_DIR,TMP_DIR,API_DIR,TMP_DIR)
CMP_IMG = '../../raw_data/face_raw/live_b_2.jpg'

def cos_sim(x1, x2):
    cos = (x1*x2).sum() / np.sqrt( (x1**2).sum() * (x2**2).sum() )
    return cos

def parse_feature(path):
    features = []
    with open(path) as inf:
        for i, line in enumerate(inf):
            if (i >= 4):
                if ('feature=') in line:
                    line = line[line.index('feature='):]
                    ed = line.index(',')
                    val_str = line[8:ed]
                    decoded = base64.b64decode(val_str)
                    x = np.array(unpack('512f', decoded))
                    features.append(x)
                else:
                    features.append(None)
    return features

class FaceRecogModel(Model):
    def __init__(self, bounds, channel_axis=2):
        super(FaceRecogModel, self).__init__(bounds=bounds,
                                             channel_axis=channel_axis)
        for files in os.listdir(TMP_DIR):
            os.remove(TMP_DIR+'/'+files)
        with open(TMP_DIR+'/list.txt', 'w') as outf:
            outf.write('%s pano\n'%CMP_IMG)
        os.system(API_COMMAND)
        features = parse_feature('%s/out.txt'%TMP_DIR)
        assert len(features) == 1
        self.cmp_feature = features[0]


    def forward(self, inputs):
        assert len(inputs.shape) == 4

        for files in os.listdir(TMP_DIR):
            os.remove(TMP_DIR+'/'+files)

        with open(TMP_DIR+'/list.txt', 'w') as outf:
            for i in range(len(inputs)):
                outf.write('inp%d.png pano\n'%i)

        for i, inp in enumerate(inputs):
            image = Image.fromarray(inp.astype('uint8'), 'RGB')
            image.save(TMP_DIR+'/inp%d.png'%i)

        os.system(API_COMMAND)
        features = parse_feature('%s/out.txt'%TMP_DIR)
        scores = []
        for f in features:
            if f is not None:
                cos = cos_sim(f, self.cmp_feature)
                flag = int(cos < 0.5)
                scores.append((flag, 1-flag))
            else:
                scores.append((1, 0))
        return (np.array(scores))

    def num_classes(self):
        return 2

    # Dummy
    def gradient_one(self, *args, **kwargs):
        return 0.0
