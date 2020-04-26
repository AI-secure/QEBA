import numpy as np
from .base import Model
import base64
from struct import unpack
from PIL import Image
import os

TMP_DIR = 'face_tmp'
API_DIR = '/home/xiaoyu.ft/zface_algo'
API_COMMAND = 'LD_LIBRARY_PATH=%s/lib_cpu/:%s/lib_common/:$LD_LIBRARY_PATH %s/bin_cpu/ClientZFace 9112 1 %s/ %s/list.txt %s/bin_cpu/config.txt %s/out.txt > %s/log.txt'%(API_DIR,API_DIR,API_DIR,TMP_DIR,TMP_DIR,API_DIR,TMP_DIR,TMP_DIR)

def parse_liveness(path):
    scores = []
    with open(path) as inf:
        for i, line in enumerate(inf):
            if (i >= 4):
                line = line[line.index("\"livenessBeagle\":"):]
                ed = line.index('}')
                val_str = line[17:ed]
                x = float(val_str)
                scores.append((x,1-x))
    return scores

class FaceLiveModel(Model):
    def __init__(self, bounds, channel_axis=2):
        super(FaceLiveModel, self).__init__(bounds=bounds,
                                            channel_axis=channel_axis)

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
        scores = parse_liveness('%s/out.txt'%TMP_DIR)
        return (np.array(scores))

    def num_classes(self):
        return 2

    # Dummy
    def gradient_one(self, *args, **kwargs):
        return 0.0
