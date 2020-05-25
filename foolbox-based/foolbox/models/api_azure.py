import numpy as np
from .base import Model
import os
from io import BytesIO
from PIL import Image
import time


class AzureModel(Model):
    def __init__(self, bounds, src_img_path, channel_axis=2):
        super(AzureModel, self).__init__(bounds=bounds, channel_axis=channel_axis)

        from azure.cognitiveservices.vision.face import FaceClient
        from msrest.authentication import CognitiveServicesCredentials

        # todo: add your own key to query Azure API
        KEY = ""
        ENDPOINT = "https://fast.cognitiveservices.azure.com"

        self.face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

        # get face_id for target image for later comparison
        src_fr = open(src_img_path, 'rb')
        src_img_faces = self.face_client.face.detect_with_stream(image=src_fr, return_face_id=True,
                                                                      return_face_landmarks=False,
                                               return_face_attributes=None, recognition_model='recognition_01',
                                               return_recognition_model=False, detection_model='detection_01',
                                               custom_headers=None, raw=False, callback=None)
        self.src_img_id = src_img_faces[0].face_id

    def forward(self, inputs):
        assert len(inputs.shape) == 4
        preds = []
        for _i in range(inputs.shape[0]):
            x = inputs[_i, :]
            # qry_bytesio = BytesIO(x)
            image = Image.fromarray(x.astype('uint8'), 'RGB')
            qry_img_path = './azure_tmp.jpg'
            image.save(qry_img_path)
            qry_img_fr = open(qry_img_path, 'rb')

            try:
                qry_faces = self.face_client.face.detect_with_stream(image=qry_img_fr, return_face_id=True,
                                                                 return_face_landmarks=False,
                                                                 return_face_attributes=None,
                                                                 recognition_model='recognition_01',
                                                                 return_recognition_model=False,
                                                                 detection_model='detection_01',
                                                                 custom_headers=None, raw=False, callback=None)
                qry_img_id = qry_faces[0].face_id
                verify_pred = self.face_client.face.verify_face_to_face(self.src_img_id, qry_img_id)
                flag = int(verify_pred.is_identical)
                preds.append((1-flag, flag))
                # time.sleep(6)
            except Exception as e:
                # print(e)
                # assert 0
                preds.append((1, 0))
                # time.sleep(6)
        return np.array(preds)

    def num_classes(self):
        return 2

    # Dummy
    def gradient_one(self, *args, **kwargs):
        return 0.0