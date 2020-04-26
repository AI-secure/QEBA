import asyncio, io, glob, os, sys, time, uuid, requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
import base64


# pip3 install --upgrade azure-cognitiveservices-vision-face
# export COGNITIVE_SERVICE_KEY="23741675ab464882b686de2ab3c843bb"
KEY = os.environ['FACE_SUBSCRIPTION_KEY']
ENDPOINT = os.environ['FACE_ENDPOINT']


def query_azure(filepath1, filepath2):
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    fr1 = open(filepath1,'rb')
    # img1 = base64.b64encode(fr1.read())
    fr2 = open(filepath2, 'rb')
    # img2 = base64.b64encode(fr2.read())
    # print(fr1)
    # assert 0

    img1_faces = face_client.face.detect_with_stream(image=fr1, return_face_id=True, return_face_landmarks=False,
                                               return_face_attributes=None, recognition_model='recognition_01',
                                               return_recognition_model=False, detection_model='detection_01',
                                               custom_headers=None, raw=False, callback=None)
    img2_faces = face_client.face.detect_with_stream(image=fr2, return_face_id=True, return_face_landmarks=False,
                                                return_face_attributes=None, recognition_model='recognition_01',
                                                return_recognition_model=False, detection_model='detection_01',
                                                custom_headers=None, raw=False, callback=None)

    print(img1_faces)
    print(img2_faces)
    img1_faceid = img1_faces[0].face_id
    img2_faceid = img2_faces[0].face_id

    verify_pred = face_client.face.verify_face_to_face(img1_faceid, img2_faceid)
    print(verify_pred)
    print(verify_pred.is_identical)
    print(verify_pred.confidence)


if __name__ == '__main__':
    src_img_path = '../src.jpg'
    tgt_img_path = '../tgt.jpg'
    query_azure(filepath1=src_img_path, filepath2=tgt_img_path)