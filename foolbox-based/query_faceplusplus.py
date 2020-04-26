# -*- coding: utf-8 -*-
# import urllib.request
# import urllib.error
# import time
#
# key = "Lc-XYlfuvVhLFc61ev9eQgHTWxgggChn"
# secret = "nz0C7jw360yeKY3r3IJKfdaWG1iOYAwC"
#
# # def compare(src_img, qry_img):
# http_url = 'https://api-us.faceplusplus.com/facepp/v3/compare'
#
#
# # filepath = r"本地图片的路径"
# filepath1 = r'../282_tigercat.jpg'
# filepath2 = r'../18_blackmagpie.jpg'
#
# boundary = '----------%s' % hex(int(time.time() * 1000))
# data = []
# data.append('--%s' % boundary)
# data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
# data.append(key)
# data.append('--%s' % boundary)
# data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
# data.append(secret)
# data.append('--%s' % boundary)
#
# fr = open(filepath1, 'rb')
# data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file1')
# data.append('Content-Type: %s\r\n' % 'application/octet-stream')
# data.append(fr.read())
# fr.close()
#
# fr = open(filepath2, 'rb')
# data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file2')
# data.append('Content-Type: %s\r\n' % 'application/octet-stream')
# data.append(fr.read())
# fr.close()
#
# # data.append('--%s' % boundary)
# # data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
# # data.append('1')
# # data.append('--%s' % boundary)
# # data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
# # data.append(
# #     "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus")
# data.append('--%s--\r\n' % boundary)
#
# for i, d in enumerate(data):
#     if isinstance(d, str):
#         data[i] = d.encode('utf-8')
#
# http_body = b'\r\n'.join(data)
#
# # build http request
# req = urllib.request.Request(url=http_url, data=http_body)
#
# # header
# req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
#
# # print(http_body)
# # print(req)
# # assert 0
#
# try:
#     # post data to server
#     resp = urllib.request.urlopen(req, timeout=5)
#     # get response
#     qrcont = resp.read()
#     # if you want to load as json, you should decode first,
#     # for example: json.loads(qrount.decode('utf-8'))
#     print(qrcont.decode('utf-8'))
# except urllib.error.HTTPError as e:
#     print(e.read().decode('utf-8'))


import requests
import base64

key = "Lc-XYlfuvVhLFc61ev9eQgHTWxgggChn"
secret = "nz0C7jw360yeKY3r3IJKfdaWG1iOYAwC"

def query_faceplusplus(filepath1, filepath2):
    http_url = 'https://api-us.faceplusplus.com/facepp/v3/compare'

    fr1 = open(filepath1,'rb')
    img1 = base64.b64encode(fr1.read())
    fr2 = open(filepath2, 'rb')
    img2 = base64.b64encode(fr2.read())
    payload = {
                'api_key': key,
                'api_secret': secret,
                'image_base64_1':img1,
                'image_base64_2':img2
                }
    fr1.close()
    fr2.close()
    try:
        res = requests.post(http_url, data=payload)
        return res
    except Exception as e:
        print('Error:')
        print(e)


if __name__ == '__main__':
    src_img_path = '../src.jpg'
    tgt_img_path = '../tgt.jpg'
    response = query_faceplusplus(src_img_path, tgt_img_path)
    print(response.text)
