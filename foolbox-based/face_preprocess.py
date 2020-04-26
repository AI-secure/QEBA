from PIL import Image
import numpy as np

if __name__ == '__main__':
    image = Image.open('../raw_data/face_raw/live_a_1.png')
    print (np.array(image).shape)
    image = image.resize((240, 320))
    image.save('../raw_data/face/live_a_1.png')

    image = Image.open('../raw_data/face_raw/live_a_2.png')
    print (np.array(image).shape)
    image = image.resize((240, 320))
    image.save('../raw_data/face/live_a_2.png')

    image = Image.open('../raw_data/face_raw/live_b_1.jpg')
    print (np.array(image).shape)
    image = image.crop((249,0,891,855))
    image = image.resize((240, 320))
    image.save('../raw_data/face/live_b_1.png')

    image = Image.open('../raw_data/face_raw/live_b_2.jpg')
    print (np.array(image).shape)
    image = image.crop((200,0,478,371))
    image = image.resize((240, 320))
    image.save('../raw_data/face/live_b_2.png')

    image = Image.open('../raw_data/face_raw/live_c_1.png')
    print (np.array(image).shape)
    image = image.resize((240, 320))
    image.save('../raw_data/face/live_c_1.png')

    image = Image.open('../raw_data/face_raw/live_c_2.png')
    print (np.array(image).shape)
    image = image.resize((240, 320))
    image.save('../raw_data/face/live_c_2.png')

    image = Image.open('../raw_data/face_raw/spoof_a_1.jpg')
    print (np.array(image).shape)
    image = image.crop((0,80,360,560))
    print (np.array(image).shape)
    image = image.resize((240, 320))
    image.save('../raw_data/face/spoof_a_1.png')
