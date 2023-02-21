import numpy as np
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2

import warnings

warnings.filterwarnings("ignore")


class Recog:
    def Recognition(self, img):
        def prediction(img):
            json_file = open('middle.json', 'r')

            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("middle.h5")
            loaded_model.save('middle.hdf5')
            loaded_model = load_model('middle.hdf5')
            roi = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            roi = np.array(roi)
            roi.reshape(1, 1024)
            prediction = loaded_model.predict(roi.reshape(1, 1024))
            characters = 'ा,क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ,०,१,२,३,४,५,६,७,८,९'
            characters = characters.split(',')

            output = characters[np.argmax(prediction.reshape(47))]
            return output

        def classifier(segment):
            pred_lbl = ""
            acc = []

            segment = cv2.resize(segment, (32, 32))
            segment = cv2.erode(segment, (3, 3), 1)
            lbl = prediction(segment)
            pred_lbl += lbl
            plt.imshow(segment)
            plt.show()
            return pred_lbl

        return classifier(img)

    def Segmentation(self, ximg):
        ximg = cv2.bitwise_not(ximg)
        ximg = cv2.dilate(ximg, np.ones((3, 3), np.uint8))
        dict = {}
        for i in range(0, len(ximg[0])):
            sum = 0
            for j in range(0, len(ximg)):
                if ximg[j][i] > 100:
                    sum += 1
            dict[i] = sum
        x = dict.values()
        x = list(set(x))
        greater_than_threshold = {k: v for k, v in dict.items() if v > max(x[0:5])}
        current_key = min(greater_than_threshold.keys())
        current_part = {current_key: greater_than_threshold[current_key]}
        parts = [current_part]
        keys = []
        for key in sorted(greater_than_threshold.keys()):
            if key == current_key + 1:
                current_key = key
                current_part[key] = greater_than_threshold[key]
            else:
                current_key = key
                keys.append(key)
                current_part = {current_key: greater_than_threshold[key]}
                parts.append(current_part)
        keys.append(ximg.shape[1])
        char = ""
        segmentedimages = []
        for key in range(0, len(keys) - 1):
            imgx = ximg[0:ximg.shape[0], keys[key]:keys[key + 1]]
            sum = 0
            for i in imgx:
                for j in imgx:
                    sum += j
            if(len(sum) < 20):
                continue
            recognizedcharacter  = self.Recognition(imgx)
            if recognizedcharacter == '८':
                pass
            else:
                char += recognizedcharacter

            segmentedimages.append(imgx)
        return char
