import warnings

import cv2
import numpy as np
from keras.models import load_model
from keras.models import model_from_json

from SegmentationCharacter import SegmentationCharacter

warnings.filterwarnings("ignore")


class Recog (SegmentationCharacter):
    def Recognition(self, img):
        def prediction(img):
            json_file = open('middle (1).json', 'r')

            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("middle (1).h5")
            loaded_model.save('middle.hdf5')
            loaded_model = load_model('middle.hdf5')
            roi = cv2.resize(img, (29, 29), interpolation=cv2.INTER_AREA)
            imgxwithpadding = np.zeros((32, 32), np.uint8)
            imgxwithpadding[1:30, 1:30] = roi
            roi = imgxwithpadding
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

            return pred_lbl

        return classifier(img)
