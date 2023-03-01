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

    def Segmentation(self, ximg, counter):
        keys2, keys, ximg = self.getVerticalProjectionProfile(ximg)
        copyForSegmentationShow = ximg.copy()
        char = ""
        segmentedimages = []
        for key in range(0, len(keys) - 1):
            imgx = ximg[0:ximg.shape[0], keys[key] - 2:(keys2[key + 1] + keys[key + 1]) // 2 + 4]
            cv2.rectangle(copyForSegmentationShow, (keys[key] - 2, 0),
                          ((keys2[key + 1] + keys[key + 1]) // 2 + 4, ximg.shape[0]), (255, 0, 0), 1)
            sum = 0
            for i in imgx:
                for j in imgx:
                    sum += j
            if (len(sum) < 20):
                continue
            recognizedcharacter = self.Recognition(imgx)
            if recognizedcharacter == '८':
                pass
            else:
                char += recognizedcharacter

            segmentedimages.append(imgx)
        cv2.imwrite("character segmentation/first" + str(counter) + ".png", copyForSegmentationShow)
        return char

    def getVerticalProjectionProfile(self, ximg):
        ximg = cv2.bitwise_not(ximg)
        ximg = cv2.dilate(ximg, np.ones((3, 3), np.uint8))
        org = ximg.copy()
        kernel = np.ones((5, 5), np.float32) / 30
        ximg = cv2.filter2D(ximg, -1, kernel)
        ximg = cv2.rotate(ximg,cv2.ROTATE_90_CLOCKWISE)
        std = {}
        positions  = {i:[] for i in range(0,ximg.shape[0])}
        plt.imshow(ximg)
        plt.show()
        for i in range(0,len(ximg)):
            for j in range(0,len(ximg[0])):
                if ximg[i][j] > 100:
                    positions[i].append(j)

        for rowkey in positions:
            std[rowkey] = np.std(positions[rowkey])
        print(std)
        dict = {}
        for i in range(0, len(ximg[0])):
            sum = 0
            for j in range(0, len(ximg)):
                if ximg[j][i] > 100:
                    sum += 1
            dict[i] = sum
        x = list(set(dict.values()))
        print(dict)
        greater_than_threshold = {k: v for k, v in dict.items() if v >= max(x[0:5]) * 1.25}
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
        keys2 = []
        for key in sorted(greater_than_threshold.keys(), reverse=True):
            if key == current_key - 1:
                current_key = key
                current_part[key] = greater_than_threshold[key]
            else:
                current_key = key
                keys2.append(key)
                current_part = {current_key: greater_than_threshold[key]}
                parts.append(current_part)
        keys2.append(0)
        ximg = org
        return keys2[::-1], keys, ximg
