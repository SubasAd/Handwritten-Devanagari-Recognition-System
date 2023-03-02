import numpy as np
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import math

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
        segmentingPosition = self.getVerticalProjectionProfile(ximg)



        ximg = cv2.bitwise_not(ximg)

        copyForSegmentationShow = ximg.copy()

        for i in range(0, len(ximg)):
            for j in range(0, len(ximg[1])):
                if ximg[i][j] > 50 :
                    ximg[i][j] = 255
                else:
                    ximg[i][j] = 0
        char = ""
        segmentedimages = []
        for pos in range(0, len(segmentingPosition) - 1):
            imgx = ximg[0:ximg.shape[0], segmentingPosition[pos]:segmentingPosition[pos + 1]]
            cv2.rectangle(copyForSegmentationShow, (segmentingPosition[pos], 0),
                          (segmentingPosition[pos + 1], ximg.shape[0]), (255, 0, 0), 1)
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

        segmentingPosition = self.getSegmentationPosition(ximg)
        return segmentingPosition

    def getSegmentationPosition(self, ximg):
        std, average = self.getStdandAverageofVericallines(ximg)
        sum = self.getSumofVerticalLines(ximg)
        x = list(set(sum.values()))
        averageValues = [x if not math.isnan(x) else -1 for x in average.values()]
        sortedAverage = sorted(set(sorted(list(averageValues))[::-1]))[::-1]
        resultOfStd = {key: val for key, val in std.items() if val > 3}
        print(resultOfStd)
        resultOfAverage = {key: val for key, val in average.items() if val < min(sortedAverage[:10])}
        commonKeys = sorted(
            set(resultOfStd.keys()).intersection(set(resultOfAverage.keys())).union(set(resultOfStd.keys())))
        discontinuities = self.find_discontinuities(commonKeys)
        segmentingPosition = [0]
        for each in discontinuities:
            segmentingPosition.append((each[0] + each[1]) // 2)
        segmentingPosition.append(ximg.shape[1])
        print(segmentingPosition)
        return segmentingPosition

    def getSumofVerticalLines(self, ximg):
        dict = {}
        for i in range(0, len(ximg[0])):
            sum = 0
            for j in range(0, len(ximg)):
                if ximg[j][i] > 100:
                    sum += 1
            dict[i] = sum
        return dict

    def getStdandAverageofVericallines(self, ximg):
        std = {}
        average = {}
        positions = self.GetVerticalPositions(ximg)
        for rowkey in positions:
            std[rowkey] = np.std(positions[rowkey])
            average[rowkey] = np.mean(positions[rowkey])
        return std, average

    def GetVerticalPositions(self, ximg):
        ximg = cv2.rotate(ximg, cv2.ROTATE_90_CLOCKWISE)
        positions = {i: [] for i in range(0, ximg.shape[0])}
        plt.imshow(ximg)
        plt.show()
        for i in range(0, len(ximg)):
            for j in range(0, len(ximg[0])):
                if ximg[i][j] > 100:
                    positions[i].append(j)
        ximg = cv2.rotate(ximg, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return positions

    def find_discontinuities(self, nums):
        start = end = None
        discontinuities = []

        for i in range(len(nums) - 1):
            if nums[i + 1] != nums[i] + 1:
                if start is None:
                    start = nums[i]
                end = nums[i + 1]
            else:
                if start is not None:
                    discontinuities.append((start, end))
                    start = end = None

        # If the loop ends with a discontinuity
        if start is not None:
            discontinuities.append((start, end))
        finaldiscontinuities = []
        for each in discontinuities:
            if each[1] - each[0] > 3:
                finaldiscontinuities.append(each)

        return finaldiscontinuities
