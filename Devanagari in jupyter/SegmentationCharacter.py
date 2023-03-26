import numpy as np
import cv2
import math
import pandas as pd
import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")


class SegmentationCharacter:
    def containsOnlyShirorekha(self, positions, averages, stds):
        data = {'position': positions, 'average': averages, 'std_dev': stds}
        dataframe = pd.DataFrame(data)

        dataframe2 = dataframe[dataframe['std_dev'] < 3]

        dataframe2 = dataframe2[dataframe2['std_dev'] > 0]
        is_sequential = []
        for pos in list(dataframe2['position']):
            is_sequential.append(all(second - first < 3 for first, second in zip(pos, pos[1:])))
        dataframe2['is_sequential'] = is_sequential
        meanShirorekhaPosition = np.mean(dataframe2['average'])

        lower_bound = meanShirorekhaPosition - 6
        upper_bound = meanShirorekhaPosition + 6
        filtered_df = dataframe2[(dataframe2['is_sequential'] == True) & (dataframe2['std_dev'] < 4) & (
                dataframe2['average'] >= lower_bound) & (dataframe2['average'] <= upper_bound)]
        l1 = list(dataframe.index)
        l2 = list(filtered_df.index)
        return [x for x in l1 if x not in l2]

    def Segmentation(self, ximg, counter):

        ximg = cv2.bitwise_not(ximg)
        copyForSegmentationShow = ximg.copy()
        kernel = np.ones((5, 5), np.float32) / 30



        for i in range(0, len(copyForSegmentationShow)):
            for j in range(0, len(copyForSegmentationShow[1])):
                if copyForSegmentationShow[i][j] < 100:
                    copyForSegmentationShow[i][j] = 0
        for i in range(0, len(ximg)):
            for j in range(0, len(ximg[1])):
                if ximg[i][j] > 100:
                    ximg[i][j] = 255
                else:
                    ximg[i][j] = 0
        char = ""
        segmentingPosition = self.getVerticalProjectionProfile(ximg)
        segmentedimages = []
        for pos in range(0, len(segmentingPosition) - 1):
            imgx = copyForSegmentationShow[0:ximg.shape[0], segmentingPosition[pos]:segmentingPosition[pos + 1]]
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
        ximg = cv2.dilate(ximg, np.ones((2, 2), np.uint8))
        org = ximg.copy()
        kernel = np.ones((5, 5), np.float32) / 30
        ximg = cv2.filter2D(ximg, -1, kernel)

        segmentingPosition = self.getSegmentationPosition(ximg)
        return segmentingPosition

    def getSegmentationPosition(self, ximg):
        positions, std, average = self.getStdandAverageofVericallines(ximg)
        sum = self.getSumofVerticalLines(ximg)
        x = list(set(sum.values()))
        averageValues = [x if not math.isnan(x) else -1 for x in average.values()]
        sortedAverage = sorted(set(sorted(list(averageValues))[::-1]))[::-1]
        resultOfStd = {key: val for key, val in std.items() if val > 3}

        resultOfAverage = {key: val for key, val in average.items() if val < min(sortedAverage[:10])}
        commonKeys = sorted(set(resultOfStd.keys()).intersection(set(resultOfAverage.keys())).union(set(resultOfStd.keys())))
        discontinuities = self.find_discontinuities(commonKeys)
        discontinuities2 = self.find_discontinuities(self.containsOnlyShirorekha(positions, average, std))

        segmentingPosition = [0]
        for each in discontinuities2:
            segmentingPosition.append((each[0] + each[1]) // 2)
        segmentingPosition.append(ximg.shape[1])

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

        return positions, std, average

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
