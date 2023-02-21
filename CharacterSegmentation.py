import cv2
import numpy as np
import matplotlib.pyplot as plt
def Segmentation(ximg):
    ximg = cv2.bitwise_not(ximg)
    ximg = cv2.dilate(ximg, np.ones((3, 3), np.uint8))
    current_key, current_part, greater_than_threshold = getPositionofCharacterSegmentation(ximg)
    keys = getKeysforCharacterSegmentation(current_key, current_part, greater_than_threshold, ximg)
    char = ""
    segmentedimages = []
    for key in range(0, len(keys) - 1):
        imgx = ximg[0:ximg.shape[0], keys[key]:keys[key + 1]]
        down, matraPositions, sudden_increases = getDownmatrapositionsandShirorekhaposition(imgx)
        plt.imshow(down)
        plt.show()
        if isTopNonEmpty(matraPositions, sudden_increases):
            current_key2,current_part2,greater_than_threshold2 = getPositionofCharacterSegmentation(down)
            keys2 = getKeysforCharacterSegmentation(current_key2,current_part2,greater_than_threshold2)



def isTopNonEmpty(matraPositions, sudden_increases):
    sum = 0
    for i in range(0, sudden_increases[0]):
        sum += matraPositions[i]
    return sum > 10


def getDownmatrapositionsandShirorekhaposition(imgx):
    f270 = cv2.rotate(imgx, cv2.ROTATE_90_COUNTERCLOCKWISE)
    matraPositions = {}
    for i in range(0, len(f270[0])):
        sum = 0
        for j in range(0, len(f270)):
            if (f270[j][i] > 100):
                sum += 1
        matraPositions[i] = sum
    x = matraPositions.values()
    x = list(set(x))
    greater_than_threshold = {k: v for k, v in matraPositions.items() if v > max(x[0:5])}
    print(matraPositions)
    threshold = 9  # the amount by which the value must increase to be considered a sudden increase
    sudden_increases = []  # a matraPositionsionary to store the sudden increases in each input matraPositionsionary
    for key, value in matraPositions.items():
        if key == 0:  # skip the first key since there is no previous value to compare to
            continue
        prev_value = matraPositions[key - 1]
        if value > prev_value + threshold:
            sudden_increases.append(key)
    up = f270[0:len(f270) - 1, 0:sudden_increases[0] - 2]
    down = f270[0:len(f270), sudden_increases[0]:len(matraPositions.items()) - 1]
    up = cv2.erode(cv2.rotate(up, cv2.ROTATE_90_CLOCKWISE), np.ones((3, 3), np.uint8))
    down = cv2.erode(cv2.rotate(down, cv2.ROTATE_90_CLOCKWISE), np.ones((3, 3), np.uint8))
    return down, matraPositions, sudden_increases


def getKeysforCharacterSegmentation(current_key, current_part, greater_than_threshold, ximg):
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
    return keys


def getPositionofCharacterSegmentation(ximg):
    characterpositions = {}
    for i in range(0, len(ximg[0])):
        sum = 0
        for j in range(0, len(ximg)):
            if (ximg[j][i] > 100):
                sum += 1
        characterpositions[i] = sum
    x = characterpositions.values()
    x = list(set(x))
    greater_than_threshold = {k: v for k, v in characterpositions.items() if v > max(x[0:5])}
    current_key = min(greater_than_threshold.keys())
    current_part = {current_key: greater_than_threshold[current_key]}
    return current_key, current_part, greater_than_threshold
Segmentation(cv2.imread('5.png',0))
