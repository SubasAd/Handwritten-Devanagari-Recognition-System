import cv2
import numpy as np
import matplotlib.pyplot as plt
import Recognition


def isTopNonEmpty(matraPositions, sudden_increases):
    sum = 0
    for i in range(0, sudden_increases[0]):
        sum += matraPositions[i]
    return sum > 10


def getDownmatrapositionsandShirorekhaposition(imgx):
    f270 = cv2.rotate(imgx, cv2.ROTATE_90_COUNTERCLOCKWISE)
    matraPositions = {}
    x = PixelValuesineachcolumn(f270, matraPositions)
    greater_than_threshold = {}
    if len(x) < 5:
        greater_than_threshold = {k:v for k,v in matraPositions.items() if v > x[0]}
    else:
        greater_than_threshold = {k: v for k, v in matraPositions.items() if v > max(x[0:5])}
    sudden_increases = getShirorekhaPosition(matraPositions, threshold=9)
    down, up = getUpandDownOfShirorekha(f270, matraPositions, sudden_increases)
    if up is not None :
        if len(up) <5 :
            up = []
    return up, down, matraPositions, sudden_increases


def getUpandDownOfShirorekha(f270, matraPositions, sudden_increases):
    if sudden_increases == []:
        return None,None
    up = f270[0:len(f270) - 1, 0:sudden_increases[0] - 2]
    down = f270[0:len(f270), sudden_increases[0]:len(matraPositions.items()) - 1]
    up = cv2.rotate(up, cv2.ROTATE_90_CLOCKWISE)
    down = cv2.rotate(down, cv2.ROTATE_90_CLOCKWISE)
    return down, up


def PixelValuesineachcolumn(f270, matraPositions):
    for i in range(0, len(f270[0])):
        sum = 0
        for j in range(0, len(f270)):
            if (f270[j][i] > 100):
                sum += 1
        matraPositions[i] = sum
    plt.imshow(f270)
    plt.show()
    x = matraPositions.values()
    x = list(set(x))
    return x


def getShirorekhaPosition(matraPositions, threshold):
    sudden_increases = []  # a matraPositionsionary to store the sudden increases in each input matraPositionsionary
    for key, value in matraPositions.items():
        if key == 0:  # skip the first key since there is no previous value to compare to
            continue
        prev_value = matraPositions[key - 1]
        if value > prev_value + threshold:
            sudden_increases.append(key)
    return sudden_increases


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
            if (ximg[j][i] > 200):
                sum += 1
        characterpositions[i] = sum
    x = characterpositions.values()
    x = list(set(x))
    plt.imshow(ximg)
    plt.show()
    greater_than_threshold = {k: v for k, v in characterpositions.items() if v > max(x[0:5])}
    current_key = min(greater_than_threshold.keys())
    current_part = {current_key: greater_than_threshold[current_key]}
    return current_key, current_part, greater_than_threshold


def SegmentationAndRecognition(ximg):
    originalcopy = ximg.copy()
    ximg = Preporocess(ximg)
    keys = getKeysforSegmentation(ximg)
    char = ""
    segmentedimages = []
    word = ""
    for key in range(0, len(keys) - 1):
        imgx = ximg[0:ximg.shape[0], keys[key]:keys[key + 1]]
        imgxchar = getLetterfromWholemage(imgx)
        word += imgxchar


    return word
def getLetterfromWholemage(imgx):
    up, down, matraPositions, sudden_increases = getDownmatrapositionsandShirorekhaposition(imgx)
    if up is None or down is None or sudden_increases == []:
        return ""
    down = cv2.dilate(down, np.zeros((5, 5), np.uint8))
    downSegments = downSegmentation(down)
    char = Recognition.Recog()
    imgxchar = getLetterwithmodifiers(char, downSegments, matraPositions, sudden_increases, up)
    return imgxchar


def getLetterwithmodifiers(char, downSegments, matraPositions, sudden_increases, up):
    imgxchar = ""
    if len(downSegments) > 1:
        mid1, mid2 = getMidCharactersWithMiddleModifiers(char, downSegments)
        if mid1 is 'ा':
            imgxchar = 'ि' + mid2
        elif mid2 is 'ा':
            modifier = ""
            if (isTopNonEmpty(matraPositions, sudden_increases)):
                modifier = char.Recognition(up, "up.json", "up.h5", "े , ै, ि, ँ")
            if modifier is 'े':
                imgxchar = mid1 + 'ो'
            if modifier is 'ै':
                imgxchar = mid1 + 'ौ'
            if modifier is 'ि':
                imgxchar = mid1 + 'ी'
            if modifier is 'ँ':
                imgxchar = mid1 + 'ाँ'
    if len(downSegments) == 1:
        modifier = ""

        if (isTopNonEmpty(matraPositions, sudden_increases)):
            modifier = char.Recognition(up, "up.json", "up.h5", "े , ै, ि, ँ")
        mid1 = char.Recognition(downSegments[0], "middle (1).json", "middle (1).h5",
                                'ा,क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ,०,१,२,३,४,५,६,७,८,९')
        imgxchar = mid1 + modifier
    return imgxchar


def getMidCharactersWithMiddleModifiers(char, downSegments):
    mid1 = char.Recognition(downSegments[0], "middle (1).json", "middle (1).h5",
                            'ा,क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ,०,१,२,३,४,५,६,७,८,९')
    mid2 = char.Recognition(downSegments[1], "middle (1).json", "middle (1).h5",
                            'ा,क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ,०,१,२,३,४,५,६,७,८,९')
    return mid1, mid2


def getKeysforSegmentation(ximg):
    current_key, current_part, greater_than_threshold = getPositionofCharacterSegmentation(ximg)
    keys = getKeysforCharacterSegmentation(current_key, current_part, greater_than_threshold, ximg)
    return keys


def Preporocess(ximg):
    ximg = Binarization(ximg)
    ximg = cv2.bitwise_not(ximg)
    ximg = cv2.dilate(ximg, np.ones((3, 3), np.uint8))
    return ximg


def Binarization(ximg):
    for i in range(0, len(ximg[0])):
        sum = 0
        for j in range(0, len(ximg)):
            print(ximg[j][i])
            if (ximg[j][i] > 200):
                ximg[j][i] = 255
            else:
                ximg[j][i] = 0
    return ximg


def downSegmentation(bordered):
    pixelPositions = {}
    PixelValuesineachcolumn(bordered,pixelPositions)
    onlyDika = []
    discontinuities = []
    for rowPixel in pixelPositions:
        if pixelPositions[rowPixel] < 10:
            onlyDika.append(rowPixel)
    print(onlyDika)
    for i in range(0,len(onlyDika)-1):
        if (onlyDika[i] - onlyDika[i+1]) < -1:
            discontinuities.append(onlyDika[i])
            discontinuities.append(onlyDika[i+1])
    #Logic for irregularities
    print("Discontinuities" ,discontinuities)
    if len(discontinuities) <= 2:
        return [bordered]
    else:
        plt.imshow(bordered[0:bordered.shape[0]-1, 0:discontinuities[1]])
        plt.show()
        plt.imshow(bordered[0:bordered.shape[0]-1,(discontinuities[2]+discontinuities[1])//2:discontinuities[3]])
        plt.show()
        return [bordered[0:bordered.shape[0]-1, 0:discontinuities[1]+(discontinuities[2]+discontinuities[1])//3], bordered[0:bordered.shape[0]-1,(discontinuities[2]+discontinuities[1])//2:discontinuities[3]]]
