import line_word_segmentation
import os
import cv2
import copy_of_charactersegementation as rec

import matplotlib.pyplot as plt
import numpy as np
x = line_word_segmentation.line_segmentation()
img = cv2.imread("51.jpg")
x.linewordSegementation(img)
images = os.listdir("i/")
strings  = [[] for i in images]
string2 = [[] for i in images]

for i in range(0,len(images)):
    x = rec.Recog()
    char = x.Segmentation(cv2.imread("i/"+str(i)+".png",0),str(i))
    img  = cv2.imread("i/" + str(i) + ".png", 0)
    img = cv2.resize(img, (int(img.shape[1]/img.shape[0])*50,50), cv2.INTER_CUBIC)
    img = cv2.dilate(img,np.zeros((3,3),np.uint8))
    plt.imshow(img)
    plt.show()
    #char2 = cs.SegmentationAndRecognition(img,i)
    # os.remove("i/"+str(i)+".png")
    strings[i].append(char)
    #string2[i].append(char2)
print(strings)


