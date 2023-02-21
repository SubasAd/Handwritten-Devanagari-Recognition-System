import line_word_segmentation
import os
import cv2
import copy_of_charactersegementation as rec

x = line_word_segmentation.line_segmentation()
x.linewordSegementation()

images = os.listdir("i/")
strings  = [[] for i in images]

for i in range(0,len(images)):
    x = rec.Recog()
    char = x.Segmentation(cv2.imread("i/"+str(i)+".png",0))
    os.remove("i/"+str(i)+".png")
    strings[i].append(char)
print(strings)

