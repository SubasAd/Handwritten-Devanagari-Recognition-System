import line_word_segmentation
import os
import cv2
import copy_of_charactersegementation as rec
import CharacterSegmentation as cs

x = line_word_segmentation.line_segmentation()
img = cv2.imread("6.jpg")
x.linewordSegementation(img)


images = os.listdir("i/")
strings  = [[] for i in images]
string2 = [[] for i in images]

for i in range(0,len(images)):
    x = rec.Recog()
    #char = x.Segmentation(cv2.imread("i/"+str(i)+".png",0))
    char2 = cs.SegmentationAndRecognition(cv2.imread("i/"+str(i)+".png",0))
    os.remove("i/"+str(i)+".png")
    #strings[i].append(char)
    string2[i].append(char2)
#print(strings)
print(string2)

