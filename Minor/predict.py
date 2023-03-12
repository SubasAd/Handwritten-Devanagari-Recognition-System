from utils import add_masks, crf
from config import imshape, model_name, n_classes
from models import preprocess_input, dice
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from utils import VideoStream
import cv2
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import time
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


model = load_model(os.path.join('models', model_name +'1'+
                   '.model'), custom_objects={'dice': dice})

inimg = cv2.imread("Hello.jpg", 3)
h=inimg.shape[0]
w=inimg.shape[1]
inimg=cv2.resize(inimg,(256,256))
ret,inimg = cv2.threshold(inimg, 100, 255, cv2.THRESH_BINARY)
im = inimg.copy()
tmp = np.expand_dims(im, axis=0)
roi_pred = model.predict(tmp)
roi_max = np.argmax(roi_pred.squeeze(), axis=2)
roi_pred = to_categorical(roi_max)
roi_mask = crf(roi_pred.squeeze(), im)
roi_mask = np.array(roi_mask, dtype=np.uint8)
roi_mask = cv2.addWeighted(im, 1.0, roi_mask, 1.0, 0)
roi_mask[np.where((roi_mask==[255,255,0]).all(axis=2))]=[255,255,255]
roi_mask[np.where((roi_mask == [255, 0, 0]).all(axis=2))] = [0, 0, 0]
roi_mask[np.where((roi_mask == [0, 0, 255]).all(axis=2))] = [0, 0, 0]
roi_mask = cv2.fastNlMeansDenoising(roi_mask, None, 20, 7, 21)
roi_mask=cv2.resize(roi_mask,(w,h))
plt.imshow(roi_mask)
plt.show()

edged = cv2.Canny(roi_mask, 30, 200)
contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(roi_mask, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', roi_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
