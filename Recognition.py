import numpy as np
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings("ignore")
class Recog:
    def Recognition(self, img, json_file,weights,characters):
        def prediction(img,json,weights,characters):
            json_file = open(json, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(weights)
            loaded_model.save('temporary.hdf5')
            loaded_model = load_model('temporary.hdf5')
            roi = cv2.resize(img, (29, 29), interpolation=cv2.INTER_AREA)
            imgxwithpadding  = np.zeros((32,32),np.uint8)
            imgxwithpadding[1:30,1:30] = roi
            roi = imgxwithpadding
            plt.imshow(roi)
            plt.show()
            roi = np.array(roi)
            roi.reshape(1, 1024)
            prediction = loaded_model.predict(roi.reshape(1, 1024))
            characters = characters.split(',')
            output = characters[np.argmax(prediction.reshape(len(characters)))]
            return output,prediction.reshape(len(characters))

        def classifier(segment,json,weights,characters):
            pred_lbl = ""
            acc = []
            if segment == [] :
                return ["",0]
            if segment is not None :
                segment = cv2.resize(segment, (32, 32))
                segment = cv2.erode(segment, (3, 3), 1)
                lbl = prediction(segment,json,weights,characters)
                pred_lbl += lbl[0]
            else :
                pred_lbl+= ""
                lbl = ["", 0]
            return pred_lbl,lbl[1]

        return classifier(img,json_file,weights,characters)


