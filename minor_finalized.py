
import numpy as np

import cv2

import warnings
warnings.filterwarnings("ignore")


class Recog:

    def _scheck(self,img):
        print(img)
        img = img[0:len(img) // 2]
        array = np.array(img)

        max = -1
        dict = {}
        dict2 = {}
        for i in range(0, len(img)):
            dict[i] = []
            for j in range(0, len(img[i])):
                if img[i][j] < 100:
                    dict[i].append(j)
        max = -1
        for i in range(0, len(dict)):
            p1 = 0
            for j in range(0, len(img[i])):
                for k in range(i - 3, i + 4):
                    if k in dict and j in dict[k]:
                        p1 += 1

                dict2[i] = p1

        print(dict2.items())
        items = []
        for each in dict2.items():
            items.append(each)
        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:5]
        sum = 0
        for i in items:
            sum += i[0]
        return (sum / 5) / img.shape[0] + 0.05
    def Recognition(self,img):
        img  = cv2.imread("i/"+img)
        def borders(here_img, thresh, bthresh=0.092):
            shape = here_img.shape
            check= int(bthresh*shape[0])
            image = here_img[:]
            top, bottom = 0, shape[0] - 1
            bg = np.repeat(thresh, shape[1])
            count = 0
            for row in range(1, shape[0]):
                if  (np.equal(bg, image[row]).any()) == True:
                    count += 1
                else:
                    count = 0
                if count >= check:
                    top = row - check
                    break


            bg = np.repeat(thresh, shape[1])
            count = 0
            rows = np.arange(1, shape[0])
            #print(rows)
            for row in rows[::-1]:
                if  (np.equal(bg, image[row]).any()) == True:
                    count += 1
                else:
                    count = 0
                if count >= check:
                    bottom = row + count
                    break

            d1 = (top - 2) >= 0
            d2 = (bottom + 2) < shape[0]
            d = d1 and d2
            if(d):
                b = 2
            else:
                b = 0

            return (top, bottom, b)
        def preprocess(bgr_img):
            bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            ret, th_img = cv2.threshold(bgr_img, 80, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            return th_img
        prepimg = preprocess(img)
        def segmentation(bordered, thresh=255, min_seg=10, scheck=0.15):
            scheck = 0.20
            try:
                shape = bordered.shape
                print(shape)
                check = int(scheck * shape[0])
                print(check)
                image = bordered[:]
                image = image[check:].T
                shape = image.shape
                print(shape)

                bg = np.repeat(255 - thresh, shape[1])
                bg_keys = []
                for row in range(1, shape[0]):
                    print(row)
                    print(np.equal(bg,image[row]).all())
                    if  (np.equal(bg, image[row]).all()):
                        bg_keys.append(row)
                print("bg_keys",bg_keys)
                lenkeys = len(bg_keys)-1
                new_keys = [bg_keys[1], bg_keys[-1]]
                print(new_keys)
                for i in range(1, lenkeys):
                    if (bg_keys[i+1] - bg_keys[i]) > 5:
                        new_keys.append(bg_keys[i])
                        print(i)
                print("New Keys ",new_keys)
                new_keys = sorted(new_keys)
                print(new_keys)
                segmented_templates = []
                first = 0
                bounding_boxes = []
                for key in new_keys[1:]:
                    segment = bordered.T[first:key]
                    if segment.shape[0]>=min_seg and segment.shape[1]>=min_seg:
                        segmented_templates.append(segment.T)
                        bounding_boxes.append((first, key))
                    first = key

                last_segment = bordered.T[new_keys[-1]:]
                if last_segment.shape[0]>=min_seg and last_segment.shape[1]>=min_seg:
                    segmented_templates.append(last_segment.T)
                    bounding_boxes.append((new_keys[-1], new_keys[-1]+last_segment.shape[0]))


                return(segmented_templates, bounding_boxes)
            except:
                return [bordered, (0, bordered.shape[1])]
        segments=segmentation(prepimg, scheck = self._scheck(prepimg))

        def localize(orig_img, tb, lr, segments):
            d=5
            rimg = orig_img.copy()
            boxes = []
            for simg, bb in zip(segments[0], segments[1]):
                print("bb",bb)
                bb = np.array(bb)
                bb += lr[0]
                rimg[tb[0]-d:tb[0], bb[0]-d:bb[1]+d] = 0
                rimg[tb[1]:tb[1]+d, bb[0]-d:bb[1]+d] = 0
                # draw cols
                rimg[tb[0]-d:tb[1]+d, bb[0]-d:bb[0]+d] = 0
                rimg[tb[0]-d:tb[1]+d, bb[1]-d:bb[1]+d] = 0

                boxes.append((tb[0]-d, tb[1]+d, bb[0], bb[1]))
            rimg = img.copy()
            print(boxes)
            for box in boxes:
                t, b, l, r = box
                cv2.rectangle(rimg, (l, t), (r, b), (0, 0, 0), 2)
            return rimg, boxes
        from keras.models import model_from_json
        from keras.models import load_model

        def prediction(img):
            json_file = open('modelCNN (5).json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("modelCNN (6).h5")
            loaded_model.save('modelCnn.hdf5')
            loaded_model=load_model('modelCnn.hdf5')


            roi = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
            roi = np.array(roi)
            roi.reshape(1,1024)
            prediction = loaded_model.predict(roi.reshape(1,1024))
            characters = 'क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ,०,१,२,३,४,५,६,७,८,९'
            characters = characters.split(',')

            output = characters[np.argmax(prediction.reshape(46))]
            return output
        def classifier(segments):
            pred_lbl = ""
            acc = []
            try:
             for segment in segments:
                segment = cv2.resize(segment, (32, 32))
                cv2.dilate(segment,np.ones((3, 3), np.uint8))
                segment = cv2.erode(segment, (3, 3), 1)
                lbl = prediction(segment)

                pred_lbl+=lbl
            except:
                print("Error")
                print(segment)
            return pred_lbl
        return classifier(segments[0])


