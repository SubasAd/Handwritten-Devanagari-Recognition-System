
import cv2
import numpy as np


class line_segmentation:
    def __init__(self):
        pass
    def linewordSegementation(self,img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape

        if w > 1000:
            new_w = 1000
            ar = w / h
            new_h = int(new_w / ar)

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        def thresholding(image):
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            return thresh

        thresh_img = thresholding(img);

        # dilation
        kernel = np.ones((3, 85), np.uint8)
        dilated = cv2.dilate(thresh_img, kernel, iterations=1)

        (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
        for i in sorted_contours_lines:
            print(cv2.boundingRect(i))

        img2 = img.copy()
        lines = []
        for ctr in sorted_contours_lines:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (40, 100, 250), 2)
        # dilation
        kernel = np.ones((7, 8), np.uint8)
        dilated2 = cv2.dilate(thresh_img, kernel, iterations=1)
        cv2.imwrite("dilated.png", dilated)
        cv2.imwrite("dilated2.png", dilated2)

        img3 = img.copy()
        words_list = []

        for line in sorted_contours_lines:

            # roi of each line
            x, y, w, h = cv2.boundingRect(line)
            roi_line = dilated2[y:y + h, x:x + w]
            # draw contours on each word
            (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            sorted_contour_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])
            for i in sorted_contour_words:
                print(cv2.boundingRect(i)[0])
            for word in sorted_contour_words:

                if cv2.contourArea(word) < 205:
                    continue

                x2, y2, w2, h2 = cv2.boundingRect(word)
                words_list.append([x + x2, y + y2, x + x2 + w2, y + y2 + h2])
                cv2.rectangle(img3, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 255, 100), 2)

        cv2.imwrite("img.png", img3)


        ninth_word = words_list[5]

        for i in range(0, len(words_list)):

            roi_9 = img[words_list[i][1]:words_list[i][3], words_list[i][0]:words_list[i][2]]
            cv2.imwrite("i/" + str(i) + ".png", roi_9)
            print("roi_9",roi_9)

