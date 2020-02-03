import os
from os.path import dirname

import cv2
import numpy as np
# Exibe imagem
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class DetectPattern:

    def __init__(self, can_show=True):
        self.can_show = can_show

    def get_contour_of_tray(self):
        tray_contour = [[[389, 34]], [[108, 40]],
                        [[113, 248]], [[386, 235]]]
        return np.asarray(tray_contour)

    def clean_noise(self, image):
        new_img = image.copy()
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        new_img = cv2.fastNlMeansDenoising(new_img, None, 20, 7, 21)
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))
        new_img = clahe.apply(new_img)
        # Threshold image
        ret, thresh = cv2.threshold(new_img, 100, 255, cv2.THRESH_BINARY_INV)
        # Efeito de Blur
        # new_img = cv.GaussianBlur(new_img, (25, 25), 0)
        return thresh

    def find_more_similar_contour(self, contours, template_contour):
        try:
            match_value = []
            area = cv2.contourArea(template_contour)
            threshold = 0.1 * area
            for (i, contour) in enumerate(contours):
                temp_area = cv2.contourArea(contour)
                if abs(temp_area - area) < threshold:
                    # Iterate through each contour in the target image and
                    # use cv.matchShapes to compare contour shapes
                    match = cv2.matchShapes(template_contour, contour, 1, 0.0)
                    match_value.append([contour, match])
                    # print(match)

            match_value = sorted(match_value, key=lambda kv: (kv[1]), reverse=False)
            return match_value[0][0]
        except:
            return None

    def save_img_contour(self, image, contour, output_file_name):
        (x, y, w, h) = cv2.boundingRect(contour)
        # Let's now crop each contour and save these images
        cropped_contour = image[y:y + h, x:x + w]
        cv2.imwrite(output_file_name, cropped_contour)

    def plot_image(self, image, title=''):
        fig = plt.figure()
        fig.suptitle(title, fontsize=20)
        plt.imshow(image, cmap=cm.gray)
        plt.axis("off")
        plt.show()

    def detect_contours(self, image):
        new_image = self.clean_noise(image)
        # Computa contorno apos imagem tratada e realiza calculo de aproximação de poligono
        im2, contours, hierarchy = cv2.findContours(new_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #print('Quantity contours: ', len(contours))
        return contours

    def crop_min_area_rect(self, img, rect):
        # rotate img
        angle = rect[2]
        rows, cols = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rot = cv2.warpAffine(img, M, (cols, rows))

        # rotate bounding box
        # rect0 = (rect[0], rect[1], angle)
        box = cv2.boxPoints(rect)

        pts = np.int0(cv2.transform(np.array([box]), M))[0]
        pts[pts < 0] = 0

        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1],
                   pts[1][0]:pts[2][0]]

        # box = np.int0(box)
        # cv2.drawContours(img_rot, [pts], 0, (255,0,255), 2)
        return img_crop


    def detect_tray(self, frame):
        contours = self.detect_contours(frame)
        contour_found = self.find_more_similar_contour(contours, self.get_contour_of_tray())
        img_crop = None
        description = {"tray": False}
        if contour_found is not None:
            rect = cv2.minAreaRect(contour_found)
            print(contour_found, rect)
            angle = rect[-1]
            print("angle "+str(angle))
            img_crop = self.crop_min_area_rect(frame, rect)
            if self.can_show:
                self.plot_image(img_crop, "Img W/ Crop")
            cv2.drawContours(frame, contour_found, -3, [0, 255, 0], 3)
            description["tray"] = True
        else:
            template = cv2.imread('../../data/bandeja.png')
            res = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            w, h, c = template.shape
            top_left = max_loc
            bottom_right = [top_left[0] + w, top_left[1] + h]
            # crop
            img_crop = frame[top_left[0]:bottom_right[0],
                       top_left[1]:bottom_right[1]]
            self.plot_image(img_crop, "Img W/ Crop")
            contour_found=None
            description["tray"] = True
            print('Bandeja não encontrada na imagem ',max_val)

        return contour_found, img_crop, description



if __name__ == '__main__':

    # path current of projet
    path_project = dirname(dirname(os.getcwd()))
    path_project = "{}{}{}{}".format(path_project, os.sep, "data", os.sep)
    print(path_project)

    for i in range(1, 18):


        detect_image = DetectPattern(can_show=True)
        name_file = "{}seg{}{}.png".format(path_project, os.sep, i)

        other_image = cv2.imread(name_file)

        contour_found, tray, description = detect_image.detect_tray(other_image)
        output_file_name = "{}tray{}tray_{}.png".format(path_project, os.sep, i)
        if tray is not None:
            #detect_image.plot_image(tray, output_file_name)
            cv2.imwrite(output_file_name, tray)
        print('\n')
