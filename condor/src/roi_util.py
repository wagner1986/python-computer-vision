import os
from os.path import dirname

import cv2
from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity

class RoiUtil:

    rois = {
        'thread': ((111, 41, 96, 33), None),
        'batery': ((88, 27, 126, 59), 'thread'),
        'charger': ((217, 121, 44, 40), None),
        'next_battery': ((229, 36, 27, 52), None),
        'key_chain': ((54, 98, 29, 37), None),
        'weapon': ((59, 137, 143, 53), None)
    }

    def extract_img_roi(self, im, roi):
        # Crop image
        imCrop = im[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        return imCrop

    def extract_img_roi_two_levels(self, im, roi_int, roi_ext):
        # Crop image
        im[int(roi_int[1]):int(roi_int[1] + roi_int[3]), int(roi_int[0]):int(roi_int[0] + roi_int[2])] = [50, 50, 50]
        imCrop = im[int(roi_ext[1]):int(roi_ext[1] + roi_ext[3]), int(roi_ext[0]):int(roi_ext[0] + roi_ext[2])]
        return imCrop

    def diff_images(self, original, duplicate):
        if original.shape == duplicate.shape:
            print("The images have same size and channels")
        difference = cv2.subtract(original, duplicate)
        b, g, r = cv2.split(difference)
        print(cv2.countNonZero(b), cv2.countNonZero(g), cv2.countNonZero(r))
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("The images are completely Equal")


    def diff_images2(self, original, duplicate):
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(duplicate, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(gray, gray2, full=True)
        diff = (diff * 255).astype("uint8")
        #print("SSIM: {} Piece found {}".format(score, score < 0.6))
        return score, diff


    def diff_images3(self, original, duplicate):
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(duplicate, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(gray, gray2, full=True)
        diff = (diff * 255).astype("uint8")
        #print("SSIM: {} Piece found {}".format(score, score < 0.6))
        return score, diff

    def analyse(self, img_original, img_final):
        description = {}
        for key in self.rois:
            #print(key)
            if self.rois[key][1] is None:
                base_img = self.extract_img_roi(img_original, self.rois[key][0])
                new_img = self.extract_img_roi(img_final, self.rois[key][0])
            else:
                key_int = self.rois[key][1]
                base_img = self.extract_img_roi_two_levels(img_original, self.rois[key_int][0], self.rois[key][0])
                new_img = self.extract_img_roi_two_levels(img_final, self.rois[key_int][0], self.rois[key][0])

            # diff_images(base_img,new_img)
            score, diff = self.diff_images3(base_img, new_img)
            found = score < 0.6

            description[key] = found
            if (self.rois[key][1] is not None) and found:
                key_int = self.rois[key][1]
                description[key_int] = False

            # Display cropped image
            #cv2.imshow('base_' + key, base_img)
            # Display cropped image
            #cv2.imshow(key, new_img)
            #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return description

if __name__ == '__main__':
    path_project = dirname(dirname(os.getcwd()))
    path_project = "{}{}{}{}".format(path_project, os.sep, "data", os.sep)

    file_base = "{}tray{}tray_2.png".format(path_project, os.sep)
    file_final = "{}tray{}tray_10.png".format(path_project, os.sep)

    roi=RoiUtil()
    img_base = cv2.imread(file_base)
    img_analise = cv2.imread(file_final)
    description = roi.analyse(img_base, img_analise)
    print(description)