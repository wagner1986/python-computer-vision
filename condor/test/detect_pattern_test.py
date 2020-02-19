import os
import unittest
from os.path import dirname

import cv2

from condor.src.detect_pattern import DetectPattern
from condor.src.roi_util import RoiUtil


class DetectPatternTestCase(unittest.TestCase):

    def test_detecta_bandeja_inexistente(self):
        path_project = dirname(dirname(os.getcwd()))
        image_name = "{}{}{}{}seg{}1.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_test = cv2.imread(image_name)
        self.assertIsNotNone(image_test)
        detect_image = DetectPattern()
        _,_, description = detect_image.detect_tray(image_test)
        self.assertFalse(description["tray"])

    def test_detecta_bandeja(self):
        path_project = dirname(dirname(os.getcwd()))
        image_name = "{}{}{}{}seg{}2.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_test = cv2.imread(image_name)
        self.assertIsNotNone(image_test)
        detect_image = DetectPattern()
        contour_found, tray, description = detect_image.detect_tray(image_test)
        print(description)
        self.assertTrue(description["tray"])

    def test_detecta_bandeja_na_caixa(self):
        path_project = dirname(dirname(os.getcwd()))
        image_name = "{}{}{}{}seg{}16.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_test = cv2.imread(image_name)
        self.assertIsNotNone(image_test)
        detect_image = DetectPattern()
        _, _, description = detect_image.detect_tray(image_test)
        self.assertTrue(description["tray"])

    def test_detecta_arma(self):
        path_project = dirname(dirname(os.getcwd()))
        # IMAGE BASE
        image_base_name = "{}{}{}{}tray{}tray_2.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_base_test = cv2.imread(image_base_name)

        image_name = "{}{}{}{}seg{}23.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_test = cv2.imread(image_name)
        self.assertIsNotNone(image_test)
        detect_image = DetectPattern()
        contour_found, tray, description = detect_image.detect_tray(image_test)

        self.assertTrue(description["tray"])

        roi = RoiUtil()
        description = roi.analyse(image_base_test, tray)
        self.assertTrue(description["weapon"])
        print(description)

    def test_detecta_nenhuma_arma(self):
        path_project = dirname(dirname(os.getcwd()))
        # IMAGE BASE
        image_base_name = "{}{}{}{}tray{}tray_2.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_base_test = cv2.imread(image_base_name)

        image_name = "{}{}{}{}seg{}12.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_test = cv2.imread(image_name)
        self.assertIsNotNone(image_test)
        detect_image = DetectPattern()
        contour_found, tray, description = detect_image.detect_tray(image_test)

        self.assertTrue(description["tray"])

        roi = RoiUtil()
        description = roi.analyse(image_base_test, tray)
        self.assertFalse(description["weapon"])

    def test_detecta_1_cartucho(self):

        self.assertEqual(1, 0)

    def test_detecta_6_cartuchos(self):
        self.assertEqual(6, 0)

    def test_detecta_nenhum_cartucho(self):
        self.assertEqual(0, 1)

    def test_detecta_bateria(self):
        path_project = dirname(dirname(os.getcwd()))
        # IMAGE BASE
        image_base_name = "{}{}{}{}tray{}tray_2.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_base_test = cv2.imread(image_base_name)

        image_name = "{}{}{}{}seg{}10.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_test = cv2.imread(image_name)
        self.assertIsNotNone(image_test)
        detect_image = DetectPattern()
        contour_found, tray, description = detect_image.detect_tray(image_test)

        self.assertTrue(description["tray"])

        roi = RoiUtil()
        description = roi.analyse(image_base_test, tray)
        self.assertTrue(description["batery"])

    def test_detecta_nenhuma_bateria(self):
        path_project = dirname(dirname(os.getcwd()))
        # IMAGE BASE
        image_base_name = "{}{}{}{}tray{}tray_2.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_base_test = cv2.imread(image_base_name)

        image_name = "{}{}{}{}seg{}9.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_test = cv2.imread(image_name)
        self.assertIsNotNone(image_test)
        detect_image = DetectPattern()
        contour_found, tray, description = detect_image.detect_tray(image_test)

        self.assertTrue(description["tray"])

        roi = RoiUtil()
        description = roi.analyse(image_base_test, tray)
        self.assertFalse(description["batery"])

if __name__ == '__main__':
    unittest.main()
