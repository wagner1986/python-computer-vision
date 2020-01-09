import os
import unittest
from os.path import dirname

import cv2

from condor.src.detect_pattern import DetectPattern


class DetectPatternTestCase(unittest.TestCase):

    def test_detecta_bandeja_inexistente(self):
        path_project = dirname(dirname(os.getcwd()))
        image_name = "{}{}{}{}seg{}1.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_test = cv2.imread(image_name)
        self.assertIsNotNone(image_test)
        detect_image = DetectPattern()
        _,_, description = detect_image.detect_tray(image_test)
        self.assertFalse(description["tray"])

    def test_detecta_nenhuma_bandeja(self):
        path_project = dirname(dirname(os.getcwd()))
        image_name = "{}{}{}{}seg{}2.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_test = cv2.imread(image_name)
        self.assertIsNotNone(image_test)
        detect_image = DetectPattern()
        _,_, description = detect_image.detect_tray(image_test)
        self.assertTrue(description["tray"])

    def test_detecta_bandeja_na_caixa(self):
        path_project = dirname(dirname(os.getcwd()))
        image_name = "{}{}{}{}seg{}35.png".format(path_project, os.sep, "data", os.sep, os.sep)
        image_test = cv2.imread(image_name)
        self.assertIsNotNone(image_test)
        detect_image = DetectPattern()
        _, _, description = detect_image.detect_tray(image_test)
        self.assertTrue(description["tray"])

    def test_detecta_arma(self):
        self.assertTrue(False)

    def test_detecta_nenhuma_arma(self):
        self.assertFalse(True)

    def test_detecta_1_cartucho(self):
        self.assertEqual(1, 0)

    def test_detecta_6_cartuchos(self):
        self.assertEqual(6, 0)

    def test_detecta_nenhum_cartucho(self):
        self.assertEqual(0, 1)

    def test_detecta_bateria(self):
        self.assertTrue(False)

    def test_detecta_nenhuma_bateria(self):
        self.assertFalse(True)

if __name__ == '__main__':
    unittest.main()
