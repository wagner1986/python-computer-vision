import os
import unittest
from os.path import dirname

from condor.src.cv_util import UtilCV


class UtilCVTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

    def test_detecta_movimento(self):
        path_project = dirname(dirname(os.getcwd()))
        path_project = "{}{}{}{}".format(path_project, os.sep, "data", os.sep)
        util = UtilCV(destiny=path_project,can_write=True)
        frame_list = util.segment_movement_video(file_name=path_project + "kit1.mp4")
        print(" movimentos detectados " + str(len(frame_list)))

        self.assertGreater(len(frame_list), 0, " lista igual a zero")

if __name__ == '__main__':
    unittest.main()
