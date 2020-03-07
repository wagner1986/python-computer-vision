import os
from datetime import datetime
from os.path import dirname

import cv2
import numpy as np

from condor.src.detect_pattern import DetectPattern


class UtilCV:
    min_frame_std = 0

    def __init__(self, destiny="", percent=50, can_show=False, can_write=False):
        self.percent = percent
        self.destiny = destiny
        self.can_show = can_show
        self.can_write = can_write

    def define_dimension(self, frame):
        height, width, depth = frame.shape
        img_scale = (self.percent * 10.) / width
        new_x, new_y = frame.shape[1] * img_scale, frame.shape[0] * img_scale
        return new_x, new_y

    def load_config_detect_pattern(self):
        # path current of projet
        path_project = dirname(dirname(os.getcwd()))
        path_project = "{}{}{}{}".format(path_project, os.sep, "data", os.sep)
        print(path_project)
        model_h5_file = path_project + 'model.h5'
        model_json_file = path_project + 'model.json'
        print(model_h5_file, model_json_file)
        labels = ['adptador', 'bandeja', 'bateria', 'cabo', 'carregador', 'cartucho', 'coldre', 'pendrive', 'spark']
        detect_image = DetectPattern(model_h5_file=model_h5_file, model_json_file=model_json_file, can_show=False,
                                     labels=labels)
        return detect_image

    def segment_movement_video(self, file_name=0):
        current_frame = 0
        num_frames_std = 0
        last_status = None
        frames_stoped = []
        detect_image = self.load_config_detect_pattern()
        if os.path.isfile(file_name) or file_name in (0,1) :
            cap = cv2.VideoCapture(file_name)

            ret, last_frame = cap.read()
            #print('isOpened ', cap.isOpened())
            if last_frame is None:
                print('last_frame ', last_frame)
                exit()
            new_x, new_y = self.define_dimension(last_frame)

            status_motion = "STOP"
            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    break

                frame = cv2.resize(frame, (int(new_x), int(new_y)))
                result=detect_image.detect_deep_learning(frame)

                if self.can_show:
                    # escreve msg na tela
                    cv2.putText(frame, "Objetos visiveis {}".format(result),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 0), 1)
                    cv2.putText(frame, datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                    cv2.imshow('original', frame)

                if cv2.waitKey(33) >= 0:
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            print("video não existe")
        return frames_stoped


if __name__ == '__main__':
    # path current of projet

    path_project = dirname(dirname(os.getcwd()))
    path_project = "{}{}{}{}".format(path_project, os.sep, "data", os.sep)
    print(path_project)
    util = UtilCV(destiny=path_project, can_write=False,can_show=True)

    video1 = path_project + "kit3.mp4"

    print(video1)

    list1 = util.segment_movement_video(file_name = video1)
