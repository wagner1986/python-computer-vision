import os
from datetime import datetime
from os.path import dirname

import cv2
import numpy as np


class UtilCV:
    min_frame_std = 0

    def __init__(self, destiny="", percent=50):
        self.percent = percent
        self.destiny = destiny

    def define_dimension(self, frame):
        height, width, depth = frame.shape
        img_scale = (self.percent * 10.) / width
        new_x, new_y = frame.shape[1] * img_scale, frame.shape[0] * img_scale
        return new_x, new_y

    def join_images(self, img1, img2):
        new_img = np.hstack((img1, img2))
        return new_img

    def normalize_blur(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (25, 25), 0)
        return gray

    def segment_movement_video(self, file_name=0):
        current_frame = 0
        num_frames_std = 0
        last_status = None
        if os.path.isfile(file_name) or file_name == 0:
            cap = cv2.VideoCapture(file_name)

            ret, last_frame = cap.read()
            print('isOpened ', cap.isOpened())
            if last_frame is None:
                print('last_frame ', last_frame)
                exit()

            new_x, new_y = self.define_dimension(last_frame)
            last_frame = cv2.resize(last_frame, (int(new_x), int(new_y)))
            last_frame_norm = self.normalize_blur(last_frame)

            frame_parados = 0
            status_motion = "STOP"
            print('video ', cap.isOpened())
            while cap.isOpened():
                ret, frame = cap.read()
                if frame is None:
                    break

                frame = cv2.resize(frame, (int(new_x), int(new_y)))
                newFrame = self.normalize_blur(frame)

                diff = cv2.absdiff(last_frame_norm, newFrame)

                media_de_cor = diff.mean()
                if media_de_cor < 0.8:
                    status_motion = "STOP"
                    current_frame = current_frame + 1
                else:
                    status_motion = "MOTION"
                    if current_frame > self.min_frame_std:
                        num_frames_std = num_frames_std + 1
                        cv2.imwrite('{}{}seg{}{}{}'.format(self.destiny, os.sep, os.sep, num_frames_std, '.png'),
                                    last_frame)
                        print("Grava Frame")
                    current_frame = 0

                if last_status != status_motion:
                    last_status = status_motion
                print("current_frame", current_frame, " status_motion ", status_motion)

                # atualiza frame
                last_frame_norm = newFrame.copy()
                last_frame = frame.copy()

                # escreve msg na tela
                cv2.putText(frame, "Room Status: {}; Number frames stopped: {}".format(status_motion, num_frames_std),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                motion_detection = self.join_images(newFrame, diff)
                cv2.imshow('original', frame)
                cv2.imshow('motion detection', motion_detection)

                if cv2.waitKey(33) >= 0:
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            print("video não existe")


if __name__ == '__main__':
    # path current of projet
    path_project = dirname(dirname(os.getcwd()))
    path_project = "{}{}{}{}".format(path_project, os.sep, "data", os.sep)
    print(path_project)
    util = UtilCV(destiny=path_project)
    util.segment_movement_video()
