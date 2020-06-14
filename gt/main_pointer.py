import sys
import os
import argparse
import time
import datetime
import cv2
import numpy as np

from gaze_tracker import GazeTracker
from calibration import calibrate
from screen import Screen

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

def main():

    url = "http://192.168.1.2:8080" # Your url might be different, check the app
#    camera = cv2.VideoCapture(url+"/video")

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    gaze_tracker = GazeTracker()
#    screen = Screen()
    screen = Screen(SCREEN_WIDTH, SCREEN_HEIGHT)

#    screen.mode = "calibration"
#    calibrate(camera, screen, gaze_tracker)

    screen.clean()
    screen.show()
    while True:

#        for i in range(4):
#            camera.grab()
#        time.sleep(3.5)
        _, frame = camera.read() 

#        print(frame.shape)

        start = time.time()

        gaze_tracker.update(frame)

        end = time.time()

        cv2.namedWindow("frame")
#        cv2.moveWindow("frame", 0,0)
        dec_frame = gaze_tracker.eye_tracker.decorate_frame()
        cv2.imshow('frame', dec_frame)

        gaze = gaze_tracker.get_gaze()
        
        print("GAZE: {}".format(gaze))
        print("DIRECTION: {}".format(gaze_tracker.get_direction()))
        if gaze:
            screen.update(gaze)
            screen.refresh()


        print("TIME: {:.3f} ms".format(end*1000 - start*1000))

        k = cv2.waitKey(1) & 0xff
        if k == 1048603 or k == 27: # esc to quit
            break
        if k == ord('c'): # c to calibrate
            screen.mode = "calibration"
            screen.draw_center()
            calibrate(camera, screen, gaze_tracker)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
