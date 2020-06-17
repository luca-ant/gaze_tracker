import cv2
import math
import time
import numpy as np
import sklearn
from threading import Thread

from eye_tracker import EyeTracker
from model import Eye
from calibration import Calibration

class GazeTracker():
    """
    GazeTracker implementation based on purkinje image
    Attributes:
        calibration: calibration parameters
    """
    def __init__(self):
        super().__init__()

        self.calibration = Calibration()
        self.eye_tracker = EyeTracker()
        self.vector = None
        self.left_eye = None
        self.right_eye = None

    def update(self, frame):
        """
        Given an image, update the tuple (x, y) with the coordinates of the vector between pupil center and purkinje image
        """
        self.eye_tracker.update(frame)
        self.left_eye = self.eye_tracker.left_eye()
        self.right_eye = self.eye_tracker.right_eye()
        self._calculate_vector()

    def get_looking_direction(self):
        return self.eye_tracker.get_looking_direction()

    def _calculate_vector(self):

        """
        Calulate the vector coordinates from eyes features
        """
        vector = None
        vector_left = None
        vector_right = None
#        print(self.left_eye)
#        print(self.right_eye)
        if self.left_eye and self.left_eye.purkinje and self.left_eye.pupil_center:
            vector_left = (self.left_eye.purkinje[0] - self.left_eye.pupil_center[0], self.left_eye.purkinje[1] - self.left_eye.pupil_center[1])
        if self.right_eye and self.right_eye.purkinje and self.right_eye.pupil_center:
            vector_right = (self.right_eye.purkinje[0] - self.right_eye.pupil_center[0], self.right_eye.purkinje[1] - self.right_eye.pupil_center[1])

        if vector_left:
            vector = vector_left
        elif vector_right:
            vector = vector_right
        elif vector_left and vector_right:
            vector = ((vector_left[0] + vector_right[0]) // 2, (vector_left[1] + vector_right[1]) // 2)
        #TODO CHECK
        self.vector = vector

    def get_vector(self):
        return self.vector

    def get_gaze(self):

        gaze = None
        if self.vector:
            try:
                gaze = self.calibration.compute(self.vector)
            except sklearn.exceptions.NotFittedError:
#            except:
                print("CALIBRATION REQUIRED!")

        return gaze 
