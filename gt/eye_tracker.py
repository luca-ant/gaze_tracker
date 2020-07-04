import os
import cv2
import math
import numpy as np
#import matplotlib.pyplot as plt
from model import Eye

class EyeTracker():
    def __init__(self):
        # initialize the opencv classifier for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(os.path.join('classifiers', 'haarcascade_frontalface_default.xml'))
#        self.face_cascade = cv2.CascadeClassifier(os.path.join('classifiers', 'haarcascade_frontalface_alt.xml'))
#        self.eye_cascade = cv2.CascadeClassifier(os.path.join('classifiers', 'haarcascade_eye.xml'))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join('classifiers', 'haarcascade_eye_tree_eyeglasses.xml'))

        self.frame = None
        self.frame_gray = None
        self.left_eye_frame = None
        self.right_eye_frame = None

        self.left_eye_detected = False
        self.right_eye_detected = False

        self.face_bb = None
        self.left_eye_bb = None
        self.right_eye_bb = None

        self.left_pupil_detected = False
        self.right_pupil_detected = False
        self.left_pupil = None
        self.right_pupil = None
        self.left_pupil_radius = None
        self.right_pupil_radius = None

        self.left_iris_detected = False
        self.right_iris_detected = False
        self.left_iris = None
        self.right_iris = None
        self.left_iris_radius = None
        self.right_iris_radius = None

        self.left_purkinje = None
        self.right_purkinje = None


    def update(self, frame):
        self.frame = frame
        self._analyze()


    def _analyze(self):
        self.frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        self._extract_face()
        self._extract_eyes()
        if self.left_eye_detected:
            self._extract_pupil("left")
            self._extract_iris("left")
            if self.left_iris_detected and self.left_pupil_detected:
                self._extract_purkinje("left")

        if self.right_eye_detected:
            self._extract_pupil("right")
            self._extract_iris("right")
            if self.right_iris_detected and self.right_pupil_detected:
                self._extract_purkinje("right")


    def left_eye(self):
        if self.left_eye_detected:
            return Eye(self.left_eye_frame.copy(), "left", self.left_pupil, self.left_pupil_radius, self.left_iris_radius, self.left_purkinje)
        return None

    def right_eye(self):
        if self.right_eye_detected:
            return Eye(self.right_eye_frame.copy(), "right", self.right_pupil, self.right_pupil_radius, self.right_iris_radius, self.right_purkinje)
        return None


    def decorate_frame(self):
        frame = self.frame.copy()

        # draw the face bounding box
        x, y, w, h = self.face_bb
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)

        if self.left_eye_bb:
##            eye_frame = frame[self.left_eye_bb[1]:self.left_eye_bb[1]+self.left_eye_bb[3], self.left_eye_bb[0]:self.left_eye_bb[0]+self.left_eye_bb[2]]
##            cv2.imwrite("images/eye_frame_start.png", eye_frame)

            if self.left_pupil and self.left_pupil_radius:
                # draw the left pupil
                x, y = self.left_pupil
                x += self.left_eye_bb[0]
                y += self.left_eye_bb[1]
                r = self.left_pupil_radius
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
#                cv2.circle(frame, (x, y), r, (0, 255, 0), 1)

##                eye_frame = frame[self.left_eye_bb[1]:self.left_eye_bb[1]+self.left_eye_bb[3], self.left_eye_bb[0]:self.left_eye_bb[0]+self.left_eye_bb[2]]
##                cv2.imwrite("images/pupil_detection_04_eye_frame.png", eye_frame)


            if self.left_iris and self.left_iris_radius:
                # draw the left iris
                x, y = self.left_iris
                x += self.left_eye_bb[0]
                y += self.left_eye_bb[1]
                r = self.left_iris_radius
#                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                cv2.circle(frame, (x, y), r, (0, 255, 0), 1)

##                eye_frame = frame[self.left_eye_bb[1]:self.left_eye_bb[1]+self.left_eye_bb[3], self.left_eye_bb[0]:self.left_eye_bb[0]+self.left_eye_bb[2]]
##                cv2.imwrite("images/iris_detection_03_eye_frame.png", eye_frame)

            if self.left_purkinje:
                # draw the left purkinje
                x, y = self.left_purkinje
                x += self.left_eye_bb[0]
                y += self.left_eye_bb[1]
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
##                eye_frame = frame[self.left_eye_bb[1]:self.left_eye_bb[1]+self.left_eye_bb[3], self.left_eye_bb[0]:self.left_eye_bb[0]+self.left_eye_bb[2]]
##                cv2.imwrite("images/purkinje_detection_03_eye_frame.png", eye_frame)

##            eye_frame = frame[self.left_eye_bb[1]:self.left_eye_bb[1]+self.left_eye_bb[3], self.left_eye_bb[0]:self.left_eye_bb[0]+self.left_eye_bb[2]]
##            cv2.imwrite("images/eye_frame_end.png", eye_frame)

            # draw the left eye bounding box
            x, y, w, h = self.left_eye_bb
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)


        if self.right_eye_bb:

            if self.right_pupil and self.right_pupil_radius:
                # draw the right pupil center
                x, y = self.right_pupil
                x += self.right_eye_bb[0]
                y += self.right_eye_bb[1]
                r = self.right_pupil_radius
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
#                cv2.circle(frame, (x, y), r, (0, 255, 0), 1)

            if self.right_iris and self.right_iris_radius:
                # draw the right iris
                x, y = self.right_iris
                x += self.right_eye_bb[0]
                y += self.right_eye_bb[1]
                r = self.right_iris_radius
#                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                cv2.circle(frame, (x, y), r, (0, 255, 0), 1)

            if self.right_purkinje:
                # draw the right purkinje
                x, y = self.right_purkinje
                x += self.right_eye_bb[0]
                y += self.right_eye_bb[1]
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # draw the right eye bounding box
            x, y, w, h = self.right_eye_bb
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)

        return frame

    def _extract_face(self):

        """
        Extract the box of the face ROI image as opencv format (x, y, w, h) from the current frame
        """
        frame_gray = cv2.GaussianBlur(self.frame_gray, (7, 7), 0)
#        frame_gray = cv2.medianBlur(frame_gray, 7)

#        faces = self.face_cascade.detectMultiScale(frame_gray) 
        faces = self.face_cascade.detectMultiScale(frame_gray, 1.3, 5) 

        # detect the best face on the image based on ROI size
        if len(faces) > 1:
            temp = (0, 0, 0, 0)
            for f in faces:
                if f[2]*f[3] > temp[2] * temp[3]:
                    temp = f
            best_face = (temp[0], temp[1], temp[2], temp[3])
        elif len(faces) == 1:
            face = faces[0]
            best_face = (face[0], face[1], face[2], face[3])
        else:
            # if no face is detected return all image as face ROI
            image_height = self.frame_gray.shape[0]
            image_width = self.frame_gray.shape[1]
            best_face = (0, 0, image_width, image_height)

        self.face_bb = best_face

    def _extract_eyes(self):

        """
        Extract the box of the eyes ROI image as opencv format (x, y, w, h) from the current frame
        """
        self.left_eye_detected = False
        self.right_eye_detected = False
        self.left_eye_bb = None
        self.right_eye_bb = None

        x, y, w, h = self.face_bb

        face_frame_gray = self.frame_gray[y:y+h, x:x+w] 
        face_frame_gray = cv2.GaussianBlur(face_frame_gray, (7, 7), 0)
#        face_frame_gray = cv2.medianBlur(face_frame_gray, 7)

#        eyes = self.eye_cascade.detectMultiScale(face_frame_gray) 
        eyes = self.eye_cascade.detectMultiScale(face_frame_gray, 1.3, 5) 

        for (ex, ey, ew, eh) in eyes:
            # do not consider false eyes detected at the bottom of the face
            if ey > 0.5 * h:
                continue

            remove_eyebrows = np.array([0, int(0.25*eh), int(0), int(-0.25*eh)])
            eye_center = ex + ew / 2
            if eye_center > w * 0.5:
                self.left_eye_detected = True
                left_bb = np.array([ex, ey, ew, eh])
                left_bb += remove_eyebrows
                self.left_eye_bb = (x + left_bb[0], y + left_bb[1], left_bb[2], left_bb[3])
                self.left_eye_frame = self.frame[self.left_eye_bb[1]:self.left_eye_bb[1]+self.left_eye_bb[3], self.left_eye_bb[0]:self.left_eye_bb[0]+self.left_eye_bb[2]]

            else:
                self.right_eye_detected = True
                right_bb = np.array([ex, ey, ew, eh])
                right_bb += remove_eyebrows
                self.right_eye_bb = (x + right_bb[0], y + right_bb[1], right_bb[2], right_bb[3])
                self.right_eye_frame = self.frame[self.right_eye_bb[1]:self.right_eye_bb[1]+self.right_eye_bb[3], self.right_eye_bb[0]:self.right_eye_bb[0]+self.right_eye_bb[2]]


    def _extract_pupil(self, position):

        """
        Extract from the eye frame the coordinates of the center of the pupil (x, y) 
        w.r.t the eye frame and the pupil radius in pixels 
        """

        self.left_pupil_detected = False
        self.right_pupil_detected = False
        pupil_center = None
        pupil_radius = None

        if position == "left":
            eye_frame_gray = self.frame_gray[self.left_eye_bb[1]:self.left_eye_bb[1]+self.left_eye_bb[3], self.left_eye_bb[0]:self.left_eye_bb[0]+self.left_eye_bb[2]]

        if position == "right":
            eye_frame_gray = self.frame_gray[self.right_eye_bb[1]:self.right_eye_bb[1]+self.right_eye_bb[3], self.right_eye_bb[0]:self.right_eye_bb[0]+self.right_eye_bb[2]]

##        if position == "left":
##            cv2.imwrite("images/pupil_detection_00_eye_frame_gray.png", eye_frame_gray)

#        eye_frame_gray = cv2.GaussianBlur(eye_frame_gray, (7, 7), 0)
#        eye_frame_gray = cv2.medianBlur(eye_frame_gray, 7)
        eye_frame_gray = cv2.equalizeHist(eye_frame_gray)

##        if position == "left":
##            cv2.imwrite("images/pupil_detection_01_equalized_hist.png", eye_frame_gray)


#        threshold = 2
        threshold = cv2.getTrackbarPos('threshold', 'frame')

        _, eye_frame_th = cv2.threshold(eye_frame_gray, threshold, 255, cv2.THRESH_BINARY)

##        if position == "left":
##            cv2.imwrite("images/pupil_detection_02_threshold.png", eye_frame_th)

        eye_frame_th = cv2.erode(eye_frame_th, None, iterations=2)
        eye_frame_th = cv2.dilate(eye_frame_th, None, iterations=4)

        eye_frame_th = cv2.medianBlur(eye_frame_th, 7)

##        if position == "left":
##            cv2.imwrite("images/pupil_detection_03_medianBlur.png", eye_frame_th)

        contours, _ = cv2.findContours(eye_frame_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x))

        for cnt in contours:

            cnt = cv2.convexHull(cnt)
            area = cv2.contourArea(cnt)
            if area == 0:
                continue
            circumference = cv2.arcLength(cnt, True)
            circularity = circumference ** 2 / (4*math.pi*area)

#            if circularity < 0.5 and circularity > 1.5:
#                continue

#            (x,y), radius = cv2.minEnclosingCircle(cnt)
#            pupil_center = (int(x),int(y))

            radius = circumference / (2 * math.pi)
            pupil_radius = int(radius)
            m = cv2.moments(cnt)
            if m['m00'] != 0:
                pupil_center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
                break


        if position == "left":
            if pupil_center != None and pupil_radius != None:
                self.left_pupil_detected = True
            self.left_pupil = pupil_center
            self.left_pupil_radius = pupil_radius

        if position == "right":
            if pupil_center != None and pupil_radius != None:
                self.right_pupil_detected = True
            self.right_pupil = pupil_center
            self.right_pupil_radius = pupil_radius
#            self.right_eye_frame = cv2.drawKeypoints(self.right_eye_frame, keypoints, self.right_eye_frame, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    def _extract_iris(self, position):
        """
        Extract from the eye frame the coordinates of the center of iris (x, y) 
        w.r.t the eye frame and the iris radius in pixels 
        """
        self.left_iris_detected = False
        self.right_iris_detected = False
        iris_center = None
        iris_radius = None

        if position == "left":
            eye_frame_gray = self.frame_gray[self.left_eye_bb[1]:self.left_eye_bb[1]+self.left_eye_bb[3], self.left_eye_bb[0]:self.left_eye_bb[0]+self.left_eye_bb[2]]

        if position == "right":
            eye_frame_gray = self.frame_gray[self.right_eye_bb[1]:self.right_eye_bb[1]+self.right_eye_bb[3], self.right_eye_bb[0]:self.right_eye_bb[0]+self.right_eye_bb[2]]
##        if position == "left":
##            cv2.imwrite("images/iris_detection_00_eye_frame_gray.png", eye_frame_gray)

        eye_frame_gray = cv2.equalizeHist(eye_frame_gray)
        eye_frame_gray = cv2.medianBlur(eye_frame_gray, 7)
        eye_frame_gray = cv2.GaussianBlur(eye_frame_gray, (11, 11), 0)

##        if position == "left":
##            cv2.imwrite("images/iris_detection_01_equalized_smoothing.png", eye_frame_gray)

        frame_height = np.size(eye_frame_gray, 0)
        frame_width = np.size(eye_frame_gray, 1)

        edged = cv2.Canny(eye_frame_gray, 100, 200) 

##        if position == "left":
##            cv2.imwrite("images/iris_detection_02_canny.png", edged)

        circles = cv2.HoughCircles(eye_frame_gray, cv2.HOUGH_GRADIENT, 1, int(frame_width),param1=100, param2=5, minRadius=5, maxRadius=int(frame_width/4)) 

#        print("Circles:", circles)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            c = circles[0][0]
            iris_center = (c[0], c[1])
            iris_radius = c[2]


        if position == "left":
            if iris_center != None and iris_radius != None:
                self.left_iris_detected = True
            self.left_iris = iris_center
            self.left_iris_radius = iris_radius

        if position == "right":
            if iris_center != None and iris_radius != None:
                self.right_iris_detected = True
            self.right_iris = iris_center
            self.right_iris_radius = iris_radius


    def _extract_purkinje(self, position):
        """
        Extract from the eye frame the coordinates of the purkinje image (x, y) 
        w.r.t the eye frame and the pupil radius in pixels 
        """

        purkinje = None

        if position == "left":
#            iris_center = self.left_iris
            pupil_center = self.left_pupil
            iris_radius = self.left_iris_radius
            eye_frame_gray = self.frame_gray[self.left_eye_bb[1]:self.left_eye_bb[1]+self.left_eye_bb[3], self.left_eye_bb[0]:self.left_eye_bb[0]+self.left_eye_bb[2]]

        if position == "right":
#            iris_center = self.right_iris
            pupil_center = self.right_pupil
            iris_radius = self.right_iris_radius
            eye_frame_gray = self.frame_gray[self.right_eye_bb[1]:self.right_eye_bb[1]+self.right_eye_bb[3], self.right_eye_bb[0]:self.right_eye_bb[0]+self.right_eye_bb[2]]

        x = pupil_center[0] - iris_radius
        y = pupil_center[1] - iris_radius
        w = iris_radius * 2
        h = iris_radius * 2
        iris_frame_gray = eye_frame_gray[y:y+h, x:x+w]
        if iris_frame_gray.shape[0] == 0 or iris_frame_gray.shape[1] == 0:
            return
            x = 0
            y = 0
            iris_frame_gray = eye_frame_gray
            if position == "left":
                self.left_purkinje = purkinje

            if position == "right":
                self.right_purkinje = purkinje

##        if position == "left":
##            cv2.imwrite("images/purkinje_detection_00_iris_frame_gray.png", iris_frame_gray)

        iris_frame_gray = cv2.equalizeHist(iris_frame_gray)
#        iris_frame_gray = cv2.medianBlur(iris_frame_gray, 7)
#        iris_frame_gray = cv2.GaussianBlur(iris_frame_gray, (7, 7), 0)

##        if position == "left":
##            cv2.imwrite("images/purkinje_detection_01_equalized.png", iris_frame_gray)


        # Iterative global thresholding
        th = 255
        count = 0
        founded = False
        while not founded and th > 127:
            _, th_global = cv2.threshold(iris_frame_gray, th, 255, cv2.THRESH_BINARY)
            count = np.count_nonzero(th_global)
            th-=1
            if count <= 0:
                continue

            contours, _ = cv2.findContours(th_global, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x))
            all_purkinjes = []
            for cnt in contours:
                cnt = cv2.convexHull(cnt)
                m = cv2.moments(cnt)
                if m['m00'] != 0:
                    purkinje = (x + int(m['m10'] / m['m00']), y + int(m['m01'] / m['m00']))

                    square_dist = pow(purkinje[0] - pupil_center[0], 2) + pow(purkinje[1] - pupil_center[1], 2)
                    all_purkinjes.append([purkinje, square_dist])
            if all_purkinjes:
                all_purkinjes.sort(key=lambda x: x[1])
                for p in all_purkinjes:
                    purkinje = p[0]
                    if pow(purkinje[0] - pupil_center[0], 2) + pow(purkinje[1] - pupil_center[1], 2) < pow(iris_radius, 2):
                        founded = True
                        break


##        if position == "left":
##            cv2.imwrite("images/purkinje_detection_02_iter_threshold.png", th_global)


        if position == "left":
            self.left_purkinje = purkinje

        if position == "right":
            self.right_purkinje = purkinje



