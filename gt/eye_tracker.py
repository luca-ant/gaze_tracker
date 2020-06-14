import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from model import Eye

class EyeTracker():
    """
    EyeTracker implementation based on threshold using OpenCV
    Attributes:
        frame: the current frame in numpy format

    Methods:

    """
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
#        self.frame_gray = cv2.GaussianBlur(self.frame_gray, (7, 7), 0)
#        self.frame_gray = cv2.equalizeHist(self.frame_gray)

        self._extract_face()
        self._extract_eyes()
        if self.left_eye_detected:
#            self._extract_pupil("left")
            self._extract_iris("left")
            if self.left_iris_detected:
                self._extract_purkinje("left")

        if self.right_eye_detected:
#            self._extract_pupil("right")
            self._extract_iris("right")
            if self.right_iris_detected:
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
            # draw the left eye bounding box
            x, y, w, h = self.left_eye_bb
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)

#            if self.left_pupil and self.left_pupil_radius:
#                # draw the left pupil
#                x, y = self.left_pupil
#                x += self.left_eye_bb[0]
#                y += self.left_eye_bb[1]
##                r = self.left_pupil_radius
#                r = 2
#                cv2.circle(frame, (x, y), r, (255, 0, 255), -1)

            if self.left_iris and self.left_iris_radius:
                # draw the left iris
                x, y = self.left_iris
                x += self.left_eye_bb[0]
                y += self.left_eye_bb[1]
                r = self.left_iris_radius
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                cv2.circle(frame, (x, y), r, (0, 255, 0), 1)

            if self.left_purkinje:
                # draw the left purkinje
                x, y = self.left_purkinje
                x += self.left_eye_bb[0]
                y += self.left_eye_bb[1]
                r = 2
                cv2.circle(frame, (x, y), r, (255, 0, 0), -1)


        if self.right_eye_bb:
            # draw the right eye bounding box
            x, y, w, h = self.right_eye_bb
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)

#            if self.right_pupil and self.right_pupil_radius:
#                # draw the right pupil center
#                x, y = self.right_pupil
#                x += self.right_eye_bb[0]
#                y += self.right_eye_bb[1]
##                r = self.right_pupil_radius
#                r = 2
#                cv2.circle(frame, (x, y), r, (255, 0, 255), -1)

            if self.right_iris and self.right_iris_radius:
                # draw the right iris
                x, y = self.right_iris
                x += self.right_eye_bb[0]
                y += self.right_eye_bb[1]
                r = self.right_iris_radius
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                cv2.circle(frame, (x, y), r, (0, 255, 0), 1)

            if self.right_purkinje:
                # draw the right purkinje
                x, y = self.right_purkinje
                x += self.right_eye_bb[0]
                y += self.right_eye_bb[1]
                r = 2
                cv2.circle(frame, (x, y), r, (255, 0, 0), -1)

        return frame

    def _extract_face(self):

        """
        Extract the box of the face ROI image as opencv format (x, y, w, h) from the current frame
        """
        frame_gray = cv2.GaussianBlur(self.frame_gray, (7, 7), 0)
        faces = self.face_cascade.detectMultiScale(frame_gray, 1.3, 5) 
#        faces = self.face_cascade.detectMultiScale(frame_gray) 

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

        eyes = self.eye_cascade.detectMultiScale(face_frame_gray) 

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

#                height = self.left_eye_frame.shape[0]
#                width = self.left_eye_frame.shape[1]
#                scale_factor = 2
#
#                self.left_eye_frame = cv2.resize(self.left_eye_frame, dsize=(width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)

            else:
                self.right_eye_detected = True
                right_bb = np.array([ex, ey, ew, eh])
                right_bb += remove_eyebrows
                self.right_eye_bb = (x + right_bb[0], y + right_bb[1], right_bb[2], right_bb[3])
                self.right_eye_frame = self.frame[self.right_eye_bb[1]:self.right_eye_bb[1]+self.right_eye_bb[3], self.right_eye_bb[0]:self.right_eye_bb[0]+self.right_eye_bb[2]]
#                height = self.right_eye_frame.shape[0]
#                width = self.right_eye_frame.shape[1]
#                scale_factor = 2
#
#                self.right_eye_frame = cv2.resize(self.right_eye_frame, dsize=(width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)



    def _extract_pupil(self, position):

        """
        Extract from the eye frame the coordinates of the center of the pupil (x, y) 
        w.r.t the eye frame and the pupil radius in pixels 
        """

        pupil_center = None
        pupil_radius = None

        if position == "left":
#            eye_frame_gray = cv2.cvtColor(self.left_eye_frame, cv2.COLOR_BGR2GRAY) 
            eye_frame_gray = self.frame_gray[self.left_eye_bb[1]:self.left_eye_bb[1]+self.left_eye_bb[3], self.left_eye_bb[0]:self.left_eye_bb[0]+self.left_eye_bb[2]]

        if position == "right":
#            eye_frame_gray = cv2.cvtColor(self.right_eye_frame, cv2.COLOR_BGR2GRAY) 
            eye_frame_gray = self.frame_gray[self.right_eye_bb[1]:self.right_eye_bb[1]+self.right_eye_bb[3], self.right_eye_bb[0]:self.right_eye_bb[0]+self.right_eye_bb[2]]

        eye_frame_gray = cv2.GaussianBlur(eye_frame_gray, (7, 7), 0)

        # Global thresholding
        _, th_global = cv2.threshold(eye_frame_gray, 30, 255, cv2.THRESH_BINARY_INV)

        # Mean thresholding
        th_mean = cv2.adaptiveThreshold(eye_frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17,2)


        # Gausian thresholding
        th_gauss = cv2.adaptiveThreshold(eye_frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17,2)

        # Otsu's thresholding
        _, th_otsu = cv2.threshold(eye_frame_gray, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


        # plot all the images and their histograms
        ths = [th_global,
                th_mean,
                th_gauss,
                th_otsu
                ]
        titles = ['Global Thresholding', 
                'Mean Thresholding',
                'Gaussian Thresholding',
                "Otsu's Thresholding"
                ]
#        if position == 'right':
#            plt.subplot(3,3,1),plt.imshow(eye_frame_gray,'gray')
#            plt.title("Original image"), plt.xticks([]), plt.yticks([])
#            plt.subplot(3,3,2),plt.hist(eye_frame_gray.ravel(),256)
#            plt.title("Histogram"), plt.xticks([]), plt.yticks([])
#            for i in range(len(ths)):
#                plt.subplot(3,3,i+3),plt.imshow(ths[i],'gray')
#                plt.title(titles[i]), plt.xticks([]), plt.yticks([])
#
#            plt.draw()
#            plt.pause(0.001)
#            plt.clf()


        threshold = th_global

        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        for cnt in contours:
#            (x, y, w, h) = cv2.boundingRect(cnt)
            m = cv2.moments(cnt)
            if m['m00'] != 0:
                pupil_center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
                break

        pupil_radius = 5
        if position == "left":
            self.left_pupil = pupil_center
            self.left_pupil_radius = pupil_radius

        if position == "right":
            self.right_pupil = pupil_center
            self.right_pupil_radius = pupil_radius


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
#            eye_frame_gray = cv2.cvtColor(self.left_eye_frame, cv2.COLOR_BGR2GRAY) 
            eye_frame_gray = self.frame_gray[self.left_eye_bb[1]:self.left_eye_bb[1]+self.left_eye_bb[3], self.left_eye_bb[0]:self.left_eye_bb[0]+self.left_eye_bb[2]]

        if position == "right":
#            eye_frame_gray = cv2.cvtColor(self.right_eye_frame, cv2.COLOR_BGR2GRAY) 
            eye_frame_gray = self.frame_gray[self.right_eye_bb[1]:self.right_eye_bb[1]+self.right_eye_bb[3], self.right_eye_bb[0]:self.right_eye_bb[0]+self.right_eye_bb[2]]
        eye_frame_gray = cv2.GaussianBlur(eye_frame_gray, (7, 7), 0)
#        eye_frame_gray = cv2.medianBlur(eye_frame_gray, 7)

#        cv2.imshow('non eq', eye_frame_gray)
#        eye_frame_gray = cv2.equalizeHist(eye_frame_gray)
#        cv2.imshow('eq', eye_frame_gray)

        frame_height = np.size(eye_frame_gray, 0)
        frame_width = np.size(eye_frame_gray, 1)

        edged = cv2.Canny(eye_frame_gray, 100, 200) 
        cv2.imshow('canny', edged)

#        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, 1,param1=100, param2=1, minRadius=1, maxRadius=20) 
#        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, int(frame_width/2), param1=200, param2=1, minRadius=5, maxRadius=int(frame_width/3)) 

        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, int(frame_width),param1=200, param2=1, minRadius=5, maxRadius=int(frame_width/4)) 

#        print("Circles:", circles)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            c = circles[0][0]
            iris_center = (c[0], c[1])
            iris_radius = c[2]


#        #draw circles debug
#        if position == "right" and circles is not None:
#
#            for cir in circles[0]:
#                print(cir)
#                x , y = cir[0], cir[1]
#                r = cir[2]
#
#                cv2.circle(self.right_eye_frame, (x, y), r, (0, 0, 255), 1)
#            cv2.circle(self.right_eye_frame, iris_center, iris_radius, (0, 255, 0), 1)
#            cv2.imshow('cirles',self.right_eye_frame)


        if position == "left":
            if iris_center != None and iris_radius != None:
                self.left_iris_detected = True
            self.left_iris = iris_center
            self.left_pupil = iris_center
            self.left_iris_radius = iris_radius

        if position == "right":
            if iris_center != None and iris_radius != None:
                self.right_iris_detected = True
            self.right_iris = iris_center
            self.right_pupil = iris_center
            self.right_iris_radius = iris_radius


    def _extract_purkinje(self, position):
        """
        Extract from the eye frame the coordinates of the purkinje image (x, y) 
        w.r.t the eye frame and the pupil radius in pixels 
        """

        purkinje = None

        if position == "left":
            iris_center = self.left_iris
            iris_radius = self.left_iris_radius
#            eye_frame_gray = cv2.cvtColor(self.left_eye_frame, cv2.COLOR_BGR2GRAY) 
            eye_frame_gray = self.frame_gray[self.left_eye_bb[1]:self.left_eye_bb[1]+self.left_eye_bb[3], self.left_eye_bb[0]:self.left_eye_bb[0]+self.left_eye_bb[2]]
#            print(self.left_iris_radius)
#            print(self.left_iris)

#            x = self.left_iris[0] - self.left_iris_radius
#            y = self.left_iris[1] - self.left_iris_radius
#            w = self.left_iris_radius * 2
#            h = self.left_iris_radius * 2
#            iris_frame_gray = eye_frame_gray[y:y+h, x:x+w]
##            print(iris_frame_gray.shape)
#            if iris_frame_gray.shape[0] == 0 or iris_frame_gray.shape[1] == 0:
#                x = 0
#                y = 0
#                iris_frame_gray = eye_frame_gray
#            cv2.imshow('iris left', iris_frame_gray)

        if position == "right":
            iris_center = self.right_iris
            iris_radius = self.right_iris_radius
#            eye_frame_gray = cv2.cvtColor(self.right_eye_frame, cv2.COLOR_BGR2GRAY) 
            eye_frame_gray = self.frame_gray[self.right_eye_bb[1]:self.right_eye_bb[1]+self.right_eye_bb[3], self.right_eye_bb[0]:self.right_eye_bb[0]+self.right_eye_bb[2]]
#            x = self.right_iris[0] - self.right_iris_radius
#            y = self.right_iris[1] - self.right_iris_radius
#            w = self.right_iris_radius * 2
#            h = self.right_iris_radius * 2
#            iris_frame_gray = eye_frame_gray[y:y+h, x:x+w]
#            if iris_frame_gray.shape[0] == 0 or iris_frame_gray.shape[1] == 0:
#                x = 0
#                y = 0
#                iris_frame_gray = eye_frame_gray
#            cv2.imshow('iris right', iris_frame_gray)

        x = iris_center[0] - iris_radius
        y = iris_center[1] - iris_radius
        w = iris_radius * 2
        h = iris_radius * 2
        iris_frame_gray = eye_frame_gray[y:y+h, x:x+w]
        if iris_frame_gray.shape[0] == 0 or iris_frame_gray.shape[1] == 0:
            x = 0
            y = 0
            iris_frame_gray = eye_frame_gray
            if position == "left":
                self.left_purkinje = purkinje

            if position == "right":
                self.right_purkinje = purkinje
            return 
        
#        iris_frame_gray = cv2.equalizeHist(iris_frame_gray)

        iris_frame_gray = cv2.GaussianBlur(iris_frame_gray, (7, 7), 0)
        if position == 'right':
            cv2.imshow('iris right', iris_frame_gray)

#        # Iterative global thresholding
#        th = 255
#        count = 0
#        founded = False
#        while not founded and th > 127:
#            _, th_global = cv2.threshold(iris_frame_gray, th, 255, cv2.THRESH_BINARY)
#            count = np.count_nonzero(th_global)
#            th-=1
#            if count <= 0:
#                continue
#
#            contours, _ = cv2.findContours(th_global, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
#            contours = sorted(contours, key=lambda x: cv2.contourArea(x))
#            all_purkinjes = []
#            for cnt in contours:
##                cnt = cv2.convexHull(cnt)
#                m = cv2.moments(cnt)
#                if m['m00'] != 0:
#                    purkinje = (x + int(m['m10'] / m['m00']), y + int(m['m01'] / m['m00']))
#
#                    square_dist = pow(purkinje[0] - iris_center[0], 2) + pow(purkinje[1] - iris_center[1], 2)
#                    all_purkinjes.append([purkinje, square_dist])
#            if all_purkinjes:
#                all_purkinjes.sort(key=lambda x: x[1])
#                for p in all_purkinjes:
#                    purkinje = p[0]
#                    if pow(purkinje[0] - iris_center[0], 2) + pow(purkinje[1] - iris_center[1], 2) < pow(iris_radius, 2):
#                        founded = True
#                        break
#
#
#        if position == "right":
#            cv2.imshow("th",th_global)



        # Canny
        edged = cv2.Canny(iris_frame_gray, 100, 200) 

#        cv2.imshow("canny purk",edged)


        num_labels, labels_image = cv2.connectedComponents(edged, connectivity=8)
        frame_height = np.size(eye_frame_gray, 0)
        frame_width = np.size(eye_frame_gray, 1)
        distance = 100
        for l in range(1,num_labels+1):
            image = np.copy(edged)
            image[labels_image == l] = 255
            image[labels_image != l] = 0

            contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x))
            
            for cnt in contours:

                cnt = cv2.convexHull(cnt)
                area = cv2.contourArea(cnt)

                if area == 0 or area > 500:
                    continue


                circumference = cv2.arcLength(cnt, True)
                circularity = circumference ** 2 / (4*math.pi*area)

#                print("circularity image {} = {}".format(l,circularity))

                if circularity < 0.3 and circularity > 1.7:
                    continue

                if abs(circularity - 1) < distance:
                    purkinje = None
#                    print("dist", abs(circularity -1))
                    distance = abs(circularity - 1)
                    m = cv2.moments(cnt)
                    if m['m00'] != 0:
                        purkinje = (x + int(m['m10'] / m['m00']), y + int(m['m01'] / m['m00']))

#                    if position == "right":
#                        x_b,y_b,w,h = cv2.boundingRect(cnt)
#                        cv2.rectangle(self.right_eye_frame,(x+x_b,y+x_b),(x+x_b+w,y_b+y+h),(0,255,0),1)


        if position == "right":
            cv2.imshow('rect',self.right_eye_frame)


#            cv2.imshow('image'+str(l), image)

        if position == "left":
            self.left_purkinje = purkinje

        if position == "right":
            self.right_purkinje = purkinje

