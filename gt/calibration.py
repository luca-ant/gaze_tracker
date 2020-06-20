import cv2
import time
import collections
import numpy as np
from sklearn.linear_model import LinearRegression

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

class Calibration:
    
    def __init__(self):
        self.reg = LinearRegression()
        self.x_reg = LinearRegression()
        self.y_reg = LinearRegression()
        self.t_x = None
        self.t_y = None
        self.m_x = None
        self.m_y = None

    def update(self, data):

        # pair-wise linear regression

        all_v = np.empty((0,2))
        all_p = np.empty((0,2))
        for point, vectors in data.items():
            v = np.array(vectors)
            mean = np.mean(v, axis=0)
            std = np.std(v, axis=0)
            filtered_v = [ [v[0], v[1]] for v in vectors if v[0] > mean[0] - 2 * std[0] and v[1] > mean[1] - 2 * std[1] ]
            filtered_v = [ [v[0], v[1]] for v in filtered_v if v[0] < mean[0] + 2 * std[0] and v[1] < mean[1] + 2 * std[1] ]

            v = np.array(filtered_v)
            p = np.full(v.shape, [point])

            all_v = np.concatenate((all_v, v))
            all_p = np.concatenate((all_p, p))

        self.reg.fit(all_v, all_p)
        print("SCORE: {}".format(self.reg.score(all_v, all_p)))
        print("COEFF: {}".format(self.reg.coef_))



#        # linear regression
#        all_x_v = np.empty((0,1))
#        all_y_v = np.empty((0,1))
#        all_x_p = np.empty((0,1))
#        all_y_p = np.empty((0,1))
#        for point, vectors in data.items():
#
#            v = np.array(vectors)
#            mean = np.mean(v, axis=0)
#            std = np.std(v, axis=0)
#            filtered_v = [ [v[0], v[1]] for v in vectors if v[0] > mean[0] - 2 * std[0] and v[1] > mean[1] - 2 * std[1] ]
#            filtered_v = [ [v[0], v[1]] for v in filtered_v if v[0] < mean[0] + 2 * std[0] and v[1] < mean[1] + 2 * std[1] ]
#
#            x_v = np.array([ [v[0]] for v in filtered_v ])
#            y_v = np.array([ [v[1]] for v in filtered_v ])
#
#            x_p = np.full(x_v.shape, [point[0]])
#            y_p = np.full(y_v.shape, [point[1]])
#
#            all_x_v = np.concatenate((all_x_v, x_v))   
#            all_y_v = np.concatenate((all_y_v, y_v))   
#            all_x_p = np.concatenate((all_x_p, x_p))   
#            all_y_p = np.concatenate((all_y_p, y_p))   
#
#        self.x_reg.fit(all_x_v, all_x_p)
#        self.y_reg.fit(all_y_v, all_y_p)
#
#        print("SCORE X: {}".format(self.x_reg.score(all_x_v, all_x_p)))
#        print("COEFF X: {}".format(self.x_reg.coef_))
#        print("SCORE Y: {}".format(self.y_reg.score(all_y_v, all_y_p)))
#        print("COEFF Y: {}".format(self.y_reg.coef_))


        # manual linear regression
#        self.t_x = list(data.keys())[0][0]
#        self.t_y = list(data.keys())[0][1]
#        m_x = []
#        m_y = []
#
#        for point, vectors in data.items():
#            mean = np.mean(vectors, axis=0)
#            m_x.append(point[0] / median[0])
#            m_y.append(point[1] / median[1])
#        self.m_x = sum(m_x) / len(m_x)
#        self.m_y = sum(m_y) / len(m_y)


    def compute(self, vector):
        # pair-wise linear regression
        np_vector = np.array([vector])
        np_gaze = self.reg.predict(np_vector)
        output = (int(np_gaze[0][0]), int(np_gaze[0][1]))


        # linear regression
#        x_vector = np.array([[vector[0]]])
#        x_out = self.x_reg.predict(x_vector)
#        y_vector = np.array([[vector[1]]])
#        y_out = self.y_reg.predict(y_vector)
#        output = (int(x_out[0][0]), int(y_out[0][0]))


        # manual linear regression
#        x_v = vector[0]
#        y_v = vector[1]
#        output = (int(self.m_x * x_v + self.t_x), int(self.m_y * y_v + self.t_y))

        return output

def calibrate(camera, screen, gaze_tracker):
    N_REQ_VECTORS = 50
    N_SKIP_VECTORS = 25

    screen.clean()

    calibration = Calibration()
    calibration_points = calculate_points(screen)

    vectors = collections.defaultdict(list)

    completed = False
    enough = 0
    skip = 0

    point = calibration_points.pop(0) 

    screen.draw(point)
    screen.show()
    while point:
        screen.draw(point)
        screen.show()

        _, frame = camera.read() 

        start = time.time()

        gaze_tracker.update(frame)

        end = time.time()

        print("TIME: {:.3f} ms".format(end*1000 - start*1000))

        cv2.namedWindow("frame")
        dec_frame = gaze_tracker.eye_tracker.decorate_frame()
        dec_frame = cv2.resize(dec_frame,(int(FRAME_WIDTH / 2), int(FRAME_HEIGHT / 2)))
        cv2.moveWindow("frame", 0 , 0)
        cv2.imshow('frame', dec_frame)

        vector = gaze_tracker.get_vector()
        print("VECTOR: {}\tPOINT: {}".format(vector, point))

        if vector and skip < N_SKIP_VECTORS:
            skip += 1
            continue


        if vector:
            vectors[point].append(vector)
            enough += 1

#        print(vectors)

        progress = len(vectors[point]) / N_REQ_VECTORS
        screen.draw(point, progress=progress)
        screen.show()

        # netx point condition
        if enough >= N_REQ_VECTORS and len(calibration_points) > 0:
            point = calibration_points.pop(0)
#            screen.clean()
            skip = 0
            enough = 0
            screen.draw(point)
            screen.show()

        # end calibration condition
        if enough >= N_REQ_VECTORS and len(calibration_points) == 0:
            screen.clean()
            completed = True
            break


        k = cv2.waitKey(1) & 0xff
        if k == 1048603 or k == 27: # esc to terminate calibration
            screen.mode = "normal"
            screen.clean()
            screen.show()
            break
#        if k == ord('n'): # n to next calibration step
##            screen.clean()
#            skip = 0
#            enough = 0
#            if len(calibration_points) == 0:
#                completed = True
#                break
#            point = calibration_points.pop(0)

    if completed:
        calibration.update(vectors)
        gaze_tracker.calibration = calibration
    screen.mode = "normal"

def calculate_points(screen):
    points = []

    # center
    p = (int(0.5 * screen.width), int(0.5 * screen.height))
    points.append(p)

    # top left
    p = (int(0.05 * screen.width), int(0.05 * screen.height))
    points.append(p)

    # top
    p = (int(0.5 * screen.width), int(0.05 * screen.height))
    points.append(p)

    # top right
    p = (int(0.95 * screen.width), int(0.05 * screen.height))
    points.append(p)

    # left
    p = (int(0.05 * screen.width), int(0.5 * screen.height))
    points.append(p)

    # right
    p = (int(0.95 * screen.width), int(0.5 * screen.height))
    points.append(p)

    # bottom left
    p = (int(0.05 * screen.width), int(0.95 * screen.height))
    points.append(p)

    # bottom
    p = (int(0.5 * screen.width), int(0.95 * screen.height))
    points.append(p)

    # bottom right
    p = (int(0.95 * screen.width), int(0.95 * screen.height))
    points.append(p)

    return points


