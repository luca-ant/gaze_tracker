class Eye:
    """
    Model class for an eye
    Attributes:
        image: eye ROI of original image
        position: string to identify left or right eye
        pupil_center: a tuple (x, y) with the coordinates of the center of the pupil w.r.t the eye ROI image
        pupil_radius: radius of the pupil in pixels
        iris_radius: radius of the iris in pixels
        purkinje: a tuple (x, y) with the coordinates of the center of the purkinje image w.r.t the eye ROI image
        landmarks: list of landmarks of the eye
    Methods:
    """
    def __init__(self, frame, position, pupil_center, pupil_radius, iris_radius, purkinje):
        self.frame = frame
        self.position = position
        self.pupil_center = pupil_center
        self.pupil_radius = pupil_radius
        self.iris_radius = iris_radius
        self.purkinje = purkinje

    def __str__(self):
        return "Eye: {}\n\tPupil center: {}\n\tPupil radius: {}\n\tIris radius: {}\n\tPurkinje: {}\n".format(self.position, self.pupil_center, self.pupil_radius, self.iris_radius, self.purkinje)

