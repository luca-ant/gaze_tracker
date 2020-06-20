class Eye:

    def __init__(self, frame, position, pupil_center, pupil_radius, iris_radius, purkinje):
        self.frame = frame
        self.position = position
        self.pupil_center = pupil_center
        self.pupil_radius = pupil_radius
        self.iris_radius = iris_radius
        self.purkinje = purkinje

    def __str__(self):
        return "Eye: {}\n\tPupil center: {}\n\tPupil radius: {}\n\tIris radius: {}\n\tPurkinje: {}\n".format(self.position, self.pupil_center, self.pupil_radius, self.iris_radius, self.purkinje)

