import cv2


class controller:
    """
    This class is used to enable the communication between the viewer and the models
    """

    def __init__(self, detector, tracker, recognizer, viewer):
        self.viewer = viewer
        self.recognizer = recognizer
        self.tracker = tracker
        self.detector = detector

    def get_tracker_bbox(self, frame):
        """
        Update the tracker bounding box

        :param frame: current frame
        :return: a boolean indicating success or failure, and p1 and p2, top left and bottom right points of the bounding box respectivley
        """
        return self.tracker.process(frame)

    def process_sign(self, image):
        """
        Gets the cropped image of the hand and returns the prediction of what letter is the hand doing (in sign language)

        :param image: the cropped image of the hand
        :return: the index of the correct label
        """
        return self.recognizer.process(image)

    def initiate(self):
        """
        Try to detect a hand for the first time then initiate the tracker and start the program properly
        """
        video = cv2.VideoCapture(-1)
        bbox = None

        while bbox is None:
            ok, frame = video.read()
            bbox = self.detector.process(frame)
            print("Trying to detect hands...")

        print("detected hand!")
        self.tracker.set_init_bbox(frame, bbox)
        video.release()
        self.viewer.view()


