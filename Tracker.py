from TrackerAlgo import TrackerAlgo
import cv2

from myModel import myModel


def get_model(algo: TrackerAlgo):
    # Set up tracker
    track_algo = None

    if algo == TrackerAlgo.KCF:
        track_algo = cv2.TrackerKCF_create()
    if algo == TrackerAlgo.TLD:
        track_algo = cv2.legacy_TrackerTLD().create()
    if algo == TrackerAlgo.MOSSE:
        track_algo = cv2.legacy_TrackerMOSSE().create()
    if algo == TrackerAlgo.CSRT:
        track_algo = cv2.TrackerCSRT_create()
    if algo == TrackerAlgo.MIL:
        track_algo = cv2.legacy_TrackerMIL().create()
    if algo == TrackerAlgo.GOTURN:
        track_algo = cv2.TrackerGOTURN_create()

    return track_algo


class tracker(myModel):
    """
    An algorithm to track an unknown object
    """

    def __init__(self, algo: TrackerAlgo):
        self.model = get_model(algo)

    def set_init_bbox(self, frame, bbox):
        """
        Set up the initial bounding box of the tracker
        :param frame: the frame on which to initiate the bounding box
        :param bbox: the initial bounding box
        :return: True of successful, else False
        """
        return self.model.init(frame, bbox)

    def process(self, image):
        ok, bbox = self.model.update(image)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        else:
            p1 = None
            p2 = None

        return ok, p1, p2
