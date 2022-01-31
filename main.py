from Controller import controller
from Detector import detector
from Recognizer import recognizer
from TrackerAlgo import TrackerAlgo
from Tracker import tracker
from viewer import viewer

if __name__ == '__main__':
    # initiate tracker
    my_tracker = tracker(TrackerAlgo.MOSSE)

    # initiate detector
    my_detector = detector()

    # initiate sign language recognizer
    sign_lang_model = recognizer()

    # initiate viewer
    view = viewer()

    # initiate controller
    control = controller(my_detector, my_tracker, sign_lang_model, view)

    # set the viewers controller
    view.set_controller(control)

    # initiate the program
    control.initiate()
