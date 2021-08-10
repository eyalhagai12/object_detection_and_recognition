import cv2
from PoseDetectorModule import PoseDetector

capture = cv2.VideoCapture(0)

pose_detector = PoseDetector()

while True:
    success, img = capture.read()

    pose_detector.process(img)
    pose_detector.get_point(31)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
