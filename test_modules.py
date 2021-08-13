import cv2
from PoseDetectorModule import PoseDetector

cap = cv2.VideoCapture(0)

pose_detector = PoseDetector()

while True:
    success, img = cap.read()

    pose_detector.process(img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

