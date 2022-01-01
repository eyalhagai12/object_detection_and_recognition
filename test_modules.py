import cv2
from ObjectronModule import Objectron

cap = cv2.VideoCapture(0)

pose_detector = Objectron(model_name="Chair")

while True:
    success, img = cap.read()

    pose_detector.process(img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

