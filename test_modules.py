import cv2
from ObjectronModule import Objectron

cap = cv2.VideoCapture(0)

objectron = Objectron(model_name="Chair")

while True:
    success, img = cap.read()

    objectron.process(img)

    cv2.imshow("Frame", img)
    cv2.waitKey(1)

