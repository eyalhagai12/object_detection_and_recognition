import cv2
import pyautogui
from ObjectronModule import Objectron
import numpy as np

# Initiate the objectron model
objectron = Objectron(model_name="Cup")

while True:
    img = pyautogui.screenshot(region=(160, 160, 1035, 720))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    objectron.process(img)

    # show
    cv2.imshow("Image", img)
    cv2.waitKey(1)
