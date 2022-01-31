import cv2
import numpy as np
import torch
from myModel import myModel

Padding = 50


class detector(myModel):
    """
    A wrapper for the yolo v5 model that i trained, but i might change it later.
    this class should make it easy for me to change this model relatively easy
    """

    def __init__(self):
        # initiate detection model
        self.model = torch.hub.load("yolov5", "custom",
                                    path="weights/weights.pt",
                                    force_reload=True, source="local")

    def process(self, image):
        out = self.model(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        bbox_test = out.pandas().xyxy[0]
        # print(bbox_test)
        if bbox_test.size > 0:
            max_conf = np.argmax(bbox_test["confidence"])
            # print(max_conf)
            best_box_row = bbox_test.iloc[max_conf]
            # print(best_box_row)
            width = int(best_box_row["xmax"] - best_box_row["xmin"])
            height = int(best_box_row["ymax"] - best_box_row["ymin"])
            return int(best_box_row["xmin"]) - Padding, int(
                best_box_row["ymin"]) - Padding, width + Padding, height + Padding
        else:
            print("No hands detected")
            return None
