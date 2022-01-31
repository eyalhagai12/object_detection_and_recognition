import cv2
import sys
import numpy as np
import torch
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

Padding = 10


def get_model(name):
    # Set up tracker
    track_algo = None

    if name == 'KCF':
        track_algo = cv2.TrackerKCF_create()
    if name == 'TLD':
        track_algo = cv2.legacy_TrackerTLD().create()
    if name == 'MOSSE':
        track_algo = cv2.legacy_TrackerMOSSE().create()
    if name == "CSRT":
        track_algo = cv2.TrackerCSRT_create()
    if name == "MIL":
        track_algo = cv2.legacy_TrackerMIL().create()

    return track_algo


def get_bbox(detection_model):
    out = detection_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    bbox_test = out.pandas().xyxy[0]
    # print(bbox_test)
    if bbox_test.size > 0:
        max_conf = np.argmax(bbox_test["confidence"])
        # print(max_conf)
        best_box_row = bbox_test.iloc[max_conf]
        # print(best_box_row)
        width = int(best_box_row["xmax"] - best_box_row["xmin"])
        height = int(best_box_row["ymax"] - best_box_row["ymin"])
        return int(best_box_row["xmin"]) - Padding, int(best_box_row["ymin"]) - Padding, width + Padding, height + Padding
    else:
        print("No hands detected")
        return None


def get_sign_lang_model():
    sign_model = keras.models.load_model("weights/best_sign_lang_model/best_sign_lang_model")
    # sign_model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="adam", metrics=["accuracy"])
    return sign_model


if __name__ == '__main__':

    # Read video
    video = cv2.VideoCapture(0)
    tracker = get_model("MOSSE")

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # initiate model
    model = torch.hub.load("yolov5", "custom",
                           path="weights/weights.pt",
                           force_reload=True, source="local")

    sign_lang_model = get_sign_lang_model()

    labels = [chr(ord("A") + i) for i in range(26)] + ["Del", "Space", "Nothing"]

    # detect hands to start
    bbox = None

    while bbox is None:
        ok, frame = video.read()
        bbox = get_bbox(model)
        cv2.imshow("Init state", frame)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)c
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            # view hand crop
            c_image = frame[p1[1]:p2[1], p1[0]:p2[0]]
            c_image = cv2.cvtColor(c_image, cv2.COLOR_BGR2GRAY)
            c_image = cv2.resize(c_image, (256, 256))
            out = sign_lang_model.predict(np.expand_dims(np.expand_dims(c_image, -1), 0))
            cv2.putText(frame, "Label : " + labels[out.argmax()], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (50, 170, 50), 2)
            print("Predicted Label: {}".format(labels[out.argmax()]))
            try:
                cv2.imshow("Crop", c_image)
            except:
                print("Error showing crop")
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
