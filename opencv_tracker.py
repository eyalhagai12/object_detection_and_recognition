import cv2
import sys
import numpy as np
import torch
import pandas as pd


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


if __name__ == '__main__':

    # Read video
    video = cv2.VideoCapture(0)
    tracker = get_model("MOSSE")

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    model = torch.hub.load("yolov5", "custom",
                           path="weights/weights.pt",
                           force_reload=True, source="local")

    # detect hands in first frame
    bbox = (0, 0, 100, 100)
    out = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    bbox_test = out.pandas().xyxy[0]
    print(bbox_test)
    if bbox_test.size > 0:
        max_conf = np.argmax(bbox_test["confidence"])
        print(max_conf)
        best_box_row = bbox_test.iloc[max_conf]
        print(best_box_row)
        width = int(best_box_row["xmax"] - best_box_row["xmin"])
        height = int(best_box_row["ymax"] - best_box_row["ymin"])
        bbox = (int(best_box_row["xmin"]), int(best_box_row["ymin"]), width, height)
    else:
        print("No hands detected")

    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)

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
            c_image = frame[p1[1]:p2[1], p1[0]:p2[0]]
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
