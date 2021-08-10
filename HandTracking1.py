import cv2
from HandTrackingModule import HandTracker

# get feed from the camera
cap = cv2.VideoCapture(0)

# initiate a hand detection model
hand_detector = HandTracker()

# show the feed
while True:
    # get the image from a frame
    success, img = cap.read()

    # detect ands in img
    hand_detector.detect_hands(img)

    # get some points
    point = hand_detector.get_finger(9, False)
    point2 = hand_detector.get_finger(12, False)

    distances = get_dist_from_fingers(point, point2)

    # draw circle around the hand
    if len(point) > 0:
        for p, data in zip(point, distances):
            x = int(p.x)
            y = int(p.y)
            pos1, pos2, dist = data

            center = (x, y)
            color = (0, 0, 0)
            radius = int(dist + 50)

            cv2.circle(img, center, radius, color, 3)

    # show the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)


