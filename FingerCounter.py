import cv2
from HandTrackingModule import HandTracker

# capture the video from the webcam
cap = cv2.VideoCapture(0)

# initiate a hand tracker
ht = HandTracker()

while True:
    # get image from webcam
    success, img = cap.read()

    # detect hands
    ht.detect_hands(img)

    # get some parts (a list of the part for each hand in the picture)
    f1 = ht.get_finger(8, False)
    f1_bottom = ht.get_finger(5, False)

    f2 = ht.get_finger(12, False)
    f2_bottom = ht.get_finger(9, False)

    f3 = ht.get_finger(16, False)
    f3_bottom = ht.get_finger(13, False)

    f4 = ht.get_finger(20, False)
    f4_bottom = ht.get_finger(17, False)

    f5 = ht.get_finger(4, False)

    num = 0

    # if there is a hand in the image get the information
    if len(f1) > 0:
        # get positions and distances from the wanted points
        f1_pos, f1_bottom_pos, dist1 = ht.get_dist_from_fingers(f1, f1_bottom)[0]
        f2_pos, f2_bottom_pos, dist2 = ht.get_dist_from_fingers(f2, f2_bottom)[0]
        f3_pos, f3_bottom_pos, dist3 = ht.get_dist_from_fingers(f3, f3_bottom)[0]
        f4_pos, f4_bottom_pos, dist4 = ht.get_dist_from_fingers(f4, f4_bottom)[0]
        f5_pos, f5_bottom_pos, dist5 = ht.get_dist_from_fingers(f5, f3_bottom)[0]

        # make a list of the distances
        distances = [dist1, dist2, dist3, dist4, dist5]

        # add one to num for every distance that is bigger than a certain amount
        for dist in distances:
            if dist > 40:
                num += 1

    print(num)

    # detect hands in the picture
    cv2.imshow("Image", img)
    cv2.waitKey(1)


