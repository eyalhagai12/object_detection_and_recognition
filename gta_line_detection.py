import math

import cv2
import numpy as np
from PIL import ImageGrab, Image
import time


def calc_gradient(line):
    x1, y1, x2, y2 = line
    gradient = (y2 - y1) / (x2 - x1)
    return gradient


def get_dist_from_center(line, center):
    p1 = (line[0], line[1])
    p2 = (line[2], line[3])

    x1, y1 = p1
    x2, y2 = p2
    x, y = center

    distance = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def process(image):
    y, x, z = image.shape  # get x, y, z from the shape of the image
    center_point = (int(x / 2), int(2 * y / 3))

    colors = {
        "left": (255, 0, 0),
        "right": (0, 255, 0)
    }

    points = np.array([[-75, 490], [215, 270], [645, 270], [915, 490]])
    img_copy = image.copy()

    # turn to gray scale
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # detect edges
    edges = cv2.Canny(gray, 100, 200)

    # smooth the picture
    smooth = cv2.GaussianBlur(edges, (9, 9), 0)

    # create mask
    mask = np.zeros_like(smooth)
    mask = cv2.fillPoly(mask, [points], (255, 255, 255))

    # apply mask
    result = cv2.bitwise_and(smooth, mask)

    # find lines
    lines = cv2.HoughLinesP(result, 1, np.pi / 180, 120, minLineLength=80)

    # draw lines
    if lines is not None:
        left = []
        right = []

        for line in lines:
            coords = line[0]

            threshold = 0.45

            gradient = calc_gradient(coords)
            if gradient > threshold or gradient < -threshold:
                p1 = (coords[0], coords[1])
                p2 = (coords[2], coords[3])

                p = p1 if p1[1] > p2[1] else p2

                if p[0] < int(x / 2):  # lines is from the left
                    left.append(coords)
                    # img_copy = cv2.line(img_copy, p1, p2, colors["left"], 15)

                else:  # lines is from the left
                    right.append(coords)
                    # img_copy = cv2.line(img_copy, p1, p2, colors["right"], 15)

        if len(left) > 0:
            left_index = 0
            for idx, line in enumerate(left):
                if get_dist_from_center(line, center_point) < get_dist_from_center(left[left_index], center_point):
                    left_index = idx

            left_line = left[left_index]

            # draw left line
            p1 = (left_line[0], left_line[1])
            p2 = (left_line[2], left_line[3])

            img_copy = cv2.line(img_copy, p1, p2, (0, 255, 0), 15)

        if len(right) > 0:
            right_index = 0
            for idx, line in enumerate(right):
                if get_dist_from_center(line, center_point) < get_dist_from_center(right[right_index], center_point):
                    right_index = idx

            right_line = right[right_index]

            # draw right line
            p1 = (right_line[0], right_line[1])
            p2 = (right_line[2], right_line[3])

            img_copy = cv2.line(img_copy, p1, p2, (0, 255, 0), 15)

    # # draw a line in the center
    # p1 = (int(x / 2), 0)
    # p2 = (int(x / 2), y)
    # img_copy = cv2.line(img_copy, p1, p2, (0, 0, 255), 1)
    #
    # # draw a point in from which the distance is calculated
    # img_copy = cv2.circle(img_copy, center_point, 3, (255, 0, 0), 2)

    return img_copy


while True:
    # keep time
    start = time.time()

    # capture image
    img = ImageGrab.grab(bbox=(460, 160, 1320, 650))  # x, y, w, h
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # process
    processed_image = process(frame)

    # end process time
    end = time.time()

    print("Detection Time: {}".format(end - start))

    cv2.imshow("frame", processed_image)
    if cv2.waitKey(1) & 0Xff == ord('q'):
        break

cv2.destroyAllWindows()
