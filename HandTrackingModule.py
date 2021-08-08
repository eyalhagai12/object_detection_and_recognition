import math
import cv2
import mediapipe as mp


class HandTracker():
    def __init__(self, image_mode=False, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(image_mode, max_hands, min_detection_confidence, min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.detect = None
        self.img = None
        self.h = 0
        self.w = 0
        self.c = 3

    def detect_hands(self, img, draw=True):
        """
        Detect hands in an image and draw points and connections

        :param img: the image
        :param draw: True by default, set to false if you don't want to draw points and connections
        """

        # convert from bgr to rgb
        self.img = img
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.h, self.w, self.c = self.img.shape

        # create an object to save the processed image
        self.detect = self.hands.process(img_rgb)

        # draw the hands
        if draw:
            hands = self.detect.multi_hand_landmarks
            if hands:
                for hand in hands:
                    self.mpDraw.draw_landmarks(self.img, hand)

    def get_finger(self, hand_location_index=0, draw=True):
        """
        Get a finger and draw a point to show which part is it

        :param hand_location_index: the index of the part of the hand
        :return: list with the co-ordinates of the part for each hand in the picture
        """

        # get all the hands in the picture
        hands = self.detect.multi_hand_landmarks

        # create an empty list of fingers
        fingers = []

        # if there are hands in the frame show it and draw a point for them
        if hands:
            # if found hands go over each hand and check the
            for hand in hands:
                finger = hand.landmark[hand_location_index]
                fingers.append(finger)

                # turn the parameters from percentages to pixels
                finger.x = int(finger.x * self.w)
                finger.y = int(finger.y * self.h)

                # draw custom point for the fingers
                if draw:
                    pos = (int(finger.x), int(finger.y))
                    color = (255, 0, 0)
                    cv2.circle(self.img, pos, 10, color, cv2.FILLED)

        return fingers

    def get_dist_from_fingers(self, fingers_1, fingers_2):
        """
        Get the distance between two fingers and return a list that contains tuples that represent
        (finger_1 position, finger_2 position, distance)

        :param fingers_1: the first finger
        :param fingers_2: the second finger
        :return: a list with tuples in the shape (finger_1 position, finger_2 position, distance)
        """
        result = []
        for finger1, finger2 in zip(fingers_1, fingers_2):
            finger2_pos = (int(finger2.x), int(finger2.y))
            finger1_pos = (int(finger1.x), int(finger1.y))
            dist = math.dist(finger1_pos, finger2_pos)
            result.append((finger1_pos, finger2_pos, dist))

        return result


def main():
    # capture from the webcam
    cap = cv2.VideoCapture(0)

    # create a hand tracker object
    hand = HandTracker()

    # show the feed
    while True:
        success, img = cap.read()
        hand.detect_hands(img, False)
        index_fingers = hand.get_finger(8)
        thumbs = hand.get_finger(4)

        for thumb, index_finger in zip(thumbs, index_fingers):
            thumb_pos = (int(thumb.x), int(thumb.y))
            index_finger_pos = (int(index_finger.x), int(index_finger.y))

            # calculate the distance
            dist = math.dist(thumb_pos, index_finger_pos)

            if dist < 35:
                print("Thin")

            if dist > 120:
                print("Wide")

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()



