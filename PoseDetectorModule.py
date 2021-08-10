import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.model = mp.solutions.pose.Pose(static_image_mode=static_image_mode, model_complexity=model_complexity,
                                            smooth_landmarks=smooth_landmarks,
                                            min_detection_confidence=min_detection_confidence,
                                            min_tracking_confidence=min_tracking_confidence)
        self.draw_tools = mp.solutions.drawing_utils
        self.img = None
        self.img_width = 0
        self.img_height = 0
        self.img_channels = 3
        self.processedImage = None

    def process(self, img, draw=True):
        """
        Detect the pose and landmarks

        :param img: the image to process
        :param draw: if to draw the landmarks in the image or not
        """

        # save the image
        self.img = img

        # save width, height, channels
        self.img_height, self.img_width, self.img_channels = img.shape

        # convert image to rgb from bgr
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # process the image
        self.processedImage = self.model.process(img_rgb)

        # draw if draw = true
        if draw:
            self.draw_tools.draw_landmarks(img, self.processedImage.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    def get_point(self, index, draw=True):
        """
        Get the positions of a single point

        :param draw: set tot false if yo don't want to draw on this point
        :param index: the index of the point to get
        :return:
        """
        if index < 0 or index > 32:
            raise IndexError("Variable 'index' must be between 0 and 32")
        if self.img is None:
            raise Exception("Must use 'process' function before using 'get_point'")

        if self.processedImage.pose_landmarks is not None:
            # get landmarks
            landmarks = self.processedImage.pose_landmarks.landmark

            # get the wanted point
            point_location = landmarks[index]

            # turn them to pixels instead of percentages
            point_location.x = point_location.x * self.img_width
            point_location.y = point_location.y * self.img_height

            if draw:
                # get the center for drawing
                center = (int(point_location.x), int(point_location.y))
                color = (255, 0, 0)  # the drawing color

                # draw the circle on the image
                cv2.circle(self.img, center, 5, color, cv2.FILLED)

            return point_location


def main():
    cap = cv2.VideoCapture(0)
    pose_detector = PoseDetector()

    while True:
        success, img = cap.read()

        pose_detector.process(img)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
